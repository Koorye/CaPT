import torch
from torch.cuda.amp import autocast

from dassl.engine import TRAINER_REGISTRY
from tqdm import tqdm

from ..base_trainers.coop import CustomCLIP as CustomCLIP_, PromptLearner as PromptLearner_, CoOp
from .losses import MarginLoss
from .ucp_base import UnifiedClassSpecificPromptLearner


class PromptLearner(PromptLearner_):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__(cfg, classnames, clip_model)
        self.ucp = UnifiedClassSpecificPromptLearner(cfg, classnames, clip_model)
        
    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        class_prompts = self.ucp()
        ctx = ctx + class_prompts.unsqueeze(1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            return torch.cat([prefix, ctx, suffix], dim=1)
        else:
            raise NotImplementedError


class CustomCLIP(CustomCLIP_):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__(cfg, classnames, clip_model)

        # reset prompt learner
        del self.learnable_params['prompt_learner']
        prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.learnable_params['prompt_learner'] = prompt_learner

        # bulid margin loss
        loss_cfg = cfg.TRAINER.UCP.LOSS
        use_sim = loss_cfg.USE_SIM
        smooth = loss_cfg.SMOOTH

        if use_sim:
            wordvecs = prompt_learner.ucp.class_tokens
            sim = torch.cosine_similarity(wordvecs.float().unsqueeze(1), 
                                          wordvecs.float().unsqueeze(0), dim=-1)
        else:
            sim = None

        self.loss_cls = MarginLoss(len(classnames), sim, smooth)

    def forward(self, image, label=None):
        image_features = self.image_encoder(image.type(self.dtype))

        prompt_learner = self.learnable_params['prompt_learner']
        prompts = prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        if prompt_learner.training:
            return self.loss_cls(logits, label)

        return logits


@TRAINER_REGISTRY.register()
class UCPCoOp(CoOp):
    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        
        prec = self.cfg.TRAINER.PREC
        if prec == "amp":
            with autocast():
                loss = self.model(image, label, self.img_feats)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            loss = self.model(image, label)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def _build_custom_clip(self, cfg, classnames, clip_model):
        if cfg.TRAINER.UCP.VERSION in [1, 2, 3]:
            img_feats, labels = self._forward_img_feats(clip_model)

        self.model = CustomCLIP(cfg, classnames, clip_model)
        ucp = self.model.learnable_params['prompt_learner'].ucp
        trainer_cfg = ucp.trainer_cfg
        dtype = ucp.dtype

        if trainer_cfg.VERSION == 1:
            # class-aware random vector N(0, 1)
            ucp.class_tokens = torch.randn(1, 512).type(dtype) 
        elif trainer_cfg.VERSION == 2:
            # mean of image features (class-agnostic)
            ucp.class_tokens = torch.mean(img_feats, dim=0, keepdim=True).type(dtype) 
        elif trainer_cfg.VERSION == 3:
            # mean of image features (class-specific)
            img_feats_list = []
            for label in labels.unique(sorted=True):
                img_feats_list.append(img_feats[labels == label].mean(dim=0))
                
            ucp.class_tokens = torch.stack(img_feats_list, dim=0).type(dtype) 
    
    @torch.no_grad()
    def _forward_img_feats(self, clip_model):
        ori_device = next(clip_model.parameters()).device
        clip_model = clip_model.to(self.device)

        img_feats_list, labels_list = [], []

        for batch in tqdm(self.train_loader_x, desc='forward image features'):
            imgs, labels = self.parse_batch_train(batch)
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            img_feats = clip_model.encode_image(imgs)
            img_feats_list.append(img_feats)
            labels_list.append(labels)
        
        img_feats = torch.cat(img_feats_list)
        labels = torch.cat(labels_list)
        
        clip_model = clip_model.to(ori_device)
        return img_feats, labels
