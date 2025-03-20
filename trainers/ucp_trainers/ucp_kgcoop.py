import torch

from dassl.engine import TRAINER_REGISTRY

from ..base_trainers.kgcoop import CustomCLIP as CustomCLIP_, PromptLearner as PromptLearner_, KgCoOp
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

        return torch.cat([prefix, ctx, suffix], dim=1)


class CustomCLIP(CustomCLIP_):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__(cfg, classnames, clip_model)
        self.w = cfg.TRAINER.KGCOOP.W

        # reset prompt learner
        del self.learnable_params['prompt_learner']
        prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.learnable_params['prompt_learner'] = prompt_learner
        self.tokenized_prompts = prompt_learner.tokenized_prompts

    def forward(self, image):
        prompt_learner = self.learnable_params['prompt_learner']
        logit_scale = self.logit_scale.exp()
        text_features_old = self.ori_embedding
        tokenized_prompts = self.tokenized_prompts

        # (B, D), (B, H, W, D)
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        prompts = prompt_learner()
        text_features = self.text_encoder(prompts, tokenized_prompts) 
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logits = logit_scale * image_features @ text_features.t()

        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-07)
        text_features_old = text_features_old / text_features_old.norm(dim=-1, keepdim=True)
        score = cos(text_features, text_features_old)
        score = 1.0 - torch.mean(score)

        return logits, score


@TRAINER_REGISTRY.register()
class UCPKgCoOp(KgCoOp):
    def build_model(self):
        super().build_model()

        classnames = self.dm.dataset.classnames

        # bulid margin loss
        loss_cfg = self.cfg.TRAINER.UCP.LOSS
        smooth = loss_cfg.SMOOTH
        wordvecs = self.model.learnable_params['prompt_learner'].ucp.class_tokens
        sim = torch.cosine_similarity(wordvecs.float().unsqueeze(1), 
                                      wordvecs.float().unsqueeze(0), dim=-1)
        self.loss_cls = MarginLoss(len(classnames), sim, smooth)
    
    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.PREC
        
        if prec == "amp":
            raise NotImplementedError
        else:
            output, score = self.model(image)

            # loss_ce = F.cross_entropy(output, label)
            loss_cls = self.loss_cls(output, label)
            loss_kg = self.w * score

            loss = loss_cls + loss_kg
            self.model_backward_and_update(loss)

        loss_summary = {
            'loss_cls': loss_cls.item(),
            'loss_kg': loss_kg.item(),
        }
        loss_summary['loss_total'] = sum(loss_summary.values())

        if (self.batch_idx + 1) == self.num_batches:
            self.sched.step()

        return loss_summary
    
    def _build_custom_clip(self, cfg, classnames, clip_model):
        self.model = CustomCLIP(cfg, classnames, clip_model)
