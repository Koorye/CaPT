import torch
import torch.nn.functional as F

from dassl.engine import TRAINER_REGISTRY
from dassl.metrics import compute_accuracy

from ..base_trainers.dapt import CustomCLIP as CustomCLIP_, PromptLearner as PromptLearner_, DAPT
from .ucp_base import UnifiedClassSpecificPromptLearner
from .losses import MarginLoss


class PromptLearner(PromptLearner_):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__(cfg, classnames, clip_model)
        self.ucp = UnifiedClassSpecificPromptLearner(cfg, classnames, clip_model)

    def forward_txt(self):
        ctx = self.txt_ctx  # [TXT_NUM_TOKENS, dim] = [16, 512] (default)
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        class_prompts = self.ucp.forward()
        ctx = ctx + class_prompts.unsqueeze(1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        return torch.cat([prefix, ctx, suffix], dim=1)


class CustomCLIP(CustomCLIP_):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__(cfg, classnames, clip_model)

        # reset prompt learner
        del self.learnable_params['prompt_learner']
        prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.learnable_params['prompt_learner'] = prompt_learner


@TRAINER_REGISTRY.register()
class UCPDAPT(DAPT):
    def build_model(self):
        super().build_model()

        # bulid margin loss
        loss_cfg = self.cfg.TRAINER.UCP.LOSS
        use_sim = loss_cfg.USE_SIM
        smooth = loss_cfg.SMOOTH

        if use_sim:
            prompt_learner = self.model.learnable_params['prompt_learner']
            wordvecs = prompt_learner.ucp.class_tokens
            sim = torch.cosine_similarity(wordvecs.float().unsqueeze(1), 
                                          wordvecs.float().unsqueeze(0), dim=-1)
        else:
            sim = None

        self.loss_cls = MarginLoss(len(self.dm.dataset.classnames), sim, smooth)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        
        output, image_features, text_features = self.model(image)
        loss_orig = self.loss_cls(output, label)

        # visual prompt dispersion loss
        batch_p = self.prototype[label]
        p = batch_p
        if self.cfg.USE_CUDA:
            p = batch_p.to('cuda')
        loss_dist_i = F.mse_loss(image_features, p)

        # text prompt dispersion loss
        loss_dist_t = torch.pdist(text_features.to(torch.float), p=2).pow(2.0).mul(-self.cfg.TRAINER.DAPT.TXT_RBF_T).exp().mean()

        # total loss
        bi = self.cfg.TRAINER.DAPT.VIS_BETA
        bt = self.cfg.TRAINER.DAPT.TXT_BETA
        loss = loss_orig + bi*loss_dist_i + bt*loss_dist_t

        self.model_backward_and_update(loss)

        accuracy = compute_accuracy(output, label)[0].item()
        loss_summary = {
            "loss": loss.item(),
            "acc": accuracy,
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary
    def _build_custom_clip(self, cfg, classnames, clip_model):
        self.model = CustomCLIP(cfg, classnames, clip_model)
