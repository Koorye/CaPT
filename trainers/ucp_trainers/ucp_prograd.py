import torch
import torch.nn.functional as F

from dassl.engine import TRAINER_REGISTRY

from ..base_trainers.prograd import CustomCLIP as CustomCLIP_, PromptLearner as PromptLearner_, \
                                    ProGrad, ProGradLoss as ProGradLoss_
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


class ProGradLoss(ProGradLoss_):
    def __init__(self, cfg, classnames, prompt_learner):
        super().__init__(cfg.LOSS.T)
        
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

    def forward(self, stu_logits, tea_logits, label):
        xe_loss = self.loss_cls(stu_logits, label)

        tea_prob = F.softmax(tea_logits / self.T, dim=-1)
        kl_loss = -tea_prob * F.log_softmax(stu_logits / self.T, -1) * self.T * self.T
        kl_loss = kl_loss.sum(1).mean()

        return xe_loss, kl_loss


@TRAINER_REGISTRY.register()
class UCPProGrad(ProGrad):
    def build_model(self):
        super().build_model()
        self.criterion = ProGradLoss(self.cfg, self.dm.dataset.classnames,
                                     self.model.learnable_params['prompt_learner'])

    def _build_custom_clip(self, cfg, classnames, clip_model):
        self.model = CustomCLIP(cfg, classnames, clip_model)
