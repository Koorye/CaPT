import torch
import torch.nn.functional as F

from dassl.engine import TRAINER_REGISTRY

from ..base_trainers.kgcoop_dept import CustomCLIP as CustomCLIP_, KgCoOpDePT
from .losses import MarginLoss
from .ucp_kgcoop import PromptLearner


class CustomCLIP(CustomCLIP_):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__(cfg, classnames, clip_model)

        # reset prompt learner
        del self.learnable_params['prompt_learner']
        prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.learnable_params['prompt_learner'] = prompt_learner
        self.tokenized_prompts = prompt_learner.tokenized_prompts   

        # bulid margin loss
        loss_cfg = self.cfg.TRAINER.UCP.LOSS
        smooth = loss_cfg.SMOOTH
        wordvecs = prompt_learner.ucp.class_tokens
        sim = torch.cosine_similarity(wordvecs.float().unsqueeze(1), 
                                      wordvecs.float().unsqueeze(0), dim=-1)
        self.loss_sim = MarginLoss(len(classnames), sim, smooth)

    def _loss(self, logits, labels, cls_logits, cls_labels, text_feats):
        loss_sim = self.loss_sim(logits, labels)
        loss_cls = F.cross_entropy(cls_logits, cls_labels)

        text_feats_old = self.ori_embedding
        text_feats_old = text_feats_old / text_feats_old.norm(dim=-1, keepdim=True)

        cos = torch.nn.CosineSimilarity(dim=1,eps=1e-07)
        score = cos(text_feats, text_feats_old)
        loss_kg = 1.0 - torch.mean(score)

        cls_weight = self.cls_weight
        loss = (1 - cls_weight) * loss_sim + cls_weight * loss_cls + self.kg_weight * loss_kg
        return loss


@TRAINER_REGISTRY.register()
class UCPKgCoOpDePT(KgCoOpDePT):
    def _build_custom_clip(self, cfg, classnames, clip_model):
        self.model = CustomCLIP(cfg, classnames, clip_model)
