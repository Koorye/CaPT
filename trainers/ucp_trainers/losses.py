import math
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from torch import nn


class AdaptiveContrastiveLoss(nn.Module):
    def __init__(self, 
                 pos_ratio=0.2, 
                 neg_ratio=0.2,
                 loss_weight=1.0):
        super().__init__()
        self.pos_ratio = pos_ratio
        self.neg_ratio = neg_ratio
        self.loss_weight = loss_weight
        
    def forward(self, x):
        # (N, 1, D), (1, N, D) -> (N, N)
        sim = torch.cosine_similarity(x.unsqueeze(1), x.unsqueeze(0), dim=-1)

        sim_flatten, _ = sim.detach().flatten().sort()
        N = len(sim_flatten)

        loss = 0.0
        
        if self.pos_ratio > 0.0:
            pos_thresh = sim_flatten[math.ceil(N * (1 - self.pos_ratio))]
            pos_mask = sim > pos_thresh
            loss = loss + (1 - sim[pos_mask]).clamp_min_(0.0).mean()
        
        if self.neg_ratio > 0.0:
            neg_thresh = sim_flatten[math.ceil(N * self.neg_ratio)]
            neg_mask = sim < neg_thresh
            loss = loss + sim[neg_mask].clamp_min_(0.0).mean()
        
        return self.loss_weight * loss


class AdaptiveMarginCrossEntropyLoss(nn.Module):
    def __init__(self, 
                 wordvecs,
                 alpha=0.01,
                 beta=0.0):
        super().__init__()
        # (C, 1, D), (1, C, D) -> (C, C)
        wordvecs = wordvecs.detach().float()
        sim = torch.cosine_similarity(wordvecs.unsqueeze(1), 
                                      wordvecs.unsqueeze(0), dim=-1)

        sim = 1 - sim
        sim[torch.eye(sim.size(0)) > 0.0] = 0.0

        self.sim_mask = sim * alpha + beta
        sns.heatmap(self.sim_mask.detach().cpu().numpy())
        plt.savefig('viz.png')

    def forward(self, img_feats_norm, text_feats_norm, logit_scale, labels):
        sim_mask = self.sim_mask[labels].type_as(img_feats_norm).to(img_feats_norm.device)
        # (B, C)
        sim = img_feats_norm @ text_feats_norm.t()
        sim = sim + sim_mask
        logits = logit_scale * sim
        return F.cross_entropy(logits, labels)


class LabelSmoothingLoss(nn.Module):
    def __init__(self, 
                 num_classes, 
                 sim=None,
                 smoothing=0.01):
        super().__init__()
        assert 0 <= smoothing < 1, f'smoothing ({smoothing}) should be between 0 and 1!'

        self.cls = num_classes
        self.sim = sim
        self.smoothing = smoothing

    def forward(self, pred, target):
        pred = pred.log_softmax(-1)

        with torch.no_grad():
            # (C, C) -> (B, C)
            sim = self.sim[target].to(pred.device) if self.sim is not None \
                  else torch.ones_like(pred).detach()
            # scale sim as soft labels
            soft_labels = sim * (self.smoothing / (self.cls - 1))

            # set true labels as 0
            # (B,) -> (B, C)
            target_onehot = F.one_hot(target, num_classes=self.cls)
            soft_labels[target_onehot > 0] = 0
            
            # calculate true label values and set true labels
            true_labels = 1 - soft_labels.sum(1)
            soft_labels[target_onehot > 0] = true_labels

        return (-soft_labels * pred).sum(-1).mean()


class MarginLoss(nn.Module):
    def __init__(self,
                 num_classes,
                 sim=None,
                 smoothing=0.01,
                 reduction='mean'):
        super().__init__()
        self.cls = num_classes
        self.sim = sim
        self.smoothing = smoothing if smoothing != 0.0 else None
        self.reduction = reduction

    def forward(self, pred, target):
        if self.smoothing is None:
            return F.cross_entropy(pred, target, reduction=self.reduction)

        with torch.no_grad():
            # (C, C) -> (B, C)
            sim = self.sim[target].to(pred.device) if self.sim is not None \
                  else torch.ones_like(pred.detach())
            # scale sim as soft labels
            margin = sim * (self.smoothing / (self.cls - 1))

            # set true labels as 0
            # (B,) -> (B, C)
            target_onehot = F.one_hot(target.detach(), num_classes=self.cls)
            margin[target_onehot > 0] = 0
            
            # calculate true label values and set true labels
            true_pred = -margin.sum(1)
            margin[target_onehot > 0] = true_pred

        sim_pred = pred / 100
        sim_pred = sim_pred - margin
        pred = sim_pred * 100
        return F.cross_entropy(pred, target, reduction=self.reduction)
