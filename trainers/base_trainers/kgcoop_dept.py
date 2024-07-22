import os.path as osp
import torch
import torch.nn.functional as F
from dassl.engine import TRAINER_REGISTRY
from torch.cuda.amp import autocast
from torch import nn

from .kgcoop import KgCoOp
from .kgcoop import CustomCLIP as CustomCLIP_


class FiLM(nn.Module):
    def __init__(self, 
                 dim, 
                 bias=True, 
                 use_sigmoid=False):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None
        self.has_bias = bias
        self.use_sigmoid = use_sigmoid
    
    def forward(self, x):
        scale = self.scale.unsqueeze(0).type(x.dtype)
        bias = self.bias.unsqueeze(0).type(x.dtype) if self.has_bias else None
        
        x = scale * x
        if bias is not None:
            x = x + bias
        
        if self.use_sigmoid:
            return x.sigmoid()
        
        return x


class CustomCLIP(CustomCLIP_):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__(cfg, classnames, clip_model)
        self.subsample_classes = cfg.DATASET.SUBSAMPLE_CLASSES
        self.dataset = cfg.DATASET.NAME
        self.cls_weight = cfg.TRAINER.DEPT.CLS_WEIGHT
        self.kg_weight = cfg.TRAINER.KGCOOP.W

        clip_dim = clip_model.text_projection.size(1)
        self.learnable_params['film_img'] = FiLM(clip_dim)
        self.learnable_params['film_text'] = FiLM(clip_dim)
        
        if (self.subsample_classes == 'base') \
        or (self.subsample_classes == 'all' and 'ImageNet' in self.dataset):
            self.learnable_params['classifier'] = nn.Linear(clip_dim, len(classnames)).type(self.dtype)
        
    def forward(self, img, labels=None):
        if (self.subsample_classes == 'base') \
        or (self.subsample_classes == 'all' and 'ImageNet' in self.dataset):
            return self._forward_base(img, labels)
        else:
            return self._forward_new(img)

    def _forward_base(self, img, labels=None):
        text_feats, img_feats = self._forward_feats(img)
        
        logits = self._forward_sim_logits(text_feats, img_feats)
        cls_logits, cls_labels = self._forward_cls_logits(text_feats, img_feats, labels)
        
        prompt_learner = self.learnable_params['prompt_learner']
        if prompt_learner.training:
            return self._loss(logits, labels, cls_logits, cls_labels, text_feats)
        
        cls_weight = self.cls_weight
        logits = (1 - cls_weight) * logits + cls_weight * cls_logits
        return logits
    
    def _forward_new(self, img):
        prompt_learner = self.learnable_params['prompt_learner']
        assert not prompt_learner.training
        
        text_feats, img_feats = self._forward_feats(img)
        logits = self._forward_sim_logits(text_feats, img_feats)
        return logits
    
    def _forward_feats(self, img):
        prompt_learner = self.learnable_params['prompt_learner']
        prompts = prompt_learner()

        tokenized_prompts = self.tokenized_prompts
        text_feats = self.text_encoder(prompts, tokenized_prompts)
        img_feats = self.image_encoder(img.type(self.dtype))

        return text_feats, img_feats
    
    def _forward_sim_logits(self, text_feats, img_feats):
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * img_feats @ text_feats.t()
        return logits
    
    def _forward_cls_logits(self, text_feats, img_feats, labels):
        film_text = self.learnable_params['film_text']
        film_img = self.learnable_params['film_img']
        classifier = self.learnable_params['classifier']
        
        text_feats = film_text(text_feats)
        img_feats = film_img(img_feats)

        if labels is None:
            all_feats = img_feats
            all_labels = labels
        else:
            text_feats = text_feats[labels]
            all_feats = torch.cat([text_feats, img_feats])
            all_labels = torch.cat([labels, labels])

        all_logits = classifier(all_feats)
        return all_logits, all_labels
    
    def _loss(self, logits, labels, cls_logits, cls_labels, text_feats):
        loss_sim = F.cross_entropy(logits, labels)
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
class KgCoOpDePT(KgCoOp):
    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss = model(image, label)
            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {"loss": loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def model_inference(self, input):
        return self.model(input)
    
    def build_model(self):
        self.cfg.defrost()
        self.cfg.OPTIM.STAGED_LR = True
        self.cfg.freeze()
        return super().build_model()
    
    def _custom_state_dict(self, state_dict):
        if self.cfg.DATASET.NAME in ['ImageNetA', 'ImageNetR']:
            from datasets.imagenet import ImageNet
            from dassl.utils import listdir_nohidden

            dataset = self.dm.dataset
            text_file = osp.join(dataset.dataset_dir, "classnames.txt")
            all_folders = ImageNet.read_classnames(text_file).keys()

            TO_BE_IGNORED = ["README.txt"]
            folders = listdir_nohidden(dataset.image_dir, sort=True)
            folders = [f for f in folders if f not in TO_BE_IGNORED]
            is_reserves = [f in folders for f in all_folders]

            print(f'State dict is CLIPPED to match the shape of target dataset {self.cfg.DATASET.NAME}!')
            
            state_dict['classifier.weight'] = state_dict['classifier.weight'][is_reserves]
            state_dict['classifier.bias'] = state_dict['classifier.bias'][is_reserves]
        
        return state_dict

    def _build_custom_clip(self, cfg, classnames, clip_model):
        self.model = CustomCLIP(cfg, classnames, clip_model)
