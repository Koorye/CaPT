import math
import random
import torch
from torch import nn
from tqdm import tqdm

import clip

from ..base_trainers.coop import PromptLearner, TextEncoder


_CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
}

_IMAGENET_TEMPLATES = [
    "a photo of a {}.",
    "a bad photo of a {}.",
    "a photo of many {}.",
    "a sculpture of a {}.",
    "a photo of the hard to see {}.",
    "a low resolution photo of the {}.",
    "a rendering of a {}.",
    "graffiti of a {}.",
    "a bad photo of the {}.",
    "a cropped photo of the {}.",
    "a tattoo of a {}.",
    "the embroidered {}.",
    "a photo of a hard to see {}.",
    "a bright photo of a {}.",
    "a photo of a clean {}.",
    "a photo of a dirty {}.",
    "a dark photo of the {}.",
    "a drawing of a {}.",
    "a photo of my {}.",
    "the plastic {}.",
    "a photo of the cool {}.",
    "a close-up photo of a {}.",
    "a black and white photo of the {}.",
    "a painting of the {}.",
    "a painting of a {}.",
    "a pixelated photo of the {}.",
    "a sculpture of the {}.",
    "a bright photo of the {}.",
    "a cropped photo of a {}.",
    "a plastic {}.",
    "a photo of the dirty {}.",
    "a jpeg corrupted photo of a {}.",
    "a blurry photo of the {}.",
    "a photo of the {}.",
    "a good photo of the {}.",
    "a rendering of the {}.",
    "a {} in a video game.",
    "a photo of one {}.",
    "a doodle of a {}.",
    "a close-up photo of the {}.",
    "the origami {}.",
    "the {} in a video game.",
    "a sketch of a {}.",
    "a doodle of the {}.",
    "a origami {}.",
    "a low resolution photo of a {}.",
    "the toy {}.",
    "a rendition of the {}.",
    "a photo of the clean {}.",
    "a photo of a large {}.",
    "a rendition of a {}.",
    "a photo of a nice {}.",
    "a photo of a weird {}.",
    "a blurry photo of a {}.",
    "a cartoon {}.",
    "art of a {}.",
    "a sketch of the {}.",
    "a embroidered {}.",
    "a pixelated photo of a {}.",
    "itap of the {}.",
]


_IMAGENET_TEMPLATES_SELECTED = [
    "itap of a {}.",
    "a bad photo of the {}.",
    "a origami {}.",
    "a photo of the large {}.",
    "a {} in a video game.",
    "art of the {}.",
    "a photo of the small {}.",
]


class SimpleClassnameTokenizer(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.trainer_cfg = cfg.TRAINER.UCP
        self.dataset_name = self.cfg.DATASET.NAME
        self.classnames = classnames
        
        self.init_wordvecs(classnames, clip_model)

    @property
    def token_dim(self):
        return self.wordvecs.size(-1)
        
    @torch.no_grad()
    def forward(self):
        return self.wordvecs

    @torch.no_grad()
    def init_wordvecs(self, classnames, clip_model):
        device = clip_model.text_projection.device
        model = clip_model.cuda()

        template = _CUSTOM_TEMPLATES[self.dataset_name]
        texts = [template.format(name) for name in classnames]
        texts = clip.tokenize(texts).cuda()
        wordvecs = clip_model.encode_text(texts)
        
        model = model.to(device)

        # (N, C, D)
        self.wordvecs = wordvecs



class DataDiversityClassnameTokenizer(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.trainer_cfg = cfg.TRAINER.UCP
        self.dataset_name = self.cfg.DATASET.NAME
        self.init_wordvecs(classnames, clip_model)
        
    @torch.no_grad()
    def forward(self, training=True):
        if training:
            ind = random.randint(0, len(self.wordvecs) - 1)
            return self.wordvecs[ind]
        else:
            return self.wordvecs.mean(0)

    @torch.no_grad()
    def init_wordvecs(self, classnames, clip_model):
        device = clip_model.text_projection.device
        model = clip_model.cuda()

        wordvecs_list = []
        for template in tqdm(_IMAGENET_TEMPLATES_SELECTED, desc='loading wordvecs'):
            texts = [template.format(name) for name in classnames]
            texts = clip.tokenize(texts).cuda()
            vecs = clip_model.encode_text(texts)
            wordvecs_list.append(vecs)
        
        model = model.to(device)

        # (N, C, D)
        self.wordvecs = torch.stack(wordvecs_list)

    @property
    def token_dim(self):
        return self.wordvecs.size(-1)


class SimilarityClassnameTokenizer(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.trainer_cfg = cfg.TRAINER.UCP
        self.dataset_name = self.cfg.DATASET.NAME
        
        classnames = self.refine_classnames(classnames, clip_model)
        self.init_wordvecs(classnames, clip_model)
        
    @torch.no_grad()
    def forward(self):
        return self.wordvecs

    @torch.no_grad()
    def refine_classnames(self, classnames, clip_model):
        template = _CUSTOM_TEMPLATES[self.dataset_name]
        texts = [template.format(name) for name in classnames]
        tokenized_texts = clip.tokenize(texts).cuda()
        
        clip_model = clip_model.cuda()
        text_feats = clip_model.encode_text(tokenized_texts)
        sim = torch.cosine_similarity(text_feats.unsqueeze(1), 
                                      text_feats.unsqueeze(0), dim=-1)
        
        k = min(math.ceil(sim.size(0) * 0.25), 3)
        _, inds = torch.topk(sim, k)
        inds = inds[:, 1:]
        
        inds = inds.cpu()
        clip_model = clip_model.cpu()
        
        new_texts = []
        for name, ind in zip(classnames, inds):
            sim_classnames = [classnames[i] for i in ind]
            s = ', '.join(sim_classnames)
            idx = s.rfind(',')

            if idx != -1:
                s = s[:idx] + ' and' + s[idx + 1:]

            new_text = f'{name}. Notice: {name} is different from {s}.'
            new_texts.append(new_text)

        print('Refined classnames:')
        for text in new_texts:
            print(f'- {text}')
        print()

        return new_texts

    @torch.no_grad()
    def init_wordvecs(self, classnames, clip_model):
        device = clip_model.text_projection.device
        model = clip_model.cuda()

        template = _CUSTOM_TEMPLATES[self.dataset_name]
        texts = [template.format(name) for name in classnames]
        texts = clip.tokenize(texts).cuda()
        self.wordvecs = clip_model.encode_text(texts)
        
        model = model.to(device)

    @property
    def token_dim(self):
        return self.wordvecs.size(-1)


class LearnableClassnameTokenizer(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.classnames = classnames
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.text_encoder = TextEncoder(clip_model)
    
    def forward(self):
        prompts = self.prompt_learner()
        return self.text_encoder(prompts, self.prompt_learner.tokenized_prompts)
    
    @property
    def token_dim(self):
        return self.text_encoder.text_projection.size(-1)
