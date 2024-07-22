import fasttext
import math
import os
import os.path as osp
import pickle
import torch
import torch.nn.functional as F
from fasttext.util import reduce_model
from torch import nn
from torchtext.vocab import GloVe
from tqdm import tqdm
from wikipedia2vec import Wikipedia2Vec

import clip


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


class Linear(nn.Module):
    def __init__(self, 
                 in_features, 
                 out_features, 
                 bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            stdv = 1. / math.sqrt(self.bias.size(0))
            self.bias.data.uniform_(-stdv, stdv)
    
    def forward(self, x):
        x = torch.matmul(x, self.weight)
        if self.bias is not None:
            x =  x + self.bias
        return x

    def __repr__(self):
        return f'Linear({self.in_features}, {self.out_features})'


class ChannelModulation(nn.Module):
    def __init__(self, dim, bias=True):
        super().__init__()
        self.dim = dim
        self.scale = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None
    
    def forward(self, x):
        return self.scale.unsqueeze(0) * x + self.bias.unsqueeze(0)
    
    def __repr__(self):
        return f'ChannelModulation({self.dim})'


class Attention(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 share_kv=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.q = nn.Linear(in_features, out_features, bias=bias)
        self.k = nn.Linear(in_features, out_features, bias=bias)
        self.v = nn.Linear(in_features, out_features, bias=bias) if not share_kv else self.k
    
    def forward(self, x):
        q, k, v = self.q(x), self.k(x), self.v(x)
        attn = q @ k.t()
        attn = torch.softmax(attn / math.sqrt(k.size(1)), dim=-1)
        return attn @ v
        
    def __repr__(self):
        return f'Attention({self.in_features}, {self.out_features})'


class MultiLayerPerceptron(nn.Module):
    def __init__(self,
                 in_feats,
                 hidden_feats,
                 out_feats,
                 bias=True,
                 dropout=0.0):
        super().__init__()
        self.in_feats = in_feats
        self.hidden_feats = hidden_feats
        self.out_feats = out_feats
        self.dropout = dropout

        self.down = Linear(in_feats, hidden_feats, bias)
        self.up = Linear(hidden_feats, out_feats, bias)
    
    def forward(self, x):
        # (C, D), (N, D), (N, D)
        x = self.down(x)
        x = F.leaky_relu(x, inplace=True)
        x = F.dropout(x, self.dropout, self.training)
        return self.up(x)
        
    def __repr__(self):
        return f'MLP({self.in_feats}, {self.hidden_feats}, {self.out_feats})'


class ClassnamesTokenizer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.trainer_cfg = cfg.TRAINER.UCP
        self.dataset_name = self.cfg.DATASET.NAME

    @torch.no_grad()
    def forward(self, classnames, clip_model):
        cache_path = self.get_cache_path()

        if osp.exists(cache_path):
            # if cache exists, load wordvecs from cache
            with open(cache_path, 'rb') as f:
                print(f'Loading wordvecs from cache {cache_path}...')
                wordvecs = pickle.load(f)
        else:
            # if cache does not exist, load wordvecs from model
            wordvecs = self.load_wordvecs(classnames, clip_model)

            # save wordvecs to cache
            os.makedirs('cache', exist_ok=True)
            print(f'Saving base wordvecs to {cache_path}...')
            with open(cache_path, 'wb') as f:
                pickle.dump(wordvecs, f)

        return wordvecs
        
    def get_cache_path(self):
        dataset_name = self.cfg.DATASET.NAME
        seed = self.cfg.SEED
        subsample = self.cfg.DATASET.SUBSAMPLE_CLASSES
        tokenizer = self.trainer_cfg.TOKENIZER
        mode = self.trainer_cfg.TOKENIZER_MODE
        reduce_dim = self.trainer_cfg.REDUCE_DIM

        if mode == 'clip':
            return f'cache/clip_{subsample}_{dataset_name}_seed{seed}.pkl'

        return f'cache/{tokenizer}_reduce{reduce_dim}_{subsample}_{dataset_name}_seed{seed}.pkl'
            
    def load_wordvecs(self, classnames, clip_model):
        mode = self.trainer_cfg.TOKENIZER_MODE
        tokenizer_path = osp.join('pretrained', self.trainer_cfg.TOKENIZER)
        reduce_dim = self.trainer_cfg.REDUCE_DIM

        # load models
        if mode == 'glove':
            _, name, dim, _ = tokenizer_path.split('.')
            dim = int(dim[:-1])
            model = GloVe(name, dim, cache='pretrained')
        elif mode == 'fasttext':
            model = fasttext.load_model(tokenizer_path)
            if reduce_dim < 300:
                reduce_model(model, reduce_dim)
        elif mode == 'wikipedia2vec':
            model = Wikipedia2Vec.load(tokenizer_path)
        elif mode == 'clip':
            device = clip_model.text_projection.device
            model = clip_model.cuda()
        else:
            raise NotImplementedError

        # load class speific wordvec
        wordvecs = []
        for name in tqdm(classnames, desc='loading wordvecs'):
            print(f'get wordvec of "{name}"')
            
            if mode == 'glove':
                vec = self._get_wordvec_glove(model, name)
            elif mode == 'fasttext':
                vec = self._get_wordvec_fasttext(model, name)
            if mode == 'wikipedia2vec':
                vec = self._get_wordvec_wikipedia2vec(model, name)
            elif mode == 'clip':
                vec = self._get_wordvec_clip(model, name)
            wordvecs.append(vec)

        if mode == 'clip':
            model = model.to(device)

        # replace empty vec with mean
        wordvecs_notnan = [vec for vec in wordvecs if vec is not None]
        meanvec = torch.stack(wordvecs_notnan, dim=0).mean(0)
        wordvecs = [vec if vec is not None else meanvec for vec in wordvecs]
        # (C, D)
        return torch.stack(wordvecs, dim=0)

    def _get_wordvec_glove(self, model, name):
        words = name.lower().replace('-', ' ').replace('_', ' ') \
                .replace('\'s', '').replace('.', '').split(' ')
        vecs = model.get_vecs_by_tokens(words, True)
        return vecs.mean(0)
    
    def _get_wordvec_fasttext(self, model, name):
        words = name.lower().replace('-', ' ').replace('_', ' ') \
                .replace('\'s', '').replace('.', '').split(' ')       
        vecs = [model.get_word_vector(word) for word in words]
        return torch.tensor(vecs).mean(0)

    def _get_wordvec_wikipedia2vec(self, model, name):
        words = name.lower().replace('-', ' ').replace('_', ' ') \
                .replace('\'s', '').replace('.', '').split(' ')

        vecs = []
        for word in words:
            try:
                vecs.append(model.get_word_vector(word))
            except:
                print(f'Warning! cannot find wordvec of word {word}!')
        
        if len(vecs) == 0:
            return None

        return torch.tensor(vecs).mean(0)
    
    def _get_wordvec_clip(self, model, name):
        name = name.replace('_', ' ')
        prompts = _CUSTOM_TEMPLATES[self.dataset_name].format(name)
        prompts = clip.tokenize([prompts]).cuda()
        vec = model.encode_text(prompts).squeeze(0)
        return vec.cpu().float()


class UnifiedClassSpecificPromptLearner(nn.Module):
    def __init__(self, 
                 cfg, 
                 classnames, 
                 clip_model,
                 img_feats=None,
                 labels=None):
        super().__init__()
        trainer_cfg = cfg.TRAINER.UCP
        module = trainer_cfg.MODULE
        dropout = trainer_cfg.DROPOUT
        hidden_dim = trainer_cfg.HIDDEN_DIM
        scale = trainer_cfg.SCALE
        dtype = clip_model.dtype

        if scale > 0.0:
            self.scale = scale
        else:
            self.scale = nn.Paramteter(torch.tensor(0.0)).type(dtype)
            
        tokenizer = ClassnamesTokenizer(cfg)
        self.class_tokens = tokenizer(classnames, clip_model).type(dtype) # word vectors

        clip_dim = clip_model.text_projection.size(-1)
        token_dim = self.class_tokens.size(-1)

        if module == 'mlp':
            self.unified_transfer = MultiLayerPerceptron(token_dim, hidden_dim, clip_dim, 
                                                         dropout=dropout).type(dtype)
        elif module == 'linear':
            self.unified_transfer = Linear(token_dim, clip_dim).type(dtype)
        elif module == 'attn':
            self.unified_transfer = Attention(token_dim, clip_dim).type(dtype)

        self.trainer_cfg = trainer_cfg
        self.dtype = dtype
        self.n_cls = len(classnames)

    def forward(self, class_tokens=None):
        device = next(self.unified_transfer.parameters()).device
        class_tokens = self.class_tokens.to(device) if class_tokens is None else class_tokens

        if isinstance(self.scale, float):
            class_tokens = class_tokens * self.scale
        else:
            class_tokens = class_tokens * self.scale.sigmoid()

        return self.unified_transfer(class_tokens)
