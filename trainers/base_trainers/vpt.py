import os.path as osp

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from .._base_ import BaseCustomCLIP, BaseTrainer

_tokenizer = _Tokenizer()


class InferenceBlock(nn.Module):
    def __init__(self, input_units, d_theta, output_units):
        """
        :param d_theta: dimensionality of the intermediate hidden layers.
        :param output_units: dimensionality of the output.
        :return: batch of outputs.
        """
        super(InferenceBlock, self).__init__()
        self.module = nn.Sequential(
            nn.Linear(input_units, d_theta, bias=True),
            nn.ELU(inplace=True),
            nn.Linear(d_theta, d_theta, bias=True),
            nn.ELU(inplace=True),
            nn.Linear(d_theta, output_units, bias=True),
        )

    def forward(self, inps):
        out = self.module(inps)
        return out


class Amortized(nn.Module):
    def __init__(self, input_units=400, d_theta=400, output_units=400):
        super(Amortized, self).__init__()
        self.output_units = output_units
        self.weight_mean = InferenceBlock(input_units, d_theta, output_units)
        self.weight_log_variance = InferenceBlock(input_units, d_theta, output_units)

    def forward(self, inps):
        weight_mean = self.weight_mean(inps)
        weight_log_variance = self.weight_log_variance(inps)
        return weight_mean, weight_log_variance


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final  # maybe layer normalization
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        # print(prompts.shape, tokenized_prompts.shape)
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.VPT.N_CTX
        ctx_init = cfg.TRAINER.VPT.CTX_INIT
        self.L = cfg.TRAINER.VPT.L
        self.vpt_type = cfg.TRAINER.VPT.VPT_TYPE
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert (
            cfg_imsize == clip_imsize
        ), f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(
                ctx_init
            )  # returns a vector with dim 1 x 77 where 77 is the maximum length of the prompt, intialized the prompt with the context and pad it with zeros.
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)  # 1 x 77 x 512
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]  # 1 x (n_ctx) x 512
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(
            ctx_vectors
        )  # ctx intialized with a embedding of the prompt "a photo of cat"

        if self.vpt_type == "cocoopvpt":
            self.meta_net = Amortized(
                input_units=vis_dim, d_theta=vis_dim // 2, output_units=ctx_dim
            )
            if cfg.TRAINER.PREC == "fp16":
                self.meta_net.half()
        elif self.vpt_type == "coopvpt":
            self.mean_posterior = nn.Parameter(torch.zeros(1, ctx_dim, dtype=dtype))
            self.std_posterior = nn.Parameter(torch.rand(1, ctx_dim, dtype=dtype))
        else:
            raise ValueError(f"Type {cfg.vpt_type} is not supported.")

        classnames = [name.replace("_", " ") for name in classnames]  # remove any available _
        name_lens = [
            len(_tokenizer.encode(name)) for name in classnames
        ]  # tokenize each class name, tokenizer might generate multiple token for each class even if the classname only have one character.
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def sample(self, mu, logvar, L):
        shape = (L,) + mu.size()
        eps = torch.randn(shape).type_as(mu)
        bias = mu.unsqueeze(0) + eps * logvar.exp().sqrt().unsqueeze(0)
        return bias

    def forward(self, im_features):
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.ctx  # (n_ctx, ctx_dim)

        if self.vpt_type == "cocoopvpt":
            bias_mu, bias_logvar = self.meta_net(im_features)  # (1, ctx_dim)
        elif self.vpt_type == "coopvpt":
            bias_mu, bias_logvar = self.mean_posterior, self.std_posterior  # (1, ctx_dim)
        else:
            raise ValueError(f"Type {self.vpt_type} is not supported.")

        bias = self.sample(bias_mu, bias_logvar, self.L)  # (L, 1, ctx_dim)
        ctx = ctx.unsqueeze(0)  # (1, n_ctx, ctx_dim)
        ctx_shifted = ctx + bias  # (L, n_ctx, ctx_dim)

        # Use instance-conditioned context tokens for all classes
        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = self.construct_prompts(ctx_i, prefix, suffix)  # (n_cls, n_tkn, ctx_dim)
            prompts.append(pts_i)
        prompts = torch.stack(prompts)

        return prompts, bias_mu, bias_logvar


class CustomCLIP(BaseCustomCLIP):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__(cfg, classnames, clip_model)
        self.text_encoder = TextEncoder(clip_model)

        prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.learnable_params['prompt_learner'] = prompt_learner
        self.tokenized_prompts = prompt_learner.tokenized_prompts
        self.L = prompt_learner.L

    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts
        # (N, L)
        tokenized_prompts = torch.tile(tokenized_prompts, (self.L, 1))
        logit_scale = self.logit_scale.exp()

        # (B, D)
        image_features = self.image_encoder(image.type(self.dtype))  # 1 x 512
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # (N, C, L, D) -> (N * C, L, D) -> (N * C, D)
        prompt_learner = self.learnable_params['prompt_learner']
        prompts, mu, logvar = prompt_learner(image_features)  # L x NumClass x Length x DIM
        _, NumClass, Length, dim = prompts.shape
        prompts = prompts.view(-1, Length, dim)  # (L * NumClass) x Length x DIM
        text_features = self.text_encoder(prompts, tokenized_prompts)

        # (B, D) -> (N, B, D)
        image_features = image_features.unsqueeze(0).expand((self.L, -1, -1))
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # (N * C, D) -> (N, C, D)
        text_features = text_features.view(-1, NumClass, text_features.shape[-1])  # L * NumClass x DIM

        # (N, B, D) @ (N, D, C) -> (N, B, C)
        logits = logit_scale * torch.einsum("LBD,LCD->LBC", image_features, text_features)
        log_p_y = torch.log_softmax(logits, dim=-1)

        if prompt_learner.training:
            # (B,) -> (B, C) -> (N, B, C)
            label_one_hot = torch.nn.functional.one_hot(label, num_classes=logits.shape[-1])
            tile_label = torch.tile(label_one_hot.unsqueeze(0), (self.L, 1, 1))

            # (N, B, C), (N, B, C) -> (N, B) -> (B,) -> scaler
            task_log_py = self.nll(log_p_y, tile_label)
            task_score = torch.logsumexp(task_log_py, dim=0) \
                       - torch.log(torch.Tensor([self.L]).type_as(logits))
            task_loss = -task_score.mean(dim=-1)

            return task_loss + 0.001 * self.kl_divergence(mu, logvar)

        # (N, B, C) -> (B, C)
        average_prediction = torch.logsumexp(log_p_y, dim=0) \
                           - torch.log(torch.Tensor([self.L]).type_as(logits))
        return average_prediction

    def kl_divergence(self, mu, logvar):
        prior_mu = torch.zeros_like(mu)
        prior_std = torch.ones_like(logvar)

        prior = torch.distributions.Normal(loc=prior_mu, scale=prior_std)
        post = torch.distributions.Normal(loc=mu, scale=logvar.exp().sqrt())

        dist = torch.distributions.kl_divergence(post, prior).mean(dim=-1)
        return dist

    def nll(self, logits, targets):
        task_log_py = (logits * targets).sum(dim=-1)
        return task_log_py


@TRAINER_REGISTRY.register()
class VPT(BaseTrainer):
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
    
    def _build_custom_clip(self, cfg, classnames, clip_model):
        self.model = CustomCLIP(cfg, classnames, clip_model)
