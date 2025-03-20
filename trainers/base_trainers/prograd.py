import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from torch.nn.modules.loss import _Loss

from .._base_ import load_clip_to_cpu, BaseCustomCLIP
from .._base_.base_prograd import BaseTrainer


_tokenizer = _Tokenizer()


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]),
              tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.PROGRAD.N_CTX
        ctx_init = cfg.TRAINER.PROGRAD.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # ctx_init = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
            # ctx_init = ctx_init.replace(" {}.", "")
            # ctx_init = ctx_init.replace("_", " ")
            
            ctx_init = 'a photo of a'
            prompt_n_ctx = len(ctx_init.split(" "))

            assert n_ctx >= prompt_n_ctx, f"#tokens ({n_ctx}) should larger equal than #initial prompt tokens ({prompt_n_ctx}, {ctx_init})"

            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)

            ctx_vectors = torch.zeros(n_ctx, ctx_dim, dtype=dtype)

            ctx_vectors[n_ctx - prompt_n_ctx:, :] = embedding[0, 1:1 +
                                                              prompt_n_ctx, :]
            prompt_prefix = " ".join(["X"] * (n_ctx - prompt_n_ctx))
            prompt_prefix = f"{prompt_prefix} {ctx_init}"
        else:
            # random initialization
            if cfg.TRAINER.PROGRAD.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(
                dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix",
                             embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.class_token_position = cfg.TRAINER.PROGRAD.CLASS_TOKEN_POSITION
        self.name_lens = name_lens

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,  # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        # elif self.class_token_position == "middle":
        #     half_n_ctx = n_ctx // 2
        #     prompts = []
        #     for i in range(self.n_cls):
        #         name_len = self.name_lens[i]
        #         prefix_i = prefix[i:i + 1, :, :]
        #         class_i = suffix[i:i + 1, :name_len, :]
        #         suffix_i = suffix[i:i + 1, name_len:, :]
        #         ctx_i_half1 = ctx[i:i + 1, :half_n_ctx, :]
        #         ctx_i_half2 = ctx[i:i + 1, half_n_ctx:, :]
        #         prompt = torch.cat(
        #             [
        #                 prefix_i,  # (1, 1, dim)
        #                 ctx_i_half1,  # (1, n_ctx//2, dim)
        #                 class_i,  # (1, name_len, dim)
        #                 ctx_i_half2,  # (1, n_ctx//2, dim)
        #                 suffix_i,  # (1, *, dim)
        #             ],
        #             dim=1,
        #         )
        #         prompts.append(prompt)
        #     prompts = torch.cat(prompts, dim=0)

        # elif self.class_token_position == "front":
        #     prompts = []
        #     for i in range(self.n_cls):
        #         name_len = self.name_lens[i]
        #         prefix_i = prefix[i:i + 1, :, :]
        #         class_i = suffix[i:i + 1, :name_len, :]
        #         suffix_i = suffix[i:i + 1, name_len:, :]
        #         ctx_i = ctx[i:i + 1, :, :]
        #         prompt = torch.cat(
        #             [
        #                 prefix_i,  # (1, 1, dim)
        #                 class_i,  # (1, name_len, dim)
        #                 ctx_i,  # (1, n_ctx, dim)
        #                 suffix_i,  # (1, *, dim)
        #             ],
        #             dim=1,
        #         )
        #         prompts.append(prompt)
        #     prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


CUSTOM_TEMPLATES = {
    "OxfordPets": "a type of pet, a photo of a {}.",
    "OxfordFlowers": "a type of flower, a photo of a {}.",
    "FGVCAircraft": "a type of aircraft, a photo of a {}.",
    "DescribableTextures": "a texture of {}.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a type of food, a photo of {}.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
}


class CLIP(nn.Module):
    def __init__(self, cfg, classnames):
        super().__init__()
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.float()

        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])

        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features
        self.clip_model = clip_model

    def forward(self, image):
        image_features = self.clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()

        text_features = self.text_features
        text_features = text_features.to(image_features.device)
        logits = logit_scale * image_features @ text_features.t()
        return logits


class CustomCLIP(BaseCustomCLIP):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__(cfg, classnames, clip_model)
        
        prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.learnable_params['prompt_learner'] = prompt_learner
        
        self.tokenized_prompts = prompt_learner.tokenized_prompts
        self.text_encoder = TextEncoder(clip_model)

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))

        prompt_learner = self.learnable_params['prompt_learner']
        prompts = prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits


class ProGradLoss(_Loss):
    def __init__(self, T):
        super(ProGradLoss, self).__init__()
        self.T = T

    def forward(self, stu_logits, tea_logits, label):
        xe_loss = F.cross_entropy(stu_logits, label)

        tea_prob = F.softmax(tea_logits / self.T, dim=-1)
        kl_loss = -tea_prob * F.log_softmax(stu_logits / self.T, -1) * self.T * self.T
        kl_loss = kl_loss.sum(1).mean()

        return xe_loss, kl_loss


@TRAINER_REGISTRY.register()
class ProGrad(BaseTrainer):
    """ Projected Gradient for few-shot CLIP """
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.PREC == "fp32" or cfg.TRAINER.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building zeroshot CLIP")
        self.zs_clip = CLIP(cfg, classnames)

        print("Building custom CLIP")
        # self.model = CustomCLIP(cfg, classnames, clip_model)
        self._build_custom_clip(cfg, classnames, clip_model)

        print("Turning off gradients in ZS Clip model")
        for name, param in self.zs_clip.named_parameters():
            param.requires_grad_(False)

        print('Turning off gradients in both the image and the text encoder')
        names_to_update = cfg.TRAINER.NAMES_TO_UPDATE

        for name, param in self.model.named_parameters():
            need_update = False
            for name_to_update in names_to_update:
                if name_to_update in name:
                    need_update = True

            param.requires_grad_(need_update)

        # Double check
        enabled = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.append(name)
        enabled = list(sorted(enabled))
        print(f'Parameters to be updated:')
        [print(f'  - {p}') for p in enabled]

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.zs_clip = self.zs_clip.to(self.device)

        # NOTE: only give prompt_learner to the optimizer
        optim_cfg = cfg.OPTIM
        self.optim = build_optimizer(self.model.learnable_params, optim_cfg)
        self.sched = build_lr_scheduler(self.optim, optim_cfg)
        self.register_model('learnable_params', self.model.learnable_params, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        assert device_count == 1, 'Multiple GPUs are not supported!'

        # build criterion
        self.criterion = ProGradLoss(T=cfg.LOSS.T)
        
        return clip_model

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        prec = self.cfg.TRAINER.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                with torch.no_grad():
                    zs_clip_output = self.zs_clip(image)
                loss = self.criterion(output, zs_clip_output.detach(), label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image)
            with torch.no_grad():
                zs_clip_output = self.zs_clip(image)

            xe_loss, kl_loss = self.criterion(output, zs_clip_output.detach(), label)
            self.prograd_backward_and_update(xe_loss, kl_loss, self.cfg.LOSS.LAMBDA)

        loss_summary = {
            "xe_loss": xe_loss.item(),
            "kl_loss": kl_loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def _build_custom_clip(self, cfg, classnames, clip_model):
        self.model = CustomCLIP(cfg, classnames, clip_model)
