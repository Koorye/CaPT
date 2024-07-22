import datetime
import os
import os.path as osp
import time
import torch
from torch import nn
from torch.cuda.amp import GradScaler
from tqdm import tqdm

from dassl.engine import TrainerX
from dassl.optim import build_lr_scheduler
from dassl.utils import AverageMeter, MetricMeter, load_checkpoint, load_pretrained_weights

from clip import clip
from clip_maple import clip as clip_maple
from utils.logger import print

from .optim import build_optimizer


def load_clip_to_cpu(cfg):
    """ load clip for coop, cocoop and kgcoop """
    
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


def load_clip_to_cpu_ivlp(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip_maple._MODELS[backbone_name]
    model_path = clip_maple._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'IVLP',
                      "vision_depth": cfg.TRAINER.IVLP.PROMPT_DEPTH_VISION,
                      "language_depth": cfg.TRAINER.IVLP.PROMPT_DEPTH_TEXT, 
                      "vision_ctx": cfg.TRAINER.IVLP.N_CTX_VISION,
                      "language_ctx": cfg.TRAINER.IVLP.N_CTX_TEXT}
    model = clip_maple.build_model(state_dict or model.state_dict(), design_details)

    return model


def load_clip_to_cpu_maple(cfg):
    """ load clip for maple """
    
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip_maple._MODELS[backbone_name]
    model_path = clip_maple._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'MaPLe',
                      "vision_depth": 0,
                      "language_depth": 0, 
                      "vision_ctx": 0,
                      "language_ctx": 0,
                      "maple_length": cfg.TRAINER.MAPLE.N_CTX}
    model = clip_maple.build_model(state_dict or model.state_dict(), design_details)

    return model


class BaseCustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.learnable_params = nn.ModuleDict()

    def forward(self, image):
        raise NotImplementedError


class BaseTrainer(TrainerX):
    def __init__(self, cfg):
        super().__init__(cfg)

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def check_cfg(self, cfg):
        assert cfg.TRAINER.PREC in ['fp16', 'fp32', 'amp']

    def build_model(self):
        """ modify parameters which need to update and save """
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        trainer = cfg.TRAINER.NAME.lower()

        print(f'Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})')
        if 'maple' in trainer:
            clip_model = load_clip_to_cpu_maple(cfg)
        elif 'ivlp' in trainer:
            clip_model = load_clip_to_cpu_ivlp(cfg)
        else:
            clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.PREC == 'fp32' or cfg.TRAINER.PREC == 'amp':
            clip_model.float()

        print('Building custom CLIP')
        self._build_custom_clip(cfg, classnames, clip_model)

        print('Turning off gradients in both the image and the text encoder')

        self.model.requires_grad_(False)
        self.model.learnable_params.requires_grad_(True)

        # Double check
        enabled = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.append(name)
        enabled = list(sorted(enabled))
        print(f'Parameters to be updated:')
        [print(f'  - {p}') for p in enabled]

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)

        optim_cfg = cfg.OPTIM
        self.optim = build_optimizer(self.model.learnable_params, optim_cfg)
        self.sched = build_lr_scheduler(self.optim, optim_cfg)
        self.register_model('learnable_params', self.model.learnable_params, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.PREC == 'amp' else None

        device_count = torch.cuda.device_count()
        assert device_count == 1, 'Multiple GPUs are not supported!'
        return clip_model

    def load_model(self, directory, epoch=None):
        """ add new function: get last epoch when epoch < 0, and delete tokens """
        
        if not directory:
            print('Note that load_model() is skipped as no pretrained model is given')
            return

        names = self.get_model_names()

        model_file = 'model-best.pth.tar'

        if epoch is not None:
            epoch = self._get_last_epoch(directory, names[0]) if epoch < 0 else epoch
            model_file = 'model.pth.tar-' + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint['state_dict']
            epoch = checkpoint['epoch']

            # Ignore fixed token vectors
            state_dict = self._delete_tokens(state_dict)
            state_dict = self._custom_state_dict(state_dict)
            
            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
        
    def run_epoch(self):
        """ reset print function """
        
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader_x)

        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_x):
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch)
            # loss_summary = {'loss': 0.0}
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                    self.max_epoch - self.epoch - 1
                ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()
    
    def before_epoch(self):
        super().before_epoch()
        self.epoch_start_time = time.time()
    
    def after_epoch(self):
        """ show training time """
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = (
            (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
            if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False
        )
        
        epoch_time = time.time() - self.epoch_start_time
        speed = self.num_batches / epoch_time
        memory = self._get_gpu_memory()
        print(f"Epoch time: {epoch_time:.2f}s, Speed: {speed:.2f} batch/s, Memory: {memory:.2f} MB")

        if last_epoch:
            training_time = time.time() - self.time_start
            speed = self.num_batches * self.max_epoch / training_time
            memory = self._get_gpu_memory()
            print(f"Training time: {training_time:.2f}s, Speed: {speed:.2f} batch/s, Memory: {memory:.2f} MB")

        if do_test and self.cfg.TEST.FINAL_MODEL == "best_val":
            curr_result = self.test(split="val")
            is_best = curr_result > self.best_result
            if is_best:
                self.best_result = curr_result
                self.save_model(
                    self.epoch,
                    self.output_dir,
                    val_result=curr_result,
                    model_name="model-best.pth.tar"
                )

        if meet_checkpoint_freq or last_epoch:
            self.save_model(self.epoch, self.output_dir)
        
    @torch.no_grad()
    def test(self, split=None):
        """ show inference time """
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")
        start_time = time.time()

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input)
            self.evaluator.process(output, label)
            
        end_time = time.time()
        time_delta = end_time - start_time
        speed = len(data_loader.dataset) / time_delta
        memory = self._get_gpu_memory()
        print(f"Test time: {time_delta:.2f}s, Speed: {speed:.2f} img/s, Memory: {memory:.2f} MB")

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]

    def _build_custom_clip(self, cfg, classnames, clip_model):
        self.model = BaseCustomCLIP(cfg, classnames, clip_model)

    def _get_last_epoch(self, directory, name):
        filenames = os.listdir(osp.join(directory, name))
        filenames = [filename for filename in filenames if '.tar' in filename]
        epochs = [int(filename.split('-')[-1]) for filename in filenames]
        return max(epochs)

    def _delete_tokens(self, state_dict):
        tokens = ['token_prefix', 'token_suffix']

        for token in tokens:
            for key in list(state_dict.keys()):
                if token in key:
                    print(f'Delete key {key} from checkpoint')
                    del state_dict[key]

        return state_dict

    def _custom_state_dict(self, state_dict):
        return state_dict

    def _get_gpu_memory(self):
        """ get device memory MB """
        return torch.cuda.memory_allocated(self.device) / 1024 / 1024
