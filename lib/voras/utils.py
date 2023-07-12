import glob
import logging
import math
import os
import shutil
import socket
import sys

import ffmpeg
import matplotlib
import matplotlib.pylab as plt
import numpy as np
import torch
from scipy.io.wavfile import read
from torch.nn import functional as F
from torch.optim.lr_scheduler import _LRScheduler

from modules.shared import ROOT_DIR

from .config import TrainConfig

matplotlib.use("Agg")
logging.getLogger("matplotlib").setLevel(logging.WARNING)

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging


class AWP:
    """
    Fast AWP
    https://www.kaggle.com/code/junkoda/fast-awp
    """
    def __init__(self, model, optimizer, *, adv_param='weight',
                 adv_lr=0.01, adv_eps=0.01):
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}

    def perturb(self):
        """
        Perturb model parameters for AWP gradient
        Call before loss and loss.backward()
        """
        self._save()  # save model parameters
        self._attack_step()  # perturb weights

    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                grad = self.optimizer.state[param]['exp_avg']
                norm_grad = torch.norm(grad)
                norm_data = torch.norm(param.detach())

                if norm_grad != 0 and not torch.isnan(norm_grad):
                    # Set lower and upper limit in change
                    limit_eps = self.adv_eps * param.detach().abs()
                    param_min = param.data - limit_eps
                    param_max = param.data + limit_eps

                    # Perturb along gradient
                    # w += (adv_lr * |w| / |grad|) * grad
                    param.data.add_(grad, alpha=(self.adv_lr * (norm_data + e) / (norm_grad + e)))

                    # Apply the limit to the change
                    param.data.clamp_(param_min, param_max)

    def _save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.clone().detach()
                else:
                    self.backup[name].copy_(param.data)

    def restore(self):
        """
        Restore model parameter to correct position; AWP do not perturbe weights, it perturb gradients
        Call after loss.backward(), before optimizer.step()
        """
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])


class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
        https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup/tree/master
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(
            self,
            optimizer : torch.optim.Optimizer,
            first_cycle_steps : int,
            cycle_mult : float = 1.,
            max_lr : float = 0.1,
            min_lr : float = 0.001,
            first_lr: float = 0.0001,
            warmup_steps : int = 0,
            gamma : float = 1.,
            last_epoch : int = -1
        ):
        assert warmup_steps < first_cycle_steps

        self.first_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle_mult = cycle_mult # cycle steps magnification
        self.base_max_lr = max_lr # first max learning rate
        self.max_lr = max_lr # max learning rate in the current cycle
        self.min_lr = min_lr # min learning rate
        self.first_lr = first_lr
        self.warmup_steps = warmup_steps # warmup step size
        self.gamma = gamma # decrease rate of max learning rate by cycle

        self.cur_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle = 0 # cycle count
        self.step_in_cycle = -1 # step size of the current cycle

        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)

        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [np.exp((np.log(self.max_lr) - np.log(self.first_lr)) * self.step_in_cycle / self.warmup_steps + np.log(self.first_lr)) for _ in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(2 * math.pi * (self.step_in_cycle - self.warmup_steps) / self.cur_cycle_steps))
                    for base_lr in self.base_lrs]

    def step(self):
        epoch = self.last_epoch + 1
        self.step_in_cycle = self.step_in_cycle + 1
        self.cycle = max(0, (self.step_in_cycle - self.warmup_steps) / self.cur_cycle_steps)

        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


def load_audio(file: str, sr):
    try:
        # https://github.com/openai/whisper/blob/main/whisper/audio.py#L26
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        file = (
            file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )  # Prevent small white copy path head and tail with spaces and " and return
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load audio: {e}")

    return np.frombuffer(out, np.float32).flatten()


def find_empty_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    s.listen(1)
    port = s.getsockname()[1]
    s.close()
    return port


def load_checkpoint(checkpoint_path, model, optimizer=None, load_opt=1, device="cuda"):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")

    saved_state_dict = checkpoint_dict["model"]
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():  # 模型需要的shape
        new_state_dict[k] = saved_state_dict[k].to(device)

    if hasattr(model, "module"):
        model.module.load_state_dict(new_state_dict, strict=False)
    else:
        model.load_state_dict(new_state_dict, strict=False)
    print("Loaded model weights")

    epoch = checkpoint_dict["epoch"]
    learning_rate = checkpoint_dict["learning_rate"]
    if optimizer is not None and load_opt == 1:
        optimizer.load_state_dict(checkpoint_dict["optimizer"])
    print("Loaded checkpoint '{}' (epoch {})".format(checkpoint_path, epoch))
    return model, optimizer, learning_rate, epoch


def save_state(model, optimizer, learning_rate, epoch, checkpoint_path):
    print(
        "Saving model and optimizer state at epoch {} to {}".format(
            epoch, checkpoint_path
        )
    )
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save(
        {
            "model": state_dict,
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "learning_rate": learning_rate,
        },
        checkpoint_path,
    )


def summarize(
    writer,
    global_step,
    scalars={},
    histograms={},
    images={},
    audios={},
    audio_sampling_rate=22050,
):
    for k, v in scalars.items():
        writer.add_scalar(k, v, global_step)
    for k, v in histograms.items():
        writer.add_histogram(k, v, global_step)
    for k, v in images.items():
        writer.add_image(k, v, global_step, dataformats="HWC")
    for k, v in audios.items():
        writer.add_audio(k, v, global_step, audio_sampling_rate)


def latest_checkpoint_path(dir_path, regex="G_*.pth"):
    filelist = glob.glob(os.path.join(dir_path, regex))
    if len(filelist) == 0:
        return None
    filelist.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    filepath = filelist[-1]
    return filepath


def plot_spectrogram_to_numpy(spectrogram):
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def plot_alignment_to_numpy(alignment, info=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(
        alignment.transpose(), aspect="auto", origin="lower", interpolation="none"
    )
    fig.colorbar(im, ax=ax)
    xlabel = "Decoder timestep"
    if info is not None:
        xlabel += "\n\n" + info
    plt.xlabel(xlabel)
    plt.ylabel("Encoder timestep")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_config(training_dir: str, sample_rate: int, emb_channels: int):
    if emb_channels == 256:
        config_path = os.path.join(ROOT_DIR, "configs", f"{sample_rate}.json")
    else:
        config_path = os.path.join(
            ROOT_DIR, "configs", f"{sample_rate}-{emb_channels}.json"
        )
    config_save_path = os.path.join(training_dir, "config.json")

    shutil.copyfile(config_path, config_save_path)

    return TrainConfig.parse_file(config_save_path)
