import math
import os
import sys

import numpy as np
import torch
import torchaudio
from torch import nn
from torch.nn import Conv2d
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm, spectral_norm, weight_norm

from . import commons, modules
from .commons import get_padding
from .modules import (ConvNext2d, ISTFTHead, LayerNorm, LoRALinear1d,
                      SnakeFilter, WaveBlock)

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

sr2sr = {
    "24k": 24000,
    "32k": 32000,
    "40k": 40000,
    "48k": 48000,
}


class GeneratorVoras(torch.nn.Module):
    def __init__(
        self,
        emb_channels,
        inter_channels,
        gin_channels,
        n_layers,
        n_fft,
        hop_length,
    ):
        super(GeneratorVoras, self).__init__()
        self.n_layers = n_layers
        self.g_out_linear = weight_norm(nn.Conv1d(gin_channels, inter_channels, 1))
        self.resblocks = nn.ModuleList()
        self.init_linear = LoRALinear1d(emb_channels, inter_channels, gin_channels, r_out=12)
        for _ in range(self.n_layers):
            self.resblocks.append(WaveBlock(inter_channels, gin_channels, [5] * 3, [1] * 3, [1, 2, 4], 3, r_out=12))
        self.head = ISTFTHead(inter_channels, gin_channels, n_fft, hop_length, padding="center")
        #self.head = IMDCTSymExpHead(inter_channels, gin_channels, hop_length, padding="center", sample_rate=sr)
        self.post = SnakeFilter(4, 8, 9, 2, eps=1e-5)

    def forward(self, x, g_out):
        x = self.init_linear(x, g_out) + self.g_out_linear(g_out)
        for i in range(self.n_layers):
            x = self.resblocks[i](x, g_out)
        x = self.head(x, g_out)
        x = self.post(x)
        return torch.tanh(x)

    def remove_weight_norm(self):
        self.init_linear.remove_weight_norm()
        remove_weight_norm(self.g_out_linear)
        for l in self.resblocks:
            l.remove_weight_norm()
        self.head.remove_weight_norm()
        self.post.remove_weight_norm()

    def fix_speaker(self, i, g):
        self.init_linear.fix_speaker(i, g)
        for l in self.resblocks:
            l.fix_speaker(i, g)
        self.head.fix_speaker(i, g)

    def unfix_speaker(self, i, g):
        self.init_linear.unfix_speaker(i, g)
        for l in self.resblocks:
            l.unfix_speaker(i, g)
        self.head.unfix_speaker(i, g)


class SpeakerEmbedder(torch.nn.Module):
    def __init__(self, gin_channels):
        super(SpeakerEmbedder, self).__init__()
        self.melspec = torchaudio.transforms.MelSpectrogram(
            n_mels=128,
            sample_rate=16000,
            n_fft=960,
            win_length=960,
            hop_length=160,
            f_min=0.0,
            f_max=None,
            window_fn=torch.hann_window,
            center=True,
            power=1,
            norm="slaney",
            mel_scale="slaney"
        )
        self.dwconvs = nn.ModuleList()
        self.pwconvs1 = nn.ModuleList()
        self.pwconvs2 = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.act = nn.GELU()
        inner_channels = gin_channels
        self.inner_channels = inner_channels
        self.init_conv = weight_norm(Conv2d(1, inner_channels, (7, 7), stride=(2, 2), padding=(3, 3)))
        for i in range(4):
            if i == 3:
                k = 3
            else:
                k = 9
            self.dwconvs.append(weight_norm(Conv2d(inner_channels, inner_channels, (3, k), stride=(2, (k+1)//4), groups=inner_channels, padding=(1, k//2))))
            self.norms.append(LayerNorm(inner_channels))
            self.pwconvs1.append(weight_norm(Conv2d(inner_channels, inner_channels * 4, (1, 1))))
            self.pwconvs2.append(weight_norm(Conv2d(inner_channels * 4, inner_channels, (1, 1))))
        self.post = nn.Linear(inner_channels * 4, gin_channels)
        self.eps = 1e-7
        # self.weight = np.sqrt(2.) * np.log(self.spk_embed_dim + 1.)

    def forward(self, y_16k):
        y_mel = torch.log(torch.clamp(self.melspec(y_16k), min=self.eps))
        y = self.init_conv(y_mel)
        for i in range(4):
            y = self.dwconvs[i](y)
            y = self.norms[i](y)
            y = self.pwconvs1[i](y)
            y = self.act(y)
            y = self.pwconvs2[i](y)
        y = torch.reshape(y.mean(dim=[3]), [y.shape[0], self.inner_channels * 4])
        y = self.post(y)
        y = y / torch.clamp(torch.norm(y, p=2, dim=1, keepdim=True), min=1e-7)
        return y


class Synthesizer(nn.Module):
    def __init__(
        self,
        segment_size,
        n_fft,
        hop_length,
        inter_channels,
        n_layers,
        spk_embed_dim,
        gin_channels,
        emb_channels,
        sr,
        **kwargs
    ):
        super().__init__()
        if type(sr) == type("strr"):
            sr = sr2sr[sr]
        self.segment_size = segment_size
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.inter_channels = inter_channels
        self.n_layers = n_layers
        self.spk_embed_dim = spk_embed_dim
        self.gin_channels = gin_channels
        self.emb_channels = emb_channels
        self.sr = sr

        self.dec = GeneratorVoras(
            emb_channels,
            inter_channels,
            gin_channels,
            n_layers,
            n_fft,
            hop_length
        )
        self.speaker_embedder = SpeakerEmbedder(gin_channels)
        self.emb_g = nn.Embedding(self.spk_embed_dim, gin_channels)
        print(
            "gin_channels:",
            gin_channels,
            "self.spk_embed_dim:",
            self.spk_embed_dim,
            "emb_channels:",
            emb_channels,
        )

        self.speaker = [None, None]

    def remove_weight_norm(self):
        self.dec.remove_weight_norm()

    def change_speaker(self, target, sid: int):
        if self.speaker[target] is not None:
            g = self.emb_g(torch.tensor(self.speaker[target]).to(self.emb_g.weight.data.device)).unsqueeze(-1)
            self.dec.unfix_speaker(target, g)
        g = self.emb_g(torch.tensor(sid).to(self.emb_g.weight.data.device)).unsqueeze(-1)
        self.dec.fix_speaker(target, g)
        self.speaker[target] = sid

    def forward(
        self, phone, ds
        ):
        g_out = self.emb_g(ds).unsqueeze(-1)
        x = torch.transpose(phone, 1, -1)
        o = self.dec(x, g_out)
        return o, None, g_out

    def infer(self, phone, sid):
        g_out = self.emb_g(sid).unsqueeze(-1)
        x = torch.transpose(phone, 1, -1)
        o = self.dec(x, g_out)
        return o, None, (None, None, None, None)


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, gin_channels, upsample_rates, final_dim=256, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        self.init_kernel_size = upsample_rates[-1] * 3
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        N = len(upsample_rates)
        self.init_conv = norm_f(Conv2d(1, final_dim // (2 ** (N - 1)), (self.init_kernel_size, 1), (upsample_rates[-1], 1)))
        self.convs = nn.ModuleList()
        for i, u in enumerate(upsample_rates[::-1][1:], start=1):
            self.convs.append(
                ConvNext2d(
                    final_dim // (2 ** (N - i)),
                    final_dim // (2 ** (N - i - 1)),
                    gin_channels,
                    (u*3, 1),
                    (u, 1),
                    4,
                    r=2 + i//2
                )
            )
        self.conv_post = weight_norm(Conv2d(final_dim, 1, (3, 1), (1, 1)))

    def forward(self, x, g):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (n_pad, 0), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        x = torch.flip(x, dims=[2])
        x = F.pad(x, [0, 0, 0, self.init_kernel_size - 1], mode="constant")
        x = self.init_conv(x)
        x = F.leaky_relu(x, modules.LRELU_SLOPE)
        x = torch.flip(x, dims=[2])
        fmap.append(x)

        for i, l in enumerate(self.convs):
            x = l(x, g)
            fmap.append(x)

        x = F.pad(x, [0, 0, 2, 0], mode="constant")
        x = self.conv_post(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, upsample_rates, gin_channels, periods=[2, 3, 5, 7, 11, 17], **kwargs):
        super(MultiPeriodDiscriminator, self).__init__()

        discs = [
            DiscriminatorP(i, gin_channels, upsample_rates, use_spectral_norm=False) for i in periods
        ]
        self.ups = np.prod(upsample_rates)
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat, g):
        fmap_rs = []
        fmap_gs = []
        y_d_rs = []
        y_d_gs = []
        for d in self.discriminators:
            y_d_r, fmap_r = d(y, g)
            y_d_g, fmap_g = d(y_hat, g)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
