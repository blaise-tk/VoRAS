import math
import os
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import Conv2d
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm, spectral_norm, weight_norm

from . import commons, modules
from .commons import get_padding
from .modules import (ConvNext2d, HarmonicEmbedder, IMDCTSymExpHead, LayerNorm,
                      LoRALinear1d, LoRALinear2d, SnakeFilter, WaveBlock)

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
        sr,
        hop_length,
    ):
        super(GeneratorVoras, self).__init__()
        self.n_layers = n_layers
        self.emb_pitch = HarmonicEmbedder(768, inter_channels, gin_channels, 16, 15)  #   # pitch 256
        self.plinear = LoRALinear1d(inter_channels, inter_channels, gin_channels, r=8)
        self.glinear = weight_norm(nn.Conv1d(gin_channels, inter_channels, 1))
        self.resblocks = nn.ModuleList()
        self.init_linear = LoRALinear1d(emb_channels, inter_channels, gin_channels, r=4)
        for _ in range(self.n_layers):
            self.resblocks.append(WaveBlock(inter_channels, gin_channels, [9] * 2, [1] * 2, [1, 9], 2, r=4))
        self.head = IMDCTSymExpHead(inter_channels, gin_channels, hop_length, padding="center", sample_rate=sr)
        self.post = SnakeFilter(4, 8, 9, 2, eps=1e-5)

    def forward(self, x, pitchf, x_mask, g):
        x = self.init_linear(x, g) + self.plinear(self.emb_pitch(pitchf, g), g) + self.glinear(g)
        for i in range(self.n_layers):
            x = self.resblocks[i](x, x_mask, g)
        x = x * x_mask
        x = self.head(x, g)
        x = self.post(x)
        return torch.tanh(x)

    def remove_weight_norm(self):
        self.plinear.remove_weight_norm()
        remove_weight_norm(self.glinear)
        for l in self.resblocks:
            l.remove_weight_norm()
        self.init_linear.remove_weight_norm()
        self.head.remove_weight_norm()
        self.post.remove_weight_norm()

    def fix_speaker(self, g):
        self.plinear.fix_speaker(g)
        self.init_linear.fix_speaker(g)
        for l in self.resblocks:
            l.fix_speaker(g)
        self.head.fix_speaker(g)

    def unfix_speaker(self, g):
        self.plinear.unfix_speaker(g)
        self.init_linear.unfix_speaker(g)
        for l in self.resblocks:
            l.unfix_speaker(g)
        self.head.unfix_speaker(g)


class GeneratorVorasNoPitch(torch.nn.Module):
    def __init__(
        self,
        emb_channels,
        inter_channels,
        gin_channels,
        n_layers,
        sr,
        hop_length,
    ):
        super(GeneratorVorasNoPitch, self).__init__()
        self.n_layers = n_layers
        self.glinear = weight_norm(nn.Conv1d(gin_channels, inter_channels, 1))
        self.resblocks = nn.ModuleList()
        self.init_linear = LoRALinear1d(emb_channels, inter_channels, gin_channels, r=4)
        for _ in range(self.n_layers):
            self.resblocks.append(WaveBlock(inter_channels, gin_channels, [9] * 2, [1] * 2, [1, 9], 2, r=4))
        self.head = IMDCTSymExpHead(inter_channels, gin_channels, hop_length, padding="center", sample_rate=sr)
        self.post = SnakeFilter(4, 8, 9, 2, eps=1e-5)

    def forward(self, x, x_mask, g):
        x = self.init_linear(x, g) + self.glinear(g)
        for i in range(self.n_layers):
            x = self.resblocks[i](x, x_mask, g)
        x = x * x_mask
        x = self.head(x, g)
        x = self.post(x)
        return torch.tanh(x)

    def remove_weight_norm(self):
        remove_weight_norm(self.glinear)
        for l in self.resblocks:
            l.remove_weight_norm()
        self.init_linear.remove_weight_norm()
        self.head.remove_weight_norm()
        self.post.remove_weight_norm()

    def fix_speaker(self, g):
        self.init_linear.fix_speaker(g)
        for l in self.resblocks:
            l.fix_speaker(g)
        self.head.fix_speaker(g)

    def unfix_speaker(self, g):
        self.init_linear.unfix_speaker(g)
        for l in self.resblocks:
            l.unfix_speaker(g)
        self.head.unfix_speaker(g)


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
            sr,
            hop_length
        )

        self.emb_g = nn.Embedding(self.spk_embed_dim, gin_channels)
        print(
            "gin_channels:",
            gin_channels,
            "self.spk_embed_dim:",
            self.spk_embed_dim,
            "emb_channels:",
            emb_channels,
        )
        self.speaker = None

    def remove_weight_norm(self):
        self.dec.remove_weight_norm()

    def change_speaker(self, sid: int):
        if self.speaker is not None:
            g = self.emb_g(torch.from_numpy(np.array(self.speaker))).unsqueeze(-1)
            self.dec.unfix_speaker(g)
        g = self.emb_g(torch.from_numpy(np.array(sid))).unsqueeze(-1)
        self.dec.fix_speaker(g)
        self.speaker = sid

    def forward(
        self, phone, phone_lengths, pitch, pitchf, ds
        ):
        g = self.emb_g(ds).unsqueeze(-1)
        x = torch.transpose(phone, 1, -1)
        x_mask = torch.unsqueeze(commons.sequence_mask(phone_lengths, x.size(2)), 1).to(phone.dtype)
        x_slice, ids_slice = commons.rand_slice_segments(
            x, phone_lengths, self.segment_size
        )
        pitchf_slice = commons.slice_segments2(pitchf, ids_slice, self.segment_size)
        mask_slice = commons.slice_segments(x_mask, ids_slice, self.segment_size)
        o = self.dec(x_slice, pitchf_slice, mask_slice, g)
        return o, ids_slice, x_mask, g

    def infer(self, phone, phone_lengths, pitch, nsff0, sid, max_len=None):
        g = self.emb_g(sid).unsqueeze(-1)
        x = torch.transpose(phone, 1, -1)
        x_mask = torch.unsqueeze(commons.sequence_mask(phone_lengths, x.size(2)), 1).to(phone.dtype)
        o = self.dec((x * x_mask)[:, :, :max_len], nsff0, x_mask, g)
        return o, x_mask, (None, None, None, None)


class SynthesizerNoPitch(nn.Module):
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

        self.dec = GeneratorVorasNoPitch(
            emb_channels,
            inter_channels,
            gin_channels,
            n_layers,
            sr,
            hop_length
        )

        self.emb_g = nn.Embedding(self.spk_embed_dim, gin_channels)
        print(
            "gin_channels:",
            gin_channels,
            "self.spk_embed_dim:",
            self.spk_embed_dim,
            "emb_channels:",
            emb_channels,
        )
        self.speaker = None

    def remove_weight_norm(self):
        self.dec.remove_weight_norm()

    def change_speaker(self, sid: int):
        if self.speaker is not None:
            g = self.emb_g(torch.from_numpy(np.array(self.speaker))).unsqueeze(-1)
            self.dec.unfix_speaker(g)
        g = self.emb_g(torch.from_numpy(np.array(sid))).unsqueeze(-1)
        self.dec.fix_speaker(g)
        self.speaker = sid

    def forward(
        self, phone, phone_lengths, ds
        ):
        g = self.emb_g(ds).unsqueeze(-1)
        x = torch.transpose(phone, 1, -1)
        x_mask = torch.unsqueeze(commons.sequence_mask(phone_lengths, x.size(2)), 1).to(phone.dtype)
        x_slice, ids_slice = commons.rand_slice_segments(
            x, phone_lengths, self.segment_size
        )
        mask_slice = commons.slice_segments(x_mask, ids_slice, self.segment_size)
        o = self.dec(x_slice, mask_slice, g)
        return o, ids_slice, x_mask, g

    def infer(self, phone, phone_lengths, sid, max_len=None):
        g = self.emb_g(sid).unsqueeze(-1)
        x = torch.transpose(phone, 1, -1)
        x_mask = torch.unsqueeze(commons.sequence_mask(phone_lengths, x.size(2)), 1).to(phone.dtype)
        o = self.dec(x, x_mask, g)
        return o, x_mask, (None, None, None, None)


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


class SpeakerDiscriminator(torch.nn.Module):
    def __init__(self, gin_channels, spk_embed_dim):
        super(SpeakerDiscriminator, self).__init__()
        self.dwconvs = nn.ModuleList()
        self.pwconvs1 = nn.ModuleList()
        self.pwconvs2 = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.act = nn.GELU()
        self.spk_embed_dim = spk_embed_dim

        self.init_conv = weight_norm(Conv2d(1, 64, (7, 7), stride=(2, 2), padding=(3, 3)))
        for i in range(4):
            if i == 3:
                k = 3
            else:
                k = 9
            self.dwconvs.append(weight_norm(Conv2d(64, 64, (3, k), stride=(2, (k+1)//4), groups=64, padding=(1, k//2))))
            self.norms.append(LayerNorm(64))
            self.pwconvs1.append(weight_norm(Conv2d(64, 256, (1, 1))))
            self.pwconvs2.append(weight_norm(Conv2d(256, 64, (1, 1))))
        self.post = nn.Linear(64, gin_channels)
        self.weight = np.sqrt(2.) * np.log(self.spk_embed_dim + 1.)

    def forward(self, y_mel, sid, gs):
        y = self.infer(y_mel)
        y_normed = y / torch.clamp(torch.norm(y, p=2, dim=1, keepdim=True), min=1e-7)
        gs_normed = gs / torch.clamp(torch.norm(gs, p=2, dim=1, keepdim=True), min=1e-7)
        score = self.weight * torch.einsum("bd,nd->bn", y_normed, gs_normed)
        ix = torch.arange(y.shape[0])
        score_pos = score[ix, sid]
        score_neg = torch.exp(torch.where(sid.unsqueeze(1) != torch.arange(self.spk_embed_dim).unsqueeze(0).to(sid.device), score, -1e5)).sum(dim=1)
        loss = (-score_pos + torch.log(torch.clamp(score_neg, min=1e-7))).mean()
        return loss

    def infer(self, y_mel):
        y = self.init_conv(y_mel)
        for i in range(4):
            y = self.dwconvs[i](y)
            y = self.norms[i](y)
            y = self.pwconvs1[i](y)
            y = self.act(y)
            y = self.pwconvs2[i](y)
        y = y.mean(dim=[2, 3])
        y = self.post(y)
        return y


class MultiPeriodSynchronizedDiscriminator(torch.nn.Module):
    def __init__(self, upsample_rates, gin_channels, periods=[2, 3, 5, 7, 11, 17], **kwargs):
        super(MultiPeriodSynchronizedDiscriminator, self).__init__()
        self.num_periods = len(periods)
        self.periods = periods
        N = len(upsample_rates)
        self.num_upsamples = N
        self.init_kernel_size = upsample_rates[-1] * 3
        final_dim = 256
        self.init_convs = nn.ModuleList([weight_norm(Conv2d(1, final_dim // (2 ** (N - 1)), (self.init_kernel_size, 1), (upsample_rates[-1], 1))) for _ in range(self.num_periods)])
        self.final_convs = nn.ModuleList([weight_norm(Conv2d(final_dim, 1, (3, 1), (1, 1))) for _ in range(self.num_periods)])

        self.dwconvs = nn.ModuleList()
        self.pwconvs1 = nn.ModuleList()
        self.pwconvs2 = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.gammas = nn.ParameterList()
        self.betas = nn.ParameterList()
        self.act = nn.GELU()

        self.kernel_sizes = []
        for i, u in enumerate(upsample_rates[::-1][1:], start=1):
            in_channels = final_dim // (2 ** (N - i))
            out_channels = final_dim // (2 ** (N - i - 1))
            inner_channels = in_channels * 4
            self.kernel_sizes.append((u*3, 1))
            r=2 + i//2
            for _ in range(self.num_periods):
                self.dwconvs.append(Conv2d(in_channels, in_channels, (u*3, 1), (u, 1), groups=in_channels))
                self.pwconvs1.append(LoRALinear2d(in_channels, inner_channels, gin_channels, r=r))
                self.norms.append(LayerNorm(in_channels))
                self.pwconvs2.append(LoRALinear2d(inner_channels, out_channels, gin_channels, r=r))
                self.gammas.append(nn.Parameter(torch.zeros(1, inner_channels, 1, 1)))
                self.betas.append(nn.Parameter(torch.zeros(1, inner_channels, 1, 1)))


    def do(self, x, g):
        xs = []
        ys = []
        fmaps = [[] for _ in range(self.num_periods)]
        b, c, t = x.shape
        for j, p in enumerate(self.periods):
            if t % p != 0:  # pad first
                n_pad = p - (t % p)
                x_ = F.pad(x, (n_pad, 0), "reflect")
                l_ = t + n_pad
            else:
                x_ = x.clone()
                l_ = t
            x_ = x_.view(b, c, l_ // p, p)
            x_ = torch.flip(x_, dims=[2])
            x_ = F.pad(x_, [0, 0, 0, self.init_kernel_size - 1], mode="constant")
            x_ = self.init_convs[j](x_)
            x_ = self.act(x_)
            x_ = torch.flip(x_, dims=[2])
            fmaps[j].append(x_.clone())
            xs.append(x_)

        for i in range(self.num_upsamples-1):
            gxs = []
            dxs = []
            for j in range(self.num_periods):
                x = xs[j]
                x = F.pad(x, [0, 0, self.kernel_sizes[i][0] - 1, 0], mode="constant")
                x = self.dwconvs[i * self.num_periods + j](x)
                x = self.norms[i * self.num_periods + j](x)
                x = self.pwconvs1[i * self.num_periods + j](x, g)
                x = self.act(x)
                xs[j] = x
                gxs.append(torch.norm(x, p=2, dim=(2, 3), keepdim=True))
                dxs.append(torch.clamp(gxs[-1].mean(dim=1, keepdim=True), min=1e-6))

            d = torch.cat(dxs, dim=1).mean(dim=1, keepdim=True)
            for j in range(self.num_periods):
                nx = gxs[j] / d
                x = self.gammas[i * self.num_periods + j] * xs[j] * nx + self.betas[i * self.num_periods + j] + xs[j]
                x = self.pwconvs2[i * self.num_periods + j](x, g)
                x = self.act(x)
                fmaps[j].append(x.clone())
                xs[j] = x

        for j in range(self.num_periods):
            ys.append(self.final_convs[j](xs[j]))
        return ys, fmaps


    def forward(self, y, y_hat, g):
        fmap_rs = [[] for _ in range(self.num_periods)]
        fmap_gs = []
        y_d_rs, fmap_rs = self.do(y, g)
        y_d_gs, fmap_gs = self.do(y_hat, g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class Discriminator(torch.nn.Module):
    def __init__(self, upsample_rates, gin_channels, spk_embed_dim, multiple_speakers, periods=[2, 3, 5, 7, 11, 17], **kwargs):
        super(Discriminator, self).__init__()

        self.disc_p = MultiPeriodDiscriminator(upsample_rates, gin_channels, periods)
        self.disc_s = SpeakerDiscriminator(gin_channels, spk_embed_dim)
        self.multiple_speakers = multiple_speakers
        if not self.multiple_speakers:
            self.disc_s.requires_grad_(False)

    def forward(self, y, y_hat, g, y_mel=None, sid=None, gs=None):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = self.disc_p(y, y_hat, g)
        if self.multiple_speakers:
            spk_loss = self.disc_s(y_mel, sid, gs)
        else:
            spk_loss = 0.
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs, spk_loss