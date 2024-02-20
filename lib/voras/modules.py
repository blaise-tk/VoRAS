import math

import numpy as np
import scipy
import torch
from torch import nn
from torch.nn import Conv1d, Conv2d
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm, spectral_norm, weight_norm
from torchaudio.functional.functional import _hz_to_mel, _mel_to_hz

from . import commons, modules
from .commons import get_padding, init_weights
from .transforms import piecewise_rational_quadratic_transform

LRELU_SLOPE = 0.1


class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


class DilatedCausalConv1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        groups=1,
        dilation=1,
        bias=True,
    ):
        super(DilatedCausalConv1d, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.conv = weight_norm(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                groups=groups,
                dilation=dilation,
                bias=bias,
            )
        )

    def forward(self, x):
        x = torch.flip(x, [2])
        x = F.pad(
            x, [0, (self.kernel_size - 1) * self.dilation], mode="constant", value=0.0
        )
        size = x.shape[2] // self.stride
        x = self.conv(x)[:, :, :size]
        x = torch.flip(x, [2])
        return x

    def remove_weight_norm(self):
        remove_weight_norm(self.conv)


class CausalConvTranspose1d(nn.Module):
    """
    padding = 0, dilation = 1のとき

    Lout = (Lin - 1) * stride + kernel_rate * stride + output_padding
    Lout = Lin * stride + (kernel_rate - 1) * stride + output_padding
    output_paddingいらないね
    """

    def __init__(self, in_channels, out_channels, kernel_rate=3, stride=1, groups=1):
        super(CausalConvTranspose1d, self).__init__()
        kernel_size = kernel_rate * stride
        self.trim_size = (kernel_rate - 1) * stride
        self.conv = weight_norm(
            nn.ConvTranspose1d(
                in_channels, out_channels, kernel_size, stride=stride, groups=groups
            )
        )

    def forward(self, x):
        x = self.conv(x)
        return x[:, :, : -self.trim_size]

    def remove_weight_norm(self):
        remove_weight_norm(self.conv)


class LoRALinear1d(nn.Module):
    def __init__(self, in_channels, out_channels, info_channels, r_out):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.info_channels = info_channels
        self.main_fc = weight_norm(nn.Conv1d(in_channels, out_channels, 1))
        self.rs = [r_out]
        self.adapter_in = torch.nn.ModuleList(
            [nn.Conv1d(info_channels, in_channels * r, 1, bias=False) for r in self.rs]
        )
        self.adapter_out = torch.nn.ModuleList(
            [nn.Conv1d(info_channels, out_channels * r, 1, bias=False) for r in self.rs]
        )
        for i in range(len(self.rs)):
            nn.init.normal_(self.adapter_in[i].weight.data, 0, 0.01)
            self.adapter_in[i] = weight_norm(self.adapter_in[i])
            nn.init.constant_(self.adapter_out[i].weight.data, 1e-7)
            self.adapter_out[i] = weight_norm(self.adapter_out[i])
        self.speaker_fixed = [False] * len(self.rs)

    def forward(self, x, g_out):
        x_ = self.main_fc(x)
        for i, g in enumerate([g_out]):
            if self.speaker_fixed[i]:
                continue
            a_in = self.adapter_in[i](g).view(-1, self.in_channels, self.rs[i])
            a_out = self.adapter_out[i](g).view(-1, self.rs[i], self.out_channels)
            l = torch.einsum(
                "brl,brc->bcl", torch.einsum("bcl,bcr->brl", x, a_in), a_out
            )
            x_ = x_ + l
        return x_

    def remove_weight_norm(self):
        for l in self.adapter_in:
            remove_weight_norm(l)
        for l in self.adapter_out:
            remove_weight_norm(l)
        remove_weight_norm(self.main_fc)

    def fix_speaker(self, i, g):
        self.speaker_fixed[i] = True
        a_in = self.adapter_in[i](g).view(-1, self.in_channels, self.rs[i])
        a_out = self.adapter_out[i](g).view(-1, self.rs[i], self.out_channels)
        weight = torch.einsum("bir,bro->oi", a_in, a_out).unsqueeze(2)
        self.main_fc.weight.data.add_(weight)

    def unfix_speaker(self, i, g):
        self.speaker_fixed[i] = False
        a_in = self.adapter_in[i](g).view(-1, self.in_channels, self.rs[i])
        a_out = self.adapter_out[i](g).view(-1, self.rs[i], self.out_channels)
        weight = torch.einsum("bir,bro->oi", a_in, a_out).unsqueeze(2)
        self.main_fc.weight.data.sub_(weight)


class LoRALinear2d(nn.Module):
    def __init__(self, in_channels, out_channels, info_channels, r):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.info_channels = info_channels
        self.r = r
        self.main_fc = weight_norm(nn.Conv2d(in_channels, out_channels, (1, 1), (1, 1)))
        self.adapter_in = nn.Conv1d(info_channels, in_channels * r, 1)
        self.adapter_out = nn.Conv1d(info_channels, out_channels * r, 1)
        nn.init.normal_(self.adapter_in.weight.data, 0, 0.01)
        nn.init.constant_(self.adapter_out.weight.data, 1e-6)
        self.adapter_in = weight_norm(self.adapter_in)
        self.adapter_out = weight_norm(self.adapter_out)
        self.speaker_fixed = False

    def forward(self, x, g):
        x_ = self.main_fc(x)
        if not self.speaker_fixed:
            a_in = self.adapter_in(g).view(-1, self.in_channels, self.r)
            a_out = self.adapter_out(g).view(-1, self.r, self.out_channels)
            l = torch.einsum(
                "brhw,brc->bchw", torch.einsum("bchw,bcr->brhw", x, a_in), a_out
            )
            x_ = x_ + l
        return x_

    def remove_weight_norm(self):
        remove_weight_norm(self.main_fc)
        remove_weight_norm(self.adapter_in)
        remove_weight_norm(self.adapter_out)

    def fix_speaker(self, g):
        a_in = self.adapter_in(g).view(-1, self.in_channels, self.r)
        a_out = self.adapter_out(g).view(-1, self.r, self.out_channels)
        weight = torch.einsum("bir,bro->oi", a_in, a_out).unsqueeze(2).unsqueeze(3)
        self.main_fc.weight.data.add_(weight)

    def unfix_speaker(self, g):
        a_in = self.adapter_in(g).view(-1, self.in_channels, self.r)
        a_out = self.adapter_out(g).view(-1, self.r, self.out_channels)
        weight = torch.einsum("bir,bro->oi", a_in, a_out).unsqueeze(2).unsqueeze(3)
        self.main_fc.weight.data.sub_(weight)


class MBConv2d(torch.nn.Module):
    """
    Causal MBConv2D
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        gin_channels,
        kernel_size,
        stride,
        extend_ratio,
        r,
        use_spectral_norm=False,
    ):
        super(MBConv2d, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        inner_channels = int(in_channels * extend_ratio)
        self.kernel_size = kernel_size
        self.pwconv1 = LoRALinear2d(in_channels, inner_channels, gin_channels, r=r)
        self.dwconv = norm_f(
            Conv2d(
                inner_channels,
                inner_channels,
                kernel_size,
                stride,
                groups=inner_channels,
            )
        )
        self.pwconv2 = LoRALinear2d(inner_channels, out_channels, gin_channels, r=r)
        self.pwnorm = LayerNorm(in_channels)
        self.dwnorm = LayerNorm(inner_channels)

    def forward(self, x, g):
        x = self.pwnorm(x)
        x = self.pwconv1(x, g)
        x = F.pad(x, [0, 0, self.kernel_size[0] - 1, 0], mode="constant")
        x = self.dwnorm(x)
        x = self.dwconv(x)
        x = F.leaky_relu(x, modules.LRELU_SLOPE)
        x = self.pwconv2(x, g)
        x = F.leaky_relu(x, modules.LRELU_SLOPE)
        return x


class ConvNext2d(torch.nn.Module):
    """
    Causal ConvNext Block
    stride = 1 only
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        gin_channels,
        kernel_size,
        stride,
        extend_ratio,
        r,
        use_spectral_norm=False,
    ):
        super(ConvNext2d, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        inner_channels = int(in_channels * extend_ratio)
        self.kernel_size = kernel_size
        self.dwconv = norm_f(
            Conv2d(in_channels, in_channels, kernel_size, stride, groups=in_channels)
        )
        self.pwconv1 = LoRALinear2d(in_channels, inner_channels, gin_channels, r=r)
        self.pwconv2 = LoRALinear2d(inner_channels, out_channels, gin_channels, r=r)
        self.norm = LayerNorm(in_channels)
        self.act = nn.GELU()

    def forward(self, x, g):
        x = F.pad(x, [0, 0, self.kernel_size[0] - 1, 0], mode="constant")
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x, g)
        x = self.act(x)
        x = self.pwconv2(x, g)
        x = self.act(x)
        return x

    def remove_weight_norm(self):
        remove_weight_norm(self.dwconv)


class WaveBlock(torch.nn.Module):
    def __init__(
        self,
        inner_channels,
        gin_channels,
        kernel_sizes,
        strides,
        dilations,
        extend_rate,
        r_out,
    ):
        super(WaveBlock, self).__init__()
        norm_f = weight_norm
        extend_channels = int(inner_channels * extend_rate)
        self.dconvs = nn.ModuleList()
        self.p1convs = nn.ModuleList()
        self.p2convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.act = nn.GELU()

        # self.ses = nn.ModuleList()
        # self.norms = []
        for i, (k, s, d) in enumerate(zip(kernel_sizes, strides, dilations)):
            self.dconvs.append(
                DilatedCausalConv1d(
                    inner_channels,
                    inner_channels,
                    k,
                    stride=s,
                    dilation=d,
                    groups=inner_channels,
                )
            )
            self.p1convs.append(
                LoRALinear1d(inner_channels, extend_channels, gin_channels, r_out)
            )
            self.p2convs.append(
                LoRALinear1d(extend_channels, inner_channels, gin_channels, r_out)
            )
            self.norms.append(LayerNorm(inner_channels))

    def forward(self, x, g_out):
        for i in range(len(self.dconvs)):
            residual = x.clone()
            x = self.dconvs[i](x)
            x = self.norms[i](x)
            x = self.p1convs[i](x, g_out)
            x = self.act(x)
            x = self.p2convs[i](x, g_out)
            x = residual + x
        return x

    def remove_weight_norm(self):
        for c in self.dconvs:
            c.remove_weight_norm()
        for c in self.p1convs:
            c.remove_weight_norm()
        for c in self.p2convs:
            c.remove_weight_norm()

    def fix_speaker(self, i, g):
        for c in self.p1convs:
            c.fix_speaker(i, g)
        for c in self.p2convs:
            c.fix_speaker(i, g)

    def unfix_speaker(self, i, g):
        for c in self.p1convs:
            c.unfix_speaker(i, g)
        for c in self.p2convs:
            c.unfix_speaker(i, g)


class SnakeFilter(torch.nn.Module):
    """
    Adaptive filter using snakebeta
    """

    def __init__(self, channels, groups, kernel_size, num_layers, eps=1e-6):
        super(SnakeFilter, self).__init__()
        self.eps = eps
        self.num_layers = num_layers
        inner_channels = channels * groups
        self.init_conv = DilatedCausalConv1d(1, inner_channels, kernel_size)
        self.dconvs = torch.nn.ModuleList()
        self.pconvs = torch.nn.ModuleList()
        self.post_conv = DilatedCausalConv1d(
            inner_channels + 1, 1, kernel_size, bias=False
        )

        for i in range(self.num_layers):
            self.dconvs.append(
                DilatedCausalConv1d(
                    inner_channels,
                    inner_channels,
                    kernel_size,
                    stride=1,
                    groups=inner_channels,
                    dilation=kernel_size ** (i + 1),
                )
            )
            self.pconvs.append(
                weight_norm(Conv1d(inner_channels, inner_channels, 1, groups=groups))
            )
        self.snake_alpha = torch.nn.Parameter(
            torch.zeros(inner_channels), requires_grad=True
        )
        self.snake_beta = torch.nn.Parameter(
            torch.zeros(inner_channels), requires_grad=True
        )

    def forward(self, x):
        y = x.clone()
        x = self.init_conv(x)
        for i in range(self.num_layers):
            # snake activation
            x = self.dconvs[i](x)
            x = self.pconvs[i](x)
        x = x + (
            1.0 / torch.clip(self.snake_beta.unsqueeze(0).unsqueeze(-1), min=self.eps)
        ) * torch.pow(torch.sin(x * self.snake_alpha.unsqueeze(0).unsqueeze(-1)), 2)
        x = torch.cat([x, y], 1)
        x = self.post_conv(x)
        return x

    def remove_weight_norm(self):
        self.init_conv.remove_weight_norm()
        for c in self.dconvs:
            c.remove_weight_norm()
        for c in self.pconvs:
            remove_weight_norm(c)
        self.post_conv.remove_weight_norm()


"""
https://github.com/charactr-platform/vocos/blob/main/vocos/heads.py
"""


class FourierHead(nn.Module):
    """Base class for inverse fourier modules."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (B, L, H), where B is the batch size,
                        L is the sequence length, and H denotes the model dimension.

        Returns:
            Tensor: Reconstructed time-domain audio signal of shape (B, T), where T is the length of the output signal.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")


class IMDCT(nn.Module):
    """
    Inverse Modified Discrete Cosine Transform (IMDCT) module.

    Args:
        frame_len (int): Length of the MDCT frame.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
    """

    def __init__(self, frame_len: int, padding: str = "same"):
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        self.frame_len = frame_len * 2
        N = frame_len
        n0 = (N + 1) / 2
        window = torch.from_numpy(scipy.signal.cosine(N * 2)).float()
        self.register_buffer("window", window)

        pre_twiddle = torch.exp(1j * torch.pi * n0 * torch.arange(N * 2) / N)
        post_twiddle = torch.exp(1j * torch.pi * (torch.arange(N * 2) + n0) / (N * 2))
        self.register_buffer("pre_twiddle", torch.view_as_real(pre_twiddle))
        self.register_buffer("post_twiddle", torch.view_as_real(post_twiddle))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Apply the Inverse Modified Discrete Cosine Transform (IMDCT) to the input MDCT coefficients.

        Args:
            X (Tensor): Input MDCT coefficients of shape (B, N, L), where B is the batch size,
                L is the number of frames, and N is the number of frequency bins.

        Returns:
            Tensor: Reconstructed audio waveform of shape (B, T), where T is the length of the audio.
        """
        X = X.transpose(1, 2)
        B, L, N = X.shape
        Y = torch.zeros((B, L, N * 2), dtype=X.dtype, device=X.device)
        Y[..., :N] = X
        Y[..., N:] = -1 * torch.conj(torch.flip(X, dims=(-1,)))
        y = torch.fft.ifft(
            Y * torch.view_as_complex(self.pre_twiddle).expand(Y.shape), dim=-1
        )
        y = (
            torch.real(y * torch.view_as_complex(self.post_twiddle).expand(y.shape))
            * np.sqrt(N)
            * np.sqrt(2)
        )
        result = y * self.window.expand(y.shape)
        output_size = (1, (L + 1) * N)
        audio = torch.nn.functional.fold(
            result.transpose(1, 2),
            output_size=output_size,
            kernel_size=(1, self.frame_len),
            stride=(1, self.frame_len // 2),
        )[:, 0, 0, :]

        if self.padding == "center":
            pad = self.frame_len // 2
        elif self.padding == "same":
            pad = self.frame_len // 4
        else:
            raise ValueError("Padding must be 'center' or 'same'.")

        audio = audio[:, pad:-pad]
        return audio.unsqueeze(1)


class ISTFT(nn.Module):
    """
    Custom implementation of ISTFT since torch.istft doesn't allow custom padding (other than `center=True`) with
    windowing. This is because the NOLA (Nonzero Overlap Add) check fails at the edges.
    See issue: https://github.com/pytorch/pytorch/issues/62323
    Specifically, in the context of neural vocoding we are interested in "same" padding analogous to CNNs.
    The NOLA constraint is met as we trim padded samples anyway.

    Args:
        n_fft (int): Size of Fourier transform.
        hop_length (int): The distance between neighboring sliding window frames.
        win_length (int): The size of window frame and STFT filter.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
    """

    def __init__(
        self, n_fft: int, hop_length: int, win_length: int, padding: str = "same"
    ):
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        window = torch.hann_window(win_length)
        self.register_buffer("window", window)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Compute the Inverse Short Time Fourier Transform (ISTFT) of a complex spectrogram.

        Args:
            spec (Tensor): Input complex spectrogram of shape (B, N, T), where B is the batch size,
                            N is the number of frequency bins, and T is the number of time frames.

        Returns:
            Tensor: Reconstructed time-domain signal of shape (B, L), where L is the length of the output signal.
        """
        if self.padding == "center":
            # Fallback to pytorch native implementation
            return torch.istft(
                spec,
                self.n_fft,
                self.hop_length,
                self.win_length,
                self.window,
                center=True,
            )
        elif self.padding == "same":
            pad = (self.win_length - self.hop_length) // 2
        else:
            raise ValueError("Padding must be 'center' or 'same'.")

        assert spec.dim() == 3, "Expected a 3D tensor as input"
        B, N, T = spec.shape

        # Inverse FFT
        ifft = torch.fft.irfft(spec, self.n_fft, dim=1, norm="backward")
        ifft = ifft * self.window[None, :, None]

        # Overlap and Add
        output_size = (T - 1) * self.hop_length + self.win_length
        y = torch.nn.functional.fold(
            ifft,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        )[:, 0, 0, pad:-pad]

        # Window envelope
        window_sq = self.window.square().expand(1, T, -1).transpose(1, 2)
        window_envelope = torch.nn.functional.fold(
            window_sq,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        ).squeeze()[pad:-pad]

        # Normalize
        assert (window_envelope > 1e-11).all()
        y = y / window_envelope

        return y


class IMDCTSymExpHead(FourierHead):
    """
    IMDCT Head module for predicting MDCT coefficients with symmetric exponential function

    Args:
        dim (int): Hidden dimension of the model.
        mdct_frame_len (int): Length of the MDCT frame.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
        sample_rate (int, optional): The sample rate of the audio. If provided, the last layer will be initialized
                                     based on perceptual scaling. Defaults to None.
        clip_audio (bool, optional): Whether to clip the audio output within the range of [-1.0, 1.0]. Defaults to False.
    """

    def __init__(
        self,
        dim: int,
        gin_channels: int,
        mdct_frame_len: int,
        padding: str = "same",
        sample_rate: int = 24000,
    ):
        super().__init__()
        out_dim = mdct_frame_len
        self.dconv = DilatedCausalConv1d(dim, dim, 5, 1, dim, 1)
        self.pconv1 = LoRALinear1d(dim, dim * 2, gin_channels, 4, 8)
        self.pconv2 = LoRALinear1d(dim * 2, out_dim, gin_channels, 4, 12)
        # self.out = LoRALinear1d(dim, out_dim, gin_channels, 2, 12)
        self.act = torch.nn.GELU()
        self.imdct = IMDCT(frame_len=mdct_frame_len, padding=padding)

        if sample_rate is not None:
            # optionally init the last layer following mel-scale
            m_max = _hz_to_mel(sample_rate // 2)
            m_pts = torch.linspace(0, m_max, out_dim)
            f_pts = _mel_to_hz(m_pts)
            scale = 1 - (f_pts / f_pts.max())

            with torch.no_grad():
                self.pconv2.main_fc.weight.mul_(scale.view(-1, 1, 1))

    def forward(
        self, x: torch.Tensor, g_in: torch.Tensor, g_out: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the IMDCTSymExpHead module.

        Args:
            x (Tensor): Input tensor of shape (B, L, H), where B is the batch size,
                        L is the sequence length, and H denotes the model dimension.

        Returns:
            Tensor: Reconstructed time-domain audio signal of shape (B, T), where T is the length of the output signal.
        """
        x = self.dconv(x)
        x = self.pconv1(x, g_in, g_out)
        x = self.act(x)
        x = self.pconv2(x, g_in, g_out)
        # x = self.act(x)
        # x = self.out(x, g_in, g_out)
        x = symexp(x)
        x = torch.clip(
            x, min=-1e2, max=1e2
        )  # safeguard to prevent excessively large magnitudes
        audio = self.imdct(x)
        return audio

    def remove_weight_norm(self):
        self.dconv.remove_weight_norm()
        self.pconv1.remove_weight_norm()
        self.pconv2.remove_weight_norm()

    def fix_speaker(self, i, g):
        self.pconv1.fix_speaker(i, g)
        self.pconv2.fix_speaker(i, g)

    def unfix_speaker(self, i, g):
        self.pconv1.unfix_speaker(i, g)
        self.pconv2.unfix_speaker(i, g)


class ISTFTHead(FourierHead):
    """
    ISTFT Head module for predicting STFT complex coefficients.

    Args:
        dim (int): Hidden dimension of the model.
        n_fft (int): Size of Fourier transform.
        hop_length (int): The distance between neighboring sliding window frames, which should align with
                          the resolution of the input features.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
    """

    def __init__(
        self,
        dim: int,
        gin_channels: int,
        n_fft: int,
        hop_length: int,
        padding: str = "same",
    ):
        super().__init__()
        out_dim = n_fft + 2
        self.dconv = DilatedCausalConv1d(dim, dim, 5, 1, dim, 1)
        self.pconv1 = LoRALinear1d(dim, dim * 3, gin_channels, 12)
        self.pconv2 = LoRALinear1d(dim * 3, dim, gin_channels, 12)
        self.out = LoRALinear1d(dim, out_dim, gin_channels, 16)
        self.act = torch.nn.GELU()
        self.istft = ISTFT(
            n_fft=n_fft, hop_length=hop_length, win_length=n_fft, padding=padding
        )

    def forward(self, x: torch.Tensor, g_out: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ISTFTHead module.

        Args:
            x (Tensor): Input tensor of shape (B, L, H), where B is the batch size,
                        L is the sequence length, and H denotes the model dimension.

        Returns:
            Tensor: Reconstructed time-domain audio signal of shape (B, T), where T is the length of the output signal.
        """
        x = self.dconv(x)
        x = self.pconv1(x, g_out)
        x = self.act(x)
        x = self.pconv2(x, g_out)
        x = self.act(x)
        x = self.out(x, g_out)
        mag, p = x.chunk(2, dim=1)
        mag = torch.exp(mag)
        mag = torch.clip(
            mag, max=1e2
        )  # safeguard to prevent excessively large magnitudes
        # wrapping happens here. These two lines produce real and imaginary value
        x = torch.cos(p)
        y = torch.sin(p)
        # recalculating phase here does not produce anything new
        # only costs time
        # phase = torch.atan2(y, x)
        # S = mag * torch.exp(phase * 1j)
        # better directly produce the complex value
        S = mag * (x + 1j * y)
        audio = self.istft(S)
        return audio.unsqueeze(1)

    def remove_weight_norm(self):
        self.dconv.remove_weight_norm()
        self.pconv1.remove_weight_norm()
        self.pconv2.remove_weight_norm()
        self.out.remove_weight_norm()

    def fix_speaker(self, i, g):
        self.pconv1.fix_speaker(i, g)
        self.pconv2.fix_speaker(i, g)
        self.out.fix_speaker(i, g)

    def unfix_speaker(self, i, g):
        self.pconv1.fix_speaker(i, g)
        self.pconv2.fix_speaker(i, g)
        self.out.unfix_speaker(i, g)


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.exp(x.abs()) - 1)


def safe_power(a, x):
    return torch.sign(x) * torch.pow(a, torch.clamp(torch.abs(x), min=1e-5))


def gap2d(x):
    return safe_power(1 / 3, safe_power(3.0, x).mean(dim=(2, 3)))
