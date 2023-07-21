import os
import re
from typing import *

import faiss
import numpy as np
import pyworld
import scipy.signal as signal
import torch
import torch.nn.functional as F
import torchaudio
import torchcrepe
from fairseq import checkpoint_utils
from fairseq.models.hubert.hubert import HubertModel
from pydub import AudioSegment
from torch import Tensor

from lib.voras.models import Synthesizer
from modules.cmd_opts import opts
from modules.models import (EMBEDDINGS_LIST, MODELS_DIR, get_embedder,
                            get_vc_model, update_state_dict)
from modules.shared import ROOT_DIR, device, is_half

MODELS_DIR = opts.models_dir or os.path.join(ROOT_DIR, "models")
vc_model: Optional["VoiceServerModel"] = None
embedder_model: Optional[HubertModel] = None
loaded_embedder_model = ""


class VoiceServerModel:
    def __init__(self, rvc_model_file: str) -> None:
        # setting vram
        global device, is_half
        if isinstance(device, str):
            device = torch.device(device)
        if device.type == "cuda":
            vram = torch.cuda.get_device_properties(device).total_memory / 1024**3
        else:
            vram = None
        if vram is not None and vram <= 4:
            self.x_pad = 1
            self.x_query = 5
            self.x_center = 30
            self.x_max = 32
        elif vram is not None and vram <= 5:
            self.x_pad = 1
            self.x_query = 6
            self.x_center = 38
            self.x_max = 41
        else:
            self.x_pad = 3
            self.x_query = 10
            self.x_center = 60
            self.x_max = 65

        # load_model
        state_dict = torch.load(rvc_model_file, map_location="cpu")
        update_state_dict(state_dict)
        self.state_dict = state_dict
        self.tgt_sr = state_dict["params"]["sr"]
        self.f0 = state_dict.get("f0", 1)
        state_dict["params"]["spk_embed_dim"] = state_dict["weight"][
            "emb_g.weight"
        ].shape[0]
        self.net_g = Synthesizer(**state_dict["params"])
        self.net_g.load_state_dict(state_dict["weight"], strict=False)
        self.net_g.eval().to(device)

        emb_name = state_dict.get("embedder_name", "contentvec")
        if emb_name == "hubert_base":
            emb_name = "contentvec"
        emb_file = os.path.join(MODELS_DIR, "embeddings", EMBEDDINGS_LIST[emb_name][0])
        models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
            [emb_file],
            suffix="",
        )
        embedder_model = models[0]
        embedder_model = embedder_model.to(device)
        embedder_model = embedder_model.float()
        embedder_model.eval()
        self.embedder_model = embedder_model

        self.embedder_output_layer = state_dict["embedder_output_layer"]

        self.n_spk = state_dict["params"]["spk_embed_dim"]

        self.sr = 16000  # hubert input sample rate
        self.window = 160  # hubert input window
        self.t_pad = self.sr * self.x_pad  # padding time for each utterance
        self.t_pad_tgt = self.tgt_sr * self.x_pad
        self.t_pad2 = self.t_pad * 2
        self.t_query = self.sr * self.x_query  # query time before and after query point
        self.t_center = self.sr * self.x_center  # query cut point position
        self.t_max = self.sr * self.x_max  # max time for no query
        self.device = device
        self.is_half = is_half

    def __call__(
        self,
        audio: np.ndarray,
        sr: int,
        sid: int
    ):
        # bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=16000)
        # audio = signal.filtfilt(bh, ah, audio)
        print(sr, audio.shape)
        if sr != self.sr:
            audio = torchaudio.functional.resample(torch.from_numpy(audio), sr, self.sr, rolloff=0.99).detach().cpu().numpy()
        print(sr, audio.shape)
        audio = (audio / np.maximum(np.max(np.abs(audio), keepdims=True), 1e-7) * (.95 * .8)) + 0.2 * audio
        audio_pad = np.pad(audio, (self.window // 2, self.window // 2), mode="reflect" if audio.shape[0] > self.window // 2 else "constant")
        print(audio_pad.shape)

        opt_ts = []
        if audio_pad.shape[0] > self.t_max:
            audio_sum = np.zeros_like(audio)
            for i in range(self.window):
                audio_sum += audio_pad[i : i - self.window]
            for t in range(self.t_center, audio.shape[0], self.t_center):
                opt_ts.append(
                    t
                    - self.t_query
                    + np.where(
                        np.abs(audio_sum[t - self.t_query : t + self.t_query])
                        == np.abs(audio_sum[t - self.t_query : t + self.t_query]).min()
                    )[0][0]
                )
        print(audio_pad.shape)

        sid = torch.tensor(sid, device=self.device).unsqueeze(0).long()
        pitch, pitchf = None, None
        audio_opt = []

        audio_opt.append(
            self._convert(
                sid,
                audio_pad,
            )
        )
        audio_opt = np.concatenate(audio_opt)
        del pitch, pitchf, sid
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return audio_opt


    def _convert(
        self,
        sid: int,
        audio: np.ndarray,
    ):
        audio = torch.from_numpy(audio).float().to(device)
        if audio.dim() == 2:  # double channels
            audio = audio.mean(-1)
        feats = audio.view(1, -1).detach().to(device)
        padding_mask = torch.BoolTensor(feats.shape).to(self.device).fill_(False)
        is_feats_dim_768 = self.net_g.emb_channels == 768

        if isinstance(self.embedder_model, tuple):
            feats = self.embedder_model[0](
                feats.squeeze(0).squeeze(0).to(self.device),
                return_tensors="pt",
                sampling_rate=16000,
            )
            feats = feats.input_values.to(self.device)
            with torch.no_grad():
                if is_feats_dim_768:
                    feats = self.embedder_model[1](feats).last_hidden_state
        else:
            inputs = {
                "source": feats.to(self.device),
                "padding_mask": padding_mask.to(self.device),
                "output_layer": self.embedder_output_layer,
            }
            with torch.no_grad():
                logits = self.embedder_model.extract_features(**inputs)
                if is_feats_dim_768:
                    feats = logits[0]
                else:
                    feats = self.embedder_model.final_proj(logits[0])

        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)

        # sid = torch.L        print(sid)
        with torch.no_grad():
            audio1 = (
                (self.net_g.infer(feats, sid)[0][0, 0] * 32768)
                .data.cpu()
                .float()
                .numpy()
                .astype(np.int16)
            )
        del feats, padding_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return audio1