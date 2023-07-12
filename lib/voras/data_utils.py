import os
import traceback
from random import randint

import numpy as np
import torch
import torch.utils.data
import torchaudio

from .config import DatasetMetadata, DatasetMetaItem, TrainConfigData
from .utils import load_audio, load_wav_to_torch


class AudioLabelLoader(torch.utils.data.Dataset):

    def __init__(self, dataset_meta: DatasetMetadata, data: TrainConfigData):
        self.dataset_meta = dataset_meta
        self.max_wav_value = data.max_wav_value
        self.sampling_rate = data.sampling_rate
        self.filter_length = data.filter_length
        self.hop_length = data.hop_length
        self.win_length = data.win_length
        self.sampling_rate = data.sampling_rate
        self.segment_size = data.segment_size
        self.pre_silence = data.pre_silence
        self.min_text_len = getattr(data, "min_text_len", 1)
        self.max_text_len = getattr(data, "max_text_len", 5000)
        self._filter()

    def _filter(self):
        """
        Filter text & store spec lengths
        """
        # Store spectrogram lengths for Bucketing
        # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
        # spec_length = wav_length // hop_length
        lengths = []
        for data in self.dataset_meta.files:
            lengths.append(os.path.getsize(data.raw_file) // (2 * self.hop_length))
        self.lengths = lengths

    def get_sid(self, sid):
        sid = torch.LongTensor([int(sid)])
        return sid

    def get_audio_text_pair(self, data: DatasetMetaItem):
        # separate filename and text
        file = data.raw_file
        dv = data.speaker_id

        wav = self.get_audio(file)
        dv = self.get_sid(dv)
        return (wav, dv)

    def get_audio(self, filename):
        if filename.endswith(".wav"):
            audio, sampling_rate = load_wav_to_torch(filename)
            if sampling_rate != self.sampling_rate:
                audio = torchaudio.functional.resample(audio, sampling_rate, self.sampling_rate, rolloff=0.99)
        else:
            audio = load_audio(filename, self.sampling_rate)
            audio = torch.FloatTensor(audio)

        audio_normed = audio / self.max_wav_value
        if len(audio_normed.shape) == 1:
            audio_normed = audio_normed.unsqueeze(0)
        elif audio_normed.shape[0] == 2:
            audio_normed = audio_normed.mean(dim=0, keepdim=True)

        audio_normed = (audio_normed / torch.clamp(audio_normed.abs().max(), min=1e-7) * (.95 * .8)) + 0.2 * audio_normed
        audio_trimed = torch.zeros([1, self.segment_size])
        start =max(0, randint(-self.pre_silence, max(0, audio_normed.shape[1] - self.segment_size))) // (self.sampling_rate//100) * (self.sampling_rate//100)
        audio_normed = audio_normed[:, start:start+self.segment_size]
        audio_trimed[:, -audio_normed.shape[1]:] = audio_normed
        return audio_trimed

    def __getitem__(self, index):
        data = self.dataset_meta.files[index]
        return self.get_audio_text_pair(data)

    def __len__(self):
        return len(self.dataset_meta.files)


class AudioLabelCollate:
    """Zero-pads model inputs and targets"""

    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """Collate's training batch from normalized text and aduio
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length

        max_wave_len = max([x[0].size(1) for x in batch])
        wave_padded = torch.FloatTensor(len(batch), 1, max_wave_len)
        wave_padded.zero_()

        sid = torch.LongTensor(len(batch))

        for i in range(len(batch)):
            row = batch[i]
            wave_padded[i, 0, -row[0].size(1):] = row[0]
            sid[i] = row[1]

        return (
            wave_padded,
            sid,
        )