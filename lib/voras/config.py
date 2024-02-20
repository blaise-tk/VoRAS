from typing import *

from pydantic import BaseModel


class TrainConfigTrain(BaseModel):
    log_interval: int
    seed: int
    epochs: int
    learning_rate: float
    betas: List[float]
    eps: float
    batch_size: int
    fp16_run: bool
    lr_decay: float
    init_lr_ratio: int
    warmup_epochs: int
    augment_start_steps: int
    c_mel: int
    c_spk: float


class TrainConfigData(BaseModel):
    max_wav_value: float
    sampling_rate: int
    filter_length: int
    hop_length: int
    win_length: int
    segment_size: int
    pre_silence: int
    n_mel_channels: int
    mel_fmin: float
    mel_fmax: Any


class TrainConfigModel(BaseModel):
    emb_channels: int
    inter_channels: int
    n_layers: int
    upsample_rates: List[int]
    use_spectral_norm: bool
    gin_channels: int
    spk_embed_dim: int


class TrainConfig(BaseModel):
    version: Literal["voras"] = "voras"
    train: TrainConfigTrain
    data: TrainConfigData
    model: TrainConfigModel


class DatasetMetaItem(BaseModel):
    raw_file: str
    speaker_id: int


class DatasetMetadata(BaseModel):
    type: Optional[str]
    files: List[DatasetMetaItem]
