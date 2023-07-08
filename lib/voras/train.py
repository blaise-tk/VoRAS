import glob
import json
import operator
import os
import sys
import time
from itertools import chain
from random import shuffle
from typing import *

import faiss
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torchaudio
import tqdm
from sklearn.cluster import MiniBatchKMeans
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from . import commons, utils
from .checkpoints import save
from .config import DatasetMetadata, TrainConfig
from .data_utils import AudioLabelCollate, AudioLabelLoader
from .losses import (MelLoss, contrastive_loss, discriminator_loss,
                     feature_loss, generator_loss)
from .mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from .models import MultiPeriodDiscriminator, Synthesizer
from .preprocessing.extract_feature import (MODELS_DIR, get_embedder,
                                            load_embedder)
from .utils import AWP, CosineAnnealingWarmupRestarts


def is_audio_file(file: str):
    if "." not in file:
        return False
    ext = os.path.splitext(file)[1]
    return ext.lower() in [
        ".wav",
        ".flac",
        ".ogg",
        ".mp3",
        ".m4a",
        ".wma",
        ".aiff",
    ]


def glob_dataset(
    glob_str: str,
    multiple_speakers: bool = False,
    recursive: bool = True,
    training_dir: str = "."
):
    globs = glob_str.split(",")
    speaker_count = 0
    datasets_speakers = []
    speaker_to_id_mapping = {}
    meta = {
        "type": "raw_dataset",
        "files": []
    }
    datasets_speakers = []
    for glob_str in globs:
        if not os.path.isdir(glob_str):
            continue
        if multiple_speakers:
            # Multispeaker format:
            # dataset_path/
            # - speakername/
            #     - {wav name here}.wav
            #     - ...
            # - next_speakername/
            #     - {wav name here}.wav
            #     - ...
            # - ...
            print("Multispeaker dataset enabled; Processing speakers.")
            for dir in tqdm.tqdm(os.listdir(glob_str)):
                if not os.path.isdir(os.path.join(glob_str, dir)):
                    continue
                speaker_to_id_mapping[dir] = speaker_count
                speaker_path = os.path.join(glob_str, dir)
                datasets_speaker = [
                    (file, speaker_count)
                    for file in glob.iglob(
                        os.path.join(speaker_path, "**", "*"), recursive=recursive
                    )
                    if is_audio_file(file)
                ]
                if len(datasets_speaker):
                    print("Speaker ID " + str(speaker_count) + ": " + dir)
                    datasets_speakers.extend(datasets_speaker)
                    speaker_count += 1
        else:
            glob_str = os.path.join(glob_str, "**", "*")
            print("Single speaker dataset enabled; Processing speaker as ID " + str(0) + ".")
            datasets_speakers.extend(
                [
                    (file, 0)
                    for file in glob.iglob(glob_str, recursive=recursive)
                    if is_audio_file(file)
                ]
            )
    if len(speaker_to_id_mapping):
        with open(os.path.join(training_dir, "speaker_info.json"), "w") as outfile:
            print("Dumped speaker info to ./speaker_info.json")
            json.dump(speaker_to_id_mapping, outfile)
    return sorted(datasets_speakers)


def create_dataset_meta(
    glob_str: str,
    multiple_speakers: bool = False,
    recursive: bool = True,
    training_dir: str = ".",
    segment_size: int = 48000
):
    meta = {
        "type": "raw_dataset",
        "files": [],
    }

    for file, speaker_id in glob_dataset(glob_str, multiple_speakers, recursive, training_dir):
        count = max(1, os.path.getsize(file) // 2 // segment_size)
        for _ in range(count):
            meta["files"].append({"raw_file": file, "speaker_id": speaker_id})

    with open(os.path.join(training_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def change_speaker(net_g, embedder, embedding_output_layer, phone, wave_16k):
    """
    random change formant
    inspired by https://github.com/auspicious3000/contentvec/blob/d746688a32940f4bee410ed7c87ec9cf8ff04f74/contentvec/data/audio/audio_utils_1.py#L179
    """
    N = phone.shape[0]
    device = phone.device
    dtype = phone.dtype

    new_sid = np.random.randint(net_g.spk_embed_dim, size=N)
    new_sid = torch.from_numpy(new_sid).to(device)

    new_wave = net_g.infer(phone, wave_16k, new_sid)[0]
    new_wave_16k = torchaudio.functional.resample(new_wave, net_g.sr, 16000, rolloff=0.99)
    padding_mask = torch.zeros_like(new_wave_16k, dtype=torch.bool).to(device)

    inputs = {
        "source": new_wave_16k.squeeze(1).to(device, dtype),
        "padding_mask": padding_mask.squeeze(1).to(device),
        "output_layer": embedding_output_layer
    }

    logits = embedder.extract_features(**inputs)
    feats = logits[0]
    feats = torch.repeat_interleave(feats, 2, 1)
    return feats.to(device), new_wave, new_wave_16k


def train_model(
    gpus: List[int],
    num_cpu_process: int,
    config: TrainConfig,
    training_dir: str,
    model_name: str,
    out_dir: str,
    sample_rate: int,
    f0: bool,
    batch_size: int,
    augment: bool,
    augment_path: Optional[str],
    multiple_speakers: bool,
    total_epoch: int,
    save_every_epoch: int,
    finetuning: bool,
    pretrain_g: str,
    pretrain_d: str,
    embedder_name: str,
    embedding_output_layer: int,
    save_only_last: bool = False,
    device: Optional[Union[str, torch.device]] = None,
):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(utils.find_empty_port())

    deterministic = torch.backends.cudnn.deterministic
    benchmark = torch.backends.cudnn.benchmark
    PREV_CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", None)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(gpu) for gpu in gpus])

    start = time.perf_counter()

    # Mac(MPS)でやると、mp.spawnでなんかトラブルが出るので普通にtraining_runnerを呼び出す。
    if device is not None:
        training_runner(
            0,  # rank
            1,  # world size
            num_cpu_process,
            config,
            training_dir,
            model_name,
            out_dir,
            sample_rate,
            f0,
            batch_size,
            augment,
            augment_path,
            multiple_speakers,
            total_epoch,
            save_every_epoch,
            finetuning,
            pretrain_g,
            pretrain_d,
            embedder_name,
            embedding_output_layer,
            save_only_last,
            device,
        )
    else:
        mp.spawn(
            training_runner,
            nprocs=len(gpus),
            args=(
                len(gpus),
                num_cpu_process,
                config,
                training_dir,
                model_name,
                out_dir,
                sample_rate,
                f0,
                batch_size,
                augment,
                augment_path,
                multiple_speakers,
                total_epoch,
                save_every_epoch,
                finetuning,
                pretrain_g,
                pretrain_d,
                embedder_name,
                embedding_output_layer,
                save_only_last,
                device,
            ),
        )

    end = time.perf_counter()

    print(f"Time: {end - start}")

    if PREV_CUDA_VISIBLE_DEVICES is None:
        del os.environ["CUDA_VISIBLE_DEVICES"]
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = PREV_CUDA_VISIBLE_DEVICES

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark


def training_runner(
    rank: int,
    world_size: List[int],
    num_cpu_process: int,
    config: TrainConfig,
    training_dir: str,
    model_name: str,
    out_dir: str,
    sample_rate: int,
    f0: bool,
    batch_size: int,
    augment: bool,
    augment_path: Optional[str],
    multiple_speakers: bool,
    total_epoch: int,
    save_every_epoch: int,
    finetuning: bool,
    pretrain_g: str,
    pretrain_d: str,
    embedder_name: str,
    embedding_output_layer: int,
    save_only_last: bool = False,
    device: Optional[Union[str, torch.device]] = None,
):
    config.train.batch_size = batch_size
    log_dir = os.path.join(training_dir, "logs")
    state_dir = os.path.join(training_dir, "state")
    embedder_out_channels = config.model.emb_channels

    is_multi_process = world_size > 1

    if device is not None:
        if type(device) == str:
            device = torch.device(device)

    global_step = 0
    is_main_process = rank == 0

    if is_main_process:
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(state_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)

    dist.init_process_group(
        backend="gloo", init_method="env://", rank=rank, world_size=world_size
    )

    if is_multi_process:
        torch.cuda.set_device(rank)

    torch.manual_seed(config.train.seed)
    training_files_path = os.path.join(training_dir, "meta.json")
    with open(training_files_path, encoding="utf-8") as f:
        d = json.load(f)

    training_meta = DatasetMetadata.parse_file(training_files_path)
    train_dataset = AudioLabelLoader(training_meta, config.data)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                    num_replicas=world_size,
                                                                    rank=rank,
                                                                    shuffle=True)
    collate_fn = AudioLabelCollate()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers = num_cpu_process,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        sampler=train_sampler,
        persistent_workers=True,
        prefetch_factor=2,
    )

    speaker_info = None
    if os.path.exists(os.path.join(training_dir, "speaker_info.json")):
        with open(os.path.join(training_dir, "speaker_info.json"), "r") as f:
            speaker_info = json.load(f)
            config.model.spk_embed_dim = len(speaker_info)

    net_g = Synthesizer(
        config.data.segment_size // config.data.hop_length,
        config.data.filter_length,
        config.data.hop_length,
        **config.model.dict(),
        is_half=False,
        sr=int(sample_rate[:-1] + "000"),
    )
    if finetuning:
        for p in net_g.speaker_embedder.parameters():
            p.requires_grad = False

    if is_multi_process:
        net_g = net_g.cuda(rank)
    else:
        net_g = net_g.to(device=device)

    periods = [1, 2, 3, 5, 7, 11, 17, 23, 37]
    net_d = MultiPeriodDiscriminator(periods=periods, **config.model.dict())

    # in GAN, weight decay inn't need
    # https://github.com/juntang-zhuang/Adabelief-Optimizer
    optim_g = torch.optim.Adam(
        net_g.parameters(),
        config.train.learning_rate,
        betas=config.train.betas,
        eps=config.train.eps,
    )
    optim_d = torch.optim.Adam(
        chain(net_d.parameters(), net_g.emb_g.parameters()),
        config.train.learning_rate,
        betas=config.train.betas,
        eps=config.train.eps,
    )

    awp = AWP(net_g, optim_g, adv_lr=1e-3, adv_eps=1e-2)

    if is_multi_process:
        net_g = DDP(net_g, device_ids=[rank])
        net_d = DDP(net_d, device_ids=[rank])

    last_d_state = utils.latest_checkpoint_path(state_dir, "D_*.pth")
    last_g_state = utils.latest_checkpoint_path(state_dir, "G_*.pth")

    if last_d_state is None or last_g_state is None:
        epoch = 1
        global_step = 0
        if os.path.exists(pretrain_g) and os.path.exists(pretrain_d):
            net_g_state = torch.load(pretrain_g, map_location="cpu")["model"]

            emb_spk_size = (config.model.spk_embed_dim, config.model.gin_channels)
            if emb_spk_size != net_g_state["emb_g.weight"].size():
                original_weight = net_g_state["emb_g.weight"]
                net_g_state["emb_g.weight"] = original_weight.mean(dim=0, keepdims=True) * torch.ones(emb_spk_size, device=original_weight.device, dtype=original_weight.dtype)
            if is_multi_process:
                net_g.module.load_state_dict(net_g_state)
            else:
                net_g.load_state_dict(net_g_state)
            del net_g_state

            if is_multi_process:
                net_d.module.load_state_dict(
                    torch.load(pretrain_d, map_location="cpu")["model"]
                )
            else:
                net_d.load_state_dict(
                    torch.load(pretrain_d, map_location="cpu")["model"]
                )
            if is_main_process:
                print(f"loaded pretrained {pretrain_g} {pretrain_d}")

    else:
        _, _, _, epoch = utils.load_checkpoint(last_d_state, net_d, optim_d)
        _, _, _, epoch = utils.load_checkpoint(last_g_state, net_g, optim_g)
        if is_main_process:
            print(f"loaded last state {last_d_state} {last_g_state}")

        epoch += 1
        global_step = (epoch - 1) * len(train_loader)

    if augment:
        # load embedder
        embedder_filepath, _, embedder_load_from = get_embedder(embedder_name)

        if embedder_load_from == "local":
            embedder_filepath = os.path.join(
                MODELS_DIR, "embeddings", embedder_filepath
            )
        embedder, _ = load_embedder(embedder_filepath, device)

        if (augment_path is not None):
            state_dict = torch.load(augment_path, map_location="cpu")
            augment_net_g = Synthesizer(
                **state_dict["params"], is_half=False
            )
            augment_net_g.load_state_dict(state_dict["weight"], strict=False)
            augment_net_g.eval().to(device)
            augment_net_g.remove_weight_norm()
        else:
            augment_net_g = net_g

    if is_multi_process:
        net_d = net_d.cuda(rank)
    else:
        net_d = net_d.to(device=device)

    if finetuning:
        scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
            optim_g, gamma=config.train.lr_decay, last_epoch=epoch - 2
        )
        scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
            optim_d, gamma=config.train.lr_decay, last_epoch=epoch - 2
        )
    else:
        scheduler_g = CosineAnnealingWarmupRestarts(
            optimizer=optim_g,
            first_cycle_steps=len(train_loader),
            cycle_mult=1,
            max_lr=config.train.learning_rate * 5,
            min_lr=config.train.learning_rate,
            first_lr=config.train.learning_rate / 10,
            warmup_steps=int(len(train_loader) * .5),
            gamma=.9,
            last_epoch=-1
        )
        scheduler_d = CosineAnnealingWarmupRestarts(
            optimizer=optim_d,
            first_cycle_steps=len(train_loader),
            cycle_mult=1,
            max_lr=config.train.learning_rate * 5,
            min_lr=config.train.learning_rate,
            first_lr=config.train.learning_rate / 10,
            warmup_steps=int(len(train_loader) * .5),
            gamma=.9,
            last_epoch=-1
        )

    scaler = GradScaler(enabled=config.train.fp16_run)

    mel_loss = MelLoss(
        sample_rate=int(sample_rate[:-1] + "000"),
        n_fft=config.data.filter_length,
        win_length=config.data.win_length,
        hop_length=config.data.hop_length,
        f_min=config.data.mel_fmin,
        f_max=config.data.mel_fmax
    ).to(device=device)

    progress_bar = tqdm.tqdm(range((total_epoch - epoch + 1) * len(train_loader)))
    progress_bar.set_postfix(epoch=epoch)
    step = -1 + len(train_loader) * (epoch - 1)
    optim_g.zero_grad()
    optim_d.zero_grad()
    for epoch in range(epoch, total_epoch + 1):

        net_g.train()
        net_d.train()

        data = enumerate(train_loader)

        if is_main_process:
            lr = optim_g.param_groups[0]["lr"]

        for batch_idx, batch in data:
            step += 1
            progress_bar.update(1)
            (
                wave,
                sid,
            ) = batch
            sid = sid.to(device=device, non_blocking=True)
            wave = wave.to(device=device, non_blocking=True)
            with torch.no_grad():
                wave_16k = torchaudio.functional.resample(wave, net_g.sr, 16000, rolloff=0.99)
                padding_mask = torch.zeros_like(wave_16k.squeeze(1), dtype=torch.bool).to(device)

                inputs = {
                    "source": wave_16k.to(device).squeeze(1),
                    "padding_mask": padding_mask.to(device),
                    "output_layer": embedding_output_layer
                }

                phone = embedder.extract_features(**inputs)[0]
                phone = torch.repeat_interleave(phone, 2, 1)

            if step > 2.5 * len(train_loader):
                awp.perturb()
            with autocast(enabled=config.train.fp16_run, dtype=torch.bfloat16):
                with torch.no_grad():
                    if augment and (finetuning or (step > .5 * len(train_loader))):
                        new_phone, new_wave, new_wave_16k = change_speaker(augment_net_g, embedder, embedding_output_layer, phone, wave_16k)
                        weight = 1 - np.power(.65, (step - .5 * len(train_loader) * (1 - finetuning)) / len(train_loader)) # 学習の初期はそのままのphone embeddingを使う
                    else:
                        new_phone, new_wave, new_wave_16k = phone.detach(), wave.detach(), wave_16k.detach()
                        weight = 1.
                    phone_delta = (phone.shape[1] - new_phone.shape[1])//2
                    if phone_delta:
                        phone = phone[:, phone_delta:-phone_delta]
                    phone = phone * (1. - weight) + new_phone * weight

                    wave_16k_delta = (wave_16k.shape[2] - new_wave_16k.shape[2])//2
                    if wave_16k_delta:
                        wave_16k = wave_16k[:, :, wave_16k_delta:-wave_16k_delta]
                    wave_16k = wave_16k * (1. - weight) + new_wave_16k * weight

                    wave_delta = (wave.shape[2] - phone.shape[1] * config.data.hop_length)//2
                    if wave_delta:
                        wave = wave[:, :, wave_delta:-wave_delta]


                (
                    y_hat,
                    g_in,
                    g_out
                ) = net_g(
                    phone, wave_16k, sid
                )
                y_hat, wave = y_hat[:, :, :wave.shape[2]], wave[:, :, :y_hat.shape[2]]

            with autocast(enabled=config.train.fp16_run, dtype=torch.bfloat16):
                # Generator
                y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(wave, y_hat, g_out.detach())
                g_in = net_g.speaker_embedder(wave_16k)
                with autocast(enabled=False):
                    if finetuning or not multiple_speakers:
                        loss_spk = 0.
                    else:

                        loss_spk = contrastive_loss(g_in, sid, net_g.emb_g.weight.data)
                    loss_mel, y_mel, y_hat_mel = mel_loss(wave.float(), y_hat.float())
                    loss_mel = loss_mel * config.train.c_mel
                    loss_fm = feature_loss(fmap_r, fmap_g)
                    loss_gen, losses_gen = generator_loss(y_d_hat_g)
                    loss_gen_all = loss_gen + loss_fm  + loss_mel + loss_spk
            optim_g.zero_grad()
            #if config.train.fp16_run:
            #    scaler.scale(loss_gen_all).backward()
            #    scaler.unscale_(optim_g)
            #else:
            loss_gen_all.backward()
            if step > 2.5 * len(train_loader):
                awp.restore()
            grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
            if np.all([torch.all(torch.isfinite(p.grad)).detach().cpu().numpy() for p in net_g.parameters() if p.requires_grad and type(p.grad) is torch.Tensor]):
                #if config.train.fp16_run:
                #    scaler.step(optim_g)
                #else:
                optim_g.step()
            else:
                print("contains nan generator")
            # scaler.update()

            with autocast(enabled=config.train.fp16_run, dtype=torch.bfloat16):
                # Discriminator
                g_out = net_g.emb_g(sid).unsqueeze(-1)
                y_d_hat_r, y_d_hat_g, _, _ = net_d(wave, y_hat.detach(), g_out)
                with autocast(enabled=False):
                    loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                        y_d_hat_r, y_d_hat_g
                    )
                    loss_disc_all = loss_disc
            optim_d.zero_grad()
            #if config.train.fp16_run:
            #    scaler.scale(loss_disc_all).backward()
            #    scaler.unscale_(optim_d)
            #else:
            loss_disc_all.backward()
            grad_norm_d = commons.clip_grad_value_(chain(net_d.parameters(), net_g.emb_g.parameters()), None)
            if np.all([torch.all(torch.isfinite(p.grad)).detach().cpu().numpy() for p in chain(net_d.parameters(), net_g.emb_g.parameters()) if p.requires_grad and type(p.grad) is torch.Tensor]):
                #if config.train.fp16_run:
                #    scaler.step(optim_d)
                #else:
                optim_d.step()
            else:
                print("contains nan discriminater")
            # scaler.update()


            scheduler_g.step()
            scheduler_d.step()

            if is_main_process:
                progress_bar.set_postfix(
                    epoch=epoch,
                    loss_g=float(loss_gen_all) if loss_gen_all is not None else 0.0,
                    loss_d=float(loss_disc) if loss_disc is not None else 0.0,
                    lr=float(lr) if lr is not None else 0.0,
                    use_cache=0,
                )
                if global_step % config.train.log_interval == 0:
                    y_hat = torch.clip(y_hat, min=-1., max=1.)
                    for i in range(4):
                        torchaudio.save(filepath=os.path.join(training_dir, "logs", f"y_true_{i:02}.wav"), src=wave[i].detach().cpu().float(), sample_rate=int(sample_rate[:-1] + "000"))
                        torchaudio.save(filepath=os.path.join(training_dir, "logs", f"y_pred_{i:02}.wav"), src=y_hat[i].detach().cpu().float(), sample_rate=int(sample_rate[:-1] + "000"))
                        if augment and (finetuning or (step > .5 * len(train_loader))):
                            torchaudio.save(filepath=os.path.join(training_dir, "logs", f"y_aug_{i:02}.wav"), src=new_wave[i].detach().cpu().float(), sample_rate=int(sample_rate[:-1] + "000"))
                    lr = optim_g.param_groups[0]["lr"]
                    # Amor For Tensorboard display
                    if loss_mel > 50:
                        loss_mel = 50

                    scalar_dict = {
                        "loss/g/total": loss_gen_all,
                        "loss/d/total": loss_disc,
                        "learning_rate": lr,
                        "grad_norm_d": grad_norm_d,
                        "grad_norm_g": grad_norm_g,
                        "loss/spk": loss_spk
                    }
                    scalar_dict.update(
                        {
                            "loss/g/fm": loss_fm,
                            "loss/g/mel": loss_mel,
                        }
                    )

                    scalar_dict.update(
                        {"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)}
                    )
                    scalar_dict.update(
                        {
                            "loss/d_r/{}".format(i): v
                            for i, v in enumerate(losses_disc_r)
                        }
                    )
                    scalar_dict.update(
                        {
                            "loss/d_g/{}".format(i): v
                            for i, v in enumerate(losses_disc_g)
                        }
                    )
                    image_dict = {
                        "slice/mel_org": utils.plot_spectrogram_to_numpy(
                            y_mel[0][0].data.cpu().numpy()
                        ),
                        "slice/mel_gen": utils.plot_spectrogram_to_numpy(
                            y_hat_mel[0][0].data.cpu().numpy()
                        ),
                    }
                    utils.summarize(
                        writer=writer,
                        global_step=global_step,
                        images=image_dict,
                        scalars=scalar_dict,
                    )
            global_step += 1
        if is_main_process and save_every_epoch != 0 and epoch % save_every_epoch == 0:
            if save_only_last:
                old_g_path = os.path.join(
                    state_dir, f"G_{epoch - save_every_epoch}.pth"
                )
                old_d_path = os.path.join(
                    state_dir, f"D_{epoch - save_every_epoch}.pth"
                )
                if os.path.exists(old_g_path):
                    os.remove(old_g_path)
                if os.path.exists(old_d_path):
                    os.remove(old_d_path)
            utils.save_state(
                net_g,
                optim_g,
                config.train.learning_rate,
                epoch,
                os.path.join(state_dir, f"G_{epoch}.pth"),
            )
            utils.save_state(
                net_d,
                optim_d,
                config.train.learning_rate,
                epoch,
                os.path.join(state_dir, f"D_{epoch}.pth"),
            )

            save(
                net_g,
                config.version,
                sample_rate,
                f0,
                embedder_name,
                embedder_out_channels,
                embedding_output_layer,
                os.path.join(training_dir, "checkpoints", f"{model_name}-{epoch}.pth"),
                epoch,
                speaker_info
            )

    if is_main_process:
        print("Training is done. The program is closed.")
        save(
            net_g,
            config.version,
            sample_rate,
            f0,
            embedder_name,
            embedder_out_channels,
            embedding_output_layer,
            os.path.join(out_dir, f"{model_name}.pth"),
            epoch,
            speaker_info
        )
