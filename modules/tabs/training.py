import math
import os
import shutil
from multiprocessing import cpu_count

import gradio as gr

from lib.voras.preprocessing import extract_f0, extract_feature, split
from lib.voras.train import create_dataset_meta, glob_dataset, train_model
from modules import models, utils
from modules.shared import MODELS_DIR, device, half_support
from modules.ui import Tab

SR_DICT = {
    "24k": 24000,
    "32k": 32000,
    "40k": 40000,
    "48k": 48000,
}


class Training(Tab):
    def title(self):
        return "Training"

    def sort(self):
        return 2

    def ui(self, outlet):
        def train_all(
            model_name,
            version,
            sampling_rate_str,
            f0,
            dataset_glob,
            recursive,
            multiple_speakers,
            gpu_id,
            num_cpu_process,
            batch_size,
            augment,
            augment_from_pretrain,
            augment_path,
            num_epochs,
            save_every_epoch,
            fp16,
            finetuning,
            pre_trained_bottom_model_g,
            pre_trained_bottom_model_d,
            embedder_name,
            embedding_channels,
            embedding_output_layer,
            ignore_cache,
        ):
            batch_size = int(batch_size)
            num_epochs = int(num_epochs)
            f0 = f0 == "Yes"
            out_dir = os.path.join(MODELS_DIR, "checkpoints")
            training_dir = os.path.join(MODELS_DIR, "training", "models", model_name)
            mute_path = os.path.join(MODELS_DIR, "training", "mute", "0_gt_wavs", "mute24k.wav")
            gpu_ids = [int(x.strip()) for x in gpu_id.split(",")] if gpu_id else []

            if os.path.exists(training_dir) and ignore_cache:
                shutil.rmtree(training_dir)

            os.makedirs(training_dir, exist_ok=True)
            config = utils.load_config(
                version, training_dir, sampling_rate_str, embedding_channels, fp16
            )

            yield f"Training directory: {training_dir}"
            create_dataset_meta(dataset_glob,
                                multiple_speakers=multiple_speakers,
                                recursive=recursive,
                                training_dir=training_dir,
                                segment_size=config.data.segment_size,
                                mute_path=mute_path
                                )

            yield "Training model..."

            print(f"train_all: emb_name: {embedder_name}")



            if not augment_from_pretrain:
                augment_path = None

            train_model(
                gpu_ids,
                num_cpu_process,
                config,
                training_dir,
                model_name,
                out_dir,
                sampling_rate_str,
                f0,
                batch_size,
                augment,
                augment_path,
                multiple_speakers,
                num_epochs,
                save_every_epoch,
                finetuning,
                pre_trained_bottom_model_g,
                pre_trained_bottom_model_d,
                embedder_name,
                int(embedding_output_layer),
                False,
                None if len(gpu_ids) > 1 else device,
            )
            yield "Training index..."

        with gr.Group():
            with gr.Box():
                with gr.Column():
                    with gr.Row().style():
                        with gr.Column():
                            model_name = gr.Textbox(label="Model Name")
                            ignore_cache = gr.Checkbox(label="Ignore cache")
                        with gr.Column():
                            dataset_glob = gr.Textbox(
                                label="Dataset glob", placeholder="data/**/*.wav"
                            )
                            recursive = gr.Checkbox(label="Recursive", value=True)
                            multiple_speakers = gr.Checkbox(
                                label="Multiple speakers", value=False
                            )
                    with gr.Row().style(equal_height=False):
                        version = gr.Radio(
                            choices=["voras"],
                            value="voras",
                            label="Model version",
                        )
                        target_sr = gr.Radio(
                            choices=["24k"],
                            value="24k",
                            label="Target sampling rate",
                        )
                        f0 = gr.Radio(
                            choices=["No"],
                            value="No",
                            label="f0 Model",
                        )
                    with gr.Row().style(equal_height=False):
                        embedding_name = gr.Radio(
                            choices=list(models.EMBEDDINGS_LIST.keys()),
                            value="hubert-base-japanese",
                            label="Using phone embedder",
                        )
                        embedding_channels = gr.Radio(
                            choices=["768"],
                            value="768",
                            label="Embedding channels",
                        )
                        embedding_output_layer = gr.Radio(
                            choices=["12"],
                            value="12",
                            label="Embedding output layer",
                        )
                    with gr.Row().style(equal_height=False):
                        gpu_id = gr.Textbox(
                            label="GPU ID",
                            value=", ".join([f"{x.index}" for x in utils.get_gpus()]),
                        )
                        num_cpu_process = gr.Slider(
                            minimum=0,
                            maximum=cpu_count(),
                            step=1,
                            value=math.ceil(cpu_count() / 2),
                            label="Number of CPU processes",
                        )
                    with gr.Row().style(equal_height=False):
                        batch_size = gr.Number(value=4, label="Batch size")
                        num_epochs = gr.Number(
                            value=30,
                            label="Number of epochs",
                        )
                        save_every_epoch = gr.Slider(
                            minimum=0,
                            maximum=100,
                            value=10,
                            step=1,
                            label="Save every epoch",
                        )
                        fp16 = gr.Checkbox(
                            label="BFP16", value=half_support, disabled=not half_support
                        )
                    with gr.Row().style(equal_height=False):
                        augment = gr.Checkbox(label="Augment", value=True)
                        augment_from_pretrain = gr.Checkbox(label="Augment From Pretrain", value=True)
                        augment_path = gr.Textbox(
                            label="Pre trained checkpoint path (pth)",
                            value=os.path.join(
                                MODELS_DIR, "pretrained", "beta", "voras_pretrain_libritts_r.pth"
                            ),
                        )
                    with gr.Row().style(equal_height=False):
                        finetuning = gr.Checkbox(label="finetuning", value=True)
                        pre_trained_generator = gr.Textbox(
                            label="Pre trained generator path",
                            value=os.path.join(
                                MODELS_DIR, "pretrained", "beta", "G24k.pth"
                            ),
                        )
                        pre_trained_discriminator = gr.Textbox(
                            label="Pre trained discriminator path",
                            value=os.path.join(
                                MODELS_DIR, "pretrained", "beta", "D24k.pth"
                            ),
                        )
                    with gr.Row().style(equal_height=False):
                        status = gr.Textbox(value="", label="Status")
                    with gr.Row().style(equal_height=False):
                        train_all_button = gr.Button("Train", variant="primary")

        train_all_button.click(
            train_all,
            inputs=[
                model_name,
                version,
                target_sr,
                f0,
                dataset_glob,
                recursive,
                multiple_speakers,
                gpu_id,
                num_cpu_process,
                batch_size,
                augment,
                augment_from_pretrain,
                augment_path,
                num_epochs,
                save_every_epoch,
                fp16,
                finetuning,
                pre_trained_generator,
                pre_trained_discriminator,
                embedding_name,
                embedding_channels,
                embedding_output_layer,
                ignore_cache,
            ],
            outputs=[status],
        )
