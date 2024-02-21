import gradio as gr
import logging
import sys

# Tabs
from tabs.server import server_tab
from tabs.training import train_tab


logging.getLogger("httpx").setLevel(logging.CRITICAL)

with gr.Blocks(theme="remilia/Ghostly", title="VoRAS") as VoRAS:

    gr.Markdown("# VoRAS")
    gr.Markdown("Vocos Retrieval and self-Augmentation for Speech")
    with gr.Tabs():
        with gr.Tab("Inference"):
            server_tab()
        with gr.Tab("Training"):
            train_tab()


if __name__ == "__main__":
    VoRAS.launch(
        share="--share" in sys.argv,
        inbrowser="--open" in sys.argv,
        server_port=6969,
    )
