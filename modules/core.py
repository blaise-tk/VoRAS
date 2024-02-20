import hashlib
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor

import requests

from modules.models import MODELS_DIR
from modules.shared import ROOT_DIR
from modules.utils import download_file


def get_hf_etag(url: str):
    r = requests.head(url)

    etag = r.headers["X-Linked-ETag"] if "X-Linked-ETag" in r.headers else ""

    if etag.startswith('"') and etag.endswith('"'):
        etag = etag[1:-1]

    return etag


def calc_sha256(filepath: str):
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def download_models():
    def exist_check(url: str, out: str):
        return os.path.exists(out)

    os.makedirs(os.path.join(MODELS_DIR, "pretrained", "beta"), exist_ok=True)

    tasks = []
    for template in [
        "D{}k",
        "G{}k",
    ]:
        basename = template.format("24")
        url = (
            f"https://huggingface.co/datasets/nadare/voras/resolve/main/{basename}.pth"
        )
        out = os.path.join(MODELS_DIR, "pretrained", "beta", f"{basename}.pth")

        if exist_check(url, out):
            continue

        tasks.append((url, out))

    for filename in ["voras_pretrain_libritts_r.pth", "voras_sample_japanese.pth"]:
        out = os.path.join(MODELS_DIR, "pretrained", "beta", filename)
        url = f"https://huggingface.co/datasets/nadare/voras/resolve/main/{filename}"

        if exist_check(url, out):
            continue

        tasks.append((url, out))

    # japanese-hubert-base (Fairseq)
    # from official repo
    # NOTE: change filename?
    hubert_jp_url = f"https://huggingface.co/rinna/japanese-hubert-base/resolve/main/fairseq/model.pt"
    out = os.path.join(MODELS_DIR, "embeddings", "rinna_hubert_base_jp.pt")
    if not exist_check(hubert_jp_url, out):
        tasks.append(
            (
                hubert_jp_url,
                out,
            )
        )

    if len(tasks) < 1:
        return

    with ThreadPoolExecutor() as pool:
        pool.map(
            download_file,
            *zip(
                *[(filename, out, i, True) for i, (filename, out) in enumerate(tasks)]
            ),
        )


def install_ffmpeg():
    if os.path.exists(os.path.join(ROOT_DIR, "bin", "ffmpeg.exe")):
        return
    tmpdir = os.path.join(ROOT_DIR, "tmp")
    url = (
        "https://www.gyan.dev/ffmpeg/builds/packages/ffmpeg-5.1.2-essentials_build.zip"
    )
    out = os.path.join(tmpdir, "ffmpeg.zip")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    download_file(url, out)
    shutil.unpack_archive(out, os.path.join(tmpdir, "ffmpeg"))
    shutil.copyfile(
        os.path.join(
            tmpdir, "ffmpeg", "ffmpeg-5.1.2-essentials_build", "bin", "ffmpeg.exe"
        ),
        os.path.join(ROOT_DIR, "bin", "ffmpeg.exe"),
    )
    os.remove(os.path.join(tmpdir, "ffmpeg.zip"))
    shutil.rmtree(os.path.join(tmpdir, "ffmpeg"))


def update_modelnames():
    if not os.path.exists(os.path.join(MODELS_DIR, "embeddings")):
        os.makedirs(os.path.join(MODELS_DIR, "embeddings"))


def preload():
    update_modelnames()
    download_models()
    if sys.platform == "win32":
        install_ffmpeg()
