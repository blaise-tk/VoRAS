import os
import requests

downloads = {
    "https://huggingface.co/datasets/nadare/voras/resolve/main/D24k.pth": "models/pretrained/beta/D24k.pth",
    "https://huggingface.co/datasets/nadare/voras/resolve/main/G24k.pth": "models/pretrained/beta/G24k.pth",
    "https://huggingface.co/datasets/nadare/voras/resolve/main/voras_pretrain_libritts_r.pth": "models/pretrained/beta/voras_pretrain_libritts_r.pth",
    "https://huggingface.co/datasets/nadare/voras/resolve/main/voras_sample_japanese.pth": "models/pretrained/beta/voras_sample_japanese.pth",
    "https://huggingface.co/rinna/japanese-hubert-base/resolve/main/fairseq/model.pt": "models/embeddings/model.pt",
    "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/ffprobe.exe": "ffprobe.exe",
    "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/ffmpeg.exe": "ffmpeg.exe",
}

# Create directories if they don't exist
for path in downloads.values():
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

# Download files
for url, path in downloads.items():
    response = requests.get(url)
    if response.status_code == 200:
        with open(path, "wb") as f:
            f.write(response.content)
        print(f"Downloaded {url} to {path}")
    else:
        print(f"Failed to download {url}. Status code: {response.status_code}")
