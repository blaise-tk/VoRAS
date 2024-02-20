# VoRAS: Vocos Retrieval and self-Augmentation for Speech

VoRAS is a model designed for fast and lightweight real-time voice modification in Japanese, derived from RVC.

## Overview

VoRAS is built upon RVC, with the enhancement of utilizing Vocos as a replacement for the vocoder, and a restructuring of the overall model architecture to a more modern design.

## Note

Development of VoRAS has been discontinued due to the absence of anticipated further improvements in performance. Feel free to fork this repository if you wish to experiment with it further.

# Launch

## Windows

To start the web UI, run the `launch.py` script.

```
Tested environment: Windows 10, Python 3.10.9, torch 2.0.0+cu118
```

# Troubleshooting

## `error: Microsoft Visual C++ 14.0 or greater is required.`

If you encounter this error, you need to install Microsoft C++ Build Tools.

### Step 1: Download the installer

[Download Microsoft C++ Build Tools](https://visualstudio.microsoft.com/ja/thank-you-downloading-visual-studio/?sku=BuildTools&rel=16)

### Step 2: Install `C++ Build Tools`

Run the installer and select `C++ Build Tools` in the `Workloads` tab.

# Credits

VoRAS acknowledges the following projects and resources for their contributions:

- [liujing04/Retrieval-based-Voice-Conversion-WebUI](https://github.com/liujing04/Retrieval-based-Voice-Conversion-WebUI)
- [charactr-platform/Vocos](https://github.com/charactr-platform/vocos)
- [ddPn08/rvc-webui](https://github.com/ddPn08/rvc-webui/tree/main)
- [NVIDIA/BigVGAN](https://github.com/NVIDIA/BigVGAN)
- [rinna/japanese-hubert-base](https://huggingface.co/rinna/japanese-hubert-base)
- [teftef6220/Voice_Separation_and_Selection](https://github.com/teftef6220/Voice_Separation_and_Selection)
- [あみたろの声素材工房](https://amitaro.net/)
- [つくよみちゃん](https://tyc.rei-yumesaki.net/)
- [刻鳴時雨 ITA コーパス読み上げ音声素材](https://booth.pm/ja/items/3640133)
- [れぷりかどーる](https://kikyohiroto1227.wixsite.com/kikoto-utau)
