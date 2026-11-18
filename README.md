<div align="center">

# LongCat Avatar 1.6 for ComfyUI (PixelAI Labs)

### One photo + one voice = a lip-synced talking avatar video.

The first ComfyUI nodes for **LongCat-Video-Avatar 1.6** by Meituan: the official **INT8** model, **8 steps**, the 1.6
DMD/OPSD LoRA switch, and videos of any length, continued window by window. Runs on one RTX 5090 (32 GB).

[![Premium ComfyUI courses](https://img.shields.io/badge/Premium%20ComfyUI%20courses-pixelailabs.com-96FF3D?style=for-the-badge&labelColor=050505)](https://pixelailabs.com)
[![Support on Ko-fi](https://img.shields.io/badge/Support%20on-Ko--fi-FF6433?style=for-the-badge&logo=kofi&logoColor=white&labelColor=050505)](https://ko-fi.com/pixelailabs)

![Three AI avatars made with LongCat Avatar 1.6 in ComfyUI: a tech creator, a real estate agent and a fitness coach](docs/hero.jpg)

</div>

---

## Examples

Each video is **one run** from one photo and one ElevenLabs voice line, 480p, seed 42. No LoRA, no face swap, no editing.
Turn the sound on.

**Tech creator, 8.7 s**

https://github.com/user-attachments/assets/72ae65a2-95be-41c2-b837-c3f1b2cdbc02

**Real estate agent, 9.1 s**

https://github.com/user-attachments/assets/eb368081-24e6-488c-a35e-bc5cb860d8fe

**Fitness coach, 8.9 s**

https://github.com/user-attachments/assets/07a1ad71-f23a-4eb6-be9b-4dc3ea8ceed0

---|---|---|
| [![Tech creator](docs/example_tech.gif)](docs/example_tech.mp4) | [![Real estate agent](docs/example_realestate.gif)](docs/example_realestate.mp4) | [![Fitness coach](docs/example_fitness.gif)](docs/example_fitness.mp4) |

---

## Installation

### 1. Install the node

**ComfyUI Manager:** *Custom Nodes Manager* → *Install via Git URL* →
`https://github.com/Pixelailabs/ComfyUI-LongCat-Avatar-PixelAILabs` → restart ComfyUI.

**Manual:**

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Pixelailabs/ComfyUI-LongCat-Avatar-PixelAILabs
cd ComfyUI-LongCat-Avatar-PixelAILabs
pip install -r requirements.txt
```

Use the Python of your ComfyUI (portable: `..\..\..\python_embeded\python.exe -m pip install -r requirements.txt`).
Optional but recommended for speed: `pip install sageattention`.

### 2. Models

The **Model Loader** and **Whisper Loader** download what they need on the first run (about 25 GB) into
`models/longcat`. You only add the text encoder:

| Model | Folder | Size |
|---|---|---|
| [umt5_xxl_fp8_e4m3fn_scaled](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/blob/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors) (or the fp16 one) | `models/text_encoders` | 6.7 GB |
| LongCat-Video-Avatar-1.6 INT8 model + 2 LoRAs *(automatic)* | `models/longcat/LongCat-Video-Avatar-1.6` | 21 GB |
| Whisper large-v3 *(automatic)* | `models/longcat/LongCat-Video-Avatar-1.6/whisper-large-v3` | 3.1 GB |
| LongCat VAE *(automatic)* | `models/longcat/LongCat-Video/vae` | 0.5 GB |

Downloading by hand instead? Take only `base_model_int8/`, `loraA/`, `loraB/`, `scheduler/`, `config.json` and
`whisper-large-v3/model.safetensors` from [meituan-longcat/LongCat-Video-Avatar-1.6](https://huggingface.co/meituan-longcat/LongCat-Video-Avatar-1.6),
and `vae/` from [meituan-longcat/LongCat-Video](https://huggingface.co/meituan-longcat/LongCat-Video). Not the full
77 GB repo.

### 3. Load the workflow

[`example_workflows/LongCat_Avatar_1.6_PixelAILabs.json`](example_workflows/LongCat_Avatar_1.6_PixelAILabs.json):
drag it into ComfyUI, or find it in *Workflow → Browse Templates*.

---

## Quick start

1. **Load Image:** a photo of your avatar, framed the way you want the video (chest up works best).
2. **Load Audio:** clean speech (an ElevenLabs line is perfect). The video is as long as the audio.
3. **Text Encode:** one short paragraph: who it is, where, and "static camera on a tripod".
4. Press **Run**. The video with the voice is saved in `ComfyUI/output/`.

---

## The nodes

| Node | What it does |
|---|---|
| **Model Loader** | Loads 1.6 (or 1.5), INT8 or bf16, downloads missing files |
| **Text Encode** | Your prompt, through a *Load CLIP* (UMT5, type **wan**). Runs on the CPU |
| **Whisper Loader / Audio Encode** | Turns the voice into lip-sync features. Keep `model_version` on 1.6 |
| **Vocal Extract** | Optional: takes the voice out of a song or a noisy clip |
| **Sampler** | Generates the video, window by window (93 frames, about 3.7 s each) |

Settings worth knowing:

| Setting | Node | Default | Notes |
|---|---|---|---|
| `attention_mode` | Model Loader | auto | Picks SageAttention, then FlashAttention 2, then PyTorch |
| `mix_steps` / `mix_alpha` | Model Loader | 4 / 1.0 | The official 1.6 LoRA switch. `mix_steps` 0 is about 15% faster but uses more VRAM |
| `resolution` | Sampler | 480p | 720p is about 3x slower per step |
| `block_num` | Sampler | 1 | 1 streams the model through the GPU. 0 loads it all: faster, needs far more VRAM |

---

## Speed and memory

Tested on an RTX 5090 (32 GB) with 64 GB RAM, Windows, 480p:

| Setup | 3.8 s clip |
|---|---|
| auto attention before SageAttention | 7.7 min |
| SageAttention | 6.6 min |
| SageAttention + `mix_steps` 0 | 5.6 min |

Every extra ~3.2 s of audio adds one window (about 1.5 to 2.5 min at 480p).

**Low VRAM**
- Keep `block_num` **1**, `mix_steps` **4** and **480p**. With streaming, the model itself allocated about 13 GB at 480p
  on our card. We have not tested cards under 32 GB yet.
- Close other GPU apps (video editors, browsers with many tabs): they can take 5 GB and cause out-of-memory errors.
- Out of memory with `mix_steps` 0? Go back to 4.

**Low RAM**
- Use the **fp8 UMT5** text encoder and leave Text Encode `offload_device` on **cpu**.
- With streaming, the INT8 model and the LoRAs wait in system RAM (about 21 GB). 64 GB is comfortable; 32 GB is tight.

---

## Tips

- **Short tests first:** 4 seconds of audio = one window, a few minutes.
- **A calm presenter:** write "hands relaxed" or "hands out of frame" in the prompt.
- **Lip sync:** use clean speech that starts on the first frame, with no music.
- **A take looks off?** Change the seed, not the prompt.

---

## Learn more and support

<table>
<tr>
<td width="50%" valign="top">

### 🎓 Premium ComfyUI courses
Build consistent AI influencers and avatars, and make ultra-realistic photos and talking videos with them: the
**AI Influencer Pro** course and more on **[pixelailabs.com](https://pixelailabs.com)**.

</td>
<td width="50%" valign="top">

### ☕ Support this project
These nodes are free. If they save you hours, you can buy me a coffee:

<a href="https://ko-fi.com/pixelailabs"><img src="https://cdn.prod.website-files.com/5c14e387dab576fe667689cf/670f5a02fcf48af59c591185_support_me_on_kofi_dark.avif" alt="Support me on Ko-fi" height="48"></a>

</td>
</tr>
</table>

---

## Credits

- [LongCat-Video-Avatar 1.6](https://huggingface.co/meituan-longcat/LongCat-Video-Avatar-1.6) by the Meituan LongCat team (MIT)
- Built on the 1.5 node pack [ComfyUI-LongCat-Avatar](https://github.com/rookiestar28/ComfyUI-LongCat-Avatar) by rookiestar28 (MIT)
- [SageAttention](https://github.com/thu-ml/SageAttention), voices made with [ElevenLabs](https://elevenlabs.io)

Made by **[PixelAI Labs](https://pixelailabs.com)**. MIT licensed.
