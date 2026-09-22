# ComfyUI — Intel Arc B580 + Qwen-Image-2.1 GGUF

Custom setup for an Intel Arc B580 (Battlemage, 12 GB) using native PyTorch XPU.

## How to launch

Double-click **`run_comfyui.bat`** (or run it from a terminal). It opens the UI at
<http://127.0.0.1:8188>.

The launcher sets Arc-specific tuning:
- `PYTORCH_ENABLE_XPU_FALLBACK=1` — unsupported ops fall back to CPU instead of crashing
- `SYCL_CACHE_PERSISTENT=1` (+ cache dir on `E:`) — compiled GPU kernels are reused → fast warm starts
- `--use-pytorch-cross-attention` — Arc has no flash-attn/xformers
- **`--reserve-vram 2.0`** — **critical on this PC.** The B580 drives your display *and*
  does the compute. Without a VRAM reserve, heavy sampling consumed all 12 GB and starved
  the Windows display driver → BSOD `0x7E` (this actually happened during setup). Reserving
  2 GB keeps the desktop alive. Raise to `3.0` if you ever see another display glitch/crash.
- All caches (pip / HF / SYCL / temp) redirected to `E:\comfy-cache` to keep C: free

## The model

Everything lives on `E:` (C: is low on space).

| Component | File | Folder |
|-----------|------|--------|
| Diffusion (GGUF) | `qwen-image-2.1-Q6_K.gguf` (5.88 GB) | `models\diffusion_models` |
| Text encoder | `qwen3vl_8b_int8_convrot.safetensors` (9.35 GB) | `models\text_encoders` |
| VAE | `qwen_image_2.1_vae_bf16.safetensors` (0.68 GB) | `models\vae` |

Source: <https://huggingface.co/KasugaiSakura/Qwen-Image-2.1-Uncensored-GGUF>

## The workflow

A ready-to-run workflow is in the **Workflows** sidebar:
**`Qwen-Image-2.1-GGUF (Arc B580)`**

It uses:
- **Unet Loader (GGUF)** → `qwen-image-2.1-Q6_K.gguf`
- **CLIPLoader** → text encoder, type **`qwen_image`**
- **VAELoader** → the VAE
- **TextEncodeQwenImage21** (positive + negative in one node)
- **KSampler**: `euler` / `simple`, **cfg 1**, **25 steps**, **1024×1024** (safe default)

### Settings notes (from the official Qwen-Image 2.1 template)
- **cfg = 1** is the official path. Only raise it if you actually use a negative prompt.
- Steps: 25 is a good start; the official pipeline uses ~40–50 with euler for max quality.
- Native 2K: set 1:1 and 2048×2048 (uses more VRAM — watch the 12 GB ceiling).

## VRAM behavior on 12 GB

ComfyUI runs the stages sequentially and offloads to your 32 GB RAM between them:
1. Text encode (int8 encoder ≈ 9.4 GB) → 2. Diffusion (Q6_K ≈ 5.9 GB + activations)
→ 3. VAE decode. Peak is one stage at a time, so it fits.

**Resolution guidance (because the display shares this GPU):**
- **1024×1024 is the safe default** and is validated working.
- Going bigger (1536², native 2K 2048²) increases activation VRAM. If you push resolution,
  launch with **`--lowvram`** (edit `run_comfyui.bat`, add it after `--reserve-vram 2.0`)
  so model layers stream in/out — slower, but it won't starve VRAM.

If you ever hit an out-of-memory error or a display crash:
- Raise `--reserve-vram` to `3.0`, and/or add `--lowvram`, or
- Swap the diffusion model to a smaller quant (Q5_K_M / Q4_K_M) from the same HF repo, or
- In the workflow, set the **CLIPLoader `device` to `cpu`** — runs the 9.4 GB text encoder
  in system RAM, removing its VRAM spike entirely (encoding is a few seconds slower).

## Updating

- ComfyUI core: `git pull` in `E:\ComfyUI`
- Custom nodes / models: use **ComfyUI-Manager** (installed) from the UI menu.
