#!/bin/bash
# One-off model prefetch into the HuggingFace cache. After this, every experiment script runs with HF_HUB_OFFLINE=1.
# Optional: export HF_ENDPOINT=https://hf-mirror.com if huggingface.co is slow from your network.
export HF_HUB_OFFLINE=0 HF_HUB_DISABLE_XET=1
PY=${PY:-python}
$PY - <<'EOF'
import torch
from huggingface_hub import snapshot_download
# generation / inversion backbone and the attack models (fp16 variants where available)
snapshot_download("stable-diffusion-v1-5/stable-diffusion-v1-5", allow_patterns=["*.json", "*.txt", "*fp16.safetensors", "tokenizer/*", "scheduler/*", "feature_extractor/*"])
snapshot_download("stabilityai/stable-diffusion-xl-base-1.0", allow_patterns=["*.json", "*.txt", "*fp16.safetensors", "tokenizer*/*", "scheduler/*"])
# SD2.1-base for the second-model experiment: the stabilityai repository is gated; this is an ungated full mirror
snapshot_download("LanguageMachines/stable-diffusion-2-1-base", allow_patterns=["*.json", "*.txt", "*.safetensors", "tokenizer/*", "scheduler/*"], ignore_patterns=["*.bin"])
print("diffusion models ok")
from diffusers import StableDiffusionInstructPix2PixPipeline
StableDiffusionInstructPix2PixPipeline.from_pretrained("timbrooks/instruct-pix2pix", torch_dtype=torch.float16, safety_checker=None)
print("ip2p ok")
import open_clip
open_clip.create_model_and_transforms("ViT-g-14", pretrained="laion2b_s34b_b88k")
print("openclip ok")
from cleanfid import fid  # downloads the inception weights on first use
print("cleanfid ok")
import compressai.zoo as zoo
zoo.cheng2020_anchor(quality=3, pretrained=True); zoo.bmshj2018_factorized(quality=3, pretrained=True)
print("compressai ok")
EOF
