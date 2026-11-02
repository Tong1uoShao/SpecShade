"""Stable Signature baseline (Fernandez et al., ICCV 2023) [10] — image-level.

# PAPER: the LDM's VAE decoder is fine-tuned so that every decoded image carries a fixed
#   48-bit key readable by a HiDDeN-style extractor; the initial noise / UNet are untouched.
# ASSUMPTION: we use the officially released decoder (fine-tuned on SD2.1's kl-f8 VAE) inside
#   our SD1.5 pipeline. SD1.x and SD2.x share the kl-f8 autoencoder family; compatibility is
#   verified empirically by the clean-image bit accuracy (must be ~1.0), see tests/smoke.
# OURS: single baked-in key -> `fixed_payload`; detection score = bit agreement with the key
#   (the paper's binomial test is monotone in it); null = unwatermarked images through the
#   same extractor.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from models.base import ExtractResult, WatermarkMethod

ROOT = Path(__file__).resolve().parents[3]


def _ldm_decoder_to_diffusers(sd: dict) -> dict:
    """Map the released `sd2_decoder.pth` (LDM naming: decoder.mid.block_1, decoder.up.i.block.j ...)
    onto diffusers AutoencoderKL keys (decoder.mid_block.resnets.0, decoder.up_blocks.3-i.resnets.j ...)."""
    out = {}
    n_up = 1 + max(int(k.split(".")[2]) for k in sd if k.startswith("decoder.up."))

    def resnet(prefix_in: str, prefix_out: str) -> None:
        for a, b in (("norm1", "norm1"), ("conv1", "conv1"), ("norm2", "norm2"), ("conv2", "conv2"),
                     ("nin_shortcut", "conv_shortcut")):
            for p in ("weight", "bias"):
                k = f"{prefix_in}.{a}.{p}"
                if k in sd:
                    out[f"{prefix_out}.{b}.{p}"] = sd[k]

    for k, v in sd.items():
        if k.startswith("post_quant_conv.") or k.startswith("decoder.conv_in.") or k.startswith("decoder.conv_out."):
            out[k] = v
        elif k.startswith("decoder.norm_out."):
            out[k.replace("decoder.norm_out.", "decoder.conv_norm_out.")] = v
    resnet("decoder.mid.block_1", "decoder.mid_block.resnets.0")
    resnet("decoder.mid.block_2", "decoder.mid_block.resnets.1")
    attn_map = {"norm": "group_norm", "q": "to_q", "k": "to_k", "v": "to_v", "proj_out": "to_out.0"}
    for a, b in attn_map.items():
        for p in ("weight", "bias"):
            k = f"decoder.mid.attn_1.{a}.{p}"
            if k in sd:
                t = sd[k]
                if a != "norm" and p == "weight" and t.ndim == 4:      # 1x1 conv -> linear
                    t = t.reshape(t.shape[0], t.shape[1])
                out[f"decoder.mid_block.attentions.0.{b}.{p}"] = t
    for i in range(n_up):
        j = n_up - 1 - i                                               # LDM up.i == diffusers up_blocks[n-1-i]
        for r in range(8):
            if f"decoder.up.{i}.block.{r}.conv1.weight" in sd:
                resnet(f"decoder.up.{i}.block.{r}", f"decoder.up_blocks.{j}.resnets.{r}")
        for p in ("weight", "bias"):
            k = f"decoder.up.{i}.upsample.conv.{p}"
            if k in sd:
                out[f"decoder.up_blocks.{j}.upsamplers.0.conv.{p}"] = sd[k]
    return out


class StableSignatureWM(WatermarkMethod):
    name = "stable_signature"
    image_level = True

    def __init__(self, cfg: dict, backend) -> None:
        super().__init__(cfg, backend)
        c = cfg["methods"]["stable_signature"]
        self.key_str = str(c["key"]).strip()
        self._n_bits = len(self.key_str)
        self.fixed_payload = np.array([int(ch) for ch in self.key_str], dtype=int)
        self.decoder_path = ROOT / c["decoder"]
        self.extractor_path = ROOT / c["extractor"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._extractor = None
        self._tf = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    @property
    def n_bits(self) -> int:
        return self._n_bits

    # ---- generation: plain noise, watermark enters through the swapped VAE decoder ----
    def embed(self, payload: np.ndarray, seed: int) -> torch.Tensor:
        return self.backend.sample_noise(seed, latent=self.backend.latent)

    def prepare_backend(self, backend) -> None:
        sd = torch.load(self.decoder_path, map_location="cpu")
        mapped = _ldm_decoder_to_diffusers(sd)
        vae = backend.vae
        target = {k: v for k, v in vae.state_dict().items() if k.startswith("decoder.") or k.startswith("post_quant_conv.")}
        missing = sorted(set(target) - set(mapped))
        unexpected = sorted(set(mapped) - set(target))
        if missing or unexpected:
            raise RuntimeError(f"stable_signature decoder mapping mismatch: missing={missing[:5]} ({len(missing)}) "
                               f"unexpected={unexpected[:5]} ({len(unexpected)})")
        cast = {k: v.to(target[k].dtype) for k, v in mapped.items()}
        res = vae.load_state_dict(cast, strict=False)
        assert not res.unexpected_keys, res.unexpected_keys
        # sanity: nothing outside the decoder changed
        assert all(k.startswith("encoder.") or k.startswith("quant_conv.") for k in res.missing_keys), res.missing_keys[:5]

    def extract(self, zT_hat: torch.Tensor) -> ExtractResult:      # not a noise-level method
        raise NotImplementedError("stable_signature reads images: use extract_image")

    # ---- detection ----
    def _load_extractor(self):
        if self._extractor is None:
            self._extractor = torch.jit.load(str(self.extractor_path), map_location=self.device).eval()
        return self._extractor

    @torch.no_grad()
    def extract_image(self, image: Image.Image, ref_payload: np.ndarray | None = None) -> ExtractResult:
        ext = self._load_extractor()
        x = self._tf(image.convert("RGB")).unsqueeze(0).to(self.device)
        msg = ext(x).squeeze(0).float().cpu().numpy()                  # 48 logits
        bits = (msg > 0).astype(int)
        ref = self.fixed_payload if ref_payload is None else np.asarray(ref_payload).astype(int)
        score = float((bits == ref).mean())                            # bit agreement with the key
        return ExtractResult(bits=bits, llr=msg, accepted=None, score=score)
