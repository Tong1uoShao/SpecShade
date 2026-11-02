"""VINE baseline (Lu et al., ICLR 2025) [5] — image-level post-hoc encoder watermark (100 bit).

# PAPER: a one-step SD-Turbo generator conditioned on the message (ConditionAdaptor) writes a
#   256x256 residual that is upsampled and added to the image; a ConvNeXt-B decoder reads 100
#   bits from the 256x256 resized image. VINE-R is the editing-robust variant.
# OURS: vendored official code (src/models/baseline/vendor/VINE, read-only) driven through the
#   uniform interface; base sd-turbo UNet/VAE are built from config (weights come from the
#   VINE checkpoint), so only sd-turbo's text encoder/tokenizer/configs are needed locally.
#   Message = our per-image derived 100-bit payload; detection score = bit agreement with the
#   reference message (no ECC / reject rule in VINE); null = unwatermarked images vs their own
#   would-be message.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from models.base import ExtractResult, WatermarkMethod

VENDOR = Path(__file__).resolve().parent / "vendor" / "VINE"


def _import_vine():
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))
    import vine.src.vine_turbo as vt                              # noqa: E402
    from diffusers import AutoencoderKL, UNet2DConditionModel

    base = "stabilityai/sd-turbo"

    def unet_from_config(path=base):                              # OURS: no sd-turbo UNet weights needed
        unet = UNet2DConditionModel.from_config(UNet2DConditionModel.load_config(path, subfolder="unet"))
        return unet

    def vae_from_config(path=base):
        vae = AutoencoderKL.from_config(AutoencoderKL.load_config(path, subfolder="vae"))
        vae.encoder.forward = vt.my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
        vae.decoder.forward = vt.my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)
        for name, cin, cout in (("skip_conv_1", 512, 512), ("skip_conv_2", 256, 512),
                                ("skip_conv_3", 128, 512), ("skip_conv_4", 128, 256)):
            setattr(vae.decoder, name, torch.nn.Conv2d(cin, cout, kernel_size=3, stride=1, padding=1, bias=True))
        vae.decoder.ignore_skip = False
        vae.decoder.gamma = 1
        return vae

    vt.initialize_unet_no_lora = unet_from_config
    vt.initialize_vae_no_lora = vae_from_config

    from transformers import CLIPTokenizer

    class _Tok:                                                   # OURS: AutoTokenizer(subfolder=) needs a
        @staticmethod                                             # root config.json -> fails offline;
        def from_pretrained(path, subfolder="tokenizer", **kw):  # load the CLIP tokenizer directly
            return CLIPTokenizer.from_pretrained(path, subfolder=subfolder)

    vt.AutoTokenizer = _Tok
    from vine.src.stega_encoder_decoder import CustomConvNeXt    # noqa: E402
    return vt.VINE_Turbo, CustomConvNeXt


class VineWM(WatermarkMethod):
    name = "vine"
    image_level = True

    def __init__(self, cfg: dict, backend) -> None:
        super().__init__(cfg, backend)
        c = cfg["methods"]["vine"]
        self._n_bits = int(c.get("n_bits", 100))
        self.enc_repo, self.dec_repo = str(c["enc_repo"]), str(c["dec_repo"])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._enc = self._dec = None
        self._to256 = transforms.Compose([transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                                          transforms.ToTensor()])

    @property
    def n_bits(self) -> int:
        return self._n_bits

    def embed(self, payload: np.ndarray, seed: int) -> torch.Tensor:  # plain generation; watermark is post-hoc
        return self.backend.sample_noise(seed, latent=self.backend.latent)

    def extract(self, zT_hat: torch.Tensor) -> ExtractResult:
        raise NotImplementedError("vine reads images: use extract_image")

    # ---- lazy model loading (encoder ~4 GB fp32; decoder ConvNeXt-B) ----
    def _encoder(self):
        if self._enc is None:
            VINE_Turbo, _ = _import_vine()
            self._enc = VINE_Turbo.from_pretrained(self.enc_repo).to(self.device).eval()
        return self._enc

    def _decoder(self):
        if self._dec is None:
            _, CustomConvNeXt = _import_vine()
            self._dec = CustomConvNeXt.from_pretrained(self.dec_repo).to(self.device).eval()
        return self._dec

    def free(self) -> None:
        self._enc = self._dec = None
        torch.cuda.empty_cache()

    @torch.no_grad()
    def embed_image(self, image: Image.Image, payload: np.ndarray) -> Image.Image:
        enc = self._encoder()
        im = image.convert("RGB")
        x256 = (2.0 * self._to256(im) - 1.0).unsqueeze(0).to(self.device)
        x_full = (2.0 * transforms.ToTensor()(im) - 1.0).unsqueeze(0).to(self.device)
        secret = torch.tensor(np.asarray(payload).astype(np.float32)).unsqueeze(0).to(self.device)
        y256 = enc(x256, secret)
        residual = F.interpolate(y256 - x256, size=x_full.shape[-2:], mode="bilinear", align_corners=False)
        out = (x_full + residual).clamp(-1, 1)
        arr = ((out[0] / 2 + 0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
        return Image.fromarray(arr)

    @torch.no_grad()
    def extract_image(self, image: Image.Image, ref_payload: np.ndarray | None = None) -> ExtractResult:
        dec = self._decoder()
        x = self._to256(image.convert("RGB")).unsqueeze(0).to(self.device)
        prob = dec(x).squeeze(0).float().cpu().numpy()                 # 100 sigmoid outputs
        bits = np.round(prob).astype(int)
        score = float((bits == np.asarray(ref_payload).astype(int)).mean()) if ref_payload is not None else float("nan")
        return ExtractResult(bits=bits, llr=2 * prob - 1, accepted=None, score=score)
