"""SD backend: component-wise offline loading, generation from a given initial noise,
DDIM inversion, and an SDXL img2img wrapper (attacks).

# Ported from peccavi_repro (sd_pipeline.py / vpa.py / gen_synth.py), keeping the
#   offline-safe patterns learnt there:
#   - sub-folder component loading + StableDiffusionPipeline assembly (the HF snapshot
#     cache has no model_index.json, so whole-pipeline from_pretrained fails offline);
#   - AutoPipelineForImage2Image loaded by local snapshot path when HF_HUB_OFFLINE=1.
# PAPER-convention (Gaussian Shading / RingID / GaussMarker [2,3,4]): 512x512, latent
#   4x64x64, CFG 7.5, 50 sampling steps, 50 DDIM inversion steps with empty prompt and
#   guidance 0 (pure unconditional epsilon).
# ASSUMPTION: SD1.5 stands in for SD2.1-base (gated); DDIM sampler for generation so that
#   sampling and inversion share one scheduler config.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from diffusers import (AutoencoderKL, AutoPipelineForImage2Image, DDIMInverseScheduler,
                       DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel)
from transformers import CLIPTextModel, CLIPTokenizer

DEFAULT_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
SDXL_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
LOCAL_ONLY = os.environ.get("HF_HUB_OFFLINE") == "1"


def local_snapshot_path(model_id: str) -> str | None:
    """Snapshot dir of model_id in the HF cache (refs/main), or None."""
    hub = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache" / "huggingface" / "hub"))
    d = hub / ("models--" + model_id.replace("/", "--"))
    ref = d / "refs" / "main"
    if ref.exists():
        snap = d / "snapshots" / ref.read_text().strip()
        if snap.exists():
            return str(snap)
    return None


def pick_device(device: str | None = None) -> str:
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


class SDBackend:
    """Single owner of the diffusion model: generate(zT, prompt) and invert(image)."""

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device: str | None = None,
                 prefer_fp16: bool = True, size: int = 512):
        self.device = pick_device(device)
        self.dtype = torch.float16 if (self.device == "cuda" and prefer_fp16) else torch.float32
        self.model_id, self.size = model_id, size
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=self.dtype).to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder",
                                                          torch_dtype=self.dtype).to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet",
                                                         torch_dtype=self.dtype).to(self.device)
        self.scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
        self.inv_scheduler = DDIMInverseScheduler.from_pretrained(model_id, subfolder="scheduler")
        self.vae_scale = self.vae.config.scaling_factor
        self.latent = size // 8
        for m in (self.vae, self.text_encoder, self.unet):
            m.eval()
            m.requires_grad_(False)
        self.pipe = StableDiffusionPipeline(
            vae=self.vae, text_encoder=self.text_encoder, tokenizer=self.tokenizer,
            unet=self.unet, scheduler=self.scheduler, safety_checker=None,
            feature_extractor=None, requires_safety_checker=False).to(self.device)
        self.pipe.set_progress_bar_config(disable=True)

    # ---------- initial noise ----------
    @staticmethod
    def sample_noise(seed: int, latent: int = 64, n_ch: int = 4) -> torch.Tensor:
        """Deterministic N(0,1) initial noise [1, n_ch, latent, latent] (CPU, float32)."""
        g = torch.Generator().manual_seed(int(seed))
        return torch.randn(1, n_ch, latent, latent, generator=g)

    # ---------- generation ----------
    x0_hook = None      # optional f(x0_pred) -> x0_pred applied at every sampling step (MaXsive template injection)

    @torch.no_grad()
    def generate(self, zT: torch.Tensor, prompt: str, steps: int = 50, cfg: float = 7.5,
                 seed: int = 0) -> Image.Image:
        lat = zT.to(self.device, self.dtype)
        if self.x0_hook is not None:
            return self._generate_hooked(lat, prompt, steps, cfg)
        gen = torch.Generator(device=self.device).manual_seed(int(seed))
        out = self.pipe(prompt, latents=lat, num_inference_steps=int(steps),
                        guidance_scale=float(cfg), generator=gen,
                        height=self.size, width=self.size)
        return out.images[0]

    @torch.no_grad()
    def _generate_hooked(self, z: torch.Tensor, prompt: str, steps: int, cfg: float) -> Image.Image:
        """The same DDIM sampling as the pipeline (CFG, eta 0), written out so that the x0 prediction of every step
        can be modified by self.x0_hook before the update (MaXsive injects its template there)."""
        emb_u, emb_c = self.get_text_emb(""), self.get_text_emb(prompt)
        self.scheduler.set_timesteps(int(steps), device=self.device)
        ac = self.scheduler.alphas_cumprod
        for t in self.scheduler.timesteps:
            eps_u = self.unet(z, t, encoder_hidden_states=emb_u).sample
            eps_c = self.unet(z, t, encoder_hidden_states=emb_c).sample
            eps = eps_u + float(cfg) * (eps_c - eps_u)
            a_t = ac[int(t)].to(z.device, z.dtype)
            x0 = (z - (1 - a_t).sqrt() * eps) / a_t.sqrt()
            x0 = self.x0_hook(x0.float()).to(z.dtype)
            eps = (z - a_t.sqrt() * x0) / (1 - a_t).sqrt()
            z = self.scheduler.step(eps, t, z).prev_sample
        return self.decode_latent(z)

    # ---------- VAE ----------
    @torch.no_grad()
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        t = pil_to_tensor(image.resize((self.size, self.size))).to(self.device, self.dtype) * 2.0 - 1.0
        return self.vae.encode(t).latent_dist.mean * self.vae_scale   # ASSUMPTION: posterior mean (deterministic)

    @torch.no_grad()
    def decode_latent(self, z: torch.Tensor) -> Image.Image:
        img = self.vae.decode(z.to(self.device, self.dtype) / self.vae_scale).sample
        img = (img / 2.0 + 0.5).clamp(0.0, 1.0)
        arr = (img[0].permute(1, 2, 0).cpu().float().numpy() * 255.0).round().astype(np.uint8)
        return Image.fromarray(arr)

    # ---------- text ----------
    @torch.no_grad()
    def get_text_emb(self, prompt: str = "") -> torch.Tensor:
        tok = self.tokenizer(prompt, padding="max_length", max_length=self.tokenizer.model_max_length,
                             truncation=True, return_tensors="pt")
        return self.text_encoder(tok.input_ids.to(self.device))[0]

    # ---------- inversion ----------
    @torch.no_grad()
    def invert(self, image: Image.Image, steps: int = 50, prompt: str = "",
               guidance: float = 0.0) -> torch.Tensor:
        """image -> estimated initial noise [1,4,latent,latent] (float32, on device)."""
        z = self.encode_image(image)
        emb_u = self.get_text_emb("")
        emb_c = self.get_text_emb(prompt) if (prompt and guidance > 1.0) else None
        self.inv_scheduler.set_timesteps(int(steps), device=self.device)
        for t in self.inv_scheduler.timesteps:
            eps = self.unet(z, t, encoder_hidden_states=emb_u).sample
            if emb_c is not None:   # classifier-free guidance at inversion (not used by default)
                eps_c = self.unet(z, t, encoder_hidden_states=emb_c).sample
                eps = eps + float(guidance) * (eps_c - eps)
            z = self.inv_scheduler.step(eps, t, z).prev_sample
        return z.float()

    @torch.no_grad()
    def invert_batch(self, images: list[Image.Image], steps: int = 50) -> torch.Tensor:
        """Batched unconditional DDIM inversion -> [B,4,latent,latent] float32 (on device).
        # OURS: numerically identical to invert() per image (unconditional UNet, no CFG);
        #   batching raises GPU utilisation ~2-3x on the vGPU."""
        t = torch.cat([pil_to_tensor(im.resize((self.size, self.size))) for im in images]).to(self.device, self.dtype)
        z = self.vae.encode(t * 2.0 - 1.0).latent_dist.mean * self.vae_scale
        emb = self.get_text_emb("").expand(z.shape[0], -1, -1)
        self.inv_scheduler.set_timesteps(int(steps), device=self.device)
        for tt in self.inv_scheduler.timesteps:
            eps = self.unet(z, tt, encoder_hidden_states=emb).sample
            z = self.inv_scheduler.step(eps, tt, z).prev_sample
        return z.float()

    def free(self) -> None:
        import gc
        for name in ("pipe", "unet", "vae", "text_encoder"):
            if hasattr(self, name):
                delattr(self, name)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class SDXLImg2Img:
    """SDXL image-to-image (style re-painting / regeneration attacks)."""

    def __init__(self, model_id: str = SDXL_MODEL_ID, device: str | None = None, size: int = 512):
        self.device, self.size = pick_device(device), size
        kw = dict(torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                  safety_checker=None, requires_safety_checker=False)
        mid = model_id
        if LOCAL_ONLY:
            mid = local_snapshot_path(model_id) or model_id
            kw["local_files_only"] = True
        if self.device == "cuda":
            kw["variant"] = "fp16"
        self.pipe = AutoPipelineForImage2Image.from_pretrained(mid, **kw)
        if self.device == "cuda" and os.environ.get("SDXL_OFFLOAD") == "1":
            # OURS: shared GPU with little free memory -> keep the weights in CPU memory and move one module at a time
            #   (same weights and arithmetic, only the placement differs; ~5 GB peak instead of ~8 GB)
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe = self.pipe.to(self.device)
        self.pipe.set_progress_bar_config(disable=True)

    @torch.no_grad()
    def __call__(self, image: Image.Image, prompt: str, strength: float, seed: int,
                 guidance_scale: float = 5.0, steps: int = 50) -> Image.Image:
        gen = torch.Generator(device=self.device).manual_seed(int(seed))
        out = self.pipe(prompt=prompt, image=image.convert("RGB"), strength=float(strength),
                        guidance_scale=float(guidance_scale), num_inference_steps=int(steps),
                        generator=gen).images[0]
        return out.resize((self.size, self.size))

    @torch.no_grad()
    def batch(self, images: list[Image.Image], prompts: list[str], strength: float, seeds: list[int],
              guidance_scale: float = 5.0, steps: int = 50) -> list[Image.Image]:
        """Several images in one pipeline call (one generator per image, so each image gets the same noise as in
        a single call). At 512 px the single-image call is launch-bound and leaves the GPU mostly idle."""
        gens = [torch.Generator(device=self.device).manual_seed(int(s)) for s in seeds]
        out = self.pipe(prompt=list(prompts), image=[im.convert("RGB") for im in images], strength=float(strength),
                        guidance_scale=float(guidance_scale), num_inference_steps=int(steps), generator=gens).images
        return [o.resize((self.size, self.size)) for o in out]

    def free(self) -> None:
        import gc
        del self.pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
