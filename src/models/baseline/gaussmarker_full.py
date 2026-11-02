"""GaussMarker [4], FULL official 256-bit protocol run inside our pipeline (with the trained GNR).

# PAPER [4] / official repo (vendor/GaussMarker):
#   - spatial multi-bit: Gaussian_Shading_chacha (ch=1, w=h=8 -> 256 bits, 64 copies), ChaCha20-encrypted sign map m,
#     truncated-normal sampling (truncSampling); frequency zero-bit Tree-Ring ring (channel 3, radius 4, complex);
#   - ONE fixed message (w, m, key, nonce = w_info written by train_GNR.py): the GNR is trained for that message
#     (positives = affine/flip-augmented copies of m, negatives = random sign maps), exactly as in the official
#     train_GNR.py / gaussmarker_det.py; detection knows the message;
#   - detection: restored m = sigmoid(GNR(sign map)) > 0.5 -> pred_w_from_m (official vote decoding).
# OURS: score fusion = bit accuracy w.r.t. the known message - lambda * normalised ring L1 (the official code fuses
#   the same two statistics with a logistic classifier fitted on SD2.1, which does not transfer to our SD1.5 ring
#   scale); both components are written to meta so other fusions can be evaluated offline. The Tree-Ring part
#   reuses our _treering helpers (same as GaussMarker-lite). Requires CUDA for the GNR.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

from models.base import ExtractResult, WatermarkMethod
from models.baseline._treering import circle_mask, inject_complex, l1_complex, ring_pattern

_CODE = Path(__file__).resolve().parents[3]
_VEND = Path(__file__).resolve().parent / "vendor" / "GaussMarker"


def _load_vendor(name: str, rel: str):
    if str(_VEND) not in sys.path:
        sys.path.insert(0, str(_VEND))                      # vendor modules import their siblings
    spec = importlib.util.spec_from_file_location(name, _VEND / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class GaussMarkerFullWM(WatermarkMethod):
    name = "gaussmarker_full"

    def __init__(self, cfg: dict, backend) -> None:
        super().__init__(cfg, backend)
        m = cfg.get("methods", {}).get("gaussmarker_full", {})
        wm = _load_vendor("gm_watermark", "watermark.py")
        w_info = torch.load(_CODE / m["w_info_path"], map_location="cpu", weights_only=False)   # local, trusted
        ch, w, h = int(m.get("ch", 1)), int(m.get("w", 8)), int(m.get("h", 8))
        self.gs = wm.Gaussian_Shading_chacha(ch, w, h, float(m.get("fpr", 1e-6)), int(m.get("user_number", 1000000)),
                                             watermark=w_info["w"], m=w_info["m"], key=w_info["key"],
                                             nonce=w_info["nonce"])
        self.fixed_payload = w_info["w"].flatten().numpy().astype(int)
        self.key = str(m.get("key", "gm-key-v1")).encode()
        self.w_channel, self.w_radius = int(m.get("w_channel", 3)), int(m.get("w_radius", 4))
        self.lam = float(m.get("fuse_lambda", 1.0))
        self.size = backend.latent
        self.mask = circle_mask(self.size, self.w_radius)
        self.mask[self.size // 2, self.size // 2] = False
        self.pattern = ring_pattern(self.key + b"-ring", self.size, self.w_channel, self.w_radius)
        self.l1_norm = float(self.pattern[torch.from_numpy(self.mask)].abs().mean()) + 1e-6
        self.gnr = None
        if m.get("gnr_path"):
            unet = _load_vendor("gm_unet_model", "unet/unet_model.py")
            net = unet.UNet(4, 4, nf=int(m.get("model_nf", 128)))
            net.load_state_dict(torch.load(_CODE / m["gnr_path"], map_location="cpu", weights_only=True))
            self.gnr = net.eval().cuda()

    @property
    def n_bits(self) -> int:
        return int(self.fixed_payload.size)

    def embed(self, payload: np.ndarray, seed: int) -> torch.Tensor:
        np.random.seed(int(seed) % (2 ** 32))                 # official truncSampling draws from scipy's global RNG
        z_s = self.gs.truncSampling(self.gs.m).float()          # the fixed encrypted sign map m
        return inject_complex(z_s, self.w_channel, self.mask, self.pattern)

    def _decode(self, m_hat: torch.Tensor) -> np.ndarray:
        return self.gs.pred_w_from_m(m_hat.int().cpu()).flatten().numpy().astype(int)

    def extract(self, zT_hat: torch.Tensor) -> ExtractResult:
        z = zT_hat.detach().float().cpu()
        rev_m = (z > 0).float()
        bits_raw = self._decode(rev_m)
        if self.gnr is not None:
            with torch.no_grad():
                m_hat = (torch.sigmoid(self.gnr(rev_m.cuda())) > 0.5).int().cpu()
            bits = self._decode(m_hat)
        else:
            bits = bits_raw
        acc = float((bits == self.fixed_payload).mean())
        acc_raw = float((bits_raw == self.fixed_payload).mean())
        l1 = float(l1_complex(z, self.w_channel, self.mask, self.pattern) / self.l1_norm)
        return ExtractResult(bits=bits, llr=None, accepted=None, score=acc - self.lam * l1,
                             meta={"acc_known": acc, "acc_known_nognr": acc_raw, "ring_l1": l1})
