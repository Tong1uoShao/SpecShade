"""T2SMark baseline [yang2025t2smark] (NeurIPS 2025), ported from the official code (github.com/0xD009/T2SMark, src/t2s.py,
53 lines) so that it runs in our offline pipeline with the same SD1.5 backend and 50-step DDIM inversion as every method.

# PAPER: Tail-Truncated Sampling. For a channel group with n entries and an m-bit message, k = m r positions are selected by a
#   PRNG seeded with the key (r = floor(2 Phi(-tau) n / m) copies per bit, tau = 0.674 -> k = n / 2). The k largest magnitudes of
#   a Gaussian draw are placed on these positions with the signs of the codeword (the message bits repeated r times, multiplied by a
#   pseudo-random +-1 sequence); the remaining magnitudes go to the other positions with random signs. Two levels: channel 0
#   carries a 16-bit session key encoded with the master key, channels 1-3 carry the 256-bit message encoded with the session key.
#   Decoding projects the support onto the +-1 sequence and sums the r copies of each bit (sign -> bit). The detection statistic is
#   the L1 norm of the per-bit sums of the KEY channel decoded with the master key, which does not depend on the message.
# OURS: deterministic per generation seed (the Gaussian draw is the backend noise of that seed, the central signs come from a PRNG
#   seeded by it); the session key is derived from the seed and the master key; detection uses the same 1%-FPR calibration on
#   1,000 unwatermarked images as every other method (the official code thresholds the statistic analytically).
"""
from __future__ import annotations

import hashlib
from functools import reduce

import numpy as np
import torch
from scipy.stats import norm

from models.base import ExtractResult, WatermarkMethod


def _bits_to_int(bits) -> int:
    return int(reduce(lambda x, y: x * 2 + int(y), [int(b) for b in bits], 0))


class _TailCodec:
    """One channel group (official T2SMark class, device-agnostic and seeded)."""

    def __init__(self, m: int, tau: float, shape: tuple[int, ...]) -> None:
        self.shape, self.m = shape, m
        self.n = int(np.prod(shape))
        self.r = int(2 * norm.cdf(-tau) * self.n / m)
        self.k = self.m * self.r
        self.noise_size = self.n - self.k

    def _book(self, key_int: int):
        g = torch.Generator().manual_seed(int(key_int))
        v_value = torch.randint(0, 2, (self.k,), generator=g).float() * 2 - 1
        v_support = torch.randperm(self.n, generator=g)[:self.k]
        return v_value, v_support

    def encode(self, bits, key_int: int, z: torch.Tensor, sign_gen: torch.Generator) -> torch.Tensor:
        v_value, v_support = self._book(key_int)
        zf = z.detach().cpu().float().flatten()
        codeword = (1 - 2 * torch.as_tensor(np.asarray(bits), dtype=torch.float32)).repeat(self.r) * v_value
        w = torch.zeros(self.n, dtype=torch.bool); w[v_support] = True
        tail = torch.topk(zf.abs(), k=self.k, largest=True, sorted=False).values
        central = torch.topk(zf.abs(), k=self.noise_size, largest=False, sorted=False).values
        out = torch.zeros(self.n)
        out[w] = tail * codeword
        out[~w] = central * (torch.randint(0, 2, (self.noise_size,), generator=sign_gen).float() * 2 - 1)
        return out.reshape(self.shape)

    def decode(self, z: torch.Tensor, key_int: int):
        v_value, v_support = self._book(key_int)
        w = torch.zeros(self.n, dtype=torch.bool); w[v_support] = True
        p = (z.detach().cpu().float().flatten()[w] * v_value).reshape(self.r, self.m).sum(0)
        return (p < 0).int().numpy(), float(p.abs().sum())


class T2SMarkWM(WatermarkMethod):
    name = "t2smark"

    def __init__(self, cfg: dict, backend) -> None:
        super().__init__(cfg, backend)
        m = cfg.get("methods", {}).get("t2smark", {})
        self.msg_len = int(m.get("msg_length", 256))
        self.key_len = int(m.get("key_length", 16))
        self.tau = float(m.get("tau", 0.674))
        self.key_ch = int(m.get("key_channel", 0))
        self.latent = backend.latent
        self.msg_ch = [c for c in range(4) if c != self.key_ch]
        self.c_key = _TailCodec(self.key_len, self.tau, (1, self.latent, self.latent))
        self.c_msg = _TailCodec(self.msg_len, self.tau, (3, self.latent, self.latent))
        h = hashlib.sha256(b"t2smark-master|" + str(m.get("key", "t2s-master-key-v1")).encode()).digest()
        self.master_bits = np.unpackbits(np.frombuffer(h, dtype=np.uint8))[:self.key_len]
        self.master_int = _bits_to_int(self.master_bits)

    @property
    def n_bits(self) -> int:
        return self.msg_len

    def _session_key(self, seed: int) -> np.ndarray:
        h = hashlib.sha256(f"t2smark-session|{int(seed)}|{self.master_int}".encode()).digest()
        return np.unpackbits(np.frombuffer(h, dtype=np.uint8))[:self.key_len]

    def embed(self, payload: np.ndarray, seed: int) -> torch.Tensor:
        z = self.backend.sample_noise(seed, latent=self.latent)[0]              # (4, L, L) Gaussian draw of this seed
        g = torch.Generator().manual_seed(int(seed) * 7919 + 17)               # random signs of the central entries
        skey = self._session_key(seed)
        out = torch.zeros_like(z, dtype=torch.float32)
        out[self.key_ch] = self.c_key.encode(skey, self.master_int, z[self.key_ch:self.key_ch + 1], g)[0]
        out[self.msg_ch] = self.c_msg.encode(payload, _bits_to_int(skey), z[self.msg_ch], g)
        return out.unsqueeze(0)

    def extract(self, zT_hat: torch.Tensor) -> ExtractResult:
        z = zT_hat.detach().cpu().float()[0]
        skey, stat = self.c_key.decode(z[self.key_ch:self.key_ch + 1], self.master_int)   # detection: key channel, master key
        bits, msg_stat = self.c_msg.decode(z[self.msg_ch], _bits_to_int(skey))             # message with the recovered session key
        return ExtractResult(bits=bits, llr=None, accepted=None, score=stat, meta={"key_stat": stat, "msg_stat": msg_stat})
