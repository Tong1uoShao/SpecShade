"""Payload <-> codeword: ChaCha20 encryption, BCH encoding, Chase-II soft decoding, reject rule.

# PAPER: encrypting the payload into a uniformly random bit stream is the precondition of
#   the distribution-preserving argument (Gaussian Shading, ChaCha20 [2]).
# OURS: BCH (galois) systematic encoding; Chase-II soft-decision decoding driven by the
#   per-cell LLRs of modulation.to_llr; a syndrome-based accept flag plus a soft-distance
#   threshold implements the reject rule of Method 3.6.
"""
from __future__ import annotations

import hashlib
import itertools

import galois
import numpy as np
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms

_GF2 = galois.GF(2)


def _keystream_bits(key: bytes, n_bits: int) -> np.ndarray:
    k32 = hashlib.sha256(b"stylering-key" + key).digest()
    nonce = hashlib.sha256(b"stylering-nonce" + key).digest()[:16]
    enc = Cipher(algorithms.ChaCha20(k32, nonce), mode=None).encryptor()
    ks = enc.update(bytes((n_bits + 7) // 8))
    return np.unpackbits(np.frombuffer(ks, dtype=np.uint8))[:n_bits].astype(int)


def encrypt(payload_bits, key: bytes) -> np.ndarray:
    p = np.asarray(payload_bits).astype(int).ravel()
    return (p ^ _keystream_bits(key, len(p))).astype(int)


def decrypt(cipher_bits, key: bytes) -> np.ndarray:
    return encrypt(cipher_bits, key)  # XOR is an involution


class BCHCode:
    """Systematic binary BCH(n, k); message occupies the first k codeword positions."""

    MAX_CODEBOOK_K = 20   # 2^20 x 63 int8 = 66 MB; beyond that fall back to the galois loop

    def __init__(self, n: int, k: int):
        # OURS: shortened codes. If (n, k) is not a primitive BCH length, take the parent code
        #   (n_p = 2^m - 1 >= n, k_p = k + n_p - n) and fix the first s = n_p - n message bits to 0;
        #   those positions are dropped from the codeword (systematic: message first). Same t as parent.
        self.short = 0
        try:
            self.code = galois.BCH(n, k)
        except Exception:
            n_p = 1
            while n_p < n:
                n_p = 2 * n_p + 1
            self.short = n_p - n
            self.code = galois.BCH(n_p, k + self.short)
        self.n, self.k, self.t = n, k, int(self.code.t)
        self._G: np.ndarray | None = None
        self._codebook: np.ndarray | None = None

    def encode(self, bits_k) -> np.ndarray:
        m = np.asarray(bits_k).astype(int).ravel()
        if m.size != self.k:
            raise ValueError(f"expected {self.k} message bits, got {m.size}")
        m = _GF2(np.concatenate([np.zeros(self.short, dtype=int), m]))
        return np.asarray(self.code.encode(m), dtype=int)[self.short:]

    def decode_hard(self, bits_n) -> tuple[np.ndarray, int, bool]:
        c = np.asarray(bits_n).astype(int).ravel()
        c = _GF2(np.concatenate([np.zeros(self.short, dtype=int), c]))
        msg, n_err = self.code.decode(c, errors=True)
        n_err = int(n_err)
        return np.asarray(msg, dtype=int)[self.short:], n_err, n_err >= 0

    @property
    def generator(self) -> np.ndarray:
        """Systematic generator matrix G (k x n) = encodings of the unit messages."""
        if self._G is None:
            self._G = np.stack([self.encode(row) for row in np.eye(self.k, dtype=int)])
        return self._G

    @property
    def codebook(self) -> np.ndarray:
        """All 2^k codewords as an int8 (2^k, n) table; row index = message read MSB-first.
        Built once (~0.1 s for k=16); enables exact table-based Chase-II (see chase2_decode)."""
        if self._codebook is None:
            if self.k > self.MAX_CODEBOOK_K:
                raise ValueError(f"codebook too large for k={self.k}")
            M = (np.arange(1 << self.k)[:, None] >> (self.k - 1 - np.arange(self.k))[None, :]) & 1
            self._codebook = ((M.astype(np.int64) @ self.generator) % 2).astype(np.int8)
        return self._codebook


def soft_distance(llr: np.ndarray, codeword: np.ndarray) -> float:
    """Negative correlation between LLR and the bipolar codeword (smaller = better match)."""
    return float(-np.sum(np.asarray(llr) * (2 * np.asarray(codeword) - 1)))


def _flip_patterns(n_flip: int) -> np.ndarray:
    """(2^n_flip, n_flip) 0/1 patterns in itertools.product order (last position fastest)."""
    idx = np.arange(1 << n_flip)[:, None]
    return (idx >> (n_flip - 1 - np.arange(n_flip))[None, :]) & 1


def chase2_decode_reference(llr: np.ndarray, code: BCHCode, n_flip: int = 4
                            ) -> tuple[np.ndarray | None, bool, float, int]:
    """Literal Chase-II (2^n_flip bounded-distance decodes via galois). Reference/test only:
    ~17 ms per galois decode -> ~4 s per call at n_flip=8. See chase2_decode."""
    llr = np.asarray(llr, dtype=np.float64)
    hard = (llr > 0).astype(int)
    unreliable = np.argsort(np.abs(llr))[:n_flip]
    best = None
    for pattern in itertools.product((0, 1), repeat=len(unreliable)):
        cand = hard.copy()
        for pos, f in zip(unreliable, pattern):
            if f:
                cand[pos] ^= 1
        msg, n_err, ok = code.decode_hard(cand)
        if not ok:
            continue
        cw = code.encode(msg)
        d = soft_distance(llr, cw)
        if best is None or d < best[2]:
            best = (msg, True, d, n_err)
    if best is None:
        return None, False, float("inf"), -1
    return best


def chase2_decode(llr: np.ndarray, code: BCHCode, n_flip: int = 4
                  ) -> tuple[np.ndarray | None, bool, float, int]:
    """Chase-II: flip the n_flip least-reliable positions in all 2^n_flip patterns,
    hard-decode each, keep the successful candidate with minimum soft distance.

    Returns (message_bits | None, accepted, soft_distance, n_corrected).

    Implementation (equivalent to chase2_decode_reference up to tie-breaking between
    distinct codewords of identical soft distance -- measure zero; ~700x faster):
    a bounded-distance decoder succeeds on hard^p iff some codeword c lies within
    Hamming radius t of hard^p; minimising over the flip subsets p of the unreliable
    set U gives  min_p d_H(hard^p, c) = d_H(hard, c) restricted to positions outside U.
    Hence the Chase-II candidate set is {c : d_H^{reliable}(hard, c) <= t}, which is
    read straight off the 2^k codebook; the survivor is the candidate of minimum soft
    distance and n_corrected is the correction count of the first (product-order)
    pattern that reaches it, as in the literal loop.
    """
    llr = np.asarray(llr, dtype=np.float64)
    if code.k > BCHCode.MAX_CODEBOOK_K:
        return chase2_decode_reference(llr, code, n_flip)
    n_flip = int(max(0, min(n_flip, code.n)))
    hard = (llr > 0).astype(np.int8)
    unreliable = np.argsort(np.abs(llr))[:n_flip]
    reliable = np.ones(code.n, dtype=bool)
    reliable[unreliable] = False
    C = code.codebook
    d_rel = (C[:, reliable] != hard[reliable]).sum(1)
    cand = np.flatnonzero(d_rel <= code.t)
    if cand.size == 0:
        return None, False, float("inf"), -1
    bip = 2.0 * C[cand].astype(np.float64) - 1.0
    d = -(bip @ llr)
    i = int(np.argmin(d))
    cw = C[cand[i]]
    msg = cw[:code.k].astype(int)                         # systematic code
    # n_corrected of the first flip pattern (product order) whose BD decode yields cw
    mism_u = (hard[unreliable] != cw[unreliable]).astype(np.int64)
    dist = int(d_rel[cand[i]]) + (_flip_patterns(n_flip) ^ mism_u[None, :]).sum(1)
    first = int(np.argmax(dist <= code.t))
    return msg, True, float(d[i]), int(dist[first])


def reject_rule(accepted: bool, soft_dist: float, thr: float) -> bool:
    """Declare 'watermark present' only if the syndrome check passed and the soft
    distance is at most thr (thr calibrated on unwatermarked images at a target FPR)."""
    return bool(accepted) and soft_dist <= thr


def bits_to_hex(bits) -> str:
    b = np.asarray(bits).astype(int).ravel()
    pad = (-len(b)) % 8
    packed = np.packbits(np.concatenate([b, np.zeros(pad, int)]).astype(np.uint8))
    return f"{len(b)}:" + packed.tobytes().hex()


def hex_to_bits(s: str) -> np.ndarray:
    n, hx = s.split(":")
    return np.unpackbits(np.frombuffer(bytes.fromhex(hx), dtype=np.uint8))[:int(n)].astype(int)
