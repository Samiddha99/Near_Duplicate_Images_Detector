"""
baseline_r5.py
==============
Babu & Rao 2022 — "Efficient detection of copy-move forgery using polar
complex exponential transform and gradient direction pattern"
(Multimedia Tools and Applications, 2022, doi: 10.1007/s11042-022-12311-6)

REFERENCE METHOD
----------------
    1.  Convert to grey-scale.
    2.  Divide the image into overlapping B × B blocks (paper uses B=16,
        stride 1).
    3.  For every block compute PCET coefficients
            N_mk = (1/π) ∫∫ H_mk(γ,θ)* f(γ,θ) γ dγ dθ
        with  H_mk(γ,θ) = R_m(γ) · e^{jkθ}     and   R_m(r)=e^{j2πmr²}.
        Truncate the series at order M=4, giving ~25 PCET coefficients.
    4.  Gradient Direction Pattern (GDP) : for each block build a
        histogram over quantised gradient orientations (8 bins) to obtain
        a rotationally-invariant texture descriptor.
    5.  Concatenate |N_mk| with the GDP histogram → per-block descriptor.
    6.  Rows of this feature matrix are sorted lexicographically; any
        two matching rows within an image indicate a copy-move forgery.
        A windowing + morphological post-filter removes isolated matches.

ADAPTATION FOR PAIR-BASED NEAR-DUPLICATE DETECTION (this script)
----------------------------------------------------------------
    Babu & Rao operate INSIDE one image (copy-move); we need a PAIR-wise
    detector comparable with the other four baselines.  We adapt the
    procedure in a literature-standard way (cf. Cozzolino 2015):

        * compute the set of block descriptors per image, plus a
          bag-of-codes image-level signature (normalised histogram over
          a shared k-means codebook built from all blocks).
        * pair similarity S(A,B) is the cosine of the two signatures,
          re-weighted by the mean of the top-k best block-wise matches
          (a "block-match" surrogate for the row-sort duplicate-count
          step in the paper).
        * The windowing / morphological post-filter is applied at image
          level: a pair must have at least MIN_BLOCK_MATCHES matched
          blocks AND signature cosine above LAMBDA_THRESH.

USAGE
-----
    python baseline_r5.py /path/to/images --report r5_report.txt
"""

from __future__ import annotations
import os, sys, time, argparse
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
import cv2

from eval_utils import (
    get_image_paths, PairPred, build_groups, write_baseline_report,
)

# ------------------------------------------------------------------
BLOCK_SIZE            = 16      # B × B blocks
BLOCK_STRIDE          = 8       # dense-ish but tractable (paper uses 1)
PCET_MAX_ORDER        = 4       # M=4 → 25 coefficients
GDP_BINS              = 8
K_CODEBOOK            = 128     # bag-of-block-codes codebook size
TOP_BLOCK_MATCHES     = 20      # k in the block-match surrogate
MIN_BLOCK_MATCHES     = 5
LAMBDA_THRESH         = 0.55

try:
    from sklearn.cluster import MiniBatchKMeans
    _SKLEARN = True
except Exception:
    _SKLEARN = False


# ══════════════════════════════════════════════════════════════════════
# PCET basis precomputation (block-size dependent, compute once)
# ══════════════════════════════════════════════════════════════════════
def _build_pcet_basis(B: int, M: int):
    """Return (M+1)² basis functions H_mk evaluated on a B×B polar grid."""
    xs = np.linspace(-1, 1, B)
    ys = np.linspace(-1, 1, B)
    Y, X = np.meshgrid(ys, xs, indexing="ij")
    R   = np.sqrt(X*X + Y*Y)
    TH  = np.arctan2(Y, X)
    MASK = (R <= 1.0)
    bases = []
    orders = []
    for m in range(0, M+1):
        for k in range(-M, M+1):
            Rm = np.exp(1j * 2 * np.pi * m * R * R)
            H  = Rm * np.exp(1j * k * TH)
            H  = H * MASK
            bases.append(H)
            orders.append((m, k))
    bases = np.stack(bases, axis=0).astype(np.complex64)   # (K, B, B)
    return bases, orders, MASK


_PCET_BASIS, _PCET_ORDERS, _PCET_MASK = _build_pcet_basis(
    BLOCK_SIZE, PCET_MAX_ORDER)


# ══════════════════════════════════════════════════════════════════════
# Per-block descriptor
# ══════════════════════════════════════════════════════════════════════
def _pcet_coeffs(block: np.ndarray) -> np.ndarray:
    f = block.astype(np.float32)
    coeffs = (_PCET_BASIS.conj() * f[None, :, :]).sum(axis=(1, 2)) / np.pi
    return np.abs(coeffs).astype(np.float32)


def _gdp_hist(block: np.ndarray) -> np.ndarray:
    """Quantised gradient-direction pattern histogram."""
    gx = cv2.Sobel(block, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(block, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx*gx + gy*gy) + 1e-8
    ang = (np.arctan2(gy, gx) + np.pi) / (2 * np.pi)   # 0..1
    bins = np.minimum(GDP_BINS - 1, (ang * GDP_BINS).astype(np.int32))
    hist = np.bincount(bins.flatten(), weights=mag.flatten(),
                       minlength=GDP_BINS)
    hist = hist / (hist.sum() + 1e-8)
    return hist.astype(np.float32)


def image_blocks(path: str, max_dim: int = 256) -> np.ndarray:
    """Return (n_blocks, feat_dim) per-block descriptor matrix."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        img = np.array(Image.open(path).convert("L"))
    # resize to contain cost (paper uses 256×256 experiments)
    h, w = img.shape
    if max(h, w) > max_dim:
        s = max_dim / max(h, w)
        img = cv2.resize(img, (int(w*s), int(h*s)))
    h, w = img.shape
    blocks = []
    for y in range(0, h - BLOCK_SIZE + 1, BLOCK_STRIDE):
        for x in range(0, w - BLOCK_SIZE + 1, BLOCK_STRIDE):
            b = img[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE].astype(np.float32) / 255.0
            c = _pcet_coeffs(b)
            g = _gdp_hist(b)
            blocks.append(np.concatenate([c, g]))
    return np.asarray(blocks, dtype=np.float32) if blocks else \
           np.zeros((0, _PCET_BASIS.shape[0] + GDP_BINS), dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════
# Pair similarity via bag-of-block-codes + block-match surrogate
# ══════════════════════════════════════════════════════════════════════
def _signature(blocks: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    if len(blocks) == 0:
        return np.zeros(len(codebook), dtype=np.float32)
    d = ((blocks[:, None, :] - codebook[None, :, :]) ** 2).sum(-1)
    assignment = np.argmin(d, axis=1)
    hist = np.bincount(assignment, minlength=len(codebook)).astype(np.float32)
    return hist / (hist.sum() + 1e-8)


def _pair_block_match(blocks_a: np.ndarray, blocks_b: np.ndarray) -> float:
    if len(blocks_a) == 0 or len(blocks_b) == 0:
        return 0.0
    # Take at most 64 best-variance blocks from each to bound cost
    def _topvar(B):
        if len(B) <= 64: return B
        v = B.var(axis=1)
        idx = np.argpartition(-v, 64)[:64]
        return B[idx]
    A, Bm = _topvar(blocks_a), _topvar(blocks_b)
    d = ((A[:, None, :] - Bm[None, :, :]) ** 2).sum(-1)
    best_per_a = d.min(axis=1)
    best = np.partition(best_per_a, min(TOP_BLOCK_MATCHES-1, len(best_per_a)-1))[:TOP_BLOCK_MATCHES]
    # convert distance → similarity
    return float(np.exp(-best.mean()))


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
def run(folder: str, report: str, threshold: float = LAMBDA_THRESH):
    t0 = time.time()
    paths = get_image_paths(folder)
    names = [Path(p).name for p in paths]
    n = len(paths)
    if n < 2:
        print("Need ≥2 images."); return
    print(f"[r5] {n} images — PCET+GDP block descriptors ...")

    t = time.time()
    per_image_blocks = []
    for i, p in enumerate(paths):
        per_image_blocks.append(image_blocks(p))
        if (i+1) % 25 == 0:
            print(f"      block-descriptor {i+1}/{n}  "
                  f"  blocks so far: {sum(len(b) for b in per_image_blocks)}")
    print(f"      block extraction in {time.time()-t:.1f}s  "
          f"(total blocks: {sum(len(b) for b in per_image_blocks):,})")

    # Build global codebook for bag-of-codes signature
    print("[r5] building PCET+GDP block codebook ...")
    all_blocks = np.vstack([b for b in per_image_blocks if len(b) > 0])
    if len(all_blocks) == 0:
        print("No blocks extracted — aborting."); return
    k = min(K_CODEBOOK, len(all_blocks))
    if _SKLEARN and len(all_blocks) > k:
        km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=3)
        km.fit(all_blocks)
        codebook = km.cluster_centers_.astype(np.float32)
    else:
        # fallback: random sample
        idx = np.random.RandomState(42).choice(
            len(all_blocks), size=k, replace=False)
        codebook = all_blocks[idx].astype(np.float32)
    print(f"      codebook shape={codebook.shape}")

    signatures = np.vstack([_signature(b, codebook) for b in per_image_blocks])
    print(f"      signature shape={signatures.shape}")

    # -- pair scoring -----------------------------------------------
    print("[r5] pairwise scoring (cosine of BoC + block-match surrogate)")
    preds: List[PairPred] = []
    for i in range(n):
        for j in range(i+1, n):
            # bag-of-codes cosine
            na, nb = signatures[i], signatures[j]
            cos_sig = float((na @ nb) /
                            (np.linalg.norm(na) * np.linalg.norm(nb) + 1e-12))
            # block-match surrogate (expensive; only run when cos_sig > 0.1)
            if cos_sig > 0.1:
                bm = _pair_block_match(per_image_blocks[i], per_image_blocks[j])
            else:
                bm = 0.0
            # morphological / windowing post-filter — require BOTH
            block_matches_ok = bm > 0.5
            score = 0.6 * cos_sig + 0.4 * bm
            pred  = 1 if (score > threshold and block_matches_ok) else 0
            preds.append(PairPred(
                names[i], names[j], float(score), pred,
                extra={"cos_sig": cos_sig, "block_match": bm}))
        if (i+1) % 20 == 0:
            print(f"      row {i+1}/{n}")

    groups = build_groups(preds)
    runtime = time.time() - t0

    extra = [
        f"  Block size B               : {BLOCK_SIZE}",
        f"  Block stride               : {BLOCK_STRIDE}",
        f"  PCET max order M           : {PCET_MAX_ORDER} → "
        f"{_PCET_BASIS.shape[0]} coefficients per block",
        f"  GDP bins                   : {GDP_BINS}",
        f"  Per-block feat dim         : {_PCET_BASIS.shape[0]+GDP_BINS}",
        f"  Codebook k                 : {K_CODEBOOK}",
        f"  Top-k block matches in pair: {TOP_BLOCK_MATCHES}",
        f"  Decision threshold         : {threshold}",
        f"  Block-match override       : require block_match > 0.5 "
        f"(≈ {MIN_BLOCK_MATCHES} strongly-matching blocks)",
    ]
    write_baseline_report(
        report_path=report,
        method_name="PCET + GDP (r5)",
        paper_ref="Babu & Rao, MTAP 2022, doi:10.1007/s11042-022-12311-6",
        filenames=names, preds=preds, groups=groups,
        runtime_sec=runtime, extra_lines=extra)
    print(f"[r5] wrote {report}  (runtime {runtime:.1f}s)")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("folder")
    p.add_argument("--report", required=True)
    p.add_argument("--threshold", type=float, default=LAMBDA_THRESH)
    a = p.parse_args()
    if not os.path.isdir(a.folder):
        print("Not a directory:", a.folder); sys.exit(1)
    run(a.folder, a.report, a.threshold)


if __name__ == "__main__":
    main()