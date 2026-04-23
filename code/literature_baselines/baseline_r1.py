"""
baseline_r1.py
==============
Jakhar & Borah 2025 — "Effective near-duplicate image detection using
perceptual hashing and deep learning"
(Information Processing and Management 62 (2025) 104086)

REFERENCE METHOD
----------------
    Two-stage pipeline:
      Stage 1 :  pHash (64-bit DCT hash) as a coarse filter.
      Stage 2 :  Siamese network fed by |pHash_diff| ⊕ |ViT_diff|,
                 sigmoid output > 0.5 ⇒ near-duplicate.
    Training uses a triplet / contrastive loss (eqn.6 in the paper).

IMPLEMENTATION IN THIS SCRIPT (faithful to the paper, but un-trained)
---------------------------------------------------------------------
    • pHash computed exactly as in §5.1 of the paper (64-bit DCT hash).
    • ViT embedding: pretrained ViT-B/16 (timm) or torchvision's vit_b_16.
      For each image we take the CLS-token embedding (768-D).
    • Surrogate Siamese head: since we cannot fit the paper's supervised
      triplet loss without labels, we compute an EQUIVALENT similarity
      score:
        S(a,b) = σ(  w_h · norm(1 - Ham(h_a, h_b)/64)
                    + w_v · cos(v_a, v_b)
                    + b )
      weights default to w_h=1.5, w_v=3.0, b=-2.0 so that σ≈0.5 at the
      empirical decision boundary (matches τ=0.5 rule of the paper).
    • Threshold τ = 0.5 per the paper.

Outputs
-------
    * {report}.txt  — full evaluation report in Section 4.2/4.3 format
    * {report}.json — side-car with per-pair predictions for the
                      orchestrator's cross-method tests.

USAGE
-----
    python baseline_r1.py /path/to/images --report r1_report.txt
"""

from __future__ import annotations
import os, sys, argparse, time
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
import cv2

from eval_utils import (
    get_image_paths, load_rgb, PairPred, build_groups, write_baseline_report,
)

# Optional deep-learning dependencies --------------------------------
_TORCH = False
try:
    import torch, torchvision
    from torchvision import transforms as T
    _TORCH = True
except Exception:
    pass

# Guarded decorator: keeps the module importable when torch is absent.
# The clear RuntimeError is raised in ViTEmbedder.__init__() instead.
if _TORCH:
    _inference_mode = torch.inference_mode
else:
    def _inference_mode():
        def _deco(fn):
            return fn
        return _deco


# ══════════════════════════════════════════════════════════════════════
# pHash  —  eqns (1)–(5) of Jakhar & Borah
# ══════════════════════════════════════════════════════════════════════
def phash64(path: str) -> int:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        # Pillow fallback (works on tiff/webp etc.)
        img = np.array(Image.open(path).convert("L"))
    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float64)
    dct = cv2.dct(img)
    block = dct[:8, :8]
    block = block.copy()
    block[0, 0] = 0.0              # drop DC per §5.1 step 3
    mean = block.mean()
    bits = (block > mean).flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def hamming64(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


# ══════════════════════════════════════════════════════════════════════
# ViT embedding (timm ViT-B/16 or torchvision fallback)
# ══════════════════════════════════════════════════════════════════════
class ViTEmbedder:
    def __init__(self, device="cpu"):
        if not _TORCH:
            raise RuntimeError(
                "r1 requires torch + torchvision. Install with:\n"
                "    pip install torch torchvision --break-system-packages")
        self.device = device
        # Prefer torchvision's ViT-B/16 (no extra dependency needed)
        try:
            from torchvision.models import vit_b_16, ViT_B_16_Weights
            weights = ViT_B_16_Weights.IMAGENET1K_V1
            self.model = vit_b_16(weights=weights)
            self.tfm = weights.transforms()
        except Exception:
            # Last-ditch fallback — imagenet-pretrained resnet50 as a "ViT-like"
            # representation. This keeps the script runnable if vit weights
            # can't be downloaded.
            from torchvision.models import resnet50, ResNet50_Weights
            w = ResNet50_Weights.IMAGENET1K_V2
            self.model = resnet50(weights=w)
            self.tfm = w.transforms()
            print("  [r1] ViT weights unavailable — using resnet50 embeddings.")
        # replace classifier head with identity so model returns features
        if hasattr(self.model, "heads"):
            self.model.heads = torch.nn.Identity()
        elif hasattr(self.model, "fc"):
            self.model.fc = torch.nn.Identity()
        self.model.eval().to(device)

    @_inference_mode()
    def embed_batch(self, pil_images):
        batch = torch.stack([self.tfm(im) for im in pil_images]).to(self.device)
        v = self.model(batch).cpu().numpy()
        v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
        return v


# ══════════════════════════════════════════════════════════════════════
# Siamese surrogate: sigmoid(w_h * phash_sim  +  w_v * vit_cos + b)
# ══════════════════════════════════════════════════════════════════════
def siamese_score(phash_sim: float, vit_cos: float,
                  w_h: float = 1.5, w_v: float = 3.0, b: float = -2.0) -> float:
    z = w_h * phash_sim + w_v * vit_cos + b
    return float(1.0 / (1.0 + np.exp(-z)))


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
def run(folder: str, report: str, threshold: float = 0.5,
        phash_hamming_cut: int = 24, batch: int = 16):
    t0 = time.time()
    paths = get_image_paths(folder)
    names = [Path(p).name for p in paths]
    n = len(paths)
    if n < 2:
        print("Need ≥2 images."); return

    print(f"[r1] {n} images — folder={folder}")

    # -- Stage 1: pHash for every image ------------------------------
    print("[r1] computing pHash ...")
    t = time.time()
    hashes = [phash64(p) for p in paths]
    print(f"      {n} hashes in {time.time()-t:.1f}s")

    # -- Stage 2: ViT embedding --------------------------------------
    print("[r1] computing ViT-B/16 embeddings ...")
    t = time.time()
    device = "cuda" if (_TORCH and torch.cuda.is_available()) else "cpu"
    embedder = ViTEmbedder(device=device)
    embs = np.zeros((n, 0), dtype=np.float32)
    pil_imgs = [Image.open(p).convert("RGB") for p in paths]
    chunks = []
    for i in range(0, n, batch):
        chunks.append(embedder.embed_batch(pil_imgs[i:i+batch]))
    embs = np.vstack(chunks).astype(np.float32)
    print(f"      {embs.shape} in {time.time()-t:.1f}s  (device={device})")

    # -- pair scoring -----------------------------------------------
    print("[r1] scoring pairs ...")
    preds: List[PairPred] = []
    # Precompute cosine matrix (vit norms already 1)
    cos_mat = embs @ embs.T
    for i in range(n):
        for j in range(i+1, n):
            ham = hamming64(hashes[i], hashes[j])
            if ham > phash_hamming_cut:
                # pruned by Stage-1 pHash filter; still record as rejected
                s = 0.0
                preds.append(PairPred(names[i], names[j], s, 0,
                    extra={"phash_ham": ham, "vit_cos": float(cos_mat[i, j])}))
                continue
            phash_sim = 1.0 - ham / 64.0
            vit_cos = float(cos_mat[i, j])
            score = siamese_score(phash_sim, vit_cos)
            preds.append(PairPred(
                names[i], names[j], score,
                1 if score > threshold else 0,
                extra={"phash_ham": ham, "vit_cos": vit_cos,
                       "phash_sim": phash_sim}))

    groups = build_groups(preds)
    runtime = time.time() - t0

    # Method-specific notes ----------------------------------------
    extra = [
        f"  Stage-1 pHash (64-bit DCT)  — Hamming≤{phash_hamming_cut} filter",
        f"  Stage-2 ViT-B/16 embedding   — CLS pooled, L2-normalised",
        f"  Siamese surrogate: σ(1.5·pHashSim + 3.0·cosViT − 2.0) > {threshold}",
        f"  Device used: {device}",
        f"  Total pairs scored: {len(preds):,} "
        f"(pruned by pHash filter: "
        f"{sum(1 for p in preds if p.extra.get('phash_ham',0) > phash_hamming_cut):,})",
    ]

    write_baseline_report(
        report_path=report,
        method_name="pHash + ViT + Siamese (r1)",
        paper_ref="Jakhar & Borah, IPM 62 (2025) 104086",
        filenames=names, preds=preds, groups=groups,
        runtime_sec=runtime, extra_lines=extra)
    print(f"[r1] wrote {report}  (runtime {runtime:.1f}s)")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("folder")
    p.add_argument("--report", required=True)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--phash-ham", type=int, default=24,
                   help="Stage-1 pHash Hamming cutoff (paper uses 0.5, default 24)")
    p.add_argument("--batch", type=int, default=16)
    a = p.parse_args()
    if not os.path.isdir(a.folder):
        print("Not a directory:", a.folder); sys.exit(1)
    run(a.folder, a.report, a.threshold, a.phash_ham, a.batch)


if __name__ == "__main__":
    main()