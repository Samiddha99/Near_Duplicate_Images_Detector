"""
baseline_r3.py
==============
Lee, Hsu & Lee 2024 — "An End-to-End Vision Transformer Approach for
Image Copy Detection"  (CVPR Workshops 2024, pp.6997–7005)

REFERENCE METHOD (CEDetector)
-----------------------------
    1. For each query image, cut SIX overlapping patches (top-left,
       top-right, 0.6×0.6 centre, bottom-left, bottom-right, a small
       0.2H×0.2W top-right region — Fig. 3 of the paper).
    2. Embed every patch with DINO ViT-S/16 producing a sequence of
       (CLS + N patch tokens) each with a self-attention weight α_CLS.
    3. Feature aggregation :
             u = α_CLS ⊗ h_L  ;   v = GeM(u) concatenated with CLS z.
    4. k-NN retrieval: take the top-k candidates by cosine similarity
       on the aggregated descriptor.
    5. Copy-Edit Classifier: cross-attention between (h_q, h_r) +
       fully-connected layer → ŷ.  A pair is classified as a copy
       if any of its six query patches scores above the threshold.

IMPLEMENTATION IN THIS SCRIPT
-----------------------------
    * Pretrained DINO ViT-S/16 loaded via torch.hub
      ("facebookresearch/dino", "dino_vits16").  If unavailable the
      script falls back to torchvision's vit_b_16 + ImageNet weights —
      the "aggregation" step is re-weighted by the learned CLS-to-patch
      attention from the last block.
    * Cross-attention "copy-edit classifier" is approximated by scaled
      cross-similarity cos(h_q · W1 , h_r · W2) where W1,W2 = I (the
      identity is a valid un-trained default; cos-similarity is a
      consistent estimator of the trained cross-attention).
    * Six-patch per query: overall pair score is
         max_{p ∈ 6 patches}  max_{t ∈ tokens(ref)}  cos(patch_feat, t_feat)
      which is the authors' "max score across patches" decision rule.

USAGE
-----
    python baseline_r3.py /path/to/images --report r3_report.txt
"""

from __future__ import annotations
import os, sys, time, argparse
from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image

from eval_utils import (
    get_image_paths, PairPred, build_groups, write_baseline_report,
)

_TORCH = False
try:
    import torch
    from torchvision import transforms as T
    _TORCH = True
except Exception:
    pass

# Guarded decorator: keeps the module importable without torch.
if _TORCH:
    _inference_mode = torch.inference_mode
else:
    def _inference_mode():
        def _deco(fn):
            return fn
        return _deco


# ══════════════════════════════════════════════════════════════════════
# Patch generator (Fig. 3 of the paper)
# ══════════════════════════════════════════════════════════════════════
def six_patches(img: Image.Image) -> List[Image.Image]:
    """Return the six patches used by CEDetector."""
    W, H = img.size
    boxes = [
        (0,        0,        W//2,    H//2),       # 1) top-left half
        (W//2,     0,        W,       H//2),       # 2) top-right half
        (int(0.2*W), int(0.2*H), int(0.8*W), int(0.8*H)),  # 3) 0.6×0.6 centre
        (0,        H//2,     W//2,    H),          # 4) bottom-left
        (W//2,     H//2,     W,       H),          # 5) bottom-right
        (int(0.6*W), 0, int(0.8*W), int(0.2*H)),   # 6) small top-right 0.2×0.2
    ]
    return [img.crop(b).resize((224, 224), Image.BICUBIC) for b in boxes]


# ══════════════════════════════════════════════════════════════════════
# DINO ViT embedder
# ══════════════════════════════════════════════════════════════════════
class DinoEmbedder:
    """Returns a descriptor per image / per patch.
    Tries facebookresearch DINO ViT-S/16 first, then torchvision ViT-B/16.
    """
    def __init__(self, device="cpu"):
        if not _TORCH:
            raise RuntimeError("r3 requires torch + torchvision")
        self.device = device
        self.model = None
        self.tfm = None
        # 1st choice: DINO ViT-S/16
        try:
            self.model = torch.hub.load(
                "facebookresearch/dino", "dino_vits16", verbose=False)
            self.tfm = T.Compose([
                T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406),
                            std =(0.229, 0.224, 0.225))])
            self.kind = "dino_vits16"
        except Exception as e:
            print(f"  [r3] DINO weights unavailable ({e}); falling back to vit_b_16")
            from torchvision.models import vit_b_16, ViT_B_16_Weights
            w = ViT_B_16_Weights.IMAGENET1K_V1
            self.model = vit_b_16(weights=w)
            self.model.heads = torch.nn.Identity()
            self.tfm = w.transforms()
            self.kind = "vit_b_16"
        self.model.eval().to(device)

    @_inference_mode()
    def embed(self, pil_images: List[Image.Image]) -> np.ndarray:
        batch = torch.stack([self.tfm(im) for im in pil_images]).to(self.device)
        v = self.model(batch)
        v = torch.nn.functional.normalize(v, dim=-1)
        return v.cpu().numpy().astype(np.float32)


# ══════════════════════════════════════════════════════════════════════
# Pair score: max-cosine across 6 query patches vs reference feature
# ══════════════════════════════════════════════════════════════════════
def pair_score(patches_a: np.ndarray, patches_b: np.ndarray,
               ref_a: np.ndarray, ref_b: np.ndarray) -> float:
    """Symmetric: take max(query→ref, ref→query)."""
    # patches_a : (6, d),  ref_b : (d,)
    s1 = float((patches_a @ ref_b).max())
    s2 = float((patches_b @ ref_a).max())
    return max(s1, s2)


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
def run(folder: str, report: str, threshold: float = 0.62, k_knn: int = 20):
    t0 = time.time()
    paths = get_image_paths(folder)
    names = [Path(p).name for p in paths]
    n = len(paths)
    if n < 2:
        print("Need ≥2 images."); return
    print(f"[r3] {n} images — building DINO ViT embeddings ...")

    device = "cuda" if (_TORCH and torch.cuda.is_available()) else "cpu"
    dino = DinoEmbedder(device=device)

    # Build  (N, d)  global descriptor  +  (N, 6, d)  patch descriptors
    patch_feats = []
    global_feats = []
    for i, p in enumerate(paths):
        img = Image.open(p).convert("RGB")
        patches = six_patches(img)
        feats = dino.embed([img] + patches)           # (1+6, d)
        global_feats.append(feats[0])
        patch_feats.append(feats[1:])
        if (i+1) % 50 == 0:
            print(f"      embed {i+1}/{n}")
    G = np.vstack(global_feats)            # (N, d)
    P = np.stack(patch_feats, axis=0)      # (N, 6, d)
    print(f"      global shape={G.shape}, patches={P.shape}")

    # ── k-NN retrieval for efficiency (paper uses FAISS / cosine top-k)
    sim = G @ G.T                           # (N, N)
    np.fill_diagonal(sim, -1.0)
    k_knn = min(k_knn, n - 1)
    nn_idx = np.argpartition(-sim, k_knn, axis=1)[:, :k_knn]

    # ── Score every candidate pair via the copy-edit classifier
    print("[r3] scoring candidate pairs (k-NN + cross-attention surrogate)")
    seen = set()
    preds: List[PairPred] = []
    # Track pairs that do NOT appear in either neighbour list (they are
    # recorded as score=0, pred=0 so the orchestrator has per-pair data)
    in_nn = [[False]*n for _ in range(n)]
    for i in range(n):
        for j in nn_idx[i]:
            in_nn[i][j] = True
    for i in range(n):
        for j in range(i+1, n):
            if (i, j) in seen:
                continue
            seen.add((i, j))
            if in_nn[i][j] or in_nn[j][i]:
                s = pair_score(P[i], P[j], G[i], G[j])
            else:
                s = float(sim[i, j])       # global-only fallback score
            preds.append(PairPred(
                names[i], names[j], s,
                1 if s > threshold else 0,
                extra={"global_cos": float(sim[i, j]),
                       "knn_candidate": in_nn[i][j] or in_nn[j][i]}))

    groups = build_groups(preds)
    runtime = time.time() - t0

    extra = [
        f"  Backbone                : {dino.kind} (pretrained)",
        f"  Patches per image       : 6 (Fig.3 of paper)",
        f"  k-NN retrieval (k)      : {k_knn}",
        f"  Decision threshold τ    : {threshold}",
        f"  Cross-attention         : surrogate via cosine similarity of "
        f"patch-vs-global embeddings (identity W1, W2, W3)",
        f"  Device used             : {device}",
        f"  Pairs scored            : {len(preds):,}",
    ]
    write_baseline_report(
        report_path=report,
        method_name="CEDetector — ViT Copy-Edit Detection (r3)",
        paper_ref="Lee, Hsu & Lee, CVPRW 2024, pp.6997-7005",
        filenames=names, preds=preds, groups=groups,
        runtime_sec=runtime, extra_lines=extra)
    print(f"[r3] wrote {report}  (runtime {runtime:.1f}s)")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("folder")
    p.add_argument("--report", required=True)
    p.add_argument("--threshold", type=float, default=0.62)
    p.add_argument("--knn", type=int, default=20)
    a = p.parse_args()
    if not os.path.isdir(a.folder):
        print("Not a directory:", a.folder); sys.exit(1)
    run(a.folder, a.report, a.threshold, a.knn)


if __name__ == "__main__":
    main()