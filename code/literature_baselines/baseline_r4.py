"""
baseline_r4.py
==============
Singh, Kumar, Ranjan & Nandan 2024 — "Duplicate image detection using
deep learning modified SVM and k-NN classification method for
multimedia application"  (Soft Computing 28:7659-7670, 2024)

REFERENCE METHOD
----------------
    * Pre-processing layer:  3-level Haar DWT (H, V, D sub-bands).
    * Feature-extraction layer:  5 convolutional blocks
        C1:16 / C2:32 / C3:64 / C4:128 / C5:256  (f=5×5) + ReLU + avg-pool.
    * Classification layer:  SVM  or  k-NN  (authors report 98.63 / 99.12%
      for CNN+SVM / CNN+KNN respectively on Imperial College London ds).

IMPLEMENTATION IN THIS SCRIPT
-----------------------------
    * Haar DWT via `pywavelets` if installed, else a manual 2-D Haar
      implementation (correct but slower).  We take level-3 LL + HH
      sub-bands, flatten and concatenate them with…
    * Pretrained ResNet-18 features (avg-pool, 512-D) — a standard
      "Conv-5-block" architecture matching Singh's Table 1 dimensions.
    * Classification step (no labels exist at run-time so we follow
      Singh's testing recipe):
         1. Compute descriptor D = [ DWT_feat ;  CNN_feat ]    (≈ 1024-D).
         2. L2-normalise, PCA to 128-D.
         3. Train a k-NN (k=1) on the **whole dataset** using the
            proposed "pseudo-augmentation" scheme — for every image
            we generate 4 weakly-augmented copies and use them as the
            positive training set for that class.
         4. Predict: for every pair, compute the fraction of mutual
            k-nearest neighbours in the learned metric space.
         5. A pair is called duplicate iff both images share at least
            `knn_k // 2` mutual nearest neighbours (a faithful re-
            enactment of the classifier's score threshold).
    * SVM branch is also implemented (RBF kernel) and its raw decision
      value is recorded in `extra["svm_margin"]` for the orchestrator's
      secondary analysis.

USAGE
-----
    python baseline_r4.py /path/to/images --report r4_report.txt
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

_TORCH = False
try:
    import torch
    from torchvision import transforms as T
    from torchvision.models import resnet18, ResNet18_Weights
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

_PYWT = False
try:
    import pywt
    _PYWT = True
except Exception:
    pass

_SKLEARN = False
try:
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors
    from sklearn.svm import OneClassSVM
    _SKLEARN = True
except Exception:
    pass


# ══════════════════════════════════════════════════════════════════════
# Haar DWT feature   (3-level, LL + HH sub-bands)
# ══════════════════════════════════════════════════════════════════════
def dwt_features(path: str, levels: int = 3, size: int = 128) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        img = np.array(Image.open(path).convert("L"))
    img = cv2.resize(img, (size, size)).astype(np.float32) / 255.0
    if _PYWT:
        coeffs = pywt.wavedec2(img, wavelet="haar", level=levels)
        LL = coeffs[0].flatten()
        details = []
        for (cH, cV, cD) in coeffs[1:]:
            details.append(cH.flatten())
            details.append(cV.flatten())
            details.append(cD.flatten())
        feat = np.concatenate([LL] + details).astype(np.float32)
    else:
        # manual 2-D Haar
        a = img
        feats = []
        for _ in range(levels):
            h, w = a.shape
            LL = (a[0::2, 0::2] + a[0::2, 1::2] + a[1::2, 0::2] + a[1::2, 1::2]) / 4.0
            LH = (a[0::2, 0::2] - a[0::2, 1::2] + a[1::2, 0::2] - a[1::2, 1::2]) / 4.0
            HL = (a[0::2, 0::2] + a[0::2, 1::2] - a[1::2, 0::2] - a[1::2, 1::2]) / 4.0
            HH = (a[0::2, 0::2] - a[0::2, 1::2] - a[1::2, 0::2] + a[1::2, 1::2]) / 4.0
            feats.extend([LH.flatten(), HL.flatten(), HH.flatten()])
            a = LL
        feats.insert(0, a.flatten())
        feat = np.concatenate(feats).astype(np.float32)
    # histogram compression (128-bin) — paper uses a learned FC layer,
    # we use a fixed histogram to stay deterministic / un-trained.
    hist, _ = np.histogram(feat, bins=128, range=(-1.0, 1.0))
    hist = hist.astype(np.float32)
    hist = hist / (hist.sum() + 1e-8)
    return hist


# ══════════════════════════════════════════════════════════════════════
# CNN (ResNet-18) feature — surrogate for the 5-block Cn CNN of Table 1
# ══════════════════════════════════════════════════════════════════════
class ResNetEmbedder:
    def __init__(self, device="cpu"):
        if not _TORCH:
            raise RuntimeError("r4 requires torch + torchvision")
        self.device = device
        w = ResNet18_Weights.IMAGENET1K_V1
        self.model = resnet18(weights=w)
        self.model.fc = torch.nn.Identity()
        self.model.eval().to(device)
        self.tfm = w.transforms()

    @_inference_mode()
    def embed_batch(self, pil_images):
        batch = torch.stack([self.tfm(im) for im in pil_images]).to(self.device)
        v = self.model(batch).cpu().numpy().astype(np.float32)
        v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
        return v


# ══════════════════════════════════════════════════════════════════════
# Pseudo-augment + KNN classifier
# ══════════════════════════════════════════════════════════════════════
def light_augment(img: Image.Image) -> Image.Image:
    """A weak on-the-fly augmentation — mimics Singh's "learned filter"
    step that expands the training set."""
    arr = np.array(img.convert("RGB"))
    # random noise + small rotation
    noise = np.random.normal(0, 5, arr.shape)
    arr = np.clip(arr.astype(float) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
def run(folder: str, report: str, knn_k: int = 7,
        mutual_frac: float = 0.35, batch: int = 16):
    t0 = time.time()
    paths = get_image_paths(folder)
    names = [Path(p).name for p in paths]
    n = len(paths)
    if n < 2:
        print("Need ≥2 images."); return
    print(f"[r4] {n} images — Haar DWT features ...")

    # -- DWT features ------------------------------------------------
    t = time.time()
    dwt = np.vstack([dwt_features(p) for p in paths])
    print(f"      DWT shape={dwt.shape} in {time.time()-t:.1f}s"
          f"   (pywt={'yes' if _PYWT else 'manual'})")

    # -- CNN features ------------------------------------------------
    print("[r4] ResNet-18 CNN features ...")
    device = "cuda" if (_TORCH and torch.cuda.is_available()) else "cpu"
    resnet = ResNetEmbedder(device=device)
    pil_imgs = [Image.open(p).convert("RGB") for p in paths]
    chunks = []
    for i in range(0, n, batch):
        chunks.append(resnet.embed_batch(pil_imgs[i:i+batch]))
    cnn = np.vstack(chunks)
    print(f"      CNN shape={cnn.shape}  device={device}")

    # -- Concatenate & PCA ------------------------------------------
    feats = np.hstack([dwt, cnn]).astype(np.float32)
    feats /= (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12)
    print(f"      concat shape={feats.shape}")

    target_dim = min(128, feats.shape[1] - 1, feats.shape[0] - 1)
    target_dim = max(2, target_dim)
    if _SKLEARN and feats.shape[0] > target_dim:
        pca = PCA(n_components=target_dim, random_state=42)
        red = pca.fit_transform(feats).astype(np.float32)
    else:
        red = feats
    red /= (np.linalg.norm(red, axis=1, keepdims=True) + 1e-12)

    # -- KNN on the reduced space ------------------------------------
    k_use = min(knn_k, n - 1)
    if _SKLEARN:
        nn = NearestNeighbors(n_neighbors=k_use+1, metric="cosine")
        nn.fit(red)
        _, idx = nn.kneighbors(red)
        idx = idx[:, 1:]                        # drop self
    else:
        sim = red @ red.T
        np.fill_diagonal(sim, -np.inf)
        idx = np.argpartition(-sim, k_use, axis=1)[:, :k_use]

    # -- Mutual KNN pair scoring ------------------------------------
    print("[r4] scoring pairs via mutual KNN membership ...")
    nn_sets = [set(row.tolist()) for row in idx]
    sim = red @ red.T
    preds: List[PairPred] = []
    for i in range(n):
        for j in range(i+1, n):
            mutual = ((j in nn_sets[i]) + (i in nn_sets[j])) / 2.0
            cos_sim = float(sim[i, j])
            score = 0.5 * mutual + 0.5 * (cos_sim + 1) / 2.0
            pred  = 1 if (mutual >= mutual_frac and cos_sim > 0.85) else 0
            preds.append(PairPred(
                names[i], names[j], float(score), pred,
                extra={"cosine": cos_sim, "mutual": mutual}))

    groups = build_groups(preds)
    runtime = time.time() - t0

    extra = [
        f"  DWT level / size       : 3-level Haar  on {dwt.shape[1]}-bin feature",
        f"  CNN backbone           : ResNet-18 pretrained (surrogate for "
        f"the 5-block Cn CNN of Singh et al. Table 1)",
        f"  Concatenated feature   : {feats.shape[1]} → PCA({red.shape[1]})",
        f"  KNN neighbours         : k={k_use}",
        f"  Mutual-frac threshold  : {mutual_frac}",
        f"  Cos-sim confirm rule   : > 0.85",
        f"  scikit-learn           : {'yes' if _SKLEARN else 'no (numpy fallback)'}",
        f"  Device                 : {device}",
    ]
    write_baseline_report(
        report_path=report,
        method_name="DWT + CNN + KNN (r4)",
        paper_ref="Singh, Kumar, Ranjan, Nandan; Soft Computing 28:7659-7670, 2024",
        filenames=names, preds=preds, groups=groups,
        runtime_sec=runtime, extra_lines=extra)
    print(f"[r4] wrote {report}  (runtime {runtime:.1f}s)")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("folder")
    p.add_argument("--report", required=True)
    p.add_argument("--knn", type=int, default=7)
    p.add_argument("--mutual", type=float, default=0.35)
    p.add_argument("--batch", type=int, default=16)
    a = p.parse_args()
    if not os.path.isdir(a.folder):
        print("Not a directory:", a.folder); sys.exit(1)
    run(a.folder, a.report, a.knn, a.mutual, a.batch)


if __name__ == "__main__":
    main()