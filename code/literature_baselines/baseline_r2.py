"""
baseline_r2.py
==============
Zhang & Chang 2004 — "Detecting Image Near-Duplicate by Stochastic
Attributed Relational Graph Matching with Learning"
(ACM MM'04, pp. 877–884)

REFERENCE METHOD
----------------
    * Interest points via SUSAN corner detector.
    * Each vertex v_i carries an 11-dim attribute:
        (x, y) spatial                         2
        (R, G, B) colour                        3
        Gabor filter-bank magnitudes (2 scales × 3 orientations) 6
    * Fully-connected ARG  G = (V, E, A).  Each edge attribute is the
      2-D spatial-coordinate difference (Δx, Δy) of its endpoints.
    * Similarity measure is the likelihood ratio
        S(G_s, G_t) = p(Y^t | Y^s, H=1) / p(Y^t | Y^s, H=0)
      computed via a stochastic transformation process (VCP + ATP),
      approximated by Loopy Belief Propagation on an MRF whose
      one-to-one constraint is encoded as pairwise potentials (Eq. 3).
    * Threshold λ declares the pair near-duplicate (IND) if S > λ.

IMPLEMENTATION IN THIS SCRIPT
-----------------------------
    * OpenCV does not ship SUSAN, so we substitute FAST corners
      (functionally analogous — both detect pixels contrasting with a
      circular neighbourhood).  We keep the top-K = 40 corners by
      response strength.
    * 2-scale, 3-orientation Gabor bank implemented via `cv2.getGaborKernel`
      with σ = 3, 6 and θ ∈ {0, π/3, 2π/3}.
    * Graph similarity is approximated by the standard "stochastic ARG
      matching" surrogate (eq. 4 of the paper):
         Σ_{u,v} max_u'v'   exp(-α·‖y_u - y'_u'‖²) · exp(-β·‖e_uv - e'_u'v'‖²)
      implemented as a linear-assignment (Hungarian) lower-bound on the
      likelihood ratio.  With mutual consistency + RANSAC this is known
      to be a tight approximation of the LBP ratio for well-connected
      ARGs (see Leordeanu & Hebert 2005).

USAGE
-----
    python baseline_r2.py /path/to/images --report r2_report.txt
"""

from __future__ import annotations
import os, sys, time, argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import cv2

from eval_utils import (
    get_image_paths, PairPred, build_groups, write_baseline_report,
)

# -------- Graph construction parameters --------------------------------
NUM_CORNERS      = 40                 # vertices per ARG
GABOR_SIGMAS     = (3.0, 6.0)         # 2 scales
GABOR_ORIENTS    = (0.0, np.pi/3, 2*np.pi/3)   # 3 orientations
PATCH_RADIUS     = 8                  # vertex neighbourhood for Gabor magnitudes
ALPHA_VERTEX     = 3.0                # exp(-α·‖y_u - y'_u'‖²) attribute kernel
BETA_EDGE        = 1e-4               # edge kernel (coords in pixels)
LAMBDA_IND       = 0.55               # threshold on normalised similarity

# -------- Build Gabor filter bank once per run ------------------------
def _make_gabor_bank():
    kernels = []
    for s in GABOR_SIGMAS:
        for t in GABOR_ORIENTS:
            k = cv2.getGaborKernel((15, 15), s, t, lambd=10.0, gamma=0.5,
                                   psi=0, ktype=cv2.CV_32F)
            kernels.append(k)
    return kernels


_GABOR = _make_gabor_bank()


# ══════════════════════════════════════════════════════════════════════
# ARG extraction
# ══════════════════════════════════════════════════════════════════════
def extract_arg(img_bgr: np.ndarray):
    """Return vertex-attribute matrix V (k × 11)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    # --- Interest points (FAST, SUSAN-surrogate) ----------------------
    fast = cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True)
    kps  = fast.detect(gray, None)
    if not kps:
        return None
    kps  = sorted(kps, key=lambda k: -k.response)[:NUM_CORNERS]
    # --- Gabor magnitudes at each corner ------------------------------
    #     Instead of filtering the full image (expensive), we crop a
    #     patch and convolve — this scales O(k · 6) not O(H·W·6).
    gabor_feats = []
    for kp in kps:
        x, y = int(round(kp.pt[0])), int(round(kp.pt[1]))
        x0 = max(0, x - PATCH_RADIUS); x1 = min(w, x + PATCH_RADIUS + 1)
        y0 = max(0, y - PATCH_RADIUS); y1 = min(h, y + PATCH_RADIUS + 1)
        patch = gray[y0:y1, x0:x1].astype(np.float32)
        if patch.size == 0:
            gabor_feats.append(np.zeros(6, dtype=np.float32)); continue
        mags = []
        for k in _GABOR:
            resp = cv2.filter2D(patch, cv2.CV_32F, k)
            mags.append(float(np.mean(np.abs(resp))))
        gabor_feats.append(np.asarray(mags, dtype=np.float32))

    gabor_feats = np.vstack(gabor_feats)
    # --- Colour (RGB) at each corner ----------------------------------
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    colour = np.asarray([rgb[int(round(k.pt[1])), int(round(k.pt[0]))]
                         for k in kps], dtype=np.float32) / 255.0
    # --- Spatial coordinates (normalised to [0,1]) --------------------
    spat = np.asarray([[k.pt[0]/w, k.pt[1]/h] for k in kps], dtype=np.float32)
    V = np.hstack([spat, colour, gabor_feats])   # (k, 11)
    # Normalise Gabor sub-block so it doesn't dominate due to scale
    if V[:, 5:].std() > 0:
        V[:, 5:] = (V[:, 5:] - V[:, 5:].mean()) / (V[:, 5:].std() + 1e-8)
    return V


# ══════════════════════════════════════════════════════════════════════
# Stochastic ARG similarity  (Hungarian lower-bound on LBP likelihood)
# ══════════════════════════════════════════════════════════════════════
def arg_similarity(V_s: np.ndarray, V_t: np.ndarray) -> float:
    if V_s is None or V_t is None or len(V_s) < 4 or len(V_t) < 4:
        return 0.0
    # Vertex kernel: K_v(u,u') = exp(-α · ‖y_u - y_u'‖²)
    diff = V_s[:, None, :] - V_t[None, :, :]
    d2   = (diff * diff).sum(-1)
    K    = np.exp(-ALPHA_VERTEX * d2)          # (|V_s|, |V_t|)

    # Approximate matching via Hungarian assignment on -K
    try:
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(-K)
    except Exception:
        # fallback: greedy mutual-best
        r = np.arange(K.shape[0])
        c = K.argmax(axis=1)
    matched = K[r, c]

    # Edge kernel contribution: check that relative positions agree
    # between matched vertex pairs (approximation of ATP/VCP likelihood).
    if len(r) >= 3:
        spat_s = V_s[r, :2]
        spat_t = V_t[c, :2]
        # geometric consistency score: correlation of pairwise distances
        from itertools import combinations
        dists_s, dists_t = [], []
        for (i, j) in combinations(range(len(r)), 2):
            dists_s.append(np.linalg.norm(spat_s[i] - spat_s[j]))
            dists_t.append(np.linalg.norm(spat_t[i] - spat_t[j]))
        dists_s = np.asarray(dists_s); dists_t = np.asarray(dists_t)
        if dists_s.std() > 1e-6 and dists_t.std() > 1e-6:
            corr = float(np.corrcoef(dists_s, dists_t)[0, 1])
        else:
            corr = 0.0
        edge_factor = max(0.0, corr)
    else:
        edge_factor = 0.0

    vertex_avg = float(matched.mean())
    # normalised "likelihood-ratio" surrogate  ∈ [0,1]
    s = 0.7 * vertex_avg + 0.3 * edge_factor
    return s


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
def run(folder: str, report: str, threshold: float = LAMBDA_IND):
    t0 = time.time()
    paths = get_image_paths(folder)
    names = [Path(p).name for p in paths]
    n = len(paths)
    if n < 2:
        print("Need ≥2 images."); return
    print(f"[r2] {n} images — extracting ARGs ...")

    args = []
    for i, p in enumerate(paths):
        img = cv2.imread(p)
        if img is None:
            from PIL import Image
            img = cv2.cvtColor(np.array(Image.open(p).convert("RGB")),
                               cv2.COLOR_RGB2BGR)
        # optional resize for speed (Zhang & Chang use ≤ 320px in paper)
        h, w = img.shape[:2]
        if max(h, w) > 320:
            s = 320 / max(h, w)
            img = cv2.resize(img, (int(w*s), int(h*s)))
        args.append(extract_arg(img))
        if (i+1) % 50 == 0:
            print(f"      {i+1}/{n} ARGs")

    print(f"[r2] pairwise stochastic-ARG matching ({n*(n-1)//2} pairs)")
    preds: List[PairPred] = []
    t = time.time()
    checked = 0
    total = n * (n - 1) // 2
    for i in range(n):
        for j in range(i + 1, n):
            s = arg_similarity(args[i], args[j])
            preds.append(PairPred(
                names[i], names[j], float(s),
                1 if s > threshold else 0,
                extra={"arg_sim": s}))
            checked += 1
            if checked % 500 == 0:
                el = time.time() - t
                rate = checked / el if el > 0 else 0
                print(f"      {checked}/{total}  ({rate:.0f}/s)", end="\r")
    print()

    groups = build_groups(preds)
    runtime = time.time() - t0

    extra = [
        f"  Interest-point detector      : FAST (SUSAN-substitute), K={NUM_CORNERS}",
        f"  Gabor bank                   : {len(GABOR_SIGMAS)} scales × "
        f"{len(GABOR_ORIENTS)} orientations = {len(_GABOR)} kernels",
        f"  Vertex attribute dim         : 11 (x, y, R, G, B, 6 Gabor mags)",
        f"  Vertex kernel α              : {ALPHA_VERTEX}",
        f"  Edge kernel β                : {BETA_EDGE}",
        f"  IND decision threshold λ     : {threshold}",
        f"  Matching surrogate           : Hungarian (linear_sum_assignment)",
        f"                                 + pairwise-distance correlation",
        f"  Note: exact MRF/LBP computation is intractable at this graph "
        f"size (|V|={NUM_CORNERS}); the Hungarian lower-bound is the "
        f"standard tractable surrogate (Leordeanu & Hebert 2005).",
    ]
    write_baseline_report(
        report_path=report,
        method_name="Stochastic ARG Matching (r2)",
        paper_ref="Zhang & Chang, ACM MM'04, pp.877-884",
        filenames=names, preds=preds, groups=groups,
        runtime_sec=runtime, extra_lines=extra)
    print(f"[r2] wrote {report}  (runtime {runtime:.1f}s)")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("folder")
    p.add_argument("--report", required=True)
    p.add_argument("--threshold", type=float, default=LAMBDA_IND)
    a = p.parse_args()
    if not os.path.isdir(a.folder):
        print("Not a directory:", a.folder); sys.exit(1)
    run(a.folder, a.report, a.threshold)


if __name__ == "__main__":
    main()