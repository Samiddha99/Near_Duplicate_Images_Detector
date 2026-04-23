"""
Near-Duplicate Image Detector v5.2.2 — Threaded + GPU (AMD/Intel/NVIDIA)
=========================================================================
Uses ThreadPoolExecutor instead of multiprocessing — no freeze, no pickle,
no spawn overhead. OpenCV releases the GIL so threads run truly parallel.

Usage:
  python dup.py /path/to/images                         # auto threads
  python dup.py /path/to/images --gpu                   # GPU accelerated
  python dup.py /path/to/images --workers 8 --gpu       # 8 threads + GPU
  python dup.py /path/to/images --fast --workers 8      # fast + threaded
"""

import os
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading
import numpy as np
from PIL import Image
import cv2
from skimage.metrics import structural_similarity as compare_ssim


# ═══════════════════════════════════════════════════════════════════════════
# GPU BACKEND: OpenCL via UMat (AMD Radeon / Intel / NVIDIA)
# ═══════════════════════════════════════════════════════════════════════════

_GPU_ENABLED = False

def init_gpu():
    global _GPU_ENABLED
    if not cv2.ocl.haveOpenCL():
        return False
    cv2.ocl.setUseOpenCL(True)
    if not cv2.ocl.useOpenCL():
        return False
    try:
        cv2.cvtColor(cv2.UMat(np.zeros((10, 10, 3), dtype=np.uint8)), cv2.COLOR_BGR2GRAY)
        _GPU_ENABLED = True
        return True
    except Exception:
        return False

def _u(a): return cv2.UMat(a) if _GPU_ENABLED else a
def _n(u): return u.get() if isinstance(u, cv2.UMat) else u

def gpu_gray(img):
    if len(img.shape) != 3: return img
    return _n(cv2.cvtColor(_u(img), cv2.COLOR_BGR2GRAY)) if _GPU_ENABLED else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def gpu_clahe(gray):
    c = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    try: return _n(c.apply(_u(gray)))
    except: return c.apply(gray)

def gpu_warp(img, H, size):
    try: return _n(cv2.warpPerspective(_u(img), H, size))
    except: return cv2.warpPerspective(img, H, size)

def gpu_orb(gray, n=3000):
    orb = cv2.ORB_create(n)
    try:
        kps, desc = orb.detectAndCompute(_u(gray), None)
        return kps, _n(desc) if desc is not None else None
    except: return orb.detectAndCompute(gray, None)

def gpu_flip(img, code):
    try: return _n(cv2.flip(_u(img), code))
    except: return cv2.flip(img, code)

def gpu_tmatch(a, b):
    try: return _n(cv2.matchTemplate(_u(a), _u(b), cv2.TM_CCOEFF_NORMED))
    except: return cv2.matchTemplate(a, b, cv2.TM_CCOEFF_NORMED)


# ═══════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def auto_crop_black(img, thresh=15):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    mask = gray > thresh
    coords = np.column_stack(np.where(mask))
    if coords.size == 0: return img
    y0, x0 = coords.min(axis=0); y1, x1 = coords.max(axis=0)
    h, w = gray.shape
    if y0 > h * .03 or x0 > w * .03 or y1 < h * .97 or x1 < w * .97:
        c = img[y0:y1 + 1, x0:x1 + 1]
        if c.size > img.size * 0.25: return c
    return img

def load_image(path, max_dim=512):
    pil = Image.open(path).convert("RGB")
    pil.thumbnail((max_dim, max_dim), Image.LANCZOS)
    img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    del pil
    return auto_crop_black(img)

def get_paths(folder):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    return [str(f) for f in sorted(Path(folder).rglob("*"))
            if f.suffix.lower() in exts and f.is_file()]


# ═══════════════════════════════════════════════════════════════════════════
# FINGERPRINT
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Fingerprint:
    path: str
    filename: str
    file_size: int
    hu_moments: np.ndarray
    hu_moments_eq: np.ndarray
    color_hist: np.ndarray
    gray_mean: float
    gray_std: float
    texture_density: float
    color_diversity: float
    keypoint_count: int


def compute_fingerprint(path, max_dim=512):
    try:
        img = load_image(path, max_dim)
        gray = gpu_gray(img)
        hu = cv2.HuMoments(cv2.moments(gray)).flatten()
        hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)
        gray_eq = gpu_clahe(gray)
        hu_eq = cv2.HuMoments(cv2.moments(gray_eq)).flatten()
        hu_eq_log = -np.sign(hu_eq) * np.log10(np.abs(hu_eq) + 1e-30)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hh = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
        hs = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
        hv = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()
        color_hist = np.concatenate([hh, hs, hv])
        color_hist = color_hist / (color_hist.sum() + 1e-8)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        texture_density = float(lap.var())
        ch = color_hist[color_hist > 0]
        color_diversity = float(-np.sum(ch * np.log2(ch + 1e-10)))
        kps, _ = gpu_orb(gray, 500)
        fp = Fingerprint(
            path=path, filename=Path(path).name,
            file_size=os.path.getsize(path),
            hu_moments=hu_log, hu_moments_eq=hu_eq_log,
            color_hist=color_hist,
            gray_mean=float(gray.mean()), gray_std=float(gray.std()),
            texture_density=texture_density,
            color_diversity=color_diversity,
            keypoint_count=len(kps) if kps else 0,
        )
        del img, gray, gray_eq, hsv
        return fp
    except Exception:
        return None


def adaptive_screen_threshold(fp_a, fp_b, base_thresh=0.30):
    avg_texture = (fp_a.texture_density + fp_b.texture_density) / 2.0
    texture_factor = min(1.0, avg_texture / 2000.0)
    adapted = base_thresh + 0.05 * texture_factor
    return max(0.15, min(0.36, adapted))


def fingerprint_similarity(a, b):
    hu_dist = np.sqrt(np.sum((a.hu_moments - b.hu_moments) ** 2))
    hu_sim = 1.0 / (1.0 + hu_dist * 0.3)
    hu_eq_dist = np.sqrt(np.sum((a.hu_moments_eq - b.hu_moments_eq) ** 2))
    hu_eq_sim = 1.0 / (1.0 + hu_eq_dist * 0.3)
    best_hu = max(hu_sim, hu_eq_sim)
    hist_corr = float(cv2.compareHist(
        a.color_hist.astype(np.float32),
        b.color_hist.astype(np.float32),
        cv2.HISTCMP_CORREL))
    hist_sim = max(0.0, hist_corr)
    std_ratio = min(a.gray_std, b.gray_std) / (max(a.gray_std, b.gray_std) + 1e-8)
    combined = 0.5 * best_hu + 0.3 * hist_sim + 0.2 * std_ratio
    return {"hu_sim": round(best_hu, 4), "hist_sim": round(hist_sim, 4),
            "std_ratio": round(std_ratio, 4), "screen_score": round(combined, 4)}


# ═══════════════════════════════════════════════════════════════════════════
# HIERARCHICAL HOMOGRAPHY
# ═══════════════════════════════════════════════════════════════════════════

def find_homography_single(ga, gb, n_features=3000):
    kp_a, desc_a = gpu_orb(ga, n_features)
    kp_b, desc_b = gpu_orb(gb, n_features)
    if desc_a is None or desc_b is None or len(kp_a) < 6 or len(kp_b) < 6:
        return None, 0, 0.0, 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw = bf.knnMatch(desc_a, desc_b, k=2)
    good = [(m, kp_a[m.queryIdx].pt, kp_b[m.trainIdx].pt)
            for m, n in raw if m.distance < 0.75 * n.distance]
    if len(good) < 6: return None, 0, 0.0, len(good)
    pts_a = np.float32([g[1] for g in good]).reshape(-1, 1, 2)
    pts_b = np.float32([g[2] for g in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(pts_b, pts_a, cv2.RANSAC, 5.0)
    if H is None or mask is None: return None, 0, 0.0, len(good)
    return H, int(mask.sum()), int(mask.sum()) / len(good), len(good)


def find_homography_hierarchical(cv_a, cv_b, n_features=3000):
    ga, gb = gpu_gray(cv_a), gpu_gray(cv_b)
    gb_flip = gpu_flip(gb, 1)
    best = {"H": None, "inliers": 0, "inlier_ratio": 0.0,
            "total_matches": 0, "flipped": False, "level": "none"}
    for gray_b, is_flipped in [(gb, False), (gb_flip, True)]:
        H, inl, ratio, nm = find_homography_single(ga, gray_b, n_features)
        if inl > best["inliers"]:
            best = {"H": H, "inliers": inl, "inlier_ratio": round(ratio, 4),
                    "total_matches": nm, "flipped": is_flipped, "level": "global"}
        if best["inliers"] < 15 or best["inlier_ratio"] < 0.4:
            ha, wa = ga.shape[:2]
            for qx0, qy0, qx1, qy1 in [
                (0, 0, wa // 2, ha // 2), (wa // 2, 0, wa, ha // 2),
                (0, ha // 2, wa // 2, ha), (wa // 2, ha // 2, wa, ha),
                (wa // 4, ha // 4, 3 * wa // 4, 3 * ha // 4)
            ]:
                qa = ga[qy0:qy1, qx0:qx1]
                if qa.shape[0] < 32 or qa.shape[1] < 32: continue
                Hq, iq, rq, nq = find_homography_single(qa, gray_b, n_features // 2)
                if iq > best["inliers"] and Hq is not None:
                    off = np.eye(3, dtype=np.float64)
                    off[0, 2], off[1, 2] = qx0, qy0
                    best = {"H": off @ Hq, "inliers": iq, "inlier_ratio": round(rq, 4),
                            "total_matches": nq, "flipped": is_flipped, "level": "quadrant"}
    return best


# ═══════════════════════════════════════════════════════════════════════════
# WARP-AND-COMPARE
# ═══════════════════════════════════════════════════════════════════════════

def warp_and_compare(cv_a, cv_b, H, flipped):
    h_a, w_a = cv_a.shape[:2]
    if flipped: cv_b = gpu_flip(cv_b, 1)
    warped_b = gpu_warp(cv_b, H, (w_a, h_a))
    gray_w = gpu_gray(warped_b)
    overlap_mask = gray_w > 10
    overlap_px = overlap_mask.sum()
    total_px = h_a * w_a
    if overlap_px < total_px * 0.10:
        return {"warp_ssim": 0.0, "warp_pixel_sim": 0.0, "overlap_ratio": 0.0}
    overlap_ratio = overlap_px / total_px
    ga, gb = gpu_gray(cv_a), gpu_gray(warped_b)
    ga_eq, gb_eq = gpu_clahe(ga), gpu_clahe(gb)

    def znorm(img, mask):
        px = img[mask].astype(np.float64)
        if len(px) == 0 or px.std() < 1: return img.astype(np.float64)
        return (img.astype(np.float64) - px.mean()) / (px.std() + 1e-8)

    ga_n, gb_n = znorm(ga_eq, overlap_mask), znorm(gb_eq, overlap_mask)
    rows, cols = np.any(overlap_mask, axis=1), np.any(overlap_mask, axis=0)
    if not np.any(rows) or not np.any(cols):
        return {"warp_ssim": 0.0, "warp_pixel_sim": 0.0, "overlap_ratio": 0.0}
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    pa, pb = ga_n[r0:r1 + 1, c0:c1 + 1], gb_n[r0:r1 + 1, c0:c1 + 1]
    ms = min(pa.shape); win = min(7, ms if ms % 2 == 1 else ms - 1)
    if win < 3: win = 3
    ssim_val = compare_ssim(pa, pb, win_size=win,
        data_range=pa.max() - pa.min() + 1e-8) if pa.shape[0] >= win and pa.shape[1] >= win else 0.0
    diff = np.abs(ga_n - gb_n)
    mae = diff[overlap_mask].mean() if overlap_px > 0 else 999
    return {"warp_ssim": round(max(0.0, ssim_val), 4),
            "warp_pixel_sim": round(1.0 / (1.0 + mae * 0.5), 4),
            "overlap_ratio": round(overlap_ratio, 4)}


# ═══════════════════════════════════════════════════════════════════════════
# LOW-TEXTURE FALLBACK
# ═══════════════════════════════════════════════════════════════════════════

def _low_texture_fallback(cv_a, cv_b, size=(128, 128)):
    def prep(img):
        c = auto_crop_black(img, 15)
        return gpu_clahe(cv2.resize(gpu_gray(c), size))

    a, b_base = prep(cv_a), prep(cv_b)
    best = 0.0
    variants = [b_base, gpu_flip(b_base, 1), gpu_flip(b_base, 0),
                gpu_flip(gpu_flip(b_base, 1), 0)]
    for k in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]:
        r = cv2.resize(cv2.rotate(b_base, k), size)
        variants.extend([r, gpu_flip(r, 1)])
    pil_b = Image.fromarray(b_base)
    for angle in range(15, 360, 15):
        variants.append(np.array(pil_b.rotate(angle, expand=False, fillcolor=0)))
    for b in variants:
        if b.shape != a.shape: b = cv2.resize(b, size)
        score = float(gpu_tmatch(a, b).max())
        if score > best: best = score
    return round(max(0.0, best), 4)


# ═══════════════════════════════════════════════════════════════════════════
# VERIFY + DECIDE
# ═══════════════════════════════════════════════════════════════════════════

def verify_pair(path_a, path_b, fp_a, fp_b, max_dim=512):
    cv_a, cv_b = load_image(path_a, max_dim), load_image(path_b, max_dim)
    homo = find_homography_hierarchical(cv_a, cv_b)
    result = {"orb_inliers": homo["inliers"], "orb_inlier_ratio": homo["inlier_ratio"],
              "orb_matches": homo["total_matches"], "flipped": homo["flipped"],
              "match_level": homo["level"], "warp_ssim": 0.0, "warp_pixel_sim": 0.0,
              "overlap_ratio": 0.0, "fallback_ssim": 0.0}
    if homo["H"] is not None and homo["inliers"] >= 4:
        result.update(warp_and_compare(cv_a, cv_b, homo["H"], homo["flipped"]))
    max_kp = max(fp_a.keypoint_count, fp_b.keypoint_count)
    if max_kp < 80 and result["orb_inliers"] < 15:
        result["fallback_ssim"] = _low_texture_fallback(cv_a, cv_b)
        if result["fallback_ssim"] > 0.5: result["match_level"] = "fallback"
    del cv_a, cv_b
    return result


def decide_duplicate(screen, detail, fp_a, fp_b, threshold=0.50):
    scores = {
        "warp_ssim": detail.get("warp_ssim", 0.0),
        "warp_pixel_sim": detail.get("warp_pixel_sim", 0.0),
        "orb_inlier_ratio": detail.get("orb_inlier_ratio", 0.0),
        "hu_sim": screen.get("hu_sim", 0.0),
        "hist_sim": screen.get("hist_sim", 0.0),
        "overlap_ratio": detail.get("overlap_ratio", 0.0),
        "fallback_ssim": detail.get("fallback_ssim", 0.0),
    }
    min_kp = min(fp_a.keypoint_count, fp_b.keypoint_count)
    kp_factor = min(1.0, min_kp / 300.0)
    if kp_factor < 0.3:
        weights = {"warp_ssim": .40, "warp_pixel_sim": .25, "orb_inlier_ratio": .10,
                   "hu_sim": .10, "hist_sim": .10, "overlap_ratio": .05}
        min_inlier_req = 4
    elif kp_factor < 0.6:
        weights = {"warp_ssim": .35, "warp_pixel_sim": .20, "orb_inlier_ratio": .20,
                   "hu_sim": .10, "hist_sim": .10, "overlap_ratio": .05}
        min_inlier_req = 6
    else:
        weights = {"warp_ssim": .30, "warp_pixel_sim": .15, "orb_inlier_ratio": .25,
                   "hu_sim": .10, "hist_sim": .10, "overlap_ratio": .10}
        min_inlier_req = 8
    confidence = sum(scores.get(k, 0) * w for k, w in weights.items())
    ni = detail.get("orb_inliers", 0)
    ir = detail.get("orb_inlier_ratio", 0.0)
    n_matches = detail.get("orb_matches", ni)
    has_inliers = ni >= min_inlier_req
    has_warp = scores["warp_ssim"] >= 0.35

    if n_matches < 30 and ni < 30 and ir < 0.90:
        has_warp = scores["warp_ssim"] >= 0.50
    if ni >= 100 and ir >= 0.50:
        gc_ = 0.3 * min(1.0, ni / 200.0) + 0.4 * ir + 0.3 * scores.get("overlap_ratio", 0.5)
        confidence = max(confidence, gc_); has_warp = True
    elif ni >= 50 and ir >= 0.50 and scores["warp_ssim"] >= 0.20:
        gc_ = 0.25 * min(1.0, ni / 100.0) + 0.35 * ir + 0.2 * scores["warp_ssim"] + 0.2 * scores.get("overlap_ratio", 0.5)
        confidence = max(confidence, gc_); has_warp = True
    fb = detail.get("fallback_ssim", 0.0)
    max_kp = max(fp_a.keypoint_count, fp_b.keypoint_count)
    if fb >= 0.75 and max_kp < 60:
        fbc = 0.5 * fb + 0.3 * scores["hu_sim"] + 0.2 * scores["hist_sim"]
        confidence = max(confidence, fbc); has_inliers = True; has_warp = True
    return confidence >= threshold and has_inliers and has_warp


# ═══════════════════════════════════════════════════════════════════════════
# UNION-FIND
# ═══════════════════════════════════════════════════════════════════════════

class UnionFind:
    def __init__(self):
        self.parent = {}; self.rank = {}
        self._lock = threading.Lock()

    def find(self, x):
        if x not in self.parent: self.parent[x] = x; self.rank[x] = 0
        if self.parent[x] != x: self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        with self._lock:
            rx, ry = self.find(x), self.find(y)
            if rx == ry: return
            if self.rank[rx] < self.rank[ry]: rx, ry = ry, rx
            self.parent[ry] = rx
            if self.rank[rx] == self.rank[ry]: self.rank[rx] += 1

    def groups(self):
        clusters = {}
        for x in self.parent: clusters.setdefault(self.find(x), set()).add(x)
        return {r: m for r, m in clusters.items() if len(m) > 1}


# ═══════════════════════════════════════════════════════════════════════════
# PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def run(folder, threshold=0.50, max_dim=512, base_screen=0.30,
        fast=False, workers=0, use_gpu=False) -> dict[str, list[str]]:
    """
    Find near-duplicate images in a folder.

    Returns a dict mapping group labels to lists of filenames, e.g.:
        {
            "group 1": ["img1.png", "img2.jpg"],
            "group 2": ["img3.jpg", "img4.png", "img5.png"],
        }
    """
    if use_gpu:
        init_gpu()

    nw = workers if workers > 0 else max(1, os.cpu_count() or 4)
    paths = get_paths(folder)
    n = len(paths)
    if n < 2:
        return {}

    # ── Fingerprint (threaded) ──
    fps = [None] * n

    def fp_task(idx, path):
        fps[idx] = compute_fingerprint(path, max_dim)

    with ThreadPoolExecutor(max_workers=nw) as pool:
        futures = [pool.submit(fp_task, i, p) for i, p in enumerate(paths)]
        for f in futures:
            f.result()

    fps = [f for f in fps if f is not None]
    if len(fps) < 2:
        return {}

    # ── Screen ──
    cands = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            sc = fingerprint_similarity(fps[i], fps[j])
            pt = adaptive_screen_threshold(fps[i], fps[j], base_screen)
            if sc["screen_score"] >= pt:
                cands.append((i, j, sc))

    if not cands:
        return {}

    uf = UnionFind()

    if fast:
        for i, j, sc in cands:
            if sc["screen_score"] >= threshold:
                uf.union(fps[i].filename, fps[j].filename)
    else:
        results_lock = threading.Lock()

        def verify_task(idx):
            i, j, sc = cands[idx]
            detail = verify_pair(fps[i].path, fps[j].path, fps[i], fps[j], max_dim)
            is_dup = decide_duplicate(sc, detail, fps[i], fps[j], threshold)
            if is_dup:
                with results_lock:
                    uf.union(fps[i].filename, fps[j].filename)

        with ThreadPoolExecutor(max_workers=nw) as pool:
            futures = [pool.submit(verify_task, idx) for idx in range(len(cands))]
            for f in futures:
                f.result()

    raw_groups = uf.groups()
    return {
        f"group {gid}": sorted(members)
        for gid, (_, members) in enumerate(
            sorted(raw_groups.items(), key=lambda x: -len(x[1])), 1
        )
    }

# ═══════════════════════════════════════════════════════════════════════════
# Display two images side by side
# ═══════════════════════════════════════════════════════════════════════════

import matplotlib.pyplot as plt
from PIL import Image
import os

def display_images_side_by_side(img_path1, title1, img_path2, title2):
    """
    Displays two images side-by-side with custom titles.
    """
    # 1. Verify both images exist before attempting to load them
    if not os.path.exists(img_path1):
        print(f"Error: Image not found at '{img_path1}'")
        return
    if not os.path.exists(img_path2):
        print(f"Error: Image not found at '{img_path2}'")
        return

    try:
        # 2. Load the images using Pillow
        img1 = Image.open(img_path1)
        img2 = Image.open(img_path2)

        # 3. Create a figure with 1 row and 2 columns
        # figsize=(width, height) in inches. Adjust as needed.
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # 4. Display the first image
        axes[0].imshow(img1)
        axes[0].set_title(title1, fontsize=14)
        axes[0].axis('off') # Hides the x and y axes for a cleaner look

        # 5. Display the second image
        axes[1].imshow(img2)
        axes[1].set_title(title2, fontsize=14)
        axes[1].axis('off')

        # Adjust layout so titles don't overlap
        plt.tight_layout()

        # 6. Render the plot
        plt.show()

    except Exception as e:
        print(f"An error occurred while displaying the images: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════
import os
import shutil

def main():
    import argparse, json
    p = argparse.ArgumentParser(description="Near-duplicate image finder")
    p.add_argument("image")
    p.add_argument("--threshold", type=float, default=0.50)
    p.add_argument("--max-dim", type=int, default=512)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--fast", action="store_true")
    p.add_argument("--gpu", action="store_true")
    args = p.parse_args()

    desired_name = "renamed_image.jpg"
    target_dir = './dataset/real'
    target_path = os.path.join(target_dir, desired_name)

    if not os.path.exists(args.image):
        print(f"Error: The source file '{args.image}' does not exist.")
        return
    
    try:
        shutil.copy(args.image, target_path)
        print(f"Successfully copied to: {target_path}")
    except Exception as e:
        print(f"Failed to copy the file: {e}")
        return

    groups = run(
        target_dir,
        threshold=args.threshold,
        max_dim=args.max_dim,
        fast=args.fast,
        workers=args.workers,
        use_gpu=args.gpu,
    )
    if groups:
        paired_images_g1 = groups['group 1']
        image1 = os.path.join(target_dir, paired_images_g1[0])
        image2 = os.path.join(target_dir, paired_images_g1[1])
        display_images_side_by_side(image1, "Uploaded", image2, "Detected Real")

    print(json.dumps(groups, indent=2))

    if os.path.exists(target_path):
        os.remove(target_path)
        print(f"Cleanup complete. Deleted: {target_path}")


if __name__ == "__main__":
    main()