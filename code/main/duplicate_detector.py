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

import os, sys, gc, time, argparse, re
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import numpy as np
from PIL import Image, ImageOps
import cv2
from skimage.metrics import structural_similarity as compare_ssim


# ═══════════════════════════════════════════════════════════════════════════
# GPU BACKEND: OpenCL via UMat (AMD Radeon / Intel / NVIDIA)
# ═══════════════════════════════════════════════════════════════════════════

_GPU_ENABLED = False
_GPU_INFO = "CPU"

def init_gpu():
    global _GPU_ENABLED, _GPU_INFO
    if not cv2.ocl.haveOpenCL():
        _GPU_INFO = "CPU (no OpenCL)"; return False
    cv2.ocl.setUseOpenCL(True)
    if not cv2.ocl.useOpenCL():
        _GPU_INFO = "CPU (OpenCL failed)"; return False
    try:
        _ = cv2.cvtColor(cv2.UMat(np.zeros((10,10,3), dtype=np.uint8)),
                         cv2.COLOR_BGR2GRAY)
        _GPU_ENABLED = True
        try:
            d = cv2.ocl.Device.getDefault()
            _GPU_INFO = f"{d.vendorName()} {d.name()} (OpenCL)"
        except:
            _GPU_INFO = "OpenCL GPU"
        return True
    except Exception as e:
        _GPU_INFO = f"CPU ({e})"; return False

def _u(a): return cv2.UMat(a) if _GPU_ENABLED else a
def _n(u): return u.get() if isinstance(u, cv2.UMat) else u

def gpu_gray(img):
    if len(img.shape) != 3: return img
    return _n(cv2.cvtColor(_u(img), cv2.COLOR_BGR2GRAY)) if _GPU_ENABLED else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def gpu_clahe(gray):
    c = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
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
    if y0 > h*.03 or x0 > w*.03 or y1 < h*.97 or x1 < w*.97:
        c = img[y0:y1+1, x0:x1+1]
        if c.size > img.size * 0.25: return c
    return img

def load_image(path, max_dim=512):
    pil = Image.open(path).convert("RGB")
    pil.thumbnail((max_dim, max_dim), Image.LANCZOS)
    img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    del pil
    return auto_crop_black(img)

def fmt(s):
    if s < 60: return f"{s:.0f}s"
    if s < 3600: return f"{s/60:.1f}m"
    return f"{s/3600:.1f}h"

def get_paths(folder):
    exts = {".jpg",".jpeg",".png",".bmp",".tiff",".tif",".webp"}
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
    phash: int
    dhash: int


def _compute_phash(gray, hash_size=8):
    resized = cv2.resize(gray, (hash_size * 4, hash_size * 4), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(resized.astype(np.float64))
    dct_low = dct[:hash_size, :hash_size]
    median = np.median(dct_low)
    bits = (dct_low > median).flatten()
    h = 0
    for b in bits: h = (h << 1) | int(b)
    return h

def _compute_dhash(gray, hash_size=8):
    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    bits = diff.flatten()
    h = 0
    for b in bits: h = (h << 1) | int(b)
    return h

def _hamming_distance(h1, h2):
    return bin(h1 ^ h2).count('1')


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
        ph = _compute_phash(gray)
        dh = _compute_dhash(gray)
        fp = Fingerprint(
            path=path, filename=Path(path).name,
            file_size=os.path.getsize(path),
            hu_moments=hu_log, hu_moments_eq=hu_eq_log,
            color_hist=color_hist,
            gray_mean=float(gray.mean()), gray_std=float(gray.std()),
            texture_density=texture_density,
            color_diversity=color_diversity,
            keypoint_count=len(kps) if kps else 0,
            phash=ph,
            dhash=dh,
        )
        del img, gray, gray_eq, hsv
        return fp
    except Exception as e:
        print(f"  [WARN] Skip {Path(path).name}: {e}")
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
            for qx0,qy0,qx1,qy1 in [(0,0,wa//2,ha//2),(wa//2,0,wa,ha//2),
                (0,ha//2,wa//2,ha),(wa//2,ha//2,wa,ha),(wa//4,ha//4,3*wa//4,3*ha//4)]:
                qa = ga[qy0:qy1, qx0:qx1]
                if qa.shape[0] < 32 or qa.shape[1] < 32: continue
                Hq, iq, rq, nq = find_homography_single(qa, gray_b, n_features//2)
                if iq > best["inliers"] and Hq is not None:
                    off = np.eye(3, dtype=np.float64)
                    off[0,2], off[1,2] = qx0, qy0
                    best = {"H": off @ Hq, "inliers": iq, "inlier_ratio": round(rq,4),
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
    r0, r1 = np.where(rows)[0][[0,-1]]
    c0, c1 = np.where(cols)[0][[0,-1]]
    pa, pb = ga_n[r0:r1+1,c0:c1+1], gb_n[r0:r1+1,c0:c1+1]
    ms = min(pa.shape); win = min(7, ms if ms%2==1 else ms-1)
    if win < 3: win = 3
    ssim_val = compare_ssim(pa, pb, win_size=win,
        data_range=pa.max()-pa.min()+1e-8) if pa.shape[0]>=win and pa.shape[1]>=win else 0.0
    diff = np.abs(ga_n - gb_n)
    mae = diff[overlap_mask].mean() if overlap_px > 0 else 999
    return {"warp_ssim": round(max(0.0, ssim_val), 4),
            "warp_pixel_sim": round(1.0/(1.0+mae*0.5), 4),
            "overlap_ratio": round(overlap_ratio, 4)}


# ═══════════════════════════════════════════════════════════════════════════
# LOW-TEXTURE FALLBACK
# ═══════════════════════════════════════════════════════════════════════════

def _low_texture_fallback(cv_a, cv_b, size=(128,128)):
    def prep(img):
        c = auto_crop_black(img, 15)
        return gpu_clahe(cv2.resize(gpu_gray(c), size))
    a, b_base = prep(cv_a), prep(cv_b)
    best = 0.0
    variants = [b_base, gpu_flip(b_base,1), gpu_flip(b_base,0),
                gpu_flip(gpu_flip(b_base,1),0)]
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
        weights = {"warp_ssim":.40,"warp_pixel_sim":.25,"orb_inlier_ratio":.10,
                   "hu_sim":.10,"hist_sim":.10,"overlap_ratio":.05}
        min_inlier_req = 4
    elif kp_factor < 0.6:
        weights = {"warp_ssim":.35,"warp_pixel_sim":.20,"orb_inlier_ratio":.20,
                   "hu_sim":.10,"hist_sim":.10,"overlap_ratio":.05}
        min_inlier_req = 6
    else:
        weights = {"warp_ssim":.30,"warp_pixel_sim":.15,"orb_inlier_ratio":.25,
                   "hu_sim":.10,"hist_sim":.10,"overlap_ratio":.10}
        min_inlier_req = 8
    confidence = sum(scores.get(k,0)*w for k,w in weights.items())
    ni = detail.get("orb_inliers", 0)
    ir = detail.get("orb_inlier_ratio", 0.0)
    n_matches = detail.get("orb_matches", ni)
    has_inliers = ni >= min_inlier_req
    has_warp = scores["warp_ssim"] >= 0.35

    # Low-match guard: with few total matches, high inlier ratio is unreliable
    # UNLESS ratio is near-perfect (≥0.90) — that's strong evidence even with few matches
    if n_matches < 30 and ni < 30 and ir < 0.90:
        has_warp = scores["warp_ssim"] >= 0.50

    # High geometric override: overwhelming evidence (100+ inliers)
    if ni >= 100 and ir >= 0.50:
        gc_ = 0.3*min(1.0,ni/200.0)+0.4*ir+0.3*scores.get("overlap_ratio",0.5)
        confidence = max(confidence, gc_); has_warp = True
    # Medium geometric override: good evidence but needs pixel confirmation
    elif ni >= 50 and ir >= 0.50 and scores["warp_ssim"] >= 0.20:
        gc_ = 0.25*min(1.0,ni/100.0)+0.35*ir+0.2*scores["warp_ssim"]+0.2*scores.get("overlap_ratio",0.5)
        confidence = max(confidence, gc_); has_warp = True
    fb = detail.get("fallback_ssim", 0.0)
    max_kp = max(fp_a.keypoint_count, fp_b.keypoint_count)
    if fb >= 0.75 and max_kp < 60:
        fbc = 0.5*fb+0.3*scores["hu_sim"]+0.2*scores["hist_sim"]
        confidence = max(confidence, fbc); has_inliers=True; has_warp=True
    return {"is_duplicate": confidence >= threshold and has_inliers and has_warp,
            "confidence": round(confidence,4),
            "scores": {k: round(v,4) for k,v in scores.items()},
            "orb_inliers": ni, "flipped": detail.get("flipped",False),
            "match_level": detail.get("match_level","")}


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
# GROUND-TRUTH LABEL EXTRACTION (for evaluation metrics)
# ═══════════════════════════════════════════════════════════════════════════

def extract_gt_label(filename):
    """Extract ground-truth group label from filename using naming conventions.
    Returns the base label, or the filename itself if no pattern matches.
    Files starting with 'NEG_' are labelled as unique singletons."""
    if filename.startswith("NEG_"):
        return "NEG__" + filename  # unique label per NEG file

    # Pattern: "teeth-XX (Y).jpg" or "teethXX (Y).jpg"
    m = re.match(r'(teeth-?\d+)', filename, re.IGNORECASE)
    if m:
        return m.group(1).lower()

    # Pattern: "BaseClass_aug_N.jpg" (COCO-style)
    m = re.match(r'(.+?)_aug_\d+\.\w+$', filename)
    if m:
        return m.group(1)

    # Pattern: "base_<augmentation>.ext" (Geometrical-style)
    # Find the EARLIEST augmentation keyword to get the true base
    known_augs = ["ORIGINAL", "blur", "bright", "combo", "contrast", "crop",
                  "edge", "flip", "jpeg", "rot", "saturate", "scale",
                  "sharpen", "shift", "stretch", "zoom"]
    earliest_idx = len(filename)
    for aug in known_augs:
        pat = f"_{aug}"
        idx = filename.find(pat)
        if 0 < idx < earliest_idx:
            earliest_idx = idx
    if earliest_idx < len(filename):
        return filename[:earliest_idx]

    # Pattern: "BaseClass (N).ext" (numbered variants)
    m = re.match(r'(.+?)\s*\(\d+\)\.\w+$', filename)
    if m:
        return m.group(1).strip()

    # No pattern matched — treat as unique
    return filename


def _resolve_gt_classes(gt_classes, n_images):
    """Validate GT classes by checking for augmentation evidence.
    1) If ALL images map to one label → treat each as unique (Face/Melanoma).
    2) If a class has multiple members but NONE contain augmentation keywords
       and NONE follow the _aug_N pattern → treat each member as unique."""
    # Case 1: single label for entire dataset
    if len(gt_classes) == 1:
        label, members = next(iter(gt_classes.items()))
        if len(members) == n_images and n_images > 3:
            new_classes = {}
            new_map = {}
            for fn in members:
                unique = f"__unique__{fn}"
                new_classes[unique] = {fn}
                new_map[fn] = unique
            return new_classes, new_map

    # Case 2: classes with no augmentation evidence
    # EXCEPTION: labels matched by explicit group-ID patterns (e.g., teeth-XX)
    # encode group membership directly in the filename — they don't need
    # augmentation keywords to prove they are duplicate groups.
    aug_keywords = ["_aug_", "_ORIGINAL", "_blur", "_bright", "_combo",
                    "_contrast", "_crop", "_edge", "_flip", "_jpeg",
                    "_rot", "_saturate", "_scale", "_sharpen", "_shift",
                    "_stretch", "_zoom"]
    explicit_group_pattern = re.compile(r'^teeth-?\d+$', re.IGNORECASE)
    remap = {}
    for label, members in list(gt_classes.items()):
        if label.startswith("NEG__"):
            continue
        if len(members) < 2:
            continue
        # Skip augmentation check for explicitly named group labels
        if explicit_group_pattern.match(label):
            continue
        has_aug = any(kw in fn for fn in members for kw in aug_keywords)
        if not has_aug:
            # No augmentation evidence — treat each member as unique
            for fn in members:
                unique = f"__unique__{fn}"
                remap[fn] = unique

    if remap:
        new_classes = {}
        new_map = {}
        for label, members in gt_classes.items():
            remaining = set()
            for fn in members:
                if fn in remap:
                    u = remap[fn]
                    new_classes[u] = {fn}
                    new_map[fn] = u
                else:
                    remaining.add(fn)
                    new_map[fn] = label
            if remaining:
                new_classes[label] = remaining
        return new_classes, new_map

    return None, None


def compute_evaluation_metrics(fps, cands, dups, rejected, groups, total_pairs):
    """Compute all evaluation metrics as defined in Section 4.2.
    Returns a dict of metric names to values and a list of formatted report lines."""

    # ── Build ground-truth label map ──
    gt_map = {}  # filename -> gt_label
    gt_classes = {}  # gt_label -> set of filenames
    for fp in fps:
        label = extract_gt_label(fp.filename)
        gt_map[fp.filename] = label
        gt_classes.setdefault(label, set()).add(fp.filename)

    # ── Handle single-class datasets (Face, Melanoma: all unique images) ──
    resolved_classes, resolved_map = _resolve_gt_classes(gt_classes, len(fps))
    if resolved_classes is not None:
        gt_classes = resolved_classes
        gt_map = resolved_map

    # ── Build filename index for screening recall ──
    fn_to_idx = {fp.filename: i for i, fp in enumerate(fps)}

    # ── Compute ground-truth pair counts ──
    gt_pair_count = 0  # total true duplicate pairs across all GT classes
    for label, members in gt_classes.items():
        n = len(members)
        if n >= 2:
            gt_pair_count += n * (n - 1) // 2

    # ── Identify which GT pairs passed screening ──
    screened_gt_pairs = 0
    for i, j, sc in cands:
        la = gt_map.get(fps[i].filename, "")
        lb = gt_map.get(fps[j].filename, "")
        if la == lb and la != "" and len(gt_classes.get(la, set())) >= 2:
            screened_gt_pairs += 1

    # ── Pair-level: TP, FP, FN ──
    tp = 0; fp = 0
    tp_confidences = []
    fp_pairs_detail = []
    for d in dups:
        la = gt_map.get(d["file_a"], "?a")
        lb = gt_map.get(d["file_b"], "?b")
        if la == lb:
            tp += 1
            tp_confidences.append(d["confidence"])
        else:
            fp += 1
            fp_pairs_detail.append((d["file_a"], d["file_b"],
                                    d["confidence"], la, lb))

    fn = gt_pair_count - tp  # GT pairs not confirmed
    tn = total_pairs - tp - fp - fn  # remaining pairs correctly rejected

    # ── Rejected candidate confidences ──
    rej_confidences = [r["confidence"] for r in rejected]

    # ── Pair precision, recall, F1 ──
    pair_precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 100.0
    pair_recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 100.0
    pair_f1 = (2 * pair_precision * pair_recall / (pair_precision + pair_recall)
               if (pair_precision + pair_recall) > 0 else 0.0)

    # ── Group purity: for each detected group, max(|G_i ∩ C_k|) / |G_i| ──
    purities = []
    pure_count = 0; contaminated_count = 0
    contaminated_details = []
    for root, members in groups.items():
        label_counts = {}
        for m in members:
            label = gt_map.get(m, m)
            label_counts[label] = label_counts.get(label, 0) + 1
        max_count = max(label_counts.values())
        purity = max_count / len(members)
        purities.append(purity)
        if len(label_counts) == 1:
            pure_count += 1
        else:
            contaminated_count += 1
            contaminated_details.append((members, label_counts))

    avg_purity = sum(purities) / len(purities) if purities else 1.0

    # ── Group completeness: for each GT class, max(|G_i ∩ C_k|) / |C_k| ──
    completeness_vals = []
    for label, gt_members in gt_classes.items():
        if label.startswith("NEG__"):
            continue  # skip NEG singletons
        if len(gt_members) < 2:
            continue  # skip singletons
        # Find how many of this class's members appear in each detected group
        best_in_one_group = 0
        for root, g_members in groups.items():
            overlap = len(gt_members & g_members)
            if overlap > best_in_one_group:
                best_in_one_group = overlap
        completeness = best_in_one_group / len(gt_members)
        completeness_vals.append(completeness)

    avg_completeness = (sum(completeness_vals) / len(completeness_vals)
                        if completeness_vals else 1.0)

    # ── NEG leakage ──
    neg_leaked = 0
    neg_leaked_files = []
    for root, members in groups.items():
        for m in members:
            if m.startswith("NEG_"):
                neg_leaked += 1
                neg_leaked_files.append(m)
    neg_total = sum(1 for f in gt_map if f.startswith("NEG_"))

    # ── Screening metrics ──
    screening_pass_rate = len(cands) / total_pairs * 100 if total_pairs > 0 else 0
    screening_recall = (screened_gt_pairs / gt_pair_count * 100
                        if gt_pair_count > 0 else 100.0)

    # ── Confidence distribution ──
    mu_tp = sum(tp_confidences) / len(tp_confidences) if tp_confidences else 0.0
    mu_rej = sum(rej_confidences) / len(rej_confidences) if rej_confidences else 0.0
    min_tp = min(tp_confidences) if tp_confidences else 0.0
    max_rej = max(rej_confidences) if rej_confidences else 0.0
    decision_margin = (min_tp - max_rej) if tp_confidences and rej_confidences else float('inf')

    # ── Coverage ──
    grouped_images = set()
    for root, members in groups.items():
        grouped_images |= members
    ungrouped = len(fps) - len(grouped_images)

    # ── Format report lines ──
    rpt = []
    rpt.append(f"{'═'*70}")
    rpt.append(f"  EVALUATION METRICS (Section 4.2)")
    rpt.append(f"{'═'*70}")

    rpt.append(f"\n  ── 4.2.1 Pair-Level Metrics ──")
    rpt.append(f"    True Positives  (TP)  : {tp:,}")
    rpt.append(f"    False Positives (FP)  : {fp:,}")
    rpt.append(f"    False Negatives (FN)  : {fn:,}")
    rpt.append(f"    True Negatives  (TN)  : {tn:,}")
    rpt.append(f"    Pair Precision        : {pair_precision:.2f}%")
    rpt.append(f"    Pair Recall           : {pair_recall:.2f}%")
    rpt.append(f"    Pair F1-Score         : {pair_f1:.2f}%")

    if fp_pairs_detail:
        rpt.append(f"\n    False-Positive Pairs ({fp}):")
        for fa, fb, conf, la, lb in fp_pairs_detail:
            rpt.append(f"      {fa}  [{la}]")
            rpt.append(f"      {fb}  [{lb}]  conf={conf:.4f}")
            rpt.append(f"")

    rpt.append(f"\n  ── 4.2.2 Group-Level Metrics ──")
    rpt.append(f"    Groups detected       : {len(groups)}")
    rpt.append(f"    GT classes (≥2 imgs)  : {sum(1 for v in gt_classes.values() if len(v)>=2)}")
    rpt.append(f"    Pure groups           : {pure_count}/{len(groups)}")
    rpt.append(f"    Contaminated groups   : {contaminated_count}")
    rpt.append(f"    Avg group purity      : {avg_purity:.4f}")
    rpt.append(f"    Avg group completeness: {avg_completeness:.4f}")
    rpt.append(f"    Grouped images        : {len(grouped_images)}/{len(fps)}")
    rpt.append(f"    Ungrouped images      : {ungrouped}")

    if contaminated_details:
        rpt.append(f"\n    Contaminated Group Details:")
        for members, label_counts in contaminated_details:
            labels_str = ", ".join(f"{l}: {c}" for l, c in
                                   sorted(label_counts.items(), key=lambda x: -x[1]))
            rpt.append(f"      [{len(members)} members] {labels_str}")

    rpt.append(f"\n  ── 4.2.3 NEG Leakage ──")
    rpt.append(f"    NEG images in dataset : {neg_total}")
    rpt.append(f"    NEG leaked into groups: {neg_leaked}/{neg_total}")
    if neg_leaked_files:
        for nf in neg_leaked_files:
            rpt.append(f"      Leaked: {nf}")

    rpt.append(f"\n  ── 4.2.4 Screening Efficiency ──")
    rpt.append(f"    Total pairs           : {total_pairs:,}")
    rpt.append(f"    Candidates (passed)   : {len(cands):,}")
    rpt.append(f"    Screening pass rate   : {screening_pass_rate:.1f}%")
    rpt.append(f"    GT duplicate pairs    : {gt_pair_count:,}")
    rpt.append(f"    GT pairs in candidates: {screened_gt_pairs:,}")
    rpt.append(f"    Screening recall      : {screening_recall:.2f}%")
    ver_rate = len(dups) / len(cands) * 100 if cands else 0
    rpt.append(f"    Verification confirm  : {ver_rate:.1f}%  ({len(dups)}/{len(cands)})")

    rpt.append(f"\n  ── 4.2.5 Confidence Distribution ──")
    rpt.append(f"    Confirmed pairs (TP+FP): {len(dups):,}")
    rpt.append(f"    Rejected candidates    : {len(rejected):,}")
    rpt.append(f"    μ_TP  (mean TP conf)   : {mu_tp:.4f}")
    rpt.append(f"    min TP confidence       : {min_tp:.4f}")
    rpt.append(f"    μ_REJ (mean REJ conf)  : {mu_rej:.4f}")
    rpt.append(f"    max REJ confidence      : {max_rej:.4f}")
    if decision_margin != float('inf'):
        rpt.append(f"    Decision margin (Δ)    : {decision_margin:+.4f}  "
                    f"(min_TP={min_tp:.4f} − max_REJ={max_rej:.4f})")
    else:
        rpt.append(f"    Decision margin (Δ)    : N/A (no TP or no REJ)")

    # ── Group completeness per GT class ──
    rpt.append(f"\n  ── Group Completeness per GT Class ──")
    for label in sorted(gt_classes.keys()):
        members = gt_classes[label]
        if label.startswith("NEG__") or len(members) < 2:
            continue
        best = 0
        for root, g_members in groups.items():
            overlap = len(members & g_members)
            if overlap > best: best = overlap
        comp = best / len(members)
        if comp < 1.0:
            rpt.append(f"    {label}: {best}/{len(members)} = {comp:.4f}")

    rpt.append(f"{'═'*70}\n")

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": pair_precision, "recall": pair_recall, "f1": pair_f1,
        "pure": pure_count, "contaminated": contaminated_count,
        "purity": avg_purity, "completeness": avg_completeness,
        "neg_leaked": neg_leaked,
        "spr": screening_pass_rate, "sr": screening_recall,
        "mu_tp": mu_tp, "mu_rej": mu_rej, "margin": decision_margin,
    }, rpt



def compute_baseline_f1(fps, cands, dups, rejected, gt_map, gt_classes, total_pairs):
    """Compute F1 scores for all baseline methods on verified candidates.
    Returns dict: {method_name: {"tp":, "fp":, "fn":, "precision":, "recall":, "f1":}}
    """
    fp_lookup = {fp.filename: fp for fp in fps}

    # GT pair count
    gt_pair_count = sum(len(v)*(len(v)-1)//2
                        for v in gt_classes.values() if len(v) >= 2)

    # Count GT pairs that passed screening
    screened_gt = 0
    for i, j, sc in cands:
        la = gt_map.get(fps[i].filename, "")
        lb = gt_map.get(fps[j].filename, "")
        if la == lb and la != "" and len(gt_classes.get(la, set())) >= 2:
            screened_gt += 1
    fn_from_screening = gt_pair_count - screened_gt  # GT pairs lost at screening

    # Build unified verified list with all scores
    all_verified = []
    for d in dups:
        la = gt_map.get(d["file_a"], "?")
        lb = gt_map.get(d["file_b"], "?")
        gt_same = 1 if (la == lb and la != "?" and
                        len(gt_classes.get(la, set())) >= 2) else 0
        fp_a = fp_lookup.get(d["file_a"])
        fp_b = fp_lookup.get(d["file_b"])
        ph_dist = _hamming_distance(fp_a.phash, fp_b.phash) if (fp_a and fp_b and hasattr(fp_a, 'phash')) else 64
        dh_dist = _hamming_distance(fp_a.dhash, fp_b.dhash) if (fp_a and fp_b and hasattr(fp_a, 'dhash')) else 64
        all_verified.append({
            "gt": gt_same,
            "hist_sim": d["scores"].get("hist_sim", 0),
            "orb_ratio": d["scores"].get("orb_inlier_ratio", 0),
            "warp_ssim": d["scores"].get("warp_ssim", 0),
            "pixel_sim": d["scores"].get("warp_pixel_sim", 0),
            "phash_dist": ph_dist,
            "dhash_dist": dh_dist,
        })
    for r in rejected:
        la = gt_map.get(r["file_a"], "?")
        lb = gt_map.get(r["file_b"], "?")
        gt_same = 1 if (la == lb and la != "?" and
                        len(gt_classes.get(la, set())) >= 2) else 0
        scores = r.get("scores", {})
        fp_a = fp_lookup.get(r["file_a"])
        fp_b = fp_lookup.get(r["file_b"])
        ph_dist = _hamming_distance(fp_a.phash, fp_b.phash) if (fp_a and fp_b and hasattr(fp_a, 'phash')) else 64
        dh_dist = _hamming_distance(fp_a.dhash, fp_b.dhash) if (fp_a and fp_b and hasattr(fp_a, 'dhash')) else 64
        all_verified.append({
            "gt": gt_same,
            "hist_sim": scores.get("hist_sim", 0),
            "orb_ratio": scores.get("orb_inlier_ratio", 0),
            "warp_ssim": scores.get("warp_ssim", 0),
            "pixel_sim": scores.get("warp_pixel_sim", 0),
            "phash_dist": ph_dist,
            "dhash_dist": dh_dist,
        })

    # Define baselines: (name, score_key, threshold, direction)
    # direction '>' means score > thresh => predict duplicate
    # direction '<' means score < thresh => predict duplicate (for distances)
    baselines = [
        ("pHash (Ham<=10)",       "phash_dist", 10,   "<"),
        ("dHash (Ham<=10)",       "dhash_dist", 10,   "<"),
        ("Hist-only (>0.70)",     "hist_sim",   0.70, ">"),
        ("ORB-ratio (>0.50)",     "orb_ratio",  0.50, ">"),
        ("SSIM-only (>0.50)",     "warp_ssim",  0.50, ">"),
        ("Pixel-sim (>0.85)",     "pixel_sim",  0.85, ">"),
    ]

    results = {}
    for bname, bkey, bthresh, bdir in baselines:
        b_tp = 0; b_fp = 0; b_fn_verified = 0
        for v in all_verified:
            val = v.get(bkey, 0)
            pred = 1 if (val > bthresh if bdir == ">" else val < bthresh) else 0
            if pred == 1 and v["gt"] == 1:
                b_tp += 1
            elif pred == 1 and v["gt"] == 0:
                b_fp += 1
            elif pred == 0 and v["gt"] == 1:
                b_fn_verified += 1

        b_fn = b_fn_verified + fn_from_screening
        b_prec = b_tp / (b_tp + b_fp) * 100 if (b_tp + b_fp) > 0 else 100.0
        b_rec = b_tp / (b_tp + b_fn) * 100 if (b_tp + b_fn) > 0 else 100.0
        b_f1 = 2 * b_prec * b_rec / (b_prec + b_rec) if (b_prec + b_rec) > 0 else 0.0

        results[bname] = {
            "tp": b_tp, "fp": b_fp, "fn": b_fn,
            "precision": round(b_prec, 2),
            "recall": round(b_rec, 2),
            "f1": round(b_f1, 2)
        }

    return results



def compute_baseline_f1(fps, cands, dups, rejected, gt_map, gt_classes, total_pairs):
    """Compute F1 scores for all baseline methods on verified candidates.
    Returns dict: {method_name: {"tp":, "fp":, "fn":, "precision":, "recall":, "f1":}}
    """
    fp_lookup = {fp.filename: fp for fp in fps}

    # GT pair count
    gt_pair_count = sum(len(v)*(len(v)-1)//2
                        for v in gt_classes.values() if len(v) >= 2)

    # Count GT pairs that passed screening
    screened_gt = 0
    for i, j, sc in cands:
        la = gt_map.get(fps[i].filename, "")
        lb = gt_map.get(fps[j].filename, "")
        if la == lb and la != "" and len(gt_classes.get(la, set())) >= 2:
            screened_gt += 1
    fn_from_screening = gt_pair_count - screened_gt  # GT pairs lost at screening

    # Build unified verified list with all scores
    all_verified = []
    for d in dups:
        la = gt_map.get(d["file_a"], "?")
        lb = gt_map.get(d["file_b"], "?")
        gt_same = 1 if (la == lb and la != "?" and
                        len(gt_classes.get(la, set())) >= 2) else 0
        fp_a = fp_lookup.get(d["file_a"])
        fp_b = fp_lookup.get(d["file_b"])
        ph_dist = _hamming_distance(fp_a.phash, fp_b.phash) if (fp_a and fp_b and hasattr(fp_a, 'phash')) else 64
        dh_dist = _hamming_distance(fp_a.dhash, fp_b.dhash) if (fp_a and fp_b and hasattr(fp_a, 'dhash')) else 64
        all_verified.append({
            "gt": gt_same,
            "hist_sim": d["scores"].get("hist_sim", 0),
            "orb_ratio": d["scores"].get("orb_inlier_ratio", 0),
            "warp_ssim": d["scores"].get("warp_ssim", 0),
            "pixel_sim": d["scores"].get("warp_pixel_sim", 0),
            "phash_dist": ph_dist,
            "dhash_dist": dh_dist,
        })
    for r in rejected:
        la = gt_map.get(r["file_a"], "?")
        lb = gt_map.get(r["file_b"], "?")
        gt_same = 1 if (la == lb and la != "?" and
                        len(gt_classes.get(la, set())) >= 2) else 0
        scores = r.get("scores", {})
        fp_a = fp_lookup.get(r["file_a"])
        fp_b = fp_lookup.get(r["file_b"])
        ph_dist = _hamming_distance(fp_a.phash, fp_b.phash) if (fp_a and fp_b and hasattr(fp_a, 'phash')) else 64
        dh_dist = _hamming_distance(fp_a.dhash, fp_b.dhash) if (fp_a and fp_b and hasattr(fp_a, 'dhash')) else 64
        all_verified.append({
            "gt": gt_same,
            "hist_sim": scores.get("hist_sim", 0),
            "orb_ratio": scores.get("orb_inlier_ratio", 0),
            "warp_ssim": scores.get("warp_ssim", 0),
            "pixel_sim": scores.get("warp_pixel_sim", 0),
            "phash_dist": ph_dist,
            "dhash_dist": dh_dist,
        })

    # Define baselines: (name, score_key, threshold, direction)
    # direction '>' means score > thresh => predict duplicate
    # direction '<' means score < thresh => predict duplicate (for distances)
    baselines = [
        ("pHash (Ham<=10)",       "phash_dist", 10,   "<"),
        ("dHash (Ham<=10)",       "dhash_dist", 10,   "<"),
        ("Hist-only (>0.70)",     "hist_sim",   0.70, ">"),
        ("ORB-ratio (>0.50)",     "orb_ratio",  0.50, ">"),
        ("SSIM-only (>0.50)",     "warp_ssim",  0.50, ">"),
        ("Pixel-sim (>0.85)",     "pixel_sim",  0.85, ">"),
    ]

    results = {}
    for bname, bkey, bthresh, bdir in baselines:
        b_tp = 0; b_fp = 0; b_fn_verified = 0
        for v in all_verified:
            val = v.get(bkey, 0)
            pred = 1 if (val > bthresh if bdir == ">" else val < bthresh) else 0
            if pred == 1 and v["gt"] == 1:
                b_tp += 1
            elif pred == 1 and v["gt"] == 0:
                b_fp += 1
            elif pred == 0 and v["gt"] == 1:
                b_fn_verified += 1

        b_fn = b_fn_verified + fn_from_screening
        b_prec = b_tp / (b_tp + b_fp) * 100 if (b_tp + b_fp) > 0 else 100.0
        b_rec = b_tp / (b_tp + b_fn) * 100 if (b_tp + b_fn) > 0 else 100.0
        b_f1 = 2 * b_prec * b_rec / (b_prec + b_rec) if (b_prec + b_rec) > 0 else 0.0

        results[bname] = {
            "tp": b_tp, "fp": b_fp, "fn": b_fn,
            "precision": round(b_prec, 2),
            "recall": round(b_rec, 2),
            "f1": round(b_f1, 2)
        }

    return results


# ═══════════════════════════════════════════════════════════════════════════
# STATISTICAL SIGNIFICANCE TESTS (Section 4.3)
# ═══════════════════════════════════════════════════════════════════════════

def compute_statistical_tests(fps, cands, dups, rejected, groups, total_pairs,
                              n_bootstrap=10000):
    """Compute all five statistical significance tests (Section 4.3).
    Returns formatted report lines appended to the main report.

    Tests computed:
      4.3.1 McNemar's Test — vs four single-metric baselines
      4.3.2 Wilcoxon Signed-Rank Test — confidence distributions
      4.3.3 Friedman Test placeholder — requires multi-dataset input
      4.3.4 Paired Bootstrap Confidence Intervals — P, R, F1
      4.3.5 Cohen's Kappa — chance-adjusted agreement
    """
    import math

    # ── Build GT label map ──
    gt_map = {}
    gt_classes = {}
    for fp in fps:
        label = extract_gt_label(fp.filename)
        gt_map[fp.filename] = label
        gt_classes.setdefault(label, set()).add(fp.filename)

    resolved_classes, resolved_map = _resolve_gt_classes(gt_classes, len(fps))
    if resolved_classes is not None:
        gt_classes = resolved_classes
        gt_map = resolved_map

    gt_pair_count = sum(len(v)*(len(v)-1)//2
                        for v in gt_classes.values() if len(v) >= 2)

    # ── Build unified prediction array over all verified candidates ──
    # ── Build filename → fingerprint lookup for hash baselines ──
    fp_lookup = {fp.filename: fp for fp in fps}
    # Each candidate gets: our_pred (0/1), our_conf, gt_label (same/diff),
    # and individual score components for baseline predictions.
    all_verified = []
    for d in dups:
        la = gt_map.get(d["file_a"], "?")
        lb = gt_map.get(d["file_b"], "?")
        gt_same = 1 if (la == lb and la != "?" and
                        len(gt_classes.get(la, set())) >= 2) else 0
        fp_a = fp_lookup.get(d["file_a"])
        fp_b = fp_lookup.get(d["file_b"])
        ph_dist = _hamming_distance(fp_a.phash, fp_b.phash) if (fp_a and fp_b) else 64
        dh_dist = _hamming_distance(fp_a.dhash, fp_b.dhash) if (fp_a and fp_b) else 64
        all_verified.append({
            "our_pred": 1, "our_conf": d["confidence"],
            "gt": gt_same,
            "orb_ratio": d["scores"].get("orb_inlier_ratio", 0),
            "hist_sim": d["scores"].get("hist_sim", 0),
            "warp_ssim": d["scores"].get("warp_ssim", 0),
            "pixel_sim": d["scores"].get("warp_pixel_sim", 0),
            "inliers": d.get("orb_inliers", 0),
            "phash_dist": ph_dist,
            "dhash_dist": dh_dist,
        })
    for r in rejected:
        la = gt_map.get(r["file_a"], "?")
        lb = gt_map.get(r["file_b"], "?")
        gt_same = 1 if (la == lb and la != "?" and
                        len(gt_classes.get(la, set())) >= 2) else 0
        scores = r.get("scores", {})
        fp_a = fp_lookup.get(r["file_a"])
        fp_b = fp_lookup.get(r["file_b"])
        ph_dist = _hamming_distance(fp_a.phash, fp_b.phash) if (fp_a and fp_b) else 64
        dh_dist = _hamming_distance(fp_a.dhash, fp_b.dhash) if (fp_a and fp_b) else 64
        all_verified.append({
            "our_pred": 0, "our_conf": r["confidence"],
            "gt": gt_same,
            "orb_ratio": scores.get("orb_inlier_ratio", 0),
            "hist_sim": scores.get("hist_sim", 0),
            "warp_ssim": scores.get("warp_ssim", 0),
            "pixel_sim": scores.get("warp_pixel_sim", 0),
            "inliers": r.get("orb_inliers", 0),
            "phash_dist": ph_dist,
            "dhash_dist": dh_dist,
        })

    if not all_verified:
        return []

    # ── Pair-level counts for our method ──
    tp = sum(1 for v in all_verified if v["our_pred"] == 1 and v["gt"] == 1)
    fp_count = sum(1 for v in all_verified if v["our_pred"] == 1 and v["gt"] == 0)
    fn = gt_pair_count - tp
    tn = total_pairs - tp - fp_count - fn

    # ── Define four single-metric baselines ──
    # Each baseline: (name, score_key, threshold, direction)
    # direction='>' means score > threshold => predict duplicate
    baselines = [
        ("pHash (Hamming≤10)",           "phash_dist", 10, "<"),
        ("dHash (Hamming≤10)",           "dhash_dist", 10, "<"),
        ("Histogram-only (hist>0.70)",   "hist_sim",   0.70, ">"),
        ("ORB-ratio-only (ratio>0.50)",  "orb_ratio",  0.50, ">"),
        ("SSIM-only (ssim>0.50)",        "warp_ssim",  0.50, ">"),
        ("Pixel-sim-only (pix>0.85)",    "pixel_sim",  0.85, ">"),
    ]

    def baseline_pred(entry, score_key, thresh, direction):
        val = entry.get(score_key, 0)
        if direction == ">":
            return 1 if val > thresh else 0
        else:  # "<" means distance < threshold => duplicate
            return 1 if val < thresh else 0

    rpt = []
    rpt.append(f"\n{'═'*70}")
    rpt.append(f"  STATISTICAL SIGNIFICANCE TESTS (Section 4.3)")
    rpt.append(f"{'═'*70}")

    # ═══════════════════════════════════════════════════════════════
    # 4.3.1  McNemar's Test
    # ═══════════════════════════════════════════════════════════════
    rpt.append(f"\n  ── 4.3.1 McNemar's Test (α=0.05) ──")
    rpt.append(f"    Comparison of our method vs single-metric baselines")
    rpt.append(f"    on {len(all_verified):,} verified candidate pairs.\n")
    rpt.append(f"    {'Baseline':<35s} {'n01':>6s} {'n10':>6s} {'χ²':>10s} {'p-value':>10s} {'Sig?':>5s}")
    rpt.append(f"    {'─'*35} {'─'*6} {'─'*6} {'─'*10} {'─'*10} {'─'*5}")

    for bname, bkey, bthresh, bdir in baselines:
        # Build contingency table
        n01 = 0  # ours correct, baseline wrong
        n10 = 0  # baseline correct, ours wrong
        for v in all_verified:
            our_correct = (v["our_pred"] == v["gt"])
            bp = baseline_pred(v, bkey, bthresh, bdir)
            bl_correct = (bp == v["gt"])
            if our_correct and not bl_correct:
                n01 += 1
            elif bl_correct and not our_correct:
                n10 += 1

        # McNemar's test with continuity correction
        total_discord = n01 + n10
        if total_discord == 0:
            chi2 = 0.0; p_val = 1.0
        elif total_discord < 25:
            # Exact binomial test
            from math import comb
            k = min(n01, n10)
            p_val = 0.0
            for x in range(k + 1):
                p_val += comb(total_discord, x) * (0.5 ** total_discord)
            p_val *= 2  # two-tailed
            p_val = min(p_val, 1.0)
            chi2 = float('nan')  # not applicable for exact test
        else:
            chi2 = (abs(n01 - n10) - 1) ** 2 / total_discord
            # p-value from chi-squared distribution (df=1)
            # Using survival function approximation
            p_val = _chi2_sf(chi2, 1)

        sig = "Yes" if p_val < 0.05 else "No"
        chi2_str = f"{chi2:.2f}" if not math.isnan(chi2) else "exact"
        rpt.append(f"    {bname:<35s} {n01:>6d} {n10:>6d} {chi2_str:>10s} "
                   f"{'<0.001' if p_val < 0.001 else f'{p_val:.4f}':>10s} {sig:>5s}")

    # ═══════════════════════════════════════════════════════════════
    # 4.3.2  Wilcoxon Signed-Rank Test
    # ═══════════════════════════════════════════════════════════════
    rpt.append(f"\n  ── 4.3.2 Wilcoxon Signed-Rank Test ──")
    rpt.append(f"    Comparing confidence distributions: our method vs SSIM-only.\n")

    # For confirmed TP pairs: compare our confidence vs SSIM
    tp_entries = [v for v in all_verified if v["our_pred"] == 1 and v["gt"] == 1]
    rej_entries = [v for v in all_verified if v["our_pred"] == 0]

    for subset_name, subset in [("True Positives", tp_entries),
                                ("Rejected pairs", rej_entries)]:
        if len(subset) < 5:
            rpt.append(f"    {subset_name}: insufficient data (n={len(subset)})")
            continue

        diffs = [(v["our_conf"] - v["warp_ssim"]) for v in subset]
        # Remove zeros
        nonzero = [(abs(d), 1 if d > 0 else -1) for d in diffs if d != 0.0]
        n = len(nonzero)
        if n < 5:
            rpt.append(f"    {subset_name}: insufficient non-zero differences (n={n})")
            continue

        # Rank by absolute value
        nonzero.sort(key=lambda x: x[0])
        ranks = list(range(1, n + 1))
        # Handle ties (average ranks)
        i = 0
        while i < n:
            j = i
            while j < n and nonzero[j][0] == nonzero[i][0]:
                j += 1
            avg_rank = sum(ranks[i:j]) / (j - i)
            for k in range(i, j):
                ranks[k] = avg_rank
            i = j

        W_plus = sum(ranks[i] for i in range(n) if nonzero[i][1] > 0)
        W_minus = sum(ranks[i] for i in range(n) if nonzero[i][1] < 0)
        W = min(W_plus, W_minus)

        # Normal approximation for large n
        mean_W = n * (n + 1) / 4
        std_W = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
        if std_W > 0:
            z = (W - mean_W) / std_W
            p_val = 2 * _normal_cdf(z)  # two-tailed
        else:
            z = 0; p_val = 1.0

        median_diff = sorted(diffs)[len(diffs) // 2]
        sig = "Yes" if p_val < 0.05 else "No"
        rpt.append(f"    {subset_name} (n={len(subset):,}):")
        rpt.append(f"      Median Δ (ours−warp_ssim) : {median_diff:+.4f}")
        rpt.append(f"      W+ = {W_plus:.0f},  W- = {W_minus:.0f}")
        rpt.append(f"      z = {z:.4f},  p-value = "
                   f"{'<0.001' if p_val < 0.001 else f'{p_val:.4f}'}"
                   f"  → {'Significant' if p_val < 0.05 else 'Not significant'}")

    # ═══════════════════════════════════════════════════════════════
    # 4.3.3  Friedman Test (multi-dataset placeholder)
    # ═══════════════════════════════════════════════════════════════
    rpt.append(f"\n  ── 4.3.3 Friedman Test with Nemenyi Post-Hoc ──")
    rpt.append(f"    NOTE: Friedman test requires results from multiple datasets.")
    rpt.append(f"    Use compute_friedman_test() with F1 scores from all datasets.")
    rpt.append(f"    See Section 4.6.3 in the paper for the aggregated analysis.")

    # ═══════════════════════════════════════════════════════════════
    # 4.3.4  Paired Bootstrap Confidence Intervals
    # ═══════════════════════════════════════════════════════════════
    rpt.append(f"\n  ── 4.3.4 Bootstrap 95% Confidence Intervals (B={n_bootstrap:,}) ──")

    import random
    random.seed(42)  # reproducibility

    # Build binary arrays: predictions and ground truth for all verified pairs
    # Plus account for pairs not in candidates (all are TN or FN)
    preds = [v["our_pred"] for v in all_verified]
    gts = [v["gt"] for v in all_verified]
    n_verified = len(preds)

    # For bootstrap: sample from verified pairs + unverified pairs
    # Unverified pairs = total_pairs - len(cands)
    # All unverified are predicted negative; they are TN if gt=0, FN if gt=1
    # We account for FN from unscreened GT pairs
    fn_from_screening = gt_pair_count - sum(1 for v in all_verified if v["gt"] == 1)
    tn_unverified = (total_pairs - len(all_verified)) - max(0, fn_from_screening)

    boot_prec = []
    boot_rec = []
    boot_f1 = []
    for _ in range(n_bootstrap):
        # Resample verified pairs with replacement
        indices = [random.randint(0, n_verified - 1) for _ in range(n_verified)]
        b_tp = sum(1 for idx in indices if preds[idx] == 1 and gts[idx] == 1)
        b_fp = sum(1 for idx in indices if preds[idx] == 1 and gts[idx] == 0)
        b_fn_verified = sum(1 for idx in indices if preds[idx] == 0 and gts[idx] == 1)
        b_fn = b_fn_verified + max(0, fn_from_screening)

        prec = b_tp / (b_tp + b_fp) * 100 if (b_tp + b_fp) > 0 else 100.0
        rec = b_tp / (b_tp + b_fn) * 100 if (b_tp + b_fn) > 0 else 100.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        boot_prec.append(prec)
        boot_rec.append(rec)
        boot_f1.append(f1)

    boot_prec.sort()
    boot_rec.sort()
    boot_f1.sort()

    lo = int(0.025 * n_bootstrap)
    hi = int(0.975 * n_bootstrap) - 1

    prec_actual = tp / (tp + fp_count) * 100 if (tp + fp_count) > 0 else 100.0
    rec_actual = tp / (tp + fn) * 100 if (tp + fn) > 0 else 100.0
    f1_actual = (2 * prec_actual * rec_actual / (prec_actual + rec_actual)
                 if (prec_actual + rec_actual) > 0 else 0.0)

    rpt.append(f"    {'Metric':<12s}  {'Value':>8s}   {'95% CI':>24s}")
    rpt.append(f"    {'─'*12}  {'─'*8}   {'─'*24}")
    rpt.append(f"    {'Precision':<12s}  {prec_actual:>7.2f}%   "
               f"[{boot_prec[lo]:.2f}%, {boot_prec[hi]:.2f}%]")
    rpt.append(f"    {'Recall':<12s}  {rec_actual:>7.2f}%   "
               f"[{boot_rec[lo]:.2f}%, {boot_rec[hi]:.2f}%]")
    rpt.append(f"    {'F1-Score':<12s}  {f1_actual:>7.2f}%   "
               f"[{boot_f1[lo]:.2f}%, {boot_f1[hi]:.2f}%]")

    # ═══════════════════════════════════════════════════════════════
    # 4.3.5  Cohen's Kappa
    # ═══════════════════════════════════════════════════════════════
    rpt.append(f"\n  ── 4.3.5 Cohen's Kappa (κ) ──")

    n_total = total_pairs
    p0 = (tp + tn) / n_total if n_total > 0 else 1.0

    # Expected agreement
    row_pos = tp + fp_count  # predicted positive
    row_neg = fn + tn         # predicted negative
    col_pos = tp + fn          # actual positive
    col_neg = fp_count + tn   # actual negative
    pe = (row_pos * col_pos + row_neg * col_neg) / (n_total ** 2) if n_total > 0 else 1.0

    kappa = (p0 - pe) / (1 - pe) if (1 - pe) > 0 else 1.0

    # Standard error and 95% CI
    se_kappa = (math.sqrt(p0 * (1 - p0) / (n_total * (1 - pe) ** 2))
                if n_total > 0 and (1 - pe) > 0 else 0.0)
    kappa_lo = kappa - 1.96 * se_kappa
    kappa_hi = kappa + 1.96 * se_kappa

    # Interpretation
    if kappa >= 0.81:
        interp = "Near-perfect"
    elif kappa >= 0.61:
        interp = "Substantial"
    elif kappa >= 0.41:
        interp = "Moderate"
    elif kappa >= 0.21:
        interp = "Fair"
    else:
        interp = "Slight"

    rpt.append(f"    Total pairs (n)       : {n_total:,}")
    rpt.append(f"    Observed agreement p₀ : {p0:.6f}")
    rpt.append(f"    Expected agreement pₑ : {pe:.6f}")
    rpt.append(f"    Cohen's Kappa (κ)     : {kappa:.4f}")
    rpt.append(f"    95% CI                : [{kappa_lo:.4f}, {kappa_hi:.4f}]")
    rpt.append(f"    SE(κ)                 : {se_kappa:.6f}")
    rpt.append(f"    Interpretation        : {interp} agreement")
    rpt.append(f"      (Landis & Koch, 1977: 0.81–1.00 = near-perfect)")

    rpt.append(f"{'═'*70}\n")
    return rpt


def compute_friedman_test(dataset_results):
    """Compute Friedman test + Nemenyi post-hoc across multiple datasets.

    Args:
        dataset_results: dict mapping dataset_name -> dict mapping method_name -> F1 score
            Example: {"COCO": {"Proposed": 98.08, "pHash": 45.2, ...}, ...}

    Returns:
        List of formatted report lines.
    """
    import math

    if not dataset_results:
        return ["  No multi-dataset results provided for Friedman test."]

    datasets = sorted(dataset_results.keys())
    methods = sorted(next(iter(dataset_results.values())).keys())
    k = len(methods)
    N = len(datasets)

    if N < 3 or k < 3:
        return [f"  Friedman test requires ≥3 datasets and ≥3 methods "
                f"(got {N} datasets, {k} methods)."]

    # ── Rank methods per dataset (rank 1 = best) ──
    ranks = {m: [] for m in methods}
    for ds in datasets:
        scores = [(m, dataset_results[ds].get(m, 0)) for m in methods]
        scores.sort(key=lambda x: -x[1])  # descending by F1
        # Assign ranks with tie handling
        i = 0
        while i < len(scores):
            j = i
            while j < len(scores) and scores[j][1] == scores[i][1]:
                j += 1
            avg_rank = sum(range(i + 1, j + 1)) / (j - i)
            for idx in range(i, j):
                ranks[scores[idx][0]].append(avg_rank)
            i = j

    mean_ranks = {m: sum(ranks[m]) / N for m in methods}

    # ── Friedman statistic ──
    sum_sq = sum(mean_ranks[m] ** 2 for m in methods)
    chi2_f = (12 * N / (k * (k + 1))) * (sum_sq - k * (k + 1) ** 2 / 4)
    df = k - 1
    p_val = _chi2_sf(chi2_f, df)

    rpt = []
    rpt.append(f"\n{'═'*70}")
    rpt.append(f"  FRIEDMAN TEST + NEMENYI POST-HOC (Section 4.3.3)")
    rpt.append(f"{'═'*70}")
    rpt.append(f"\n  Rankings by F1-Score across {N} datasets:\n")
    header = f"    {'Dataset':<20s}"
    for m in methods:
        header += f"  {m:<12s}"
    rpt.append(header)
    rpt.append(f"    {'─'*20}" + f"  {'─'*12}" * k)
    for ds in datasets:
        row = f"    {ds:<20s}"
        ds_scores = [(m, dataset_results[ds].get(m, 0)) for m in methods]
        ds_scores.sort(key=lambda x: -x[1])
        rank_map = {}
        i = 0
        while i < len(ds_scores):
            j = i
            while j < len(ds_scores) and ds_scores[j][1] == ds_scores[i][1]:
                j += 1
            avg_r = sum(range(i + 1, j + 1)) / (j - i)
            for idx in range(i, j):
                rank_map[ds_scores[idx][0]] = avg_r
            i = j
        for m in methods:
            row += f"  {rank_map.get(m, 0):<12.1f}"
        rpt.append(row)

    row = f"    {'Mean Rank':<20s}"
    for m in methods:
        row += f"  {mean_ranks[m]:<12.2f}"
    rpt.append(f"    {'─'*20}" + f"  {'─'*12}" * k)
    rpt.append(row)

    rpt.append(f"\n  Friedman χ²_F = {chi2_f:.4f}, df = {df}, "
               f"p = {'<0.001' if p_val < 0.001 else f'{p_val:.4f}'}")

    if p_val < 0.05:
        rpt.append(f"  → Null hypothesis REJECTED (methods differ significantly)")

        # ── Nemenyi post-hoc ──
        # q_α values for Nemenyi test (α=0.05)
        # Approximation from Studentized range / sqrt(2) for k methods
        q_alpha_table = {3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
                         7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}
        q_alpha = q_alpha_table.get(k, 2.728)
        cd = q_alpha * math.sqrt(k * (k + 1) / (6 * N))

        rpt.append(f"\n  Nemenyi Critical Difference (CD) = {cd:.4f} "
                   f"(q_α={q_alpha}, k={k}, N={N})")
        rpt.append(f"\n  Pairwise comparisons (|ΔR| > CD = significant):\n")
        rpt.append(f"    {'Method A':<18s}  {'Method B':<18s}  {'|ΔR|':>6s}  "
                   f"{'> CD?':>5s}  {'Sig?':>5s}")
        rpt.append(f"    {'─'*18}  {'─'*18}  {'─'*6}  {'─'*5}  {'─'*5}")

        for i_m, m1 in enumerate(methods):
            for m2 in methods[i_m + 1:]:
                delta = abs(mean_ranks[m1] - mean_ranks[m2])
                exceeds = delta > cd
                rpt.append(f"    {m1:<18s}  {m2:<18s}  {delta:>6.2f}  "
                           f"{'Yes' if exceeds else 'No':>5s}  "
                           f"{'Yes' if exceeds else 'No':>5s}")
    else:
        rpt.append(f"  → Null hypothesis NOT rejected (no significant difference)")

    rpt.append(f"{'═'*70}\n")
    return rpt


def _chi2_sf(x, df):
    """Survival function (1-CDF) of chi-squared distribution.
    Uses the regularised incomplete gamma function approximation."""
    import math
    if x <= 0:
        return 1.0
    if df == 1:
        # For df=1: P(X>x) = 2*(1 - Phi(sqrt(x)))
        return 2 * (1 - _normal_cdf_pos(math.sqrt(x)))
    if df == 2:
        return math.exp(-x / 2)
    # General case: series approximation for regularised gamma
    return _gamma_sf(x, df)


def _normal_cdf(z):
    """CDF of standard normal distribution."""
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def _normal_cdf_pos(z):
    """CDF of standard normal for positive z."""
    import math
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def _gamma_sf(x, df):
    """Survival function of chi-squared via incomplete gamma series."""
    import math
    a = df / 2.0
    z = x / 2.0
    # Regularised lower incomplete gamma using series expansion
    if z == 0:
        return 1.0
    term = math.exp(-z) * (z ** a) / math.gamma(a + 1)
    s = term
    for n in range(1, 300):
        term *= z / (a + n)
        s += term
        if abs(term) < 1e-12:
            break
    return max(0.0, min(1.0, 1.0 - s))

import math


# ═══════════════════════════════════════════════════════════════════════════
# PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def run(folder, threshold=0.50, max_dim=512, base_screen=0.30,
        fast=False, report=None, workers=0, use_gpu=False):

    gpu_ok = init_gpu() if use_gpu else False
    nw = workers if workers > 0 else max(1, os.cpu_count() or 4)

    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║   NEAR-DUPLICATE DETECTOR v5.2.2 — Threaded + GPU (OpenCL)        ║
╠══════════════════════════════════════════════════════════════════════╣
║  Stage 1: Hu Moments + Color Hist (adaptive per-pair threshold)    ║
║  Stage 2: Hierarchical ORB+RANSAC → CLAHE Warp → Pixel Compare    ║
║  GPU: AMD Radeon / Intel / NVIDIA via OpenCL (auto CPU fallback)   ║
╚══════════════════════════════════════════════════════════════════════╝
  Folder    : {folder}
  Threshold : {threshold}
  Screen    : {base_screen} (adaptive)
  Max dim   : {max_dim}px
  Workers   : {nw} threads
  GPU       : {_GPU_INFO}
  Fast mode : {fast}
""")

    paths = get_paths(folder)
    n = len(paths)
    print(f"[1/4] Found {n} images ({n*(n-1)//2:,} pairs)\n")
    if n < 2: print("Need >= 2."); return

    # ── Fingerprint (threaded) ──
    print(f"[2/4] Computing fingerprints ({nw} threads)...")
    t0 = time.time()
    fps = [None] * n
    done_count = [0]

    def fp_task(idx, path):
        fp = compute_fingerprint(path, max_dim)
        fps[idx] = fp
        done_count[0] += 1
        dc = done_count[0]
        if dc % 50 == 0 or dc == n:
            el = time.time()-t0; r = dc/el if el > 0 else 0
            print(f"       {dc}/{n} ({r:.1f}/s)  ", end="\r")
        return fp

    with ThreadPoolExecutor(max_workers=nw) as pool:
        futures = [pool.submit(fp_task, i, p) for i, p in enumerate(paths)]
        for f in futures: f.result()  # wait for all

    fps = [f for f in fps if f is not None]
    print(f"\n       {len(fps)} in {fmt(time.time()-t0)}.\n")

    # ── Screen ──
    print("[3/4] Adaptive screening...")
    total = len(fps)*(len(fps)-1)//2
    cands = []; t1 = time.time(); checked = 0
    for i in range(len(fps)):
        for j in range(i+1, len(fps)):
            checked += 1
            sc = fingerprint_similarity(fps[i], fps[j])
            pt = adaptive_screen_threshold(fps[i], fps[j], base_screen)
            if sc["screen_score"] >= pt:
                cands.append((i, j, sc))
            if checked % 5000 == 0 or checked == total:
                el = time.time()-t1; r = checked/el if el > 0 else 0
                print(f"       {checked:,}/{total:,}  cands: {len(cands)}  ({r:.0f}/s)  ", end="\r")
    print(f"\n       {len(cands):,} candidates in {fmt(time.time()-t1)}.\n")

    if not cands: print("  No duplicates found.\n"); return

    dups = []; rejected = []; uf = UnionFind()

    if fast:
        print("[4/4] Fast mode.\n")
        for i, j, sc in cands:
            if sc["screen_score"] >= threshold:
                uf.union(fps[i].filename, fps[j].filename)
                dups.append({"file_a": fps[i].filename, "file_b": fps[j].filename,
                             "confidence": sc["screen_score"], "scores": sc,
                             "orb_inliers": 0, "flipped": False, "match_level": "fast"})
    else:
        print(f"[4/4] Verifying {len(cands):,} candidates ({nw} threads)...")
        t2 = time.time()
        done_v = [0]
        results_lock = threading.Lock()

        def verify_task(idx):
            i, j, sc = cands[idx]
            detail = verify_pair(fps[i].path, fps[j].path, fps[i], fps[j], max_dim)
            dec = decide_duplicate(sc, detail, fps[i], fps[j], threshold)
            if dec["is_duplicate"]:
                uf.union(fps[i].filename, fps[j].filename)
                with results_lock:
                    dups.append({"file_a": fps[i].filename, "file_b": fps[j].filename,
                                 "confidence": dec["confidence"],
                                 "scores": dec["scores"],
                                 "orb_inliers": dec["orb_inliers"],
                                 "flipped": dec["flipped"],
                                 "match_level": dec.get("match_level", "")})
            else:
                with results_lock:
                    rejected.append({"file_a": fps[i].filename,
                                     "file_b": fps[j].filename,
                                     "confidence": dec["confidence"],
                                     "scores": dec.get("scores", {}),
                                     "orb_inliers": dec.get("orb_inliers", 0)})
            done_v[0] += 1
            dv = done_v[0]
            if dv % 10 == 0 or dv == len(cands):
                el = time.time()-t2; r = dv/el if el > 0 else 0
                print(f"       {dv}/{len(cands)}  dupes: {len(dups)}  "
                      f"({r:.1f}/s, ETA {fmt((len(cands)-dv)/r) if r else '?'})  ", end="\r")

        with ThreadPoolExecutor(max_workers=nw) as pool:
            futures = [pool.submit(verify_task, idx) for idx in range(len(cands))]
            for f in futures: f.result()

        print(f"\n       Done in {fmt(time.time()-t2)}.\n")

    groups = uf.groups()

    # ── Report ──
    lines = []
    def out(t=""): print(t); lines.append(t)
    out(f"{'═'*70}")
    out(f"  RESULTS")
    out(f"{'═'*70}")
    out(f"  Images    : {len(fps):,}")
    out(f"  Screened  : {total:,} pairs")
    out(f"  Candidates: {len(cands):,}")
    out(f"  Confirmed : {len(dups):,}")
    out(f"  Groups    : {len(groups)}")
    out(f"  Threshold : {threshold}")
    out(f"  Workers   : {nw} threads")
    out(f"  GPU       : {_GPU_INFO}")
    out(f"{'═'*70}\n")
    if not groups:
        out("  No near-duplicates found.\n")
    else:
        out(f"  DUPLICATE GROUPS: {len(groups)}\n")
        for gid, (root, members) in enumerate(
            sorted(groups.items(), key=lambda x: -len(x[1])), 1):
            out(f"  Group {gid} ({len(members)} images):")
            for m in sorted(members): out(f"    • {m}")
            out()
        show = sorted(dups, key=lambda x: -x["confidence"])
        if len(show) > 60:
            out(f"  Top 60 of {len(show)} pairs:\n")
            show = show[:60]
        for d in show:
            flip = " [FLIP]" if d["flipped"] else ""
            out(f"  ┌─ {d['file_a']}")
            out(f"  └─ {d['file_b']}  conf={d['confidence']:.4f}  "
                f"inliers={d['orb_inliers']}{flip}  [{d.get('match_level','')}]")
            for k, v in sorted(d["scores"].items(), key=lambda x: -x[1]):
                bar = "█" * int(v*20) + "░" * (20-int(v*20))
                out(f"     {k:>20s}  {bar}  {v:.4f}")
            out()
    # ── Evaluation Metrics ──
    metrics, metrics_lines = compute_evaluation_metrics(
        fps, cands, dups, rejected, groups, total)
    for ml in metrics_lines:
        out(ml)

    # ── Baseline F1 Scores + Friedman Test ──
    gt_map_bl = {}
    gt_classes_bl = {}
    for fp in fps:
        label = extract_gt_label(fp.filename)
        gt_map_bl[fp.filename] = label
        gt_classes_bl.setdefault(label, set()).add(fp.filename)
    resolved_c, resolved_m = _resolve_gt_classes(gt_classes_bl, len(fps))
    if resolved_c is not None:
        gt_classes_bl = resolved_c
        gt_map_bl = resolved_m

    baseline_results = compute_baseline_f1(
        fps, cands, dups, rejected, gt_map_bl, gt_classes_bl, total)

    out(f"\n{'═'*70}")
    out(f"  BASELINE COMPARISON — F1 Scores (Section 4.6.3)")
    out(f"{'═'*70}")
    out(f"\n  {'Method':<25s} {'TP':>7s} {'FP':>7s} {'FN':>7s} "
        f"{'Prec%':>8s} {'Rec%':>8s} {'F1%':>8s}")
    out(f"  {'─'*25} {'─'*7} {'─'*7} {'─'*7} {'─'*8} {'─'*8} {'─'*8}")
    out(f"  {'Proposed (full system)':<25s} {metrics['tp']:>7,} {metrics['fp']:>7,} "
        f"{metrics['fn']:>7,} {metrics['precision']:>7.2f}% "
        f"{metrics['recall']:>7.2f}% {metrics['f1']:>7.2f}%")
    for bname, br in baseline_results.items():
        out(f"  {bname:<25s} {br['tp']:>7,} {br['fp']:>7,} {br['fn']:>7,} "
            f"{br['precision']:>7.2f}% {br['recall']:>7.2f}% {br['f1']:>7.2f}%")
    out(f"{'═'*70}\n")
    
    # ── Statistical Significance Tests ──
    stat_lines = compute_statistical_tests(
        fps, cands, dups, rejected, groups, total)
    for sl in stat_lines:
        out(sl)

    if report:
        with open(report, "w", encoding="utf-8") as f: f.write("\n".join(lines))
        print(f"  Report: {report}\n")


def main():
    p = argparse.ArgumentParser(
        description="Near-duplicate detector v5.2.2 (threaded + GPU)")
    p.add_argument("folder")
    p.add_argument("--threshold", type=float, default=0.50)
    p.add_argument("--max-dim", type=int, default=512)
    p.add_argument("--screen-thresh", type=float, default=0.30)
    p.add_argument("--fast", action="store_true")
    p.add_argument("--report", type=str, default=None)
    p.add_argument("--workers", type=int, default=0,
                   help="Number of threads (0 = auto CPU count)")
    p.add_argument("--gpu", action="store_true",
                   help="GPU via OpenCL (AMD Radeon / Intel / NVIDIA)")
    args = p.parse_args()
    if not os.path.isdir(args.folder):
        print(f"Not a directory: {args.folder}"); sys.exit(1)
    run(args.folder, args.threshold, args.max_dim, args.screen_thresh,
        args.fast, args.report, args.workers, args.gpu)


if __name__ == "__main__":
    main()