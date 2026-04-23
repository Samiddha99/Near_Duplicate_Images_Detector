"""
eval_utils.py
=============
Shared evaluation utilities for the 5 literature-baseline scripts.

Provides:
  * Ground-truth label extraction (identical to duplicate_detector_final.py)
  * Union-Find for grouping duplicate pairs
  * Pair-level metrics (TP/FP/FN/TN, Precision, Recall, F1)
  * Group-level metrics (purity, completeness, NEG leakage)
  * Statistical tests: Bootstrap 95% CI, Cohen's Kappa,
    McNemar (pairwise cross-method), Wilcoxon Signed-Rank, Friedman
  * A unified report writer (`write_baseline_report`) so every literature
    baseline produces a report in EXACTLY the same format — making the
    methods directly comparable with your Section 4.2 / 4.3 results.

Author: comparison study for near-duplicate image grouping research.
"""

from __future__ import annotations
import os, re, math, json, random, time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Tuple, Set, Callable

import numpy as np


# ══════════════════════════════════════════════════════════════════════
# FILE DISCOVERY
# ══════════════════════════════════════════════════════════════════════
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

def get_image_paths(folder: str) -> List[str]:
    return [str(f) for f in sorted(Path(folder).rglob("*"))
            if f.suffix.lower() in IMG_EXTS and f.is_file()]


# ══════════════════════════════════════════════════════════════════════
# GROUND-TRUTH LABEL EXTRACTION  — identical logic to your research code
# ══════════════════════════════════════════════════════════════════════
def extract_gt_label(filename: str) -> str:
    """Return the GT group label for `filename` using your naming conventions.
    Files starting with 'NEG_' become unique singletons."""
    if filename.startswith("NEG_"):
        return "NEG__" + filename

    m = re.match(r'(teeth-?\d+)', filename, re.IGNORECASE)
    if m:
        return m.group(1).lower()

    m = re.match(r'(.+?)_aug_\d+\.\w+$', filename)
    if m:
        return m.group(1)

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

    m = re.match(r'(.+?)\s*\(\d+\)\.\w+$', filename)
    if m:
        return m.group(1).strip()

    return filename


def _resolve_gt_classes(gt_classes: Dict[str, Set[str]], n_images: int):
    """Validate GT classes — fall back to 'all unique' if no augmentation
    evidence exists in a class (mirrors your research code)."""
    if len(gt_classes) == 1:
        label, members = next(iter(gt_classes.items()))
        if len(members) == n_images and n_images > 3:
            new_classes, new_map = {}, {}
            for fn in members:
                unique = f"__unique__{fn}"
                new_classes[unique] = {fn}
                new_map[fn] = unique
            return new_classes, new_map

    aug_keywords = ["_aug_", "_ORIGINAL", "_blur", "_bright", "_combo",
                    "_contrast", "_crop", "_edge", "_flip", "_jpeg",
                    "_rot", "_saturate", "_scale", "_sharpen", "_shift",
                    "_stretch", "_zoom"]
    explicit_group_pattern = re.compile(r'^teeth-?\d+$', re.IGNORECASE)
    remap = {}
    for label, members in list(gt_classes.items()):
        if label.startswith("NEG__") or len(members) < 2:
            continue
        if explicit_group_pattern.match(label):
            continue
        has_aug = any(kw in fn for fn in members for kw in aug_keywords)
        if not has_aug:
            for fn in members:
                remap[fn] = f"__unique__{fn}"

    if remap:
        new_classes, new_map = {}, {}
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


def build_gt(filenames: List[str]) -> Tuple[Dict[str, str], Dict[str, Set[str]]]:
    """Return (gt_map filename→label, gt_classes label→{filenames})."""
    gt_map, gt_classes = {}, {}
    for fn in filenames:
        label = extract_gt_label(fn)
        gt_map[fn] = label
        gt_classes.setdefault(label, set()).add(fn)

    resolved_classes, resolved_map = _resolve_gt_classes(gt_classes, len(filenames))
    if resolved_classes is not None:
        return resolved_map, resolved_classes
    return gt_map, gt_classes


# ══════════════════════════════════════════════════════════════════════
# UNION-FIND
# ══════════════════════════════════════════════════════════════════════
class UnionFind:
    def __init__(self):
        self.parent, self.rank = {}, {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x; self.rank[x] = 0
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1

    def groups(self) -> Dict[str, Set[str]]:
        clusters = {}
        for x in self.parent:
            clusters.setdefault(self.find(x), set()).add(x)
        return {r: m for r, m in clusters.items() if len(m) > 1}


# ══════════════════════════════════════════════════════════════════════
# STAT HELPERS (no scipy dependency)
# ══════════════════════════════════════════════════════════════════════
def _normal_cdf(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))

def _chi2_sf(x: float, df: int) -> float:
    if x <= 0:
        return 1.0
    if df == 1:
        return 2 * (1 - _normal_cdf(math.sqrt(x)))
    if df == 2:
        return math.exp(-x / 2)
    a = df / 2.0
    z = x / 2.0
    term = math.exp(-z) * (z ** a) / math.gamma(a + 1)
    s = term
    for n in range(1, 500):
        term *= z / (a + n)
        s += term
        if abs(term) < 1e-14:
            break
    return max(0.0, min(1.0, 1.0 - s))


# ══════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════
@dataclass
class PairPred:
    """Single pair-level prediction from a baseline."""
    file_a: str
    file_b: str
    score: float           # raw similarity score (higher = more duplicate-like)
    predicted: int         # 0 / 1
    # optional secondary scores for analysis / baselines-within-baseline
    extra: Dict[str, float] = field(default_factory=dict)


def compute_pair_metrics(preds: List[PairPred],
                         gt_map: Dict[str, str],
                         gt_classes: Dict[str, Set[str]],
                         total_pairs: int) -> Dict:
    """Pair-level TP/FP/FN/TN + Precision/Recall/F1."""
    gt_pair_count = sum(len(v)*(len(v)-1)//2
                        for v in gt_classes.values() if len(v) >= 2)

    tp = fp = 0
    tp_scores, fp_scores, rej_scores = [], [], []
    verified_gt_positive = 0          # predictions evaluated where GT=positive
    for p in preds:
        la = gt_map.get(p.file_a, "?")
        lb = gt_map.get(p.file_b, "?")
        gt_same = (la == lb and la != "?" and
                   len(gt_classes.get(la, set())) >= 2)
        if gt_same:
            verified_gt_positive += 1
        if p.predicted == 1:
            if gt_same:
                tp += 1; tp_scores.append(p.score)
            else:
                fp += 1; fp_scores.append(p.score)
        else:
            rej_scores.append(p.score)

    fn = gt_pair_count - tp                 # positives missed everywhere
    tn = total_pairs - tp - fp - fn
    tn = max(0, tn)

    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 100.0
    recall    = tp / (tp + fn) * 100 if (tp + fn) > 0 else 100.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                precision=precision, recall=recall, f1=f1,
                gt_pair_count=gt_pair_count,
                tp_scores=tp_scores, fp_scores=fp_scores, rej_scores=rej_scores)


def compute_group_metrics(groups: Dict[str, Set[str]],
                          gt_map: Dict[str, str],
                          gt_classes: Dict[str, Set[str]],
                          n_images: int) -> Dict:
    purities, completeness_vals = [], []
    pure, contaminated = 0, 0
    contam_details = []
    for _, members in groups.items():
        counts = {}
        for m in members:
            counts[gt_map.get(m, m)] = counts.get(gt_map.get(m, m), 0) + 1
        max_c = max(counts.values())
        purities.append(max_c / len(members))
        if len(counts) == 1:
            pure += 1
        else:
            contaminated += 1
            contam_details.append((members, counts))

    for label, members in gt_classes.items():
        if label.startswith("NEG__") or len(members) < 2:
            continue
        best = 0
        for _, g in groups.items():
            best = max(best, len(members & g))
        completeness_vals.append(best / len(members))

    neg_leaked, neg_leaked_files = 0, []
    for _, members in groups.items():
        for m in members:
            if m.startswith("NEG_"):
                neg_leaked += 1
                neg_leaked_files.append(m)
    neg_total = sum(1 for f in gt_map if f.startswith("NEG_"))

    grouped = set()
    for _, m in groups.items():
        grouped |= m

    return dict(
        n_groups=len(groups),
        n_gt_classes=sum(1 for v in gt_classes.values() if len(v) >= 2),
        pure=pure, contaminated=contaminated,
        purity=float(np.mean(purities)) if purities else 1.0,
        completeness=float(np.mean(completeness_vals)) if completeness_vals else 1.0,
        grouped=len(grouped), ungrouped=n_images - len(grouped),
        neg_leaked=neg_leaked, neg_total=neg_total,
        neg_leaked_files=neg_leaked_files,
        contaminated_details=contam_details,
    )


# ══════════════════════════════════════════════════════════════════════
# STATISTICAL TESTS
# ══════════════════════════════════════════════════════════════════════
def bootstrap_ci(preds: List[PairPred], gt_map, gt_classes,
                 total_pairs: int, B: int = 10_000, seed: int = 42) -> Dict:
    rng = random.Random(seed)
    gt_pair_count = sum(len(v)*(len(v)-1)//2
                        for v in gt_classes.values() if len(v) >= 2)
    # pre-compute per-pair gt_same + predicted
    flags = []
    for p in preds:
        la = gt_map.get(p.file_a, "?"); lb = gt_map.get(p.file_b, "?")
        gt_same = 1 if (la == lb and la != "?" and
                        len(gt_classes.get(la, set())) >= 2) else 0
        flags.append((p.predicted, gt_same))

    n = len(flags)
    fn_from_screening = gt_pair_count - sum(1 for _, g in flags if g == 1)
    fn_from_screening = max(0, fn_from_screening)

    ps, rs, fs = [], [], []
    for _ in range(B):
        tp = fp = fn_v = 0
        for _ in range(n):
            pred, g = flags[rng.randrange(n)]
            if pred == 1 and g == 1: tp += 1
            elif pred == 1: fp += 1
            elif g == 1: fn_v += 1
        fn_t = fn_v + fn_from_screening
        P = tp/(tp+fp)*100 if (tp+fp) else 100.0
        R = tp/(tp+fn_t)*100 if (tp+fn_t) else 100.0
        F = 2*P*R/(P+R) if (P+R) else 0.0
        ps.append(P); rs.append(R); fs.append(F)
    ps.sort(); rs.sort(); fs.sort()
    lo, hi = int(0.025*B), int(0.975*B) - 1
    return dict(P_ci=(ps[lo], ps[hi]), R_ci=(rs[lo], rs[hi]),
                F_ci=(fs[lo], fs[hi]), B=B)


def cohens_kappa(tp: int, fp: int, fn: int, tn: int) -> Dict:
    n = tp + fp + fn + tn
    if n == 0:
        return dict(kappa=1.0, p0=1.0, pe=1.0, se=0.0, ci=(1.0, 1.0), interp="N/A")
    p0 = (tp + tn) / n
    pe = ((tp+fp)*(tp+fn) + (fn+tn)*(fp+tn)) / (n * n)
    kappa = (p0 - pe) / (1 - pe) if (1 - pe) > 0 else 1.0
    se = math.sqrt(p0*(1-p0)/(n*(1-pe)**2)) if (1-pe) > 0 else 0.0
    lo, hi = kappa - 1.96*se, kappa + 1.96*se
    if   kappa >= 0.81: interp = "Near-perfect"
    elif kappa >= 0.61: interp = "Substantial"
    elif kappa >= 0.41: interp = "Moderate"
    elif kappa >= 0.21: interp = "Fair"
    else:               interp = "Slight"
    return dict(kappa=kappa, p0=p0, pe=pe, se=se, ci=(lo, hi), interp=interp)


def mcnemar_test(pairs_a: List[int], pairs_b: List[int],
                 gts:     List[int]) -> Dict:
    """pairs_a[i], pairs_b[i] ∈ {0,1} are predictions of two methods
    on the SAME pair; gts[i] is the ground truth."""
    assert len(pairs_a) == len(pairs_b) == len(gts)
    n01 = n10 = 0
    for a, b, g in zip(pairs_a, pairs_b, gts):
        ca, cb = (a == g), (b == g)
        if ca and not cb: n01 += 1
        elif cb and not ca: n10 += 1
    total = n01 + n10
    if total == 0:
        return dict(n01=0, n10=0, chi2=0.0, p=1.0, sig=False)
    if total < 25:
        from math import comb
        k = min(n01, n10)
        p = sum(comb(total, x) * (0.5 ** total) for x in range(k+1)) * 2
        p = min(p, 1.0)
        return dict(n01=n01, n10=n10, chi2=float('nan'), p=p, sig=p<0.05)
    chi2 = (abs(n01 - n10) - 1) ** 2 / total
    p = _chi2_sf(chi2, 1)
    return dict(n01=n01, n10=n10, chi2=chi2, p=p, sig=p<0.05)


def wilcoxon_signed_rank(a: List[float], b: List[float]) -> Dict:
    diffs = [x - y for x, y in zip(a, b) if (x - y) != 0]
    n = len(diffs)
    if n < 5:
        return dict(n=n, z=0.0, p=1.0, W_plus=0, W_minus=0,
                    median_diff=0.0, sig=False)
    absd = sorted([(abs(d), 1 if d > 0 else -1) for d in diffs], key=lambda x: x[0])
    ranks = list(range(1, n+1))
    i = 0
    while i < n:
        j = i
        while j < n and absd[j][0] == absd[i][0]:
            j += 1
        avg = sum(ranks[i:j])/(j-i)
        for k in range(i, j): ranks[k] = avg
        i = j
    Wp = sum(ranks[k] for k in range(n) if absd[k][1] > 0)
    Wm = sum(ranks[k] for k in range(n) if absd[k][1] < 0)
    W = min(Wp, Wm)
    mu = n*(n+1)/4
    sig = math.sqrt(n*(n+1)*(2*n+1)/24)
    z = (W - mu) / sig if sig > 0 else 0.0
    p = 2 * _normal_cdf(z) if z < 0 else 2 * (1 - _normal_cdf(z))
    p = min(p, 1.0)
    sorted_d = sorted([x - y for x, y in zip(a, b)])
    med = sorted_d[len(sorted_d)//2] if sorted_d else 0.0
    return dict(n=n, z=z, p=p, W_plus=Wp, W_minus=Wm,
                median_diff=med, sig=p<0.05)


def friedman_test(f1_matrix: Dict[str, Dict[str, float]]) -> List[str]:
    """f1_matrix[dataset][method] = F1%. Returns formatted report lines
    with Friedman χ² + Nemenyi CD pairwise."""
    datasets = sorted(f1_matrix.keys())
    methods = sorted(next(iter(f1_matrix.values())).keys())
    k, N = len(methods), len(datasets)
    out = []
    if N < 3 or k < 3:
        out.append(f"  Friedman test needs ≥3 datasets and ≥3 methods "
                   f"(got N={N}, k={k}). Skipped.")
        return out

    ranks = {m: [] for m in methods}
    for ds in datasets:
        scored = sorted([(m, f1_matrix[ds].get(m, 0.0)) for m in methods],
                        key=lambda x: -x[1])
        i = 0
        while i < len(scored):
            j = i
            while j < len(scored) and scored[j][1] == scored[i][1]:
                j += 1
            avg = sum(range(i+1, j+1))/(j-i)
            for idx in range(i, j):
                ranks[scored[idx][0]].append(avg)
            i = j
    mean_ranks = {m: sum(ranks[m])/N for m in methods}
    sum_sq = sum(r*r for r in mean_ranks.values())
    chi2 = (12*N/(k*(k+1))) * (sum_sq - k*(k+1)**2/4)
    df = k - 1
    p = _chi2_sf(chi2, df)

    out.append(f"  Friedman χ²_F = {chi2:.4f}, df={df}, "
               f"p={'<0.001' if p<0.001 else f'{p:.4f}'}")
    if p < 0.05:
        out.append(f"  → H0 REJECTED (methods differ significantly)")
        q_table = {3:2.343, 4:2.569, 5:2.728, 6:2.850,
                   7:2.949, 8:3.031, 9:3.102, 10:3.164}
        q = q_table.get(k, 2.728)
        cd = q * math.sqrt(k*(k+1)/(6*N))
        out.append(f"  Nemenyi CD = {cd:.4f} (q_α={q}, k={k}, N={N})")
        out.append("")
        out.append(f"    {'Method A':<25s}  {'Method B':<25s}  {'|ΔR|':>6s}  {'Sig?':>5s}")
        out.append(f"    {'-'*25}  {'-'*25}  {'-'*6}  {'-'*5}")
        ms = list(methods)
        for i, m1 in enumerate(ms):
            for m2 in ms[i+1:]:
                d = abs(mean_ranks[m1] - mean_ranks[m2])
                out.append(f"    {m1:<25s}  {m2:<25s}  {d:>6.2f}  "
                           f"{'Yes' if d>cd else 'No':>5s}")
    else:
        out.append(f"  → H0 NOT rejected (no significant difference)")
    out.append("")
    out.append(f"  Mean ranks (1 = best):")
    for m in sorted(mean_ranks.keys(), key=lambda x: mean_ranks[x]):
        out.append(f"    {m:<30s}  {mean_ranks[m]:.3f}")
    return out


# ══════════════════════════════════════════════════════════════════════
# REPORT WRITER
# ══════════════════════════════════════════════════════════════════════
def write_baseline_report(report_path: str,
                          method_name: str,
                          paper_ref: str,
                          filenames: List[str],
                          preds: List[PairPred],
                          groups: Dict[str, Set[str]],
                          runtime_sec: float,
                          extra_lines: Optional[List[str]] = None,
                          bootstrap_B: int = 2000):
    """Write a fully-formatted evaluation report for one baseline.
    Produces a report comparable with your Section 4.2 / 4.3 format."""
    n = len(filenames)
    total_pairs = n * (n - 1) // 2
    gt_map, gt_classes = build_gt(filenames)

    pair_m = compute_pair_metrics(preds, gt_map, gt_classes, total_pairs)
    grp_m  = compute_group_metrics(groups, gt_map, gt_classes, n)
    ci     = bootstrap_ci(preds, gt_map, gt_classes, total_pairs, B=bootstrap_B)
    kap    = cohens_kappa(pair_m['tp'], pair_m['fp'], pair_m['fn'], pair_m['tn'])

    lines = []
    def L(s=""): lines.append(s)

    sep = "═" * 72
    L(sep); L(f"  LITERATURE BASELINE REPORT  —  {method_name}")
    L(f"  Reference: {paper_ref}"); L(sep)
    L(f"  Images              : {n:,}")
    L(f"  Total pairs         : {total_pairs:,}")
    L(f"  Pair predictions    : {len(preds):,}")
    L(f"  Runtime             : {runtime_sec:.1f} s")
    L(sep); L()

    # ---- Detection output: duplicate groups (compact) --------------
    L(f"DUPLICATE GROUPS DETECTED : {len(groups)}")
    if groups:
        for gid, (_, members) in enumerate(
                sorted(groups.items(), key=lambda x: -len(x[1])), 1):
            L(f"  Group {gid} ({len(members)} images):")
            for m in sorted(members):
                L(f"    • {m}")
            L()
    else:
        L("  No duplicate groups formed at this method's threshold.\n")

    # ---- Confirmed pairs (top 40) ----------------------------------
    confirmed = [p for p in preds if p.predicted == 1]
    L(f"CONFIRMED PAIRS : {len(confirmed):,}"
      f"   REJECTED : {len(preds) - len(confirmed):,}")
    show = sorted(confirmed, key=lambda x: -x.score)[:40]
    if show:
        L(f"\n  Top {len(show)} highest-scoring confirmed pairs:")
        for p in show:
            gt_a = gt_map.get(p.file_a, "?"); gt_b = gt_map.get(p.file_b, "?")
            tag = " [TP]" if (gt_a == gt_b and gt_a != "?" and
                              len(gt_classes.get(gt_a, set())) >= 2) else " [FP]"
            L(f"    score={p.score:.4f}{tag}  {p.file_a}  ↔  {p.file_b}")
    L()

    # ---- Section 4.2.1  pair metrics -------------------------------
    L(sep); L(f"  EVALUATION METRICS — pair-level (Section 4.2.1)")
    L(f"{sep}")
    L(f"    True Positives  (TP) : {pair_m['tp']:,}")
    L(f"    False Positives (FP) : {pair_m['fp']:,}")
    L(f"    False Negatives (FN) : {pair_m['fn']:,}")
    L(f"    True Negatives  (TN) : {pair_m['tn']:,}")
    L(f"    Precision            : {pair_m['precision']:.2f}%")
    L(f"    Recall               : {pair_m['recall']:.2f}%")
    L(f"    F1-Score             : {pair_m['f1']:.2f}%")

    # ---- 4.2.2  group metrics --------------------------------------
    L()
    L(f"  GROUP-LEVEL METRICS (Section 4.2.2)")
    L(f"    Groups detected      : {grp_m['n_groups']}")
    L(f"    GT classes (≥2 imgs) : {grp_m['n_gt_classes']}")
    L(f"    Pure groups          : {grp_m['pure']}/{grp_m['n_groups']}")
    L(f"    Contaminated groups  : {grp_m['contaminated']}")
    L(f"    Avg group purity     : {grp_m['purity']:.4f}")
    L(f"    Avg group completeness: {grp_m['completeness']:.4f}")
    L(f"    Grouped / ungrouped  : {grp_m['grouped']} / {grp_m['ungrouped']}")
    if grp_m['contaminated_details']:
        L("    Contaminated group breakdown:")
        for members, counts in grp_m['contaminated_details']:
            tag = ", ".join(f"{k}: {v}" for k,v in sorted(counts.items(), key=lambda x:-x[1]))
            L(f"      [{len(members)} members] {tag}")

    # ---- NEG leakage ----------------------------------------------
    L()
    L(f"  NEG LEAKAGE (Section 4.2.3)")
    L(f"    NEG images in dataset : {grp_m['neg_total']}")
    L(f"    NEG leaked into groups: {grp_m['neg_leaked']}/{grp_m['neg_total']}")
    for nf in grp_m['neg_leaked_files']:
        L(f"      Leaked: {nf}")

    # ---- confidence distribution -----------------------------------
    L()
    L(f"  SCORE DISTRIBUTION (Section 4.2.5)")
    def _s(x): return (sum(x)/len(x)) if x else 0.0
    L(f"    μ_TP  mean score of TP  : {_s(pair_m['tp_scores']):.4f}")
    L(f"    μ_FP  mean score of FP  : {_s(pair_m['fp_scores']):.4f}")
    L(f"    μ_REJ mean score of rej : {_s(pair_m['rej_scores']):.4f}")
    if pair_m['tp_scores'] and pair_m['rej_scores']:
        margin = min(pair_m['tp_scores']) - max(pair_m['rej_scores'])
        L(f"    Decision margin (Δ)     : {margin:+.4f}")

    # ---- 4.3.4  bootstrap CI ---------------------------------------
    L()
    L(sep); L(f"  STATISTICAL TESTS (Section 4.3)"); L(sep)
    L(f"\n  Bootstrap 95% CI (B={ci['B']:,}):")
    L(f"    Precision : {pair_m['precision']:7.2f}%   "
      f"[{ci['P_ci'][0]:.2f}%, {ci['P_ci'][1]:.2f}%]")
    L(f"    Recall    : {pair_m['recall']:7.2f}%   "
      f"[{ci['R_ci'][0]:.2f}%, {ci['R_ci'][1]:.2f}%]")
    L(f"    F1        : {pair_m['f1']:7.2f}%   "
      f"[{ci['F_ci'][0]:.2f}%, {ci['F_ci'][1]:.2f}%]")

    # ---- 4.3.5  Cohen's Kappa --------------------------------------
    L()
    L(f"  Cohen's Kappa (Section 4.3.5):")
    L(f"    p₀ (observed agreement) : {kap['p0']:.6f}")
    L(f"    pₑ (expected agreement) : {kap['pe']:.6f}")
    L(f"    κ                       : {kap['kappa']:.4f}   ({kap['interp']})")
    L(f"    95% CI                  : [{kap['ci'][0]:.4f}, {kap['ci'][1]:.4f}]")
    L(f"    SE(κ)                   : {kap['se']:.6f}")

    if extra_lines:
        L(""); L(sep)
        L(f"  METHOD-SPECIFIC NOTES")
        L(sep)
        for s in extra_lines:
            L(s)

    L(""); L(sep); L(f"  END OF REPORT — {method_name}"); L(sep)

    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    # Also write a machine-readable side-car so the orchestrator can
    # run the cross-method statistical tests without re-computing anything.
    side = {
        "method": method_name,
        "paper": paper_ref,
        "runtime_sec": runtime_sec,
        "n_images": n,
        "total_pairs": total_pairs,
        "tp": pair_m['tp'], "fp": pair_m['fp'],
        "fn": pair_m['fn'], "tn": pair_m['tn'],
        "precision": pair_m['precision'],
        "recall": pair_m['recall'],
        "f1": pair_m['f1'],
        "kappa": kap['kappa'],
        "P_ci": ci['P_ci'], "R_ci": ci['R_ci'], "F_ci": ci['F_ci'],
        "n_groups": grp_m['n_groups'], "purity": grp_m['purity'],
        "completeness": grp_m['completeness'],
        "pure_groups": grp_m['pure'],
        "contaminated_groups": grp_m['contaminated'],
        "neg_leaked": grp_m['neg_leaked'], "neg_total": grp_m['neg_total'],
        # store per-pair predictions so orchestrator can compute McNemar
        "pair_predictions": [
            {
                "a": p.file_a, "b": p.file_b,
                "score": float(p.score), "pred": int(p.predicted),
            } for p in preds
        ],
    }
    side_path = Path(report_path).with_suffix(".json")
    with open(side_path, "w", encoding="utf-8") as fh:
        json.dump(side, fh, indent=2)

    return dict(pair=pair_m, group=grp_m, ci=ci, kappa=kap, side_path=str(side_path))


# ══════════════════════════════════════════════════════════════════════
# SHARED UNION-FIND FROM PAIR PREDICTIONS
# ══════════════════════════════════════════════════════════════════════
def build_groups(preds: List[PairPred]) -> Dict[str, Set[str]]:
    uf = UnionFind()
    for p in preds:
        if p.predicted == 1:
            uf.union(p.file_a, p.file_b)
    return uf.groups()


# ══════════════════════════════════════════════════════════════════════
# IMAGE LOADING (shared)
# ══════════════════════════════════════════════════════════════════════
def load_rgb(path: str, max_dim: int = 512):
    """Cheap loader — returns HxWx3 uint8 np.array."""
    from PIL import Image, ImageOps
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_dim, max_dim))
    return np.array(img)