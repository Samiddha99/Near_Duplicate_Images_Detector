"""
run_all_baselines.py
====================
Orchestrator for the five literature baselines.

1.  Runs each baseline_rN.py on the given image folder, producing
        rN_report.txt   +   rN_report.json    side-by-side.

2.  Loads the side-car JSONs plus (optionally) the proposed method's
    predictions from --proposed-json (an identically-structured JSON
    dumped by your `duplicate_detector_final.py`).  If --proposed-json
    is omitted we skip the cross-method stats (the per-baseline metrics
    are still written).

3.  Computes cross-method statistical comparisons on the SAME dataset:
        • McNemar's Test — Proposed vs every baseline (n01 / n10 / χ² / p)
        • Wilcoxon Signed-Rank — Proposed score vs every baseline score
          on the pairs that both methods predicted.
        • Friedman + Nemenyi post-hoc — using F1-scores across
          synthetic sub-datasets (we stratify the folder by
          GT-class-prefix so that N≥3, which is required for Friedman).

4.  Writes `comparison_report.txt` (plus `comparison_report.json`) in
    the folder specified by --out-dir.

USAGE
-----
    # (a) run each baseline individually first, then orchestrate
    python baseline_r1.py  /path/to/imgs --report out/r1_report.txt
    python baseline_r2.py  /path/to/imgs --report out/r2_report.txt
    ...
    python run_all_baselines.py  /path/to/imgs  --out-dir out \\
           --proposed-json your_method.json

    # (b) one-shot: orchestrator will run every baseline that has
    # not yet produced its side-car JSON.
    python run_all_baselines.py /path/to/imgs --out-dir out --run-all
"""

from __future__ import annotations
import os, sys, time, json, argparse, subprocess, shutil
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

from eval_utils import (
    get_image_paths, build_gt, mcnemar_test, wilcoxon_signed_rank,
    friedman_test, cohens_kappa, _chi2_sf,
)


BASELINES = [
    ("r1", "pHash + ViT + Siamese",       "baseline_r1.py"),
    ("r2", "Stochastic ARG Matching",     "baseline_r2.py"),
    ("r3", "CEDetector (ViT DINO)",       "baseline_r3.py"),
    ("r4", "DWT + CNN + KNN",             "baseline_r4.py"),
    ("r5", "PCET + GDP",                  "baseline_r5.py"),
]


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════
def _load_side(json_path: str) -> Dict:
    with open(json_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _run_baseline(tag: str, script: str, folder: str, out_dir: str,
                  python_bin: str = "python") -> str:
    report = os.path.join(out_dir, f"{tag}_report.txt")
    cmd = [python_bin, os.path.join(os.path.dirname(__file__), script),
           folder, "--report", report]
    print(f"  → running {script} ...")
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  [ERROR] {script} exited {r.returncode}")
        print(r.stdout[-1000:])
        print(r.stderr[-1000:])
    print(f"  ← {script} done in {time.time()-t0:.1f}s")
    return report


def _pair_key(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a < b else (b, a)


def _build_pair_map(preds: List[Dict]) -> Dict[Tuple[str, str], Dict]:
    return {_pair_key(p["a"], p["b"]): p for p in preds}


def _align(preds_a: List[Dict], preds_b: List[Dict], gt_map, gt_classes):
    """Return aligned (pred_a, pred_b, gt) arrays over the intersection
    of pair-keys that both methods scored."""
    ma = _build_pair_map(preds_a)
    mb = _build_pair_map(preds_b)
    common = sorted(set(ma.keys()) & set(mb.keys()))
    pa, pb, gs = [], [], []
    score_a, score_b = [], []
    for k in common:
        pa.append(ma[k]["pred"])
        pb.append(mb[k]["pred"])
        score_a.append(ma[k]["score"])
        score_b.append(mb[k]["score"])
        fa, fb = k
        la = gt_map.get(fa, "?"); lb = gt_map.get(fb, "?")
        gs.append(1 if (la == lb and la != "?" and
                        len(gt_classes.get(la, set())) >= 2) else 0)
    return pa, pb, gs, score_a, score_b, common


def _stratify_into_subsets(filenames: List[str], k: int = 3) -> Dict[str, List[str]]:
    """Split the dataset into `k` balanced sub-datasets based on GT-class
    prefix (needed for Friedman's N≥3 sub-datasets)."""
    gt_map, gt_classes = build_gt(filenames)
    # Group classes into k buckets by hash of prefix
    class_to_bucket = {}
    classes = sorted(gt_classes.keys())
    for i, c in enumerate(classes):
        class_to_bucket[c] = i % k
    sub = {f"subset_{i+1}": [] for i in range(k)}
    for fn in filenames:
        b = class_to_bucket.get(gt_map[fn], 0)
        sub[f"subset_{b+1}"].append(fn)
    return sub


def _subset_pair_metrics(preds: List[Dict], fileset: Set[str],
                         gt_map, gt_classes) -> Dict:
    """Re-compute TP/FP/FN/F1 when we restrict the evaluation to a
    subset of filenames."""
    sub_classes = {}
    for fn in fileset:
        lab = gt_map[fn]
        sub_classes.setdefault(lab, set()).add(fn)
    gt_pair_count = sum(len(v)*(len(v)-1)//2
                        for v in sub_classes.values() if len(v) >= 2)
    n = len(fileset)
    total_pairs = n * (n - 1) // 2
    tp = fp = 0
    considered = 0
    for p in preds:
        if p["a"] not in fileset or p["b"] not in fileset:
            continue
        considered += 1
        la = gt_map[p["a"]]; lb = gt_map[p["b"]]
        gt_same = (la == lb and la != "" and
                   len(sub_classes.get(la, set())) >= 2)
        if p["pred"] == 1:
            if gt_same: tp += 1
            else: fp += 1
    fn = gt_pair_count - tp
    tn = max(0, total_pairs - tp - fp - fn)
    P = tp/(tp+fp)*100 if (tp+fp) else 100.0
    R = tp/(tp+fn)*100 if (tp+fn) else 100.0
    F = 2*P*R/(P+R) if (P+R) else 0.0
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, P=P, R=R, F=F,
                n_pairs=total_pairs, considered=considered)


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("folder", help="Image folder to evaluate")
    ap.add_argument("--out-dir", default="baseline_out",
                    help="Where to write reports + comparison_report.txt")
    ap.add_argument("--run-all", action="store_true",
                    help="Run every baseline script that has not yet "
                         "produced a side-car JSON")
    ap.add_argument("--proposed-json", default=None,
                    help="Path to your research method's JSON output "
                         "(same schema as the baselines).  Omit to skip "
                         "cross-method McNemar/Wilcoxon/Friedman.")
    ap.add_argument("--python-bin", default=sys.executable)
    args = ap.parse_args()

    if not os.path.isdir(args.folder):
        print("Not a directory:", args.folder); sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Optionally run baselines first ─────────────────────────────
    side_paths: Dict[str, str] = {}
    for tag, name, script in BASELINES:
        side = os.path.join(args.out_dir, f"{tag}_report.json")
        if os.path.exists(side) and not args.run_all:
            side_paths[tag] = side
            continue
        if args.run_all:
            _run_baseline(tag, script, args.folder, args.out_dir,
                          args.python_bin)
            if os.path.exists(side):
                side_paths[tag] = side
            else:
                print(f"  [WARN] {tag} did not produce {side}")
        else:
            print(f"  [skip] {tag} — {side} not found (use --run-all to execute)")

    if not side_paths:
        print("No baseline JSONs available; nothing to compare.")
        return

    # ── Load everything into memory ────────────────────────────────
    baselines: Dict[str, Dict] = {t: _load_side(p) for t, p in side_paths.items()}

    proposed = None
    if args.proposed_json:
        if os.path.exists(args.proposed_json):
            proposed = _load_side(args.proposed_json)
        else:
            print(f"  [WARN] --proposed-json not found: {args.proposed_json}")

    # Build GT from the folder
    paths = get_image_paths(args.folder)
    filenames = [Path(p).name for p in paths]
    gt_map, gt_classes = build_gt(filenames)

    # ── Start the comparison report ────────────────────────────────
    sep = "═" * 72
    out = [sep, "  CROSS-METHOD COMPARISON REPORT", sep,
           f"  Image folder : {args.folder}",
           f"  Images       : {len(filenames):,}",
           f"  Baselines    : {', '.join(side_paths.keys())}",
           f"  Proposed JSON: "
           f"{'yes' if proposed else 'no (cross-method stats reduced)'}",
           sep, ""]

    # ── Summary table: Precision / Recall / F1 / Kappa ─────────────
    out += ["SUMMARY OF DETECTION METRICS", "-"*72]
    out.append(f"  {'Method':<30s} {'TP':>6s} {'FP':>6s} {'FN':>6s} "
               f"{'P%':>6s} {'R%':>6s} {'F1%':>6s} {'κ':>6s}")
    out.append(f"  {'-'*30} {'-'*6} {'-'*6} {'-'*6} "
               f"{'-'*6} {'-'*6} {'-'*6} {'-'*6}")
    if proposed:
        out.append(f"  {'Proposed (ours)':<30s} "
                   f"{proposed['tp']:>6d} {proposed['fp']:>6d} "
                   f"{proposed['fn']:>6d} "
                   f"{proposed['precision']:>6.2f} "
                   f"{proposed['recall']:>6.2f} "
                   f"{proposed['f1']:>6.2f} "
                   f"{proposed.get('kappa', 0.0):>6.3f}")
    for tag, name, _ in BASELINES:
        if tag not in baselines: continue
        b = baselines[tag]
        out.append(f"  {name[:30]:<30s} "
                   f"{b['tp']:>6d} {b['fp']:>6d} {b['fn']:>6d} "
                   f"{b['precision']:>6.2f} {b['recall']:>6.2f} "
                   f"{b['f1']:>6.2f} {b.get('kappa',0.0):>6.3f}")
    out.append("")

    # ── Group-level table ──────────────────────────────────────────
    out += ["GROUP-LEVEL METRICS", "-"*72]
    out.append(f"  {'Method':<30s} {'#G':>5s} {'Pure':>5s} {'Contam':>6s} "
               f"{'Purity':>7s} {'Comp.':>7s} {'NEGlk':>6s}")
    out.append(f"  {'-'*30} {'-'*5} {'-'*5} {'-'*6} "
               f"{'-'*7} {'-'*7} {'-'*6}")
    if proposed:
        out.append(f"  {'Proposed (ours)':<30s} "
                   f"{proposed.get('n_groups', 0):>5d} "
                   f"{proposed.get('pure_groups', 0):>5d} "
                   f"{proposed.get('contaminated_groups', 0):>6d} "
                   f"{proposed.get('purity', 1.0):>7.4f} "
                   f"{proposed.get('completeness', 1.0):>7.4f} "
                   f"{proposed.get('neg_leaked', 0):>6d}")
    for tag, name, _ in BASELINES:
        if tag not in baselines: continue
        b = baselines[tag]
        out.append(f"  {name[:30]:<30s} "
                   f"{b['n_groups']:>5d} {b['pure_groups']:>5d} "
                   f"{b['contaminated_groups']:>6d} "
                   f"{b['purity']:>7.4f} {b['completeness']:>7.4f} "
                   f"{b['neg_leaked']:>6d}")
    out.append("")

    # ── Bootstrap CI table ─────────────────────────────────────────
    out += ["BOOTSTRAP 95% CI  (computed by each baseline on full dataset)",
            "-"*72]
    out.append(f"  {'Method':<30s}  {'Precision CI':>22s}  "
               f"{'Recall CI':>22s}  {'F1 CI':>22s}")
    out.append(f"  {'-'*30}  {'-'*22}  {'-'*22}  {'-'*22}")
    if proposed:
        out.append(f"  {'Proposed (ours)':<30s}  "
                   f"[{proposed['P_ci'][0]:.2f}, {proposed['P_ci'][1]:.2f}]".ljust(32)
                   + f"  [{proposed['R_ci'][0]:.2f}, {proposed['R_ci'][1]:.2f}]".ljust(24)
                   + f"  [{proposed['F_ci'][0]:.2f}, {proposed['F_ci'][1]:.2f}]")
    for tag, name, _ in BASELINES:
        if tag not in baselines: continue
        b = baselines[tag]
        out.append(f"  {name[:30]:<30s}  "
                   f"[{b['P_ci'][0]:.2f}, {b['P_ci'][1]:.2f}]".ljust(32)
                   + f"  [{b['R_ci'][0]:.2f}, {b['R_ci'][1]:.2f}]".ljust(24)
                   + f"  [{b['F_ci'][0]:.2f}, {b['F_ci'][1]:.2f}]")
    out.append("")

    # ── McNemar's test: Proposed vs each baseline ──────────────────
    if proposed:
        out += [sep, "  4.3.1  McNEMAR TEST  —  Proposed vs. each baseline", sep]
        out.append(f"  {'Baseline':<35s}  {'n01':>6s} {'n10':>6s} "
                   f"{'χ²':>10s} {'p-value':>10s} {'Sig?':>5s}")
        out.append(f"  {'-'*35}  {'-'*6} {'-'*6} "
                   f"{'-'*10} {'-'*10} {'-'*5}")
        for tag, name, _ in BASELINES:
            if tag not in baselines: continue
            pa, pb, gs, sa, sb, common = _align(
                proposed["pair_predictions"],
                baselines[tag]["pair_predictions"], gt_map, gt_classes)
            if not pa:
                out.append(f"  {name:<35s}  (no overlapping pairs)")
                continue
            m = mcnemar_test(pa, pb, gs)
            chi_str = f"{m['chi2']:.2f}" if not np.isnan(m['chi2']) else "exact"
            p_str = "<0.001" if m["p"] < 0.001 else f"{m['p']:.4f}"
            out.append(f"  {name:<35s}  {m['n01']:>6d} {m['n10']:>6d} "
                       f"{chi_str:>10s} {p_str:>10s} "
                       f"{'Yes' if m['sig'] else 'No':>5s}")
        out.append("")

    # ── Wilcoxon Signed-Rank: Proposed vs each baseline ────────────
    if proposed:
        out += [sep, "  4.3.2  WILCOXON SIGNED-RANK  —  Proposed vs. each baseline", sep]
        out.append(f"  (applied to scores on pairs scored by BOTH methods)")
        out.append(f"  {'Baseline':<35s}  {'n':>6s}  {'median Δ':>10s} "
                   f"{'z':>8s}  {'p-value':>10s}  {'Sig?':>5s}")
        out.append(f"  {'-'*35}  {'-'*6}  {'-'*10} {'-'*8}  {'-'*10}  {'-'*5}")
        for tag, name, _ in BASELINES:
            if tag not in baselines: continue
            pa, pb, gs, sa, sb, common = _align(
                proposed["pair_predictions"],
                baselines[tag]["pair_predictions"], gt_map, gt_classes)
            w = wilcoxon_signed_rank(sa, sb)
            p_str = "<0.001" if w["p"] < 0.001 else f"{w['p']:.4f}"
            out.append(f"  {name:<35s}  {w['n']:>6d}  "
                       f"{w['median_diff']:>+10.4f} "
                       f"{w['z']:>8.3f}  {p_str:>10s}  "
                       f"{'Yes' if w['sig'] else 'No':>5s}")
        out.append("")

    # ── Friedman Test + Nemenyi (needs multiple "datasets") ────────
    out += [sep, "  4.3.3  FRIEDMAN TEST + NEMENYI POST-HOC", sep]
    out.append(f"  Stratifying the folder into sub-datasets (by GT-class hash)")
    subs = _stratify_into_subsets(filenames, k=3)
    for sname, fns in subs.items():
        out.append(f"    {sname}: {len(fns)} images")
    out.append("")
    f1_matrix: Dict[str, Dict[str, float]] = {}
    for sname, fns in subs.items():
        fileset = set(fns)
        row = {}
        if proposed:
            row["Proposed"] = _subset_pair_metrics(
                proposed["pair_predictions"], fileset,
                gt_map, gt_classes)["F"]
        for tag, name, _ in BASELINES:
            if tag not in baselines: continue
            row[name] = _subset_pair_metrics(
                baselines[tag]["pair_predictions"], fileset,
                gt_map, gt_classes)["F"]
        f1_matrix[sname] = row

    # Print F1 table per subset
    methods = sorted(next(iter(f1_matrix.values())).keys())
    hdr = f"    {'Subset':<12s}" + "".join(f"{m[:18]:>19s}" for m in methods)
    out.append(hdr)
    out.append("    " + "-"*len(hdr))
    for sname in f1_matrix:
        row = f"    {sname:<12s}"
        for m in methods:
            row += f"{f1_matrix[sname][m]:>18.2f}%"
        out.append(row)
    out.append("")
    out += friedman_test(f1_matrix)

    # ── Write files ────────────────────────────────────────────────
    report_path = os.path.join(args.out_dir, "comparison_report.txt")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out))

    summary = {
        "folder": args.folder,
        "baselines": {t: {k: v for k, v in b.items()
                          if k != "pair_predictions"}
                      for t, b in baselines.items()},
        "proposed": ({k: v for k, v in proposed.items()
                      if k != "pair_predictions"} if proposed else None),
        "friedman_f1_matrix": f1_matrix,
    }
    with open(os.path.join(args.out_dir, "comparison_report.json"),
              "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n✔ Comparison report → {report_path}")
    print(f"✔ Summary JSON     → {os.path.join(args.out_dir, 'comparison_report.json')}")


if __name__ == "__main__":
    main()