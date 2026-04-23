"""
export_proposed_json.py
=======================
Companion script that runs your proposed pipeline
(`duplicate_detector_final.py`) on a folder of images and dumps the
side-car JSON in the exact schema expected by `run_all_baselines.py`.

This does NOT modify `duplicate_detector_final.py`.  It imports the
pipeline functions (`compute_fingerprint`, `fingerprint_similarity`,
`adaptive_screen_threshold`, `verify_pair`, `decide_duplicate`,
`UnionFind`, `extract_gt_label`, `_resolve_gt_classes`,
`compute_evaluation_metrics`, `get_paths`) and re-runs the SAME
screening / verification / grouping loop, plus writes:

    <out>.json   —  side-car for the orchestrator (--proposed-json)
    <out>.txt    —  (optional) your existing-format text report

USAGE
-----
    # adjust paths if your research code lives elsewhere
    python export_proposed_json.py  /path/to/images \
           --code /path/to/duplicate_detector_final.py \
           --json proposed.json  --report proposed_report.txt

Then feed the JSON straight into the orchestrator:

    python run_all_baselines.py  /path/to/images  --out-dir results \
           --proposed-json proposed.json --run-all
"""

from __future__ import annotations
import os, sys, json, time, argparse, math, random, importlib.util, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# ─── Tell the orchestrator which literature_baselines directory to use ─
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

# Import our shared statistics helpers (bootstrap CI, Cohen's κ) so the
# JSON we write matches the baseline side-cars exactly.
from eval_utils import (                                    # noqa: E402
    bootstrap_ci as _eval_bootstrap_ci,
    cohens_kappa as _eval_cohens_kappa,
    build_gt      as _eval_build_gt,
    PairPred      as _EvalPairPred,
    compute_pair_metrics  as _eval_pair_metrics,
    compute_group_metrics as _eval_group_metrics,
)


def _import_user_module(code_path: str):
    """Import `duplicate_detector_final.py` (or whatever you named it)
    from any path, without requiring it to be on PYTHONPATH."""
    code_path = os.path.abspath(code_path)
    if not os.path.isfile(code_path):
        raise FileNotFoundError(code_path)
    spec = importlib.util.spec_from_file_location("user_pipeline", code_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["user_pipeline"] = mod
    spec.loader.exec_module(mod)
    return mod


# ══════════════════════════════════════════════════════════════════════
# Core: replicate the screening + verification loop, recording EVERY
# pair's (score, pred) so we can dump them to JSON.
# ══════════════════════════════════════════════════════════════════════
def run_pipeline_and_collect(up, folder, threshold=0.50, max_dim=512,
                             base_screen=0.30, workers=0, use_gpu=False,
                             fast=False):
    """`up` is the imported user-pipeline module.  Returns a dict with
    `fps`, `cands`, `dups`, `rejected`, `groups`, `total_pairs`, `runtime`,
    plus `all_pairs`: list of (file_a, file_b, score, pred) for EVERY
    unordered pair — what the orchestrator needs."""

    if use_gpu and hasattr(up, "init_gpu"):
        up.init_gpu()

    nw = workers if workers > 0 else max(1, os.cpu_count() or 4)
    t_start = time.time()

    paths = up.get_paths(folder)
    n = len(paths)
    if n < 2:
        raise RuntimeError("Need >= 2 images")
    total_pairs = n * (n - 1) // 2

    # ── Fingerprints (threaded) ─────────────────────────────────────
    print(f"[1/4] {n} images ({total_pairs:,} pairs); fingerprinting ...")
    fps = [None] * n

    def fp_task(idx, path):
        fps[idx] = up.compute_fingerprint(path, max_dim)

    with ThreadPoolExecutor(max_workers=nw) as pool:
        for f in [pool.submit(fp_task, i, p) for i, p in enumerate(paths)]:
            f.result()
    fps = [f for f in fps if f is not None]

    # ── Screening (identical to your run()) ─────────────────────────
    print(f"[2/4] Screening (adaptive, base={base_screen}) ...")
    cands = []
    pair_screen_score = {}        # (i, j) -> screen_score, for pairs NOT in cands
    for i in range(len(fps)):
        for j in range(i+1, len(fps)):
            sc = up.fingerprint_similarity(fps[i], fps[j])
            pt = up.adaptive_screen_threshold(fps[i], fps[j], base_screen)
            if sc["screen_score"] >= pt:
                cands.append((i, j, sc))
            else:
                pair_screen_score[(i, j)] = sc["screen_score"]

    # ── Verification / decision ─────────────────────────────────────
    print(f"[3/4] Verifying {len(cands):,} candidates (threshold={threshold}) ...")
    dups, rejected = [], []
    uf = up.UnionFind()
    lock = threading.Lock()

    if fast:
        # same "fast mode" shortcut as in your run()
        for i, j, sc in cands:
            if sc["screen_score"] >= threshold:
                uf.union(fps[i].filename, fps[j].filename)
                dups.append({"file_a": fps[i].filename,
                             "file_b": fps[j].filename,
                             "confidence": sc["screen_score"],
                             "scores": sc, "orb_inliers": 0,
                             "flipped": False, "match_level": "fast"})
    else:
        def verify_task(idx):
            i, j, sc = cands[idx]
            detail = up.verify_pair(fps[i].path, fps[j].path,
                                    fps[i], fps[j], max_dim)
            dec = up.decide_duplicate(sc, detail, fps[i], fps[j], threshold)
            if dec["is_duplicate"]:
                with lock:
                    uf.union(fps[i].filename, fps[j].filename)
                    dups.append({"file_a": fps[i].filename,
                                 "file_b": fps[j].filename,
                                 "confidence": dec["confidence"],
                                 "scores": dec["scores"],
                                 "orb_inliers": dec["orb_inliers"],
                                 "flipped": dec["flipped"],
                                 "match_level": dec.get("match_level", "")})
            else:
                with lock:
                    rejected.append({"file_a": fps[i].filename,
                                     "file_b": fps[j].filename,
                                     "confidence": dec["confidence"],
                                     "scores": dec.get("scores", {}),
                                     "orb_inliers": dec.get("orb_inliers", 0)})

        with ThreadPoolExecutor(max_workers=nw) as pool:
            for f in [pool.submit(verify_task, idx) for idx in range(len(cands))]:
                f.result()

    groups = uf.groups()
    runtime = time.time() - t_start

    # ── Build the all-pairs list for the JSON (what the orchestrator needs) ─
    print("[4/4] Building per-pair prediction list ...")
    conf_map = {}    # (filename_a, filename_b) -> (score, pred)
    for d in dups:
        a, b = d["file_a"], d["file_b"]
        conf_map[tuple(sorted((a, b)))] = (float(d["confidence"]), 1)
    for r in rejected:
        a, b = r["file_a"], r["file_b"]
        conf_map[tuple(sorted((a, b)))] = (float(r["confidence"]), 0)
    # pairs that never passed screening → pred=0, score=screen_score
    for i in range(len(fps)):
        for j in range(i+1, len(fps)):
            key = tuple(sorted((fps[i].filename, fps[j].filename)))
            if key in conf_map:
                continue
            sc = pair_screen_score.get((i, j), 0.0)
            conf_map[key] = (float(sc), 0)

    all_pairs = [(a, b, s, p) for (a, b), (s, p) in conf_map.items()]

    print(f"  pairs scored={len(all_pairs):,}  dups={len(dups):,}  "
          f"rejected={len(rejected):,}  groups={len(groups)}  "
          f"runtime={runtime:.1f}s")

    return dict(fps=fps, cands=cands, dups=dups, rejected=rejected,
                groups=groups, total_pairs=total_pairs, runtime=runtime,
                all_pairs=all_pairs)


# ══════════════════════════════════════════════════════════════════════
# Build side-car JSON in the exact schema used by run_all_baselines.py
# ══════════════════════════════════════════════════════════════════════
def build_sidecar_dict(up, r: dict, bootstrap_B: int = 10_000) -> dict:
    fps    = r["fps"]
    groups = r["groups"]
    total_pairs = r["total_pairs"]

    # ── Ground truth (reuse eval_utils — same logic as your research code) ─
    filenames = [fp.filename for fp in fps]
    gt_map, gt_classes = _eval_build_gt(filenames)

    # ── Convert all_pairs → PairPred list ───────────────────────────
    preds = [_EvalPairPred(a, b, s, p) for a, b, s, p in r["all_pairs"]]

    pair_m = _eval_pair_metrics(preds, gt_map, gt_classes, total_pairs)
    grp_m  = _eval_group_metrics(groups, gt_map, gt_classes, len(fps))
    ci     = _eval_bootstrap_ci(preds, gt_map, gt_classes,
                                total_pairs, B=bootstrap_B)
    kap    = _eval_cohens_kappa(pair_m['tp'], pair_m['fp'],
                                pair_m['fn'], pair_m['tn'])

    side = {
        "method":  "Proposed (Hu + Hist + Hierarchical ORB + Warp-SSIM)",
        "paper":   "Ours",
        "runtime_sec": r["runtime"],
        "n_images":    len(fps),
        "total_pairs": total_pairs,
        "tp":         pair_m["tp"],
        "fp":         pair_m["fp"],
        "fn":         pair_m["fn"],
        "tn":         pair_m["tn"],
        "precision":  pair_m["precision"],
        "recall":     pair_m["recall"],
        "f1":         pair_m["f1"],
        "kappa":      kap["kappa"],
        "P_ci":       list(ci["P_ci"]),
        "R_ci":       list(ci["R_ci"]),
        "F_ci":       list(ci["F_ci"]),
        "n_groups":            grp_m["n_groups"],
        "purity":              grp_m["purity"],
        "completeness":        grp_m["completeness"],
        "pure_groups":         grp_m["pure"],
        "contaminated_groups": grp_m["contaminated"],
        "neg_leaked":          grp_m["neg_leaked"],
        "neg_total":           grp_m["neg_total"],
        "pair_predictions": [
            {"a": p.file_a, "b": p.file_b,
             "score": float(p.score), "pred": int(p.predicted)}
            for p in preds
        ],
    }
    return side


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("folder", help="Image folder (same one you pass to "
                    "duplicate_detector_final.py)")
    ap.add_argument("--code", required=True,
                    help="Path to your duplicate_detector_final.py")
    ap.add_argument("--json", required=True,
                    help="Where to write the side-car JSON for the orchestrator")
    ap.add_argument("--report", default=None,
                    help="Optional: also write the Section-4.2/4.3-style "
                         "text report here (same format as baselines)")
    ap.add_argument("--threshold",   type=float, default=0.50)
    ap.add_argument("--max-dim",     type=int,   default=512)
    ap.add_argument("--screen-thresh", type=float, default=0.30)
    ap.add_argument("--fast",        action="store_true")
    ap.add_argument("--workers",     type=int,   default=0)
    ap.add_argument("--gpu",         action="store_true")
    ap.add_argument("--bootstrap",   type=int,   default=10_000,
                    help="Bootstrap iterations for 95% CI (default 10k)")
    args = ap.parse_args()

    if not os.path.isdir(args.folder):
        print("Not a directory:", args.folder); sys.exit(1)

    print(f"Loading proposed-method pipeline from: {args.code}")
    up = _import_user_module(args.code)

    r = run_pipeline_and_collect(
        up, args.folder, args.threshold, args.max_dim, args.screen_thresh,
        args.workers, args.gpu, args.fast)

    side = build_sidecar_dict(up, r, bootstrap_B=args.bootstrap)
    with open(args.json, "w", encoding="utf-8") as fh:
        json.dump(side, fh, indent=2)
    print(f"\n✔ Side-car JSON written → {args.json}")
    print(f"    TP={side['tp']}  FP={side['fp']}  FN={side['fn']}")
    print(f"    P={side['precision']:.2f}%  R={side['recall']:.2f}%  "
          f"F1={side['f1']:.2f}%  κ={side['kappa']:.3f}")

    # Optional: also emit a unified text report (same format as baselines)
    if args.report:
        from eval_utils import write_baseline_report
        preds = [_EvalPairPred(a, b, s, p) for a, b, s, p in r["all_pairs"]]
        extra = [
            f"  Threshold              : {args.threshold}",
            f"  Screen (base)          : {args.screen_thresh}",
            f"  Max image dim          : {args.max_dim}",
            f"  Candidates screened    : {len(r['cands']):,}",
            f"  Verifications confirmed: {len(r['dups']):,}",
            f"  Verifications rejected : {len(r['rejected']):,}",
            f"  Fast mode              : {args.fast}",
        ]
        write_baseline_report(
            report_path=args.report,
            method_name="Proposed (ours) — duplicate_detector_final.py",
            paper_ref="Our research method",
            filenames=[fp.filename for fp in r["fps"]],
            preds=preds, groups=r["groups"],
            runtime_sec=r["runtime"], extra_lines=extra,
            bootstrap_B=args.bootstrap)
        print(f"✔ Text report written  → {args.report}")


if __name__ == "__main__":
    main()