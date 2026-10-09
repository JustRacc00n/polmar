#!/usr/bin/env python3
"""
PolMAR WACV — Paired Statistical Analysis
==========================================
Six pre-defined paired comparisons with moving-block bootstrap (Jitter) and
ordinary paired bootstrap (AAPM), two-sided Wilcoxon signed-rank tests, and
Holm–Bonferroni correction.

Outputs
-------
paired_statistical_results.csv  — full numerical results
paired_statistical_results.tex  — publication-ready LaTeX table
figures/diag_*.pdf              — one diagnostic figure per comparison
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import math

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════════════════════
# WILCOXON SIGNED-RANK TEST (scipy-free, normal approximation)
# ══════════════════════════════════════════════════════════════════════════════

def _norm_cdf(x: float) -> float:
    """CDF of the standard normal distribution via math.erfc."""
    return 0.5 * math.erfc(-x / math.sqrt(2))


def wilcoxon_two_sided(diffs: np.ndarray) -> tuple[float, float]:
    """
    Two-sided paired Wilcoxon signed-rank test.
    Uses normal approximation with continuity correction and tie correction.
    Zeros are dropped before ranking (wilcox method).
    Returns (W_statistic, p_value).
    """
    d = np.asarray(diffs, dtype=float)
    d = d[d != 0.0]
    n = len(d)
    if n == 0:
        return np.nan, 1.0

    abs_d = np.abs(d)
    # Rank |d| with average-rank tie resolution
    order = np.argsort(abs_d, kind="stable")
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i + 1
        while j < n and abs_d[order[j]] == abs_d[order[i]]:
            j += 1
        avg_rank = (i + j + 1) / 2.0   # 1-based average rank
        ranks[order[i:j]] = avg_rank
        i = j

    W_plus  = float(np.sum(ranks[d > 0]))
    W_minus = float(np.sum(ranks[d < 0]))
    W       = min(W_plus, W_minus)

    # Variance with tie correction
    _, counts = np.unique(abs_d, return_counts=True)
    tie_corr = float(np.sum(counts * (counts ** 2 - 1))) / 48.0
    var_W    = n * (n + 1) * (2 * n + 1) / 24.0 - tie_corr
    if var_W <= 0:
        return W, 1.0

    # Normal approximation with continuity correction
    z   = (W - n * (n + 1) / 4.0 + 0.5) / math.sqrt(var_W)
    p   = min(2.0 * _norm_cdf(z), 1.0)
    return W, p


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — edit paths here
# ══════════════════════════════════════════════════════════════════════════════

DATA_DIR = Path(__file__).parent          # directory that contains the CSVs

# Jitter test (scan04): six CSV paths
JITTER_BASE            = DATA_DIR / "metrics_refiner_scan04_base.csv"
JITTER_BASE_CIRCNET    = DATA_DIR / "metrics_refiner_scan04_base_circnet.csv"
JITTER_BASE_NMAR       = DATA_DIR / "metrics_refiner_scan04_base_nmar.csv"
JITTER_BASE_NMAR_SEG   = DATA_DIR / "metrics_refiner_scan04_base_nmar_seg.csv"

# AAPM body5: two CSV paths
AAPM_BASE              = DATA_DIR / "metrics_refiner_body5_base.csv"
AAPM_BASE_CIRCNET      = DATA_DIR / "metrics_refiner_body5_circnet.csv"

# Bootstrap parameters
N_BOOT        = 10_000
RANDOM_SEED   = 42
BLOCK_SIZE    = 16     # moving-block bootstrap block size for Jitter

# Output directory
OUT_DIR = DATA_DIR
FIG_DIR = OUT_DIR / "figures"

# ══════════════════════════════════════════════════════════════════════════════
# COMPARISON DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════
# Each entry: (label, ref_csv, imp_csv, metric_col, higher_is_better,
#              dataset_type, join_keys)
#
# dataset_type: "jitter" → moving-block bootstrap, keys=[scan_id, slice_id]
#               "aapm"   → ordinary bootstrap, key=[iid]

COMPARISONS = [
    # ── 1 & 2  Jitter Base vs Base+CircNet ────────────────────────────────────
    {
        "id":               "T1",
        "comparison_label": "Base vs Base+CircNet",
        "dataset":          "Jitter scan04",
        "ref_csv":          JITTER_BASE,
        "imp_csv":          JITTER_BASE_CIRCNET,
        "metric":           "mae_pred",
        "metric_label":     "MAE",
        "unit":             "HU",
        "higher_is_better": False,
        "dataset_type":     "jitter",
        "join_keys":        ["scan_id", "slice_id"],
    },
    {
        "id":               "T2",
        "comparison_label": "Base vs Base+CircNet",
        "dataset":          "Jitter scan04",
        "ref_csv":          JITTER_BASE,
        "imp_csv":          JITTER_BASE_CIRCNET,
        "metric":           "rmse_pred",
        "metric_label":     "RMSE",
        "unit":             "HU",
        "higher_is_better": False,
        "dataset_type":     "jitter",
        "join_keys":        ["scan_id", "slice_id"],
    },
    # ── 3 & 4  AAPM Base vs Base+CircNet ──────────────────────────────────────
    {
        "id":               "T3",
        "comparison_label": "Base vs Base+CircNet",
        "dataset":          "AAPM body5",
        "ref_csv":          AAPM_BASE,
        "imp_csv":          AAPM_BASE_CIRCNET,
        "metric":           "mae_pred",
        "metric_label":     "MAE",
        "unit":             "HU",
        "higher_is_better": False,
        "dataset_type":     "aapm",
        "join_keys":        ["iid"],
    },
    {
        "id":               "T4",
        "comparison_label": "Base vs Base+CircNet",
        "dataset":          "AAPM body5",
        "ref_csv":          AAPM_BASE,
        "imp_csv":          AAPM_BASE_CIRCNET,
        "metric":           "rmse_pred",
        "metric_label":     "RMSE",
        "unit":             "HU",
        "higher_is_better": False,
        "dataset_type":     "aapm",
        "join_keys":        ["iid"],
    },
    # ── 5 & 6  Jitter Base+NMAR vs Base+NMAR+SEG ─────────────────────────────
    {
        "id":               "T5",
        "comparison_label": "Base+NMAR vs Base+NMAR+SEG",
        "dataset":          "Jitter scan04",
        "ref_csv":          JITTER_BASE_NMAR,
        "imp_csv":          JITTER_BASE_NMAR_SEG,
        "metric":           "ssim_pred",
        "metric_label":     "SSIM",
        "unit":             "",
        "higher_is_better": True,
        "dataset_type":     "jitter",
        "join_keys":        ["scan_id", "slice_id"],
    },
    {
        "id":               "T6",
        "comparison_label": "Base+NMAR vs Base+NMAR+SEG",
        "dataset":          "Jitter scan04",
        "ref_csv":          JITTER_BASE_NMAR,
        "imp_csv":          JITTER_BASE_NMAR_SEG,
        "metric":           "grad_pred",
        "metric_label":     "Grad-MAE",
        "unit":             "",
        "higher_is_better": False,
        "dataset_type":     "jitter",
        "join_keys":        ["scan_id", "slice_id"],
    },
]

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_and_validate(path: Path, required_cols: list[str]) -> pd.DataFrame:
    """Load CSV, check required columns, and verify uniqueness of required keys."""
    if not path.exists():
        sys.exit(f"[ERROR] File not found: {path}")
    df = pd.read_csv(path)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        sys.exit(f"[ERROR] {path.name} is missing columns: {missing}\n"
                 f"  Available: {list(df.columns)}")
    # Check uniqueness of join keys (all required_cols except the metric)
    key_cols = required_cols[:-1]   # last element is metric
    if key_cols:
        n_dup = df.duplicated(subset=key_cols).sum()
        if n_dup:
            sys.exit(f"[ERROR] {path.name}: {n_dup} duplicate rows on keys {key_cols}")
    return df


def paired_merge(ref_df: pd.DataFrame, imp_df: pd.DataFrame,
                 join_keys: list[str], metric: str) -> tuple[pd.DataFrame, str]:
    """Strict 1-to-1 merge on join_keys, return merged df and info string."""
    ref_col = f"{metric}_ref"
    imp_col = f"{metric}_imp"
    merged = ref_df[join_keys + [metric]].rename(columns={metric: ref_col}).merge(
        imp_df[join_keys + [metric]].rename(columns={metric: imp_col}),
        on=join_keys, how="inner")

    n_ref = len(ref_df)
    n_imp = len(imp_df)
    n_merged = len(merged)
    info_parts = []
    if n_merged != n_ref:
        info_parts.append(
            f"[WARN] ref has {n_ref} rows, imp has {n_imp} rows, "
            f"inner-join produced {n_merged} rows ({n_ref - n_merged} unmatched in ref).")
    if n_merged == 0:
        sys.exit("[ERROR] Zero matched rows after merge — check join keys.")
    info_parts.append(f"Matched rows: {n_merged}")
    return merged, " | ".join(info_parts)


def compute_differences(merged: pd.DataFrame, metric: str,
                         higher_is_better: bool) -> np.ndarray:
    """Return per-slice differences so that positive == improved method is better."""
    ref = merged[f"{metric}_ref"].values
    imp = merged[f"{metric}_imp"].values
    if higher_is_better:
        return imp - ref       # improvement if imp > ref
    else:
        return ref - imp       # improvement if imp < ref


def drop_nonfinite(diffs: np.ndarray, ref_vals: np.ndarray,
                   imp_vals: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    mask = np.isfinite(diffs) & np.isfinite(ref_vals) & np.isfinite(imp_vals)
    n_excluded = int((~mask).sum())
    return diffs[mask], ref_vals[mask], imp_vals[mask], n_excluded


# ── Bootstrap ────────────────────────────────────────────────────────────────

def bootstrap_ci_ordinary(diffs: np.ndarray, n_boot: int, rng: np.random.Generator,
                           alpha: float = 0.05) -> tuple[float, float]:
    """Paired percentile bootstrap CI for mean of diffs (AAPM)."""
    n = len(diffs)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = diffs[idx].mean()
    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def bootstrap_ci_moving_block(diffs: np.ndarray, block_size: int,
                               n_boot: int, rng: np.random.Generator,
                               alpha: float = 0.05) -> tuple[float, float]:
    """
    Paired moving-block bootstrap CI (Jitter).
    Contiguous blocks of `block_size`; resample blocks with replacement until
    bootstrap sample >= n, then truncate to n. Use 2.5th/97.5th percentiles.
    """
    n = len(diffs)
    # All starting positions for blocks of length block_size
    starts = np.arange(0, n - block_size + 1)
    n_starts = len(starts)
    blocks_needed = int(np.ceil(n / block_size))

    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        chosen = rng.integers(0, n_starts, size=blocks_needed)
        sample = np.concatenate([diffs[starts[c]:starts[c] + block_size]
                                  for c in chosen])[:n]
        boot_means[i] = sample.mean()

    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


# ── Holm–Bonferroni ──────────────────────────────────────────────────────────

def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Return Holm–Bonferroni adjusted p-values (maintains order)."""
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [None] * n
    prev_adj = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj = p * (n - rank)
        adj = min(max(adj, prev_adj), 1.0)
        adjusted[orig_idx] = adj
        prev_adj = adj
    return adjusted


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def run_all() -> list[dict]:
    rng = np.random.default_rng(RANDOM_SEED)

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  PolMAR Statistical Analysis")
    print(f"  N_BOOT={N_BOOT}  BLOCK_SIZE={BLOCK_SIZE}  SEED={RANDOM_SEED}")
    print("=" * 70)

    # ── First pass: collect raw p-values ──────────────────────────────────────
    raw_results: list[dict] = []
    raw_p_values: list[float] = []

    for comp in COMPARISONS:
        tid   = comp["id"]
        label = comp["comparison_label"]
        ds    = comp["dataset"]
        metric = comp["metric"]
        mlabel = comp["metric_label"]
        unit   = comp["unit"]
        hib    = comp["higher_is_better"]
        dtype  = comp["dataset_type"]
        keys   = comp["join_keys"]

        print(f"\n{'─'*70}")
        print(f"  {tid}: {ds} | {label} | {mlabel}")
        print(f"{'─'*70}")

        required_ref = keys + [metric]
        required_imp = keys + [metric]

        ref_df = load_and_validate(comp["ref_csv"], required_ref)
        imp_df = load_and_validate(comp["imp_csv"], required_imp)

        merged, merge_info = paired_merge(ref_df, imp_df, keys, metric)
        print(f"  Merge: {merge_info}")

        diffs    = compute_differences(merged, metric, hib)
        ref_vals = merged[f"{metric}_ref"].values
        imp_vals = merged[f"{metric}_imp"].values

        diffs, ref_vals, imp_vals, n_excluded = drop_nonfinite(diffs, ref_vals, imp_vals)

        if n_excluded:
            print(f"  [WARN] Excluded {n_excluded} rows with non-finite values.")

        n = len(diffs)
        mean_ref  = float(np.mean(ref_vals))
        mean_imp  = float(np.mean(imp_vals))
        mean_diff = float(np.mean(diffs))
        med_diff  = float(np.median(diffs))
        frac_imp  = float(np.mean(diffs > 0))
        n_imp     = int(np.sum(diffs > 0))

        # Wilcoxon two-sided (normal approximation, zeros dropped)
        w_stat, p_raw = wilcoxon_two_sided(diffs)

        # Bootstrap CI
        if dtype == "jitter":
            ci_lo, ci_hi = bootstrap_ci_moving_block(
                diffs, BLOCK_SIZE, N_BOOT, rng)
            ci_method = f"moving-block (block={BLOCK_SIZE})"
        else:
            ci_lo, ci_hi = bootstrap_ci_ordinary(diffs, N_BOOT, rng)
            ci_method = "ordinary paired percentile"

        raw_p_values.append(p_raw)
        raw_results.append({
            "id":               tid,
            "comparison_label": label,
            "dataset":          ds,
            "metric_label":     mlabel,
            "unit":             unit,
            "higher_is_better": hib,
            "ci_method":        ci_method,
            "n":                n,
            "n_excluded":       n_excluded,
            "mean_ref":         mean_ref,
            "mean_imp":         mean_imp,
            "mean_diff":        mean_diff,
            "median_diff":      med_diff,
            "ci_lo":            ci_lo,
            "ci_hi":            ci_hi,
            "frac_improved":    frac_imp,
            "n_improved":       n_imp,
            "wilcoxon_stat":    float(w_stat),
            "p_raw":            float(p_raw),
            # placeholder — filled after Holm pass
            "p_holm":           None,
            "diffs":            diffs,   # keep for figures; removed before CSV
        })

    # ── Holm–Bonferroni ───────────────────────────────────────────────────────
    holm_adj = holm_bonferroni(raw_p_values)
    for res, p_adj in zip(raw_results, holm_adj):
        res["p_holm"] = float(p_adj)

    return raw_results


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

def print_results(results: list[dict]) -> None:
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    for r in results:
        sig = "***" if r["p_holm"] < 0.001 else ("**" if r["p_holm"] < 0.01
              else ("*" if r["p_holm"] < 0.05 else "ns"))
        unit = f" {r['unit']}" if r['unit'] else ""
        print(f"\n  [{r['id']}] {r['dataset']} | {r['comparison_label']} | {r['metric_label']}")
        print(f"    n = {r['n']}  (excluded: {r['n_excluded']})")
        print(f"    Mean ref = {r['mean_ref']:.4f}{unit}")
        print(f"    Mean imp = {r['mean_imp']:.4f}{unit}")
        print(f"    Mean improvement = {r['mean_diff']:.4f}{unit}  "
              f"[95% CI {r['ci_lo']:.4f}, {r['ci_hi']:.4f}]")
        print(f"    Median improvement = {r['median_diff']:.4f}{unit}")
        print(f"    Improved slices: {r['n_improved']}/{r['n']} ({r['frac_improved']*100:.1f}%)")
        print(f"    Wilcoxon W={r['wilcoxon_stat']:.0f}  p_raw={r['p_raw']:.2e}  "
              f"p_holm={r['p_holm']:.2e}  {sig}")
        print(f"    Bootstrap CI method: {r['ci_method']}")


def print_paper_snippets(results: list[dict]) -> None:
    print("\n" + "=" * 70)
    print("  COPY-PASTE SNIPPETS FOR THE PAPER")
    print("=" * 70)

    def fmt_p(p: float) -> str:
        if p < 0.001:
            return r"$p < 0.001$"
        elif p < 0.01:
            return f"$p = {p:.3f}$"
        else:
            return f"$p = {p:.2f}$"

    def fmt_ci(r: dict) -> str:
        unit = f"\\,\\text{{{r['unit']}}}" if r['unit'] else ""
        return (f"[{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]{unit}")

    # T1 & T2 — Jitter Base vs Base+CircNet
    t1 = next(r for r in results if r["id"] == "T1")
    t2 = next(r for r in results if r["id"] == "T2")
    print(f"""
[Jitter — Base vs Base+CircNet]
On the held-out Jitter volume (scan 4, {t1['n']} slices), adding the CircNet
polar-domain prior to the Base refiner reduced MAE by a mean of
{t1['mean_diff']:.2f}\\,HU (95\\% CI {fmt_ci(t1)}, {t1['n_improved']}/{t1['n']} slices improved,
Wilcoxon {fmt_p(t1['p_holm'])} after Holm--Bonferroni correction) and
RMSE by {t2['mean_diff']:.2f}\\,HU (95\\% CI {fmt_ci(t2)}, {t2['n_improved']}/{t2['n']} slices
improved, Wilcoxon {fmt_p(t2['p_holm'])}).
Confidence intervals are estimated with moving-block bootstrap (block size {BLOCK_SIZE})
to account for correlations between neighbouring slices.""")

    # T3 & T4 — AAPM Base vs Base+CircNet
    t3 = next(r for r in results if r["id"] == "T3")
    t4 = next(r for r in results if r["id"] == "T4")
    print(f"""
[AAPM body5 — Base vs Base+CircNet]
On the AAPM body phantom ({t3['n']} slices), Base+CircNet achieved a mean MAE
reduction of {t3['mean_diff']:.2f}\\,HU over the Base refiner (95\\% CI {fmt_ci(t3)},
{t3['n_improved']}/{t3['n']} slices improved, Wilcoxon {fmt_p(t3['p_holm'])}) and
a mean RMSE reduction of {t4['mean_diff']:.2f}\\,HU (95\\% CI {fmt_ci(t4)},
{t4['n_improved']}/{t4['n']} slices improved, Wilcoxon {fmt_p(t4['p_holm'])}).""")

    # T5 & T6 — Jitter NMAR vs NMAR+SEG
    t5 = next(r for r in results if r["id"] == "T5")
    t6 = next(r for r in results if r["id"] == "T6")
    print(f"""
[Jitter — Base+NMAR vs Base+NMAR+SEG]
Adding the semantic segmentation prior (SEG) to the NMAR configuration
improved SSIM by a mean of {t5['mean_diff']:.5f} (95\\% CI {fmt_ci(t5)},
{t5['n_improved']}/{t5['n']} slices improved, Wilcoxon {fmt_p(t5['p_holm'])}) and
reduced Grad-MAE by {t6['mean_diff']:.5f} (95\\% CI {fmt_ci(t6)},
{t6['n_improved']}/{t6['n']} slices improved, Wilcoxon {fmt_p(t6['p_holm'])}).
Confidence intervals use moving-block bootstrap (block size {BLOCK_SIZE}).""")

    # Dagger recommendations
    print("\n" + "=" * 70)
    print("  DAGGER (†) RECOMMENDATIONS")
    print("  Mark with ^\\dagger where Holm–Bonferroni p < 0.05")
    print("=" * 70)
    any_sig = False
    for r in results:
        if r["p_holm"] < 0.05:
            any_sig = True
            direction = "decrease" if not r["higher_is_better"] else "increase"
            print(f"  [{r['id']}] {r['dataset']} | {r['comparison_label']} | "
                  f"{r['metric_label']}  →  †  (p_holm={r['p_holm']:.3e}, "
                  f"mean {direction} {abs(r['mean_diff']):.3f} {r['unit']})")
    if not any_sig:
        print("  No comparison reached p_holm < 0.05 — do not place any † markers.")


def save_csv(results: list[dict]) -> None:
    rows = []
    for r in results:
        rows.append({
            "id":                   r["id"],
            "dataset":              r["dataset"],
            "comparison":           r["comparison_label"],
            "metric":               r["metric_label"],
            "unit":                 r["unit"],
            "n":                    r["n"],
            "n_excluded":           r["n_excluded"],
            "mean_ref":             r["mean_ref"],
            "mean_imp":             r["mean_imp"],
            "mean_improvement":     r["mean_diff"],
            "median_improvement":   r["median_diff"],
            "ci_lo_95":             r["ci_lo"],
            "ci_hi_95":             r["ci_hi"],
            "ci_method":            r["ci_method"],
            "n_improved":           r["n_improved"],
            "frac_improved":        r["frac_improved"],
            "wilcoxon_stat":        r["wilcoxon_stat"],
            "p_raw":                r["p_raw"],
            "p_holm":               r["p_holm"],
        })
    df = pd.DataFrame(rows)
    out = OUT_DIR / "paired_statistical_results.csv"
    df.to_csv(out, index=False)
    print(f"\n  [CSV] Saved: {out}")


def save_latex(results: list[dict]) -> None:
    def fmt_p_tex(p: float) -> str:
        if p < 0.001:
            return r"$<\!0.001$"
        elif p < 0.01:
            return f"${p:.3f}$"
        else:
            return f"${p:.2f}$"

    def fmt_val(v: float, metric: str, unit: str) -> str:
        if metric in ("SSIM", "Grad-MAE"):
            return f"{v:.4f}"
        return f"{v:.2f}"

    def fmt_improvement(r: dict) -> str:
        v = r["mean_diff"]
        if r["metric_label"] in ("SSIM", "Grad-MAE"):
            s = f"{v:+.4f}"
        else:
            s = f"{v:+.2f}\\,\\text{{HU}}"
        return s

    def fmt_ci_tex(r: dict) -> str:
        lo, hi = r["ci_lo"], r["ci_hi"]
        if r["metric_label"] in ("SSIM", "Grad-MAE"):
            return f"[{lo:.4f},\\,{hi:.4f}]"
        return f"[{lo:.2f},\\,{hi:.2f}]"

    lines = []
    lines.append(r"% PolMAR — Paired statistical analysis results")
    lines.append(r"% Auto-generated by statistical_analysis.py")
    lines.append(r"\begin{table}[tbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Paired statistical analysis of pre-defined model comparisons.")
    lines.append(r"  Mean improvement and 95\% bootstrap CI are shown.")
    lines.append(r"  $\dagger$~denotes $p_\text{Holm} < 0.05$.}")
    lines.append(r"\label{tab:statistical}")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\begin{tabular}{lllrrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Comparison & Dataset & Metric & $n$ & Mean impr. & 95\,\% CI"
                 r" & Impr.\ slices & $p_\text{Wilcoxon}$ & $p_\text{Holm}$ \\")
    lines.append(r"\midrule")

    prev_comp = None
    for r in results:
        comp_key = (r["comparison_label"], r["dataset"])
        show_comp = (comp_key != prev_comp)
        prev_comp = comp_key

        comp_str = r["comparison_label"].replace("+", r"\texttt{+}") if show_comp else ""
        ds_str   = r["dataset"] if show_comp else ""

        sig = r"$^\dagger$" if r["p_holm"] < 0.05 else ""
        improved_str = f"{r['n_improved']}/{r['n']} ({r['frac_improved']*100:.0f}\\%)"

        row = " & ".join([
            comp_str,
            ds_str,
            r["metric_label"] + sig,
            str(r["n"]),
            fmt_improvement(r),
            fmt_ci_tex(r),
            improved_str,
            fmt_p_tex(r["p_raw"]),
            fmt_p_tex(r["p_holm"]),
        ])
        lines.append(row + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    # Add footnote about CI methods
    lines.append(r"\begin{tablenotes}")
    lines.append(r"\small")
    lines.append(r"\item Jitter CI: moving-block bootstrap"
                 rf" (block={BLOCK_SIZE}, $B={N_BOOT:,}$, seed~{RANDOM_SEED}).")
    lines.append(r"\item AAPM CI: ordinary paired percentile bootstrap"
                 rf" ($B={N_BOOT:,}$, seed~{RANDOM_SEED}).")
    lines.append(r"\item $p$-values: two-sided Wilcoxon signed-rank test, Holm--Bonferroni corrected.")
    lines.append(r"\end{tablenotes}")
    lines.append(r"\end{table}")

    out = OUT_DIR / "paired_statistical_results.tex"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [LaTeX] Saved: {out}")


def save_figures(results: list[dict]) -> None:
    """One PDF per test showing distribution of per-slice paired differences."""
    COLORS = {
        "Jitter scan04": "#4c8cef",
        "AAPM body5":    "#f5a623",
    }

    for r in results:
        diffs = r["diffs"]
        color = COLORS.get(r["dataset"], "#888888")

        fig, ax = plt.subplots(figsize=(5.5, 3.5))

        ax.hist(diffs, bins=40, color=color, alpha=0.75, edgecolor="white",
                linewidth=0.4, label="Per-slice diff")
        ax.axvline(0,                 color="#333333", lw=1.0, ls="--", label="zero")
        ax.axvline(r["mean_diff"],    color="#d62728", lw=1.5, label=f"mean={r['mean_diff']:.3f}")
        ax.axvline(r["median_diff"],  color="#2ca02c", lw=1.5, ls=":", label=f"median={r['median_diff']:.3f}")
        ax.axvspan(r["ci_lo"], r["ci_hi"], alpha=0.15, color="#d62728", label="95% CI")

        direction = "decrease" if not r["higher_is_better"] else "increase"
        unit_str = f" [{r['unit']}]" if r["unit"] else ""
        ax.set_xlabel(f"Paired {direction} in {r['metric_label']}{unit_str}\n"
                      f"(positive = {r['comparison_label'].split('vs')[-1].strip()} is better)",
                      fontsize=9)
        ax.set_ylabel("Slice count", fontsize=9)
        ax.set_title(
            f"{r['id']}: {r['dataset']}  |  {r['comparison_label']}  |  {r['metric_label']}\n"
            f"n={r['n']}  p_raw={r['p_raw']:.2e}  p_holm={r['p_holm']:.2e}",
            fontsize=9)
        ax.legend(fontsize=7.5, loc="upper left")
        ax.tick_params(labelsize=8)
        fig.tight_layout()

        fname = FIG_DIR / f"diag_{r['id']}_{r['dataset'].replace(' ', '_')}_{r['metric_label']}.pdf"
        fig.savefig(fname, format="pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"  [FIG] Saved: {fname}")

    # ── Combined overview figure ──────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes = axes.flatten()
    for ax, r in zip(axes, results):
        color = COLORS.get(r["dataset"], "#888888")
        diffs = r["diffs"]
        ax.hist(diffs, bins=35, color=color, alpha=0.70, edgecolor="white", linewidth=0.3)
        ax.axvline(0,                color="#333333", lw=0.8, ls="--")
        ax.axvline(r["mean_diff"],   color="#d62728", lw=1.2)
        ax.axvspan(r["ci_lo"], r["ci_hi"], alpha=0.15, color="#d62728")
        sig_str = "†" if r["p_holm"] < 0.05 else "ns"
        ax.set_title(f"{r['id']}: {r['comparison_label']}\n"
                     f"{r['dataset']} | {r['metric_label']}  {sig_str}",
                     fontsize=7.5)
        ax.set_xlabel("Improvement (positive = better)", fontsize=7)
        ax.tick_params(labelsize=7)

    fig.suptitle("Paired per-slice improvement distributions — PolMAR", fontsize=10)
    fig.tight_layout()
    combined = FIG_DIR / "diag_overview_all.pdf"
    fig.savefig(combined, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  [FIG] Combined overview: {combined}")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    results = run_all()
    print_results(results)
    print_paper_snippets(results)
    save_csv(results)
    save_latex(results)
    save_figures(results)
    print("\n  Done.\n")
