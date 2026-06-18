#!/usr/bin/env python3
"""Multiple-comparisons correction across the full 150-run sweep.

Reads all .results.jsonl files under sweeps/artifacts, extracts the
permutation p-value for each successful run, and applies Bonferroni
and Benjamini-Hochberg (FDR) corrections.

Prints a summary table sorted by raw p-value, and reports corrected
p-values for the headline (hybrid, ws=6, ik=1) and best SVM runs.
"""

import json
import glob
import sys
import numpy as np
from pathlib import Path
from collections import OrderedDict


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction. Returns q-values."""
    n = len(pvals)
    order = np.argsort(pvals)
    ranks = np.arange(1, n + 1)
    qvals = np.minimum(1.0, pvals[order] * n / ranks)
    # Ensure monotonicity
    for i in range(n - 2, -1, -1):
        qvals[i] = min(qvals[i], qvals[i + 1])
    # Restore original order
    result = np.empty(n)
    result[order] = qvals
    return result


def main() -> None:
    artifacts_dir = Path(__file__).resolve().parent.parent / "sweeps" / "artifacts"
    jsonl_files = sorted(glob.glob(str(artifacts_dir / "*.results.jsonl")))

    if not jsonl_files:
        print("ERROR: No .results.jsonl files found in sweeps/artifacts/")
        sys.exit(1)

    records = []
    for fp in jsonl_files:
        with open(fp) as f:
            last = None
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("status") == "success":
                        last = rec
                except json.JSONDecodeError:
                    continue
        if last is None:
            continue

        m = last.get("mlflow_metrics", {})
        pval = m.get("patient_roc_auc_permutation_pvalue")
        roc = m.get("patient_roc_auc")
        if pval is None or roc is None:
            continue

        records.append(
            {
                "model": last.get("model", "?"),
                "window": int(last.get("window_seconds", 0)),
                "inner_k": int(last.get("inner_k", 0)),
                "roc_auc": float(roc),
                "p_raw": float(pval),
                "file": Path(fp).name,
            }
        )

    print(f"Found {len(records)} successful runs with permutation p-values.\n")

    # Sort by raw p-value
    records.sort(key=lambda r: r["p_raw"])

    # Compute corrections
    pvals = np.array([r["p_raw"] for r in records])
    n_tests = len(pvals)
    qvals_fdr = _bh_fdr(pvals)

    for i, r in enumerate(records):
        r["p_bonf"] = min(1.0, r["p_raw"] * n_tests)
        r["q_fdr"] = float(qvals_fdr[i])

    # ---- Find key runs ----
    headline = None
    best_svm = None
    for r in records:
        if r["model"] == "advanced_hybrid_1dcnn_lstm" and r["window"] == 6 and r["inner_k"] == 1:
            headline = r
        if r["model"] == "svm_linear" and r["roc_auc"] >= 0.5:
            if best_svm is None or r["roc_auc"] > best_svm["roc_auc"]:
                best_svm = r

    # ---- Summary stats ----
    sig_bonf = [r for r in records if r["p_bonf"] < 0.05]
    sig_fdr = [r for r in records if r["q_fdr"] < 0.05]

    print(f"Total tests: {n_tests}")
    print(f"Significant at Bonferroni (p_corr < 0.05): {len(sig_bonf)}")
    print(f"Significant at BH-FDR (q < 0.05):          {len(sig_fdr)}")
    print()

    # ---- Key results ----
    if headline:
        print("=" * 70)
        print("HEADLINE RESULT: hybrid_1dcnn_lstm, ws=6s, inner-k=1")
        print(f"  Raw p-value:       {headline['p_raw']:.6f}")
        print(f"  Bonferroni-corrected p:  {headline['p_bonf']:.6f}  {'✅ SIG' if headline['p_bonf'] < 0.05 else '❌ NS'}")
        print(f"  BH-FDR q-value:          {headline['q_fdr']:.6f}  {'✅ SIG' if headline['q_fdr'] < 0.05 else '❌ NS'}")
        print(f"  ROC-AUC:           {headline['roc_auc']:.4f}")
        print()

    if best_svm:
        print("=" * 70)
        print("BEST SVM: svm_linear, ws=8s, inner-k=30 (from paper)")
        print(f"  Raw p-value:       {best_svm['p_raw']:.6f}")
        print(f"  Bonferroni-corrected p:  {best_svm['p_bonf']:.6f}  {'✅ SIG' if best_svm['p_bonf'] < 0.05 else '❌ NS'}")
        print(f"  BH-FDR q-value:          {best_svm['q_fdr']:.6f}  {'✅ SIG' if best_svm['q_fdr'] < 0.05 else '❌ NS'}")
        print(f"  ROC-AUC:           {best_svm['roc_auc']:.4f}")
        print()

    # ---- Top-15 table ----
    print("=" * 70)
    print(f"{'Rank':<5} {'Model':<28} {'ws':<4} {'ik':<4} {'ROC-AUC':<8} {'p_raw':<10} {'p_bonf':<10} {'q_fdr':<10}")
    print("-" * 70)
    for i, r in enumerate(records[:15], 1):
        sig = "✅" if r["q_fdr"] < 0.05 else ("⚠️" if r["p_bonf"] < 0.05 else "")
        model_short = r["model"].replace("advanced_hybrid_1dcnn_lstm", "hybrid").replace("svm_linear", "svm")
        print(
            f"{i:<5} {model_short:<28} {r['window']:<4} {r['inner_k']:<4} "
            f"{r['roc_auc']:<8.4f} {r['p_raw']:<10.6f} {r['p_bonf']:<10.6f} {r['q_fdr']:<10.6f} {sig}"
        )

    # ---- Distribution summary ----
    print()
    print("=" * 70)
    print("Distribution of raw p-values:")
    bins = [0.0, 0.001, 0.01, 0.05, 0.10, 0.25, 0.50, 1.01]
    for lo, hi in zip(bins[:-1], bins[1:]):
        count = sum(1 for r in records if lo <= r["p_raw"] < hi)
        pct = count / n_tests * 100
        bar = "█" * int(pct / 2)
        print(f"  [{lo:.3f}, {hi:.3f}): {count:4d} ({pct:5.1f}%) {bar}")

    # For the paper text:
    # Bonferroni: multiply raw p by 150
    # FDR: Benjamini-Hochberg
    print()
    if headline:
        bonf_str = "significant" if headline["p_bonf"] < 0.05 else "not significant"
        fdr_str = "significant" if headline["q_fdr"] < 0.05 else "not significant"
        print(f"SUGGESTED TEXT: After Bonferroni correction for {n_tests}-fold multiple comparisons "
              f"(corrected p = {headline['p_bonf']:.4f}), the result is {bonf_str}. "
              f"Using Benjamini-Hochberg FDR (q = {headline['q_fdr']:.4f}), the result is {fdr_str}.")


if __name__ == "__main__":
    main()
