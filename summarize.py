#!/usr/bin/env python
"""Collect results/<dir>/*/summary.json (+ layers.json) into one table.

    python summarize.py --results-dir results/rtn/Meta-Llama-3-8B_w3g128 [--csv out.csv]

Columns: PPL, mean attention KL over layers (before/after QK), MQ/QF split, flips, time,
and the Spearman / Pearson correlation between predicted and exact acceptance dJ.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _rank(a):
    order = np.argsort(a, kind="mergesort")
    r = np.empty(len(a), dtype=float)
    r[order] = np.arange(len(a), dtype=float)
    return r


def _corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None, None
    pear = float(np.corrcoef(x, y)[0, 1])
    spear = float(np.corrcoef(_rank(x), _rank(y))[0, 1])
    return pear, spear


def _fmt(v, nd=4):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}g}" if abs(v) < 1e-2 or abs(v) >= 1e4 else f"{v:.{nd}f}"
    return str(v)


def collect(results_dir: Path):
    rows = []
    for sfile in sorted(results_dir.glob("*/summary.json")):
        s = json.loads(sfile.read_text())
        c = s.get("config", {})
        lay = s.get("layers", {})
        row = {
            "run": c.get("run_name", sfile.parent.name),
            "base": c.get("base"),
            "bits": c.get("bits") if c.get("base") != "none" else "-",
            "clc": c.get("clc_targets"),
            "qk": c.get("qk_correct"),
            "kl_k_after": (lay.get("diag_after") or {}).get("kl_k_only"),
            "accept": c.get("qk_accept") if c.get("qk_correct") != "none" else "-",
            "strata": c.get("qk_strata") if c.get("qk_correct") != "none" else "-",
            "ppl_wiki": (s.get("ppl", {}).get("wikitext2") or {}).get("perplexity"),
            "ppl_c4": (s.get("ppl", {}).get("c4") or {}).get("perplexity"),
            "kl_before": (lay.get("diag_before") or {}).get("kl_full"),
            "kl_after": (lay.get("diag_after") or {}).get("kl_full"),
            "kl_q_only": (lay.get("diag_before") or {}).get("kl_q_only"),
            "kl_k_only": (lay.get("diag_before") or {}).get("kl_k_only"),
            "kl_mq": (lay.get("diag_before") or {}).get("kl_mq"),
            "kl_qf": (lay.get("diag_before") or {}).get("kl_qf"),
            "sched": c.get("qk_schedule", "Q") if c.get("qk_correct") != "none" else "-",
            "q_flips": lay.get("q_flips_total", lay.get("qk_flips_total")),
            "k_flips": lay.get("k_flips_total"),
            "clc_flips": lay.get("clc_flips_total"),
            "J_acc": None,
            "qk_time_s": lay.get("qk_time_s"),
            "qk_mem_gb": lay.get("qk_peak_mem_gb"),
            "pearson": None,
            "spearman": None,
            "audit_pearson": None,
        }
        if lay.get("J_acc_before_sum") is not None:
            row["J_acc"] = f"{lay['J_acc_before_sum']:.3e}->{lay['J_acc_after_sum']:.3e}"
        lfile = sfile.parent / "layers.json"
        if lfile.exists() and c.get("qk_correct") not in (None, "none"):
            layers = json.loads(lfile.read_text())
            pred, acc, ap, ae = [], [], [], []
            for l in layers:
                h = (l.get("qk") or {}).get("history") or {}
                if h.get("acc"):
                    pred += h.get("pred", [])
                    acc += h.get("acc", [])
                ap += h.get("audit_pred", [])
                ae += h.get("audit_exact_opt", [])
            row["pearson"], row["spearman"] = _corr(pred, acc)
            row["audit_pearson"], _ = _corr(ap, ae)
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    rows = collect(Path(args.results_dir))
    if not rows:
        print("no summary.json found")
        return
    cols = list(rows[0].keys())
    print("| " + " | ".join(cols) + " |")
    print("|" + "|".join(["---"] * len(cols)) + "|")
    for r in rows:
        print("| " + " | ".join(_fmt(r[k]) for k in cols) + " |")
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
