#!/usr/bin/env python
"""Collect results/<dir>/*/summary.json (+ layers.json) into one table.

    python summarize.py --results-dir results/rtn/Llama-3.2-1B_w3g128 [--csv out.csv] [--runs a,b,c]

Predictor quality is reported per pass type (Q / K), computed per layer and then
summarised by the median over layers (pooling layers inflates Pearson because the
dJ scale differs a lot between layers):
    q_sp / k_sp     median per-layer Spearman(pred dJ, exact acceptance dJ)
    q_acc / k_acc   fraction of proposed head/group batches that were accepted
    q_bs / k_bs     mean proposed batch size (flips per head / per KV group)
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
    if len(x) < 5 or np.std(x) == 0 or np.std(y) == 0:
        return None, None
    return float(np.corrcoef(x, y)[0, 1]), float(np.corrcoef(_rank(x), _rank(y))[0, 1])


def _fmt(v, nd=4):
    if v is None:
        return "-"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        if v == 0:
            return "0"
        return f"{v:.{nd}g}" if abs(v) < 1e-2 or abs(v) >= 1e4 else f"{v:.{nd}f}"
    return str(v)


def _layer_passes(layer: dict, schedule: str):
    """Yield (pass_type, history) for one layer; falls back to merged history for old runs."""
    passes = layer.get("qk_passes") or []
    if passes and all("history" in p for p in passes):
        for p in passes:
            yield p.get("pass", "Q"), p["history"]
        return
    h = (layer.get("qk") or {}).get("history")
    if h and schedule in ("Q", "K"):
        yield schedule, h


def predictor_stats(layers: list, schedule: str) -> dict:
    out = {}
    for kind in ("Q", "K"):
        sps, n_acc, n_tot, bs = [], 0, 0, []
        for l in layers:
            for k, h in _layer_passes(l, schedule):
                if k != kind:
                    continue
                acc_flags = h.get("accepted", [])
                n_acc += int(sum(bool(a) for a in acc_flags))
                n_tot += len(acc_flags)
                bs += h.get("nsel", [])
                if h.get("acc") and len(h["acc"]) == len(h.get("pred", [])):
                    _, sp = _corr(h["pred"], h["acc"])
                    if sp is not None:
                        sps.append(sp)
        key = kind.lower()
        out[f"{key}_sp"] = float(np.median(sps)) if sps else None
        out[f"{key}_acc"] = (n_acc / n_tot) if n_tot else None
        out[f"{key}_bs"] = float(np.mean(bs)) if bs else None
    return out


LM_NORM_TASKS = {"arc_challenge", "hellaswag", "openbookqa"}  # report acc_norm for these, acc otherwise


def lm_scores(s: dict) -> dict:
    out = {}
    for task, m in (s.get("lm_eval") or {}).items():
        key = "acc_norm,none" if task in LM_NORM_TASKS and "acc_norm,none" in m else "acc,none"
        if key in m:
            out[task] = float(m[key])
    return out


def collect(results_dir: Path, only=None):
    rows = []
    for sfile in sorted(results_dir.glob("*/summary.json")):
        s = json.loads(sfile.read_text())
        c = s.get("config", {})
        name = c.get("run_name", sfile.parent.name)
        if only and name not in only:
            continue
        lay = s.get("layers", {})
        qk_on = c.get("qk_correct", "none") != "none"
        sched = c.get("qk_schedule", "Q") if qk_on else "-"
        db = lay.get("diag_before") or {}
        da = lay.get("diag_after") or {}
        row = {
            "run": name,
            "base": c.get("base"),
            "bits": c.get("bits") if c.get("base") != "none" else "-",
            "clc": c.get("clc_targets"),
            "qk": c.get("qk_correct"),
            "sched": sched,
            "ppl_wiki": (s.get("ppl", {}).get("wikitext2") or {}).get("perplexity"),
            "ppl_8k": (s.get("ppl", {}).get("wikitext2_ctx8192") or {}).get("perplexity"),
            "ppl_c4": (s.get("ppl", {}).get("c4") or {}).get("perplexity"),
            "kl_before": db.get("kl_full"),
            "kl_after": da.get("kl_full"),
            "kl_k_before": db.get("kl_k_only"),
            "kl_k_after": da.get("kl_k_only"),
            "kl_mq": db.get("kl_mq"),
            "kl_qf": db.get("kl_qf"),
            "q_flips": lay.get("q_flips_total", lay.get("qk_flips_total")) if qk_on else None,
            "k_flips": lay.get("k_flips_total") if qk_on else None,
            "clc_flips": lay.get("clc_flips_total"),
            "J_acc": None,
            "qk_time_s": lay.get("qk_time_s"),
            "q_sp": None, "q_acc": None, "q_bs": None,
            "k_sp": None, "k_acc": None, "k_bs": None,
            "audit_pearson": None,
        }
        if lay.get("J_acc_before_sum") is not None:
            row["J_acc"] = f"{lay['J_acc_before_sum']:.3e}->{lay['J_acc_after_sum']:.3e}"
        lfile = sfile.parent / "layers.json"
        if qk_on and lfile.exists():
            layers = json.loads(lfile.read_text())
            row.update(predictor_stats(layers, sched))
            ap, ae = [], []
            for l in layers:
                h = (l.get("qk") or {}).get("history") or {}
                ap += h.get("audit_pred", [])
                ae += h.get("audit_exact_opt", [])
            row["audit_pearson"], _ = _corr(ap, ae)
        row["_lm"] = lm_scores(s)
        rows.append(row)
    return rows


def add_lm_columns(rows: list):
    tasks = sorted({t for r in rows for t in r["_lm"]})
    for r in rows:
        lm = r.pop("_lm")
        for t in tasks:
            r[f"lm_{t}"] = lm.get(t)
        vals = [lm[t] for t in tasks if t in lm]
        r["lm_avg"] = (sum(vals) / len(vals)) if tasks and len(vals) == len(tasks) else None
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True, nargs="+", help="one or more result folders")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--runs", default=None, help="comma-separated run names to include")
    ap.add_argument("--lm-only", action="store_true", help="only run, bits, PPL and lm-eval columns")
    args = ap.parse_args()
    only = set(args.runs.split(",")) if args.runs else None
    rows = []
    for d in args.results_dir:
        rows += collect(Path(d), only)
    if not rows:
        print("no summary.json found")
        return
    tasks = add_lm_columns(rows)
    if args.lm_only:
        keep = ["run", "base", "bits", "clc", "qk", "sched", "ppl_wiki", "ppl_8k"] + [f"lm_{t}" for t in tasks] + ["lm_avg"]
        rows = [{k: r[k] for k in keep} for r in rows]
    elif not tasks:
        for r in rows:
            r.pop("lm_avg", None)
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