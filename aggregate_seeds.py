#!/usr/bin/env python
"""Mean +- std over seeds for runs with the same name in several result folders.

    python aggregate_seeds.py results/llama3_8b_s42/rtn_w3g128 results/llama3_8b_s1/rtn_w3g128 \
                              results/llama3_8b_s2/rtn_w3g128 [--csv out.csv]

Runs missing from some folders (e.g. deterministic `rtn`, `fp`) are aggregated over the folders
that contain them (n column).
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

METRICS = [
    ("ppl_wiki", lambda s: (s.get("ppl", {}).get("wikitext2") or {}).get("perplexity")),
    ("ppl_8k", lambda s: (s.get("ppl", {}).get("wikitext2_ctx8192") or {}).get("perplexity")),
    ("kl_before", lambda s: ((s.get("layers") or {}).get("diag_before") or {}).get("kl_full")),
    ("kl_after", lambda s: ((s.get("layers") or {}).get("diag_after") or {}).get("kl_full")),
    ("q_flips", lambda s: (s.get("layers") or {}).get("q_flips_total")),
    ("k_flips", lambda s: (s.get("layers") or {}).get("k_flips_total")),
    ("qk_time_s", lambda s: (s.get("layers") or {}).get("qk_time_s")),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    vals = defaultdict(lambda: defaultdict(list))
    order = []
    for d in args.dirs:
        for f in sorted(Path(d).glob("*/summary.json")):
            s = json.loads(f.read_text())
            name = s.get("config", {}).get("run_name", f.parent.name)
            if name not in order:
                order.append(name)
            for m, fn in METRICS:
                v = fn(s)
                if v is not None:
                    vals[name][m].append(float(v))

    rows = []
    for name in order:
        row = {"run": name, "n": max((len(v) for v in vals[name].values()), default=0)}
        for m, _ in METRICS:
            xs = vals[name].get(m, [])
            if not xs:
                row[m] = "-"
            elif len(xs) == 1:
                row[m] = f"{xs[0]:.4g}"
            else:
                row[m] = f"{np.mean(xs):.4g} ± {np.std(xs, ddof=1):.2g}"
        rows.append(row)

    cols = ["run", "n"] + [m for m, _ in METRICS]
    print("| " + " | ".join(cols) + " |")
    print("|" + "|".join(["---"] * len(cols)) + "|")
    for r in rows:
        print("| " + " | ".join(str(r[c]) for c in cols) + " |")
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
