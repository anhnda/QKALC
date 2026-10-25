#!/usr/bin/env python
"""Entry point: base PTQ (RTN / AWQ) [+ CLC] [+ attention-aware QK lattice correction] + evaluation.

Examples
--------
FP reference:
    python run_qk.py --model-path meta-llama/Meta-Llama-3-8B --base none
RTN W3 g128 + QK correction, with diagnostics:
    python run_qk.py --model-path meta-llama/Meta-Llama-3-8B --base rtn --bits 3 --qk-correct full --diag
"""

from __future__ import annotations

import argparse
import os
import random
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.calibration import get_wikitext2_calibration_data, load_calibration_data
from src.io_utils import dump_json
from src.pipeline.sequential import PipelineConfig, SequentialPipeline, parse_layer_spec
from src.post_correction.clc import CLCConfig, CLCCorrection
from src.post_correction.k_alc import KALCConfig, KLatticeCorrector
from src.post_correction.qk_alc import QKALCConfig, QKLatticeCorrector


# ----------------------------------------------------------------------------- env / io
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_runtime_env(env_path=".env"):
    p = Path(env_path)
    if not p.exists():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(p, override=False)
        return
    except ImportError:
        pass
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#") and "=" in s:
            k, v = s.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


def resolve_hf_token():
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")


def resolve_model_reference(model_ref: str, models_root: str) -> str:
    p = Path(model_ref).expanduser()
    if p.exists():
        return str(p)
    q = Path(models_root).expanduser() / model_ref
    if q.exists():
        return str(q)
    return model_ref


def model_slug(ref: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(ref).rstrip("/").split("/")[-1])


# ----------------------------------------------------------------------------- args
def build_parser():
    ap = argparse.ArgumentParser(description="RTN/AWQ + CLC + attention-aware QK lattice correction")
    g = ap.add_argument_group("model")
    g.add_argument("--model-path", required=True)
    g.add_argument("--models-root", default="/models")
    g.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    g.add_argument("--device", default="cuda")
    g.add_argument("--trust-remote-code", action="store_true")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--allow-tf32", action="store_true", help="faster GEMMs in the correction (lower precision)")

    g = ap.add_argument_group("base quantizer")
    g.add_argument("--base", choices=["none", "rtn", "awq"], default="rtn")
    g.add_argument("--bits", type=int, default=4)
    g.add_argument("--group-size", type=int, default=128)
    g.add_argument("--awq-n-grid", type=int, default=20)
    g.add_argument("--awq-max-rows", type=int, default=2048)

    g = ap.add_argument_group("CLC (original repo correction)")
    g.add_argument("--clc-targets", choices=["none", "q", "all", "all_but_q"], default="none")
    g.add_argument("--clc-knee-tolerance", type=float, default=0.0)
    g.add_argument("--clc-max-flip-percent", type=float, default=0.05)
    g.add_argument("--no-james-stein", action="store_true")

    g = ap.add_argument_group("QK lattice correction (Q-pass)")
    g.add_argument("--qk-correct", choices=["none", "full", "meanquery"], default="none")
    g.add_argument("--qk-accept", choices=["exact", "none"], default="exact")
    g.add_argument("--qk-layers", default="all", help='e.g. "all", "0-7,31"')
    g.add_argument("--qk-schedule", default="Q", help='passes in order: "Q", "K", "QK", "QKQ", ...')
    g.add_argument("--qk-k-hsub-queries", type=int, default=32,
                   help="K-pass: queries per seq used for diag H (both moments on the same subset); 0 = all")
    g.add_argument("--qk-k-refresh-every", type=int, default=5, help="K-pass: exact G refresh period")
    g.add_argument("--qk-k-batch-init", type=int, default=16, help="K-pass: initial flips per KV group per round")
    g.add_argument("--qk-k-batch-max", type=int, default=64, help="K-pass: max flips per KV group per round")
    g.add_argument("--qk-k-row-cap", type=int, default=1, help="K-pass: max flips per k_proj row per round")
    g.add_argument("--qk-k-max-rounds", type=int, default=None, help="K-pass rounds (default: --qk-max-rounds)")
    g.add_argument("--qk-max-flip-percent", type=float, default=0.05)
    g.add_argument("--qk-opt-seqs", type=int, default=64)
    g.add_argument("--qk-accept-seqs", type=int, default=16)
    g.add_argument("--qk-queries-per-seq", type=int, default=256)
    g.add_argument("--qk-cov-queries-per-seq", type=int, default=32)
    g.add_argument("--qk-accept-queries-per-seq", type=int, default=32)
    g.add_argument("--qk-strata", type=int, default=3)
    g.add_argument("--qk-entropy-bins", type=int, default=64)
    g.add_argument("--qk-damping", type=float, default=0.01)
    g.add_argument("--qk-batch-init", type=int, default=64)
    g.add_argument("--qk-batch-max", type=int, default=512)
    g.add_argument("--qk-row-cap", type=int, default=4)
    g.add_argument("--qk-max-rounds", type=int, default=200)
    g.add_argument("--qk-refresh-every", type=int, default=10)
    g.add_argument("--qk-patience", type=int, default=8)
    g.add_argument("--qk-accept-margin", type=float, default=0.0)
    g.add_argument("--qk-min-residual", type=float, default=0.0)
    g.add_argument("--qk-head-chunk", type=int, default=8)
    g.add_argument("--qk-audit-every", type=int, default=0)
    g.add_argument("--exclude-first", type=int, default=1)

    g = ap.add_argument_group("calibration / pipeline")
    g.add_argument("--calib-dataset", choices=["c4", "wikitext2"], default="c4")
    g.add_argument("--n-calib", type=int, default=128)
    g.add_argument("--calib-seqlen", type=int, default=2048)
    g.add_argument("--calib-cache-dir", default="./data/cache/calibration")
    g.add_argument("--fwd-batch", type=int, default=4)
    g.add_argument("--inps-device", default="cuda", help="cuda | cpu (hidden states of calibration seqs)")

    g = ap.add_argument_group("diagnostics")
    g.add_argument("--diag", action="store_true", help="per-layer attention KL (before/after) on held-out seqs")
    g.add_argument("--diag-seqs", type=int, default=4)
    g.add_argument("--diag-seqlen", type=int, default=None)
    g.add_argument("--diag-head-chunk", type=int, default=4)

    g = ap.add_argument_group("evaluation / output")
    g.add_argument("--no-eval-ppl", action="store_true")
    g.add_argument("--eval-c4", action="store_true")
    g.add_argument("--c4-samples", type=int, default=500)
    g.add_argument("--ppl-stride", type=int, default=2048)
    g.add_argument("--ppl-max-length", type=int, default=2048)
    g.add_argument("--eval-cache-dir", default="./data/cache/eval")
    g.add_argument("--lm-eval-tasks", nargs="*", default=[])
    g.add_argument("--lm-eval-batch-size", default="1")
    g.add_argument("--output-dir", default="./results")
    g.add_argument("--run-name", default=None)
    g.add_argument("--save-model", action="store_true")
    return ap


def auto_run_name(args) -> str:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    parts = [model_slug(args.model_path), args.base]
    if args.base != "none":
        parts.append(f"w{args.bits}g{args.group_size}")
        parts.append(f"clc-{args.clc_targets}")
        qk = args.qk_correct
        if qk != "none":
            qk += f"-{args.qk_schedule}-{args.qk_accept}-k{args.qk_strata}-f{args.qk_max_flip_percent:g}"
        parts.append(f"qk-{qk}")
    parts.append(ts)
    return "_".join(parts)


# ----------------------------------------------------------------------------- summary helpers
def summarize_layers(layers: list) -> dict:
    out = {"n_layers": len(layers)}
    if not layers:
        return out
    out["qk_flips_total"] = int(sum(l["qk"]["flips_total"] for l in layers if "qk" in l))
    out["q_flips_total"] = int(sum(l["qk"].get("q_flips", 0) for l in layers if "qk" in l))
    out["k_flips_total"] = int(sum(l["qk"].get("k_flips", 0) for l in layers if "qk" in l))
    out["qk_time_s"] = float(sum(l["qk"]["time_total_s"] for l in layers if "qk" in l))
    mems = [l["qk"]["peak_mem_gb"] for l in layers if "qk" in l and l["qk"]["peak_mem_gb"] is not None]
    out["qk_peak_mem_gb"] = max(mems) if mems else None
    out["clc_flips_total"] = int(sum(v.get("clc_flips", 0) for l in layers for v in l["linears"].values()))
    out["pipeline_time_s"] = float(sum(l["time_s"] for l in layers))
    for stage in ("diag_before", "diag_after"):
        rows = [l[stage] for l in layers if stage in l]
        if not rows:
            continue
        agg = {}
        for k in rows[0]:
            if isinstance(rows[0][k], dict) and "mean" in rows[0][k]:
                agg[k] = float(np.mean([r[k]["mean"] for r in rows]))
            elif isinstance(rows[0][k], (int, float)):
                agg[k] = float(np.mean([r[k] for r in rows]))
        out[stage] = agg
    jb = [sum(l["qk"]["J_acc_before"]) for l in layers if "qk" in l and l["qk"]["J_acc_before"] is not None]
    ja = [sum(l["qk"]["J_acc_after"]) for l in layers if "qk" in l and l["qk"]["J_acc_after"] is not None]
    if jb:
        out["J_acc_before_sum"] = float(sum(jb))
        out["J_acc_after_sum"] = float(sum(ja))
    ob = [sum(l["qk"]["J_opt_before"]) for l in layers if "qk" in l]
    oa = [sum(l["qk"]["J_opt_after"]) for l in layers if "qk" in l]
    if ob:
        out["J_opt_before_sum"] = float(sum(ob))
        out["J_opt_after_sum"] = float(sum(oa))
    return out


# ----------------------------------------------------------------------------- main
def main():
    args = build_parser().parse_args()
    load_runtime_env()
    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)

    hf_token = resolve_hf_token()
    model_ref = resolve_model_reference(args.model_path, args.models_root)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    run_name = args.run_name or auto_run_name(args)
    out_dir = Path(args.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_dump = dict(vars(args))
    cfg_dump["run_name"] = run_name
    cfg_dump["resolved_model"] = model_ref
    dump_json(out_dir / "config.json", cfg_dump)
    print(f"[run] {run_name} -> {out_dir}")

    tok = AutoTokenizer.from_pretrained(model_ref, trust_remote_code=args.trust_remote_code, token=hf_token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_ref,
        torch_dtype=dtype,
        attn_implementation="sdpa",
        trust_remote_code=args.trust_remote_code,
        token=hf_token,
    )
    model.to(args.device)
    model.eval()
    model.config.use_cache = False

    summary = {"config": cfg_dump}
    t_quant = time.time()
    if args.base != "none":
        calib = load_calibration_data(
            args.calib_dataset, tok, n_samples=args.n_calib, seqlen=args.calib_seqlen,
            seed=args.seed, return_tensors=True, cache_dir=args.calib_cache_dir,
        )
        diag_ids = []
        if args.diag:
            diag_ids = get_wikitext2_calibration_data(
                tok, n_samples=args.diag_seqs, seqlen=args.diag_seqlen or args.calib_seqlen,
                seed=args.seed + 1, split="test", return_tensors=True, cache_dir=args.calib_cache_dir,
            )

        clc = None
        if args.clc_targets != "none":
            clc = CLCCorrection(CLCConfig(
                knee_tolerance=args.clc_knee_tolerance,
                max_flip_percent=args.clc_max_flip_percent,
                use_james_stein=not args.no_james_stein,
            ))
        qk = None
        if args.qk_correct != "none":
            qk = QKLatticeCorrector(QKALCConfig(
                mode=args.qk_correct,
                accept=args.qk_accept,
                max_flip_percent=args.qk_max_flip_percent,
                queries_per_seq=args.qk_queries_per_seq,
                cov_queries_per_seq=args.qk_cov_queries_per_seq,
                accept_queries_per_seq=args.qk_accept_queries_per_seq,
                exclude_first=args.exclude_first,
                strata=args.qk_strata,
                entropy_bins=args.qk_entropy_bins,
                damping=args.qk_damping,
                batch_init=args.qk_batch_init,
                batch_max=args.qk_batch_max,
                row_cap=args.qk_row_cap,
                max_rounds=args.qk_max_rounds,
                refresh_every=args.qk_refresh_every,
                patience=args.qk_patience,
                accept_margin=args.qk_accept_margin,
                min_residual=args.qk_min_residual,
                head_chunk=args.qk_head_chunk,
                audit_every=args.qk_audit_every,
                seed=args.seed,
            ))
        kqk = None
        if args.qk_correct != "none" and "K" in args.qk_schedule.upper():
            kqk = KLatticeCorrector(KALCConfig(
                accept=args.qk_accept,
                max_flip_percent=args.qk_max_flip_percent,
                queries_per_seq=args.qk_queries_per_seq,
                h_sub_queries_per_seq=args.qk_k_hsub_queries,
                accept_queries_per_seq=args.qk_accept_queries_per_seq,
                exclude_first=args.exclude_first,
                batch_init=args.qk_k_batch_init,
                batch_max=args.qk_k_batch_max,
                row_cap=args.qk_k_row_cap,
                max_rounds=args.qk_k_max_rounds if args.qk_k_max_rounds is not None else args.qk_max_rounds,
                refresh_every=args.qk_k_refresh_every,
                patience=args.qk_patience,
                accept_margin=args.qk_accept_margin,
                min_residual=args.qk_min_residual,
                head_chunk=args.qk_head_chunk,
                seed=args.seed,
            ))

        n_layers = len(model.model.layers)
        pcfg = PipelineConfig(
            base=args.base,
            bits=args.bits,
            group_size=args.group_size,
            awq_n_grid=args.awq_n_grid,
            awq_max_rows=args.awq_max_rows,
            clc_targets=args.clc_targets,
            qk_layers=parse_layer_spec(args.qk_layers, n_layers),
            qk_schedule=args.qk_schedule.upper(),
            n_opt=args.qk_opt_seqs,
            n_acc=args.qk_accept_seqs,
            exclude_first=args.exclude_first,
            fwd_batch=args.fwd_batch,
            inps_device=args.inps_device,
            diag=args.diag,
            diag_head_chunk=args.diag_head_chunk,
        )
        pipe = SequentialPipeline(model, pcfg, clc=clc, qk=qk, kqk=kqk, device=args.device)
        layers = pipe.run(calib, diag_ids)
        dump_json(out_dir / "layers.json", layers)
        summary["layers"] = summarize_layers(layers)
        del pipe, calib
        torch.cuda.empty_cache()
    summary["quantize_time_s"] = time.time() - t_quant

    if not args.no_eval_ppl:
        from src.evaluation.sliding_window import SlidingWindowEvaluator

        ev = SlidingWindowEvaluator(
            device=args.device, seed=args.seed, stride=args.ppl_stride,
            max_length=args.ppl_max_length, cache_dir=args.eval_cache_dir, hf_token=hf_token,
        )
        ppl = {"wikitext2": ev.evaluate_sliding_window(model, tok, ev.load_wikitext2_test())}
        print(f"[ppl] wikitext2 = {ppl['wikitext2']['perplexity']:.4f}")
        if args.eval_c4:
            ppl["c4"] = ev.evaluate_sliding_window(model, tok, ev.load_c4_validation(n_samples=args.c4_samples))
            print(f"[ppl] c4 = {ppl['c4']['perplexity']:.4f}")
        summary["ppl"] = ppl
        dump_json(out_dir / "summary.json", summary)

    if args.lm_eval_tasks:
        from src.evaluation.lm_eval import LMEvalHarnessRunner

        runner = LMEvalHarnessRunner(
            tasks=args.lm_eval_tasks, device=args.device, batch_size=args.lm_eval_batch_size,
            output_dir=str(out_dir / "lm_eval"), run_name=run_name, hf_token=hf_token,
        )
        res = runner.evaluate_model(run_name, {"model": model, "tokenizer": tok})
        summary["lm_eval"] = res["summary"]

    if args.save_model:
        model.save_pretrained(out_dir / "model", safe_serialization=True)
        tok.save_pretrained(out_dir / "model")

    dump_json(out_dir / "summary.json", summary)
    print(f"[done] {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
