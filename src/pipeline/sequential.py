"""Block-sequential PTQ pipeline: base quantizer (RTN/AWQ) [+ CLC] [+ QK lattice correction].

For every decoder layer, in order:
  1. collect activation stats of all Linear inputs (only if AWQ or CLC needs them);
     inputs come from the already-quantized prefix (block-sequential, GPTQ-style);
  2. quantize every Linear of the block (keep q_proj lattice if QK correction runs);
  3. diagnostics "before" on held-out sequences (attention KL, MQ/QF split);
  4. QK correction of q_proj (K fixed, FP teacher on the same inputs);
  5. diagnostics "after";
  6. propagate calibration / diagnostic hidden states through the quantized block.
Supported families: Llama / Mistral / Qwen2 (HF transformers, rotate_half RoPE, no QK-norm).
"""

from __future__ import annotations

import gc
import inspect
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from src.attention.geometry import get_attn_geometry
from src.attention.qk_diag import attention_kl_diagnostics
from src.attention.qk_stats import query_mean
from src.attention.rope import compute_rope_tables
from src.post_correction.qk_alc import LatticeHandle, QKLatticeCorrector
from src.quantization.base import ActStats, quantize_linear, rows_per_call_for


@dataclass
class PipelineConfig:
    base: str = "rtn"                 # rtn | awq
    bits: int = 4
    group_size: int = 128
    awq_n_grid: int = 20
    awq_max_rows: int = 2048
    clc_targets: str = "none"         # none | q | all | all_but_q
    qk_layers: Optional[set] = None   # None = all layers (when a QK corrector is given)
    n_opt: int = 64                   # calibration seqs [0, n_opt) -> QK optimisation statistics
    n_acc: int = 16                   # calibration seqs [n_opt, n_opt + n_acc) -> QK acceptance set
    exclude_first: int = 1
    fwd_batch: int = 4
    inps_device: str = "cuda"
    diag: bool = False
    diag_head_chunk: int = 4
    verbose: bool = True


def parse_layer_spec(spec: Optional[str], n_layers: int) -> Optional[set]:
    if spec is None or spec.strip().lower() in ("", "all"):
        return None
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return {i for i in out if 0 <= i < n_layers}


class SequentialPipeline:
    def __init__(self, model, cfg: PipelineConfig, clc=None, qk: Optional[QKLatticeCorrector] = None, device: str = "cuda"):
        self.model = model
        self.cfg = cfg
        self.clc = clc
        self.qk = qk
        self.device = torch.device(device)
        self.dtype = next(model.parameters()).dtype
        self._mask_cache = {}
        self._layer_params = None

    # ------------------------------------------------------------ forward utils
    def _mask(self, T: int):
        if T not in self._mask_cache:
            m = torch.full((T, T), torch.finfo(self.dtype).min, device=self.device, dtype=self.dtype).triu(1)
            self._mask_cache[T] = m[None, None]
        return self._mask_cache[T]

    def _layer_forward(self, layer, hs: torch.Tensor) -> torch.Tensor:
        if self._layer_params is None:
            self._layer_params = set(inspect.signature(layer.forward).parameters)
        T = hs.shape[1]
        kw = {}
        if "attention_mask" in self._layer_params:
            kw["attention_mask"] = self._mask(T)
        if "position_ids" in self._layer_params:
            kw["position_ids"] = torch.arange(T, device=self.device)[None].expand(hs.shape[0], -1)
        if "position_embeddings" in self._layer_params:
            kw["position_embeddings"] = (self.cos_m[:, :T], self.sin_m[:, :T])
        if "use_cache" in self._layer_params:
            kw["use_cache"] = False
        out = layer(hs, **kw)
        return out[0] if isinstance(out, (tuple, list)) else out

    def _batches(self, n: int):
        b = max(1, self.cfg.fwd_batch)
        for i in range(0, n, b):
            yield slice(i, min(n, i + b))

    @torch.no_grad()
    def _embed(self, ids_list):
        if not ids_list:
            return None
        ids = torch.cat([t.reshape(1, -1) for t in ids_list], 0)
        emb = self.model.model.embed_tokens
        out = torch.empty(ids.shape[0], ids.shape[1], emb.embedding_dim, dtype=self.dtype, device=self.cfg.inps_device)
        for sl in self._batches(ids.shape[0]):
            out[sl] = emb(ids[sl].to(self.device)).to(self.cfg.inps_device)
        return out

    @torch.no_grad()
    def _propagate(self, layer, inps):
        if inps is None:
            return None
        for sl in self._batches(inps.shape[0]):
            out = self._layer_forward(layer, inps[sl].to(self.device))
            inps[sl] = out.to(inps.device)
        return inps

    def _x_fn(self, layer, inps, offset: int = 0):
        def fn(i):
            hs = inps[offset + i: offset + i + 1].to(self.device)
            return layer.input_layernorm(hs)[0].float()
        return fn

    def _use_clc(self, name: str) -> bool:
        t = self.cfg.clc_targets
        is_q = name.endswith("q_proj")
        return self.clc is not None and (t == "all" or (t == "q" and is_q) or (t == "all_but_q" and not is_q))

    # ------------------------------------------------------------ diagnostics
    @torch.no_grad()
    def _diag(self, layer, geom, diag_inps, Wq, bq, Wk, bk, mu):
        attn = layer.self_attn
        Wq_hat = attn.q_proj.weight.detach().float()
        Wk_hat = attn.k_proj.weight.detach().float()
        tot, cnt = None, 0.0
        xfn = self._x_fn(layer, diag_inps)
        for i in range(diag_inps.shape[0]):
            x = xfn(i)
            sums, c = attention_kl_diagnostics(
                x, self.cos32, self.sin32, geom, Wq, bq, Wk, bk, Wq_hat, Wk_hat,
                mu=mu, head_chunk=self.cfg.diag_head_chunk, exclude_first=self.cfg.exclude_first,
            )
            tot = sums if tot is None else {k: tot[k] + v for k, v in sums.items()}
            cnt += c
        res = {}
        for k, v in tot.items():
            per_head = (v / max(cnt, 1.0)).cpu()
            res[k] = {"mean": float(per_head.mean()), "max": float(per_head.max()), "per_head": per_head.tolist()}
        if "kl_mq" in res:
            res["kl_interaction_mean"] = res["kl_full"]["mean"] - res["kl_mq"]["mean"] - res["kl_qf"]["mean"]
        return res

    # ------------------------------------------------------------ main loop
    @torch.no_grad()
    def run(self, calib_ids, diag_ids=None) -> list:
        cfg = self.cfg
        model = self.model
        layers = model.model.layers
        n_calib = len(calib_ids)
        qk_on = self.qk is not None
        if qk_on and cfg.n_opt + (cfg.n_acc if self.qk.config.accept == "exact" else 0) > n_calib:
            raise ValueError("n_opt + n_acc exceeds the number of calibration sequences")
        if cfg.base not in ("rtn", "awq"):
            raise ValueError(f"unknown base {cfg.base}")

        inps = self._embed(calib_ids)
        diag_inps = self._embed(diag_ids) if (cfg.diag and diag_ids) else None
        T_max = inps.shape[1] if diag_inps is None else max(inps.shape[1], diag_inps.shape[1])
        self.cos_m, self.sin_m = compute_rope_tables(model, T_max, self.device, self.dtype)
        cos32, sin32 = compute_rope_tables(model, T_max, self.device, torch.float32)
        self.cos32, self.sin32 = cos32[0], sin32[0]

        n_calls = (n_calib + max(1, cfg.fwd_batch) - 1) // max(1, cfg.fwd_batch)
        report = []
        for li, layer in enumerate(layers):
            t0 = time.time()
            geom = get_attn_geometry(model, layer, seq_len=T_max)
            lin = {n: m for n, m in layer.named_modules() if isinstance(m, nn.Linear)}
            do_qk = qk_on and (cfg.qk_layers is None or li in cfg.qk_layers)

            # 1. activation statistics
            need = {n for n in lin if cfg.base == "awq" or self._use_clc(n)}
            stats = {}
            if need:
                rpc = rows_per_call_for(cfg.awq_max_rows, n_calls) if cfg.base == "awq" else 0
                handles = []
                for n in need:
                    stats[n] = ActStats(rows_per_call=rpc)

                    def hook(_m, inp, _o, key=n):
                        stats[key].update(inp[0])

                    handles.append(lin[n].register_forward_hook(hook))
                for sl in self._batches(n_calib):
                    self._layer_forward(layer, inps[sl].to(self.device))
                for hd in handles:
                    hd.remove()

            # 2. FP references for q/k, then quantize the block
            attn = layer.self_attn
            Wq = attn.q_proj.weight.detach().float().clone()
            Wk = attn.k_proj.weight.detach().float().clone()
            bq = None if attn.q_proj.bias is None else attn.q_proj.bias.detach().float().clone()
            bk = None if attn.k_proj.bias is None else attn.k_proj.bias.detach().float().clone()
            q_lat = None
            lin_info = {}
            for n, m in lin.items():
                is_q = n.endswith("q_proj")
                lat, info = quantize_linear(
                    m, cfg.base, cfg.bits, cfg.group_size,
                    act=stats.get(n), clc=self.clc if self._use_clc(n) else None,
                    awq_n_grid=cfg.awq_n_grid, awq_max_rows=cfg.awq_max_rows,
                    keep_state=is_q and do_qk,
                )
                if is_q:
                    q_lat = lat
                lin_info[n] = info
            stats.clear()

            # 3. diagnostics before QK correction
            mu = None
            if cfg.diag or do_qk:
                n_mu = cfg.n_opt if cfg.n_opt > 0 else min(16, n_calib)
                mu = query_mean(self._x_fn(layer, inps), min(n_mu, n_calib), cfg.exclude_first)
            rep = {"layer": li, "linears": lin_info}
            if diag_inps is not None:
                rep["diag_before"] = self._diag(layer, geom, diag_inps, Wq, bq, Wk, bk, mu)

            # 4. QK correction
            if do_qk:
                handle = LatticeHandle.from_lattice_state(q_lat)
                n_acc = cfg.n_acc if self.qk.config.accept == "exact" else 0
                qk_log = self.qk.correct(
                    geom=geom, cos=self.cos32, sin=self.sin32,
                    opt_x=self._x_fn(layer, inps, 0), n_opt=cfg.n_opt,
                    acc_x=self._x_fn(layer, inps, cfg.n_opt) if n_acc > 0 else None, n_acc=n_acc,
                    Wq_fp=Wq, bq=bq, Wk_fp=Wk, bk=bk, Wk_hat=attn.k_proj.weight.detach().float(),
                    lattice=handle, mu=mu,
                )
                handle.write_back(q_lat)
                attn.q_proj.weight.data.copy_(handle.weight().to(attn.q_proj.weight.dtype))
                rep["qk"] = qk_log
                del handle
                if diag_inps is not None:
                    rep["diag_after"] = self._diag(layer, geom, diag_inps, Wq, bq, Wk, bk, mu)
            del q_lat

            # 5. propagate
            inps = self._propagate(layer, inps)
            diag_inps = self._propagate(layer, diag_inps)
            rep["time_s"] = time.time() - t0
            report.append(rep)

            if cfg.verbose:
                msg = f"[layer {li:02d}] {rep['time_s']:.1f}s"
                if "diag_before" in rep:
                    msg += f" KL_full={rep['diag_before']['kl_full']['mean']:.3e}"
                if "qk" in rep:
                    q = rep["qk"]
                    msg += f" | QK flips={q['flips_total']} rounds={q['rounds']}"
                    if "diag_after" in rep:
                        msg += f" KL_full->{rep['diag_after']['kl_full']['mean']:.3e}"
                    if q["J_acc_before"] is not None:
                        jb, ja = sum(q["J_acc_before"]), sum(q["J_acc_after"])
                        msg += f" J_acc {jb:.3e}->{ja:.3e}"
                print(msg, flush=True)

            del Wq, Wk, bq, bk
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        del inps, diag_inps
        return report
