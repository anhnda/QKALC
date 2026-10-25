"""Per-sequence sufficient statistics for the Q-pass objective (K fixed, teacher attention fixed).

Objective (per head, averaged over sampled queries t):
    J = 1/2 * E_t Var_{s ~ p_t}(eps_ts)
    eps_ts = scaling * ( (P_t qe_hat_t) . (P_s k_hat_s) - (P_t qe_t) . (P_s k_s) )
where p_t is the FP teacher attention row (FP q and FP k on the same input x),
qe_t is the "error query" (the real query, or the mean query mu in mean-query mode).

With  Y_ts = P_t^T P_s k_hat_s  (h-vector, "rotated-back" quantized key):
    c_t = Cov_{p_t}(eps_t., Y_t.)          -> gradient  G[o, i] = scaling * E_t[ xe_t[i] c_t[r] ]
    v_t = diag Cov_{p_t}(Y_t.)             -> diag Hess H[o, i] = scaling^2 * E_t[ xe_t[i]^2 v_t[r] ]
    C_t = Cov_{p_t}(Y_t.)  (h x h)         -> Kronecker curvature / exact acceptance
for output row o = head * h + r.  A flip of W_q[o, i] by delta changes J of head(o) by
exactly  delta * G + 1/2 delta^2 H  (J is quadratic in W_q when K and p are fixed).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch

from src.attention.geometry import AttnGeometry, expand_kv, project_heads
from src.attention.rope import rope_apply, rope_apply_T, rope_diag_var


@dataclass
class SeqCache:
    tq: torch.Tensor              # (n_q,) sampled query positions (sorted)
    sub: Optional[torch.Tensor]   # (n_sub,) indices into tq for full C_t, or None
    cos_q: torch.Tensor           # (n_q, h)
    sin_q: torch.Tensor           # (n_q, h)
    qr_teacher: torch.Tensor      # (nh, n_q, h) rotated FP real queries (teacher)
    qr_err_fp: torch.Tensor       # (nh, n_q, h) rotated FP error queries (== teacher in full mode)
    xe: torch.Tensor              # (n_q, d) error-query inputs, or (1, d) for mean-query mode
    kr_fp: torch.Tensor           # (T, n_kv, h) rotated FP keys
    kr_hat: torch.Tensor          # (T, n_kv, h) rotated quantized keys


@torch.no_grad()
def build_seq_cache(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    geom: AttnGeometry,
    Wq: torch.Tensor,
    bq: Optional[torch.Tensor],
    Wk: torch.Tensor,
    bk: Optional[torch.Tensor],
    Wk_hat: torch.Tensor,
    tq: torch.Tensor,
    sub: Optional[torch.Tensor] = None,
    mu: Optional[torch.Tensor] = None,
) -> SeqCache:
    """x: (T, d) attention input (after input_layernorm); cos/sin: (T, h)."""
    nh, nkv, h = geom.n_heads, geom.n_kv_heads, geom.head_dim
    xq = x[tq]
    cq, sq = cos[tq], sin[tq]
    q_fp = project_heads(xq, Wq, bq, nh, h)
    qr_t = rope_apply(q_fp, cq[:, None, :], sq[:, None, :]).transpose(0, 1).contiguous()
    if mu is None:
        xe = xq
        qr_e = qr_t
    else:
        xe = mu.reshape(1, -1).to(x.dtype)
        qmu = project_heads(xe, Wq, bq, nh, h).expand(tq.numel(), nh, h)
        qr_e = rope_apply(qmu, cq[:, None, :], sq[:, None, :]).transpose(0, 1).contiguous()
    kr_fp = rope_apply(project_heads(x, Wk, bk, nkv, h), cos[:, None, :], sin[:, None, :])
    kr_hat = rope_apply(project_heads(x, Wk_hat, bk, nkv, h), cos[:, None, :], sin[:, None, :])
    return SeqCache(
        tq=tq, sub=sub, cos_q=cq, sin_q=sq, qr_teacher=qr_t, qr_err_fp=qr_e,
        xe=xe, kr_fp=kr_fp.contiguous(), kr_hat=kr_hat.contiguous(),
    )


@torch.no_grad()
def compute_seq_stats(
    sc: SeqCache,
    Wq_hat: torch.Tensor,
    bq: Optional[torch.Tensor],
    geom: AttnGeometry,
    need_v: bool = False,
    need_C: bool = False,
    need_ent: bool = False,
    head_chunk: int = 8,
) -> dict:
    """Returns dict with
        J   (nh, n_q)        1/2 Var_{p_t}(eps)
        c   (nh, n_q, h)     Cov(eps, Y)  (un-rotated)
        v   (nh, n_q, h)     diag Cov(Y)                  [need_v]
        ent (n_q,)           teacher entropy, mean over heads [need_ent]
        C   (n_sub, nh, h, h) full Cov(Y) for sc.sub       [need_C]
    """
    nh, h, s, g = geom.n_heads, geom.head_dim, geom.scaling, geom.group
    n_q = sc.tq.numel()
    T = sc.kr_fp.shape[0]
    dev, dt = sc.kr_fp.device, sc.kr_fp.dtype

    Kt = expand_kv(sc.kr_fp, g)    # (nh, T, h)
    Kh = expand_kv(sc.kr_hat, g)

    qe_hat = project_heads(sc.xe, Wq_hat.to(dt), bq, nh, h)
    if qe_hat.shape[0] != n_q:
        qe_hat = qe_hat.expand(n_q, nh, h)
    qe_hat_r = rope_apply(qe_hat, sc.cos_q[:, None, :], sc.sin_q[:, None, :]).transpose(0, 1)

    mask = torch.arange(T, device=dev)[None, :] > sc.tq[:, None]          # (n_q, T)
    logits = s * torch.matmul(sc.qr_teacher, Kt.transpose(1, 2))           # (nh, n_q, T)
    logp = torch.log_softmax(logits.masked_fill(mask, float("-inf")), dim=-1)
    p = logp.exp()

    eps = s * (torch.matmul(qe_hat_r, Kh.transpose(1, 2)) - torch.matmul(sc.qr_err_fp, Kt.transpose(1, 2)))
    eps = eps.masked_fill(mask, 0.0)
    eb = (p * eps).sum(-1, keepdim=True)
    dev_eps = (eps - eb).masked_fill(mask, 0.0)
    w = p * dev_eps
    out = {"J": 0.5 * (w * dev_eps).sum(-1)}

    c_rot = torch.matmul(w, Kh)                                             # (nh, n_q, h)
    out["c"] = rope_apply_T(c_rot, sc.cos_q[None], sc.sin_q[None])

    if need_ent:
        zero = torch.zeros((), device=dev, dtype=dt)
        plogp = torch.where(mask, zero, p * logp)
        out["ent"] = (-plogp.sum(-1)).mean(0)

    m = None
    if need_v or need_C:
        m = torch.matmul(p, Kh)
    if need_v:
        e2 = torch.matmul(p, Kh * Kh)
        h2 = h // 2
        eab = torch.matmul(p, Kh[..., :h2] * Kh[..., h2:])
        out["v"] = rope_diag_var(m, e2, eab, sc.cos_q[None], sc.sin_q[None])

    if need_C:
        sub = sc.sub if sc.sub is not None else torch.arange(n_q, device=dev)
        n_sub = sub.numel()
        C = torch.empty(n_sub, nh, h, h, device=dev, dtype=dt)
        cs = sc.cos_q[sub][None, :, None, :]
        ss = sc.sin_q[sub][None, :, None, :]
        for h0 in range(0, nh, head_chunk):
            hs = slice(h0, min(nh, h0 + head_chunk))
            ps = p[hs][:, sub]                                              # (hc, n_sub, T)
            K_ = Kh[hs]                                                     # (hc, T, h)
            S = torch.matmul((ps.unsqueeze(-1) * K_.unsqueeze(1)).transpose(-1, -2), K_.unsqueeze(1))
            mm = m[hs][:, sub]
            V = S - mm.unsqueeze(-1) * mm.unsqueeze(-2)
            A = rope_apply_T(V, cs, ss)                                     # V P
            Cc = rope_apply_T(A.transpose(-1, -2), cs, ss)                  # P^T V P
            Cc = 0.5 * (Cc + Cc.transpose(-1, -2))
            C[:, hs] = Cc.transpose(0, 1)
            del ps, S, V, A, Cc
        out["C"] = C
    return out


def sample_queries(T: int, n: int, n_sub: Optional[int], seed: int, exclude_first: int = 1):
    """Deterministic query positions in [exclude_first, T) and an optional sub-index set.
    Shared by the Q- and K-passes so both see the same queries for the same seed."""
    g = torch.Generator().manual_seed(int(seed))
    lo = min(max(exclude_first, 0), T - 1)
    cand = T - lo
    n = max(1, min(int(n), cand))
    tq = (torch.randperm(cand, generator=g)[:n] + lo).sort().values
    sub = None
    if n_sub is not None and n_sub > 0:
        sub = torch.randperm(n, generator=g)[: min(int(n_sub), n)].sort().values
    return tq, sub


@torch.no_grad()
def query_mean(x_fn: Callable[[int], torch.Tensor], n: int, exclude_first: int = 1) -> torch.Tensor:
    """Mean of the attention input over positions >= exclude_first of sequences 0..n-1."""
    total = None
    count = 0
    for i in range(n):
        x = x_fn(i)
        lo = min(exclude_first, x.shape[0] - 1)
        xs = x[lo:]
        total = xs.sum(0) if total is None else total + xs.sum(0)
        count += xs.shape[0]
    return total / max(count, 1)
