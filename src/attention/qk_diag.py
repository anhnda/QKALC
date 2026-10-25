"""Attention-distortion diagnostics on held-out sequences (go/no-go step).

For one layer and one sequence, teacher = FP q/k, student = quantized q/k, same input x.
Per head we accumulate (summed over query rows t >= exclude_first):
    kl_full    KL(p_t || softmax(q_hat k_hat))
    kl_q_only  KL with quantized Q, FP K
    kl_k_only  KL with FP Q, quantized K
    proxy_full 1/2 Var_{p_t}(eps_full)            (second-order proxy of kl_full)
and, if `mu` (query-side mean input) is given, the exact split  eps_full = eps_MQ + eps_QF:
    kl_mq      KL when only the mean-query part eps_MQ is injected
    kl_qf      KL when only the query-fluctuation part eps_QF is injected
    proxy_mq   1/2 Var_{p_t}(eps_MQ)
KL of the parts does not add up to kl_full (cross terms); report the interaction separately.
"""

from __future__ import annotations

from typing import Optional

import torch

from src.attention.geometry import AttnGeometry, expand_kv, project_heads
from src.attention.rope import rope_apply


@torch.no_grad()
def attention_kl_diagnostics(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    geom: AttnGeometry,
    Wq: torch.Tensor,
    bq: Optional[torch.Tensor],
    Wk: torch.Tensor,
    bk: Optional[torch.Tensor],
    Wq_hat: torch.Tensor,
    Wk_hat: torch.Tensor,
    mu: Optional[torch.Tensor] = None,
    head_chunk: int = 4,
    exclude_first: int = 1,
):
    T = x.shape[0]
    nh, nkv, h, s, g = geom.n_heads, geom.n_kv_heads, geom.head_dim, geom.scaling, geom.group
    dev, dt = x.device, x.dtype
    c_, s_ = cos[:T, None, :], sin[:T, None, :]

    Q = rope_apply(project_heads(x, Wq, bq, nh, h), c_, s_).transpose(0, 1)
    Qh = rope_apply(project_heads(x, Wq_hat, bq, nh, h), c_, s_).transpose(0, 1)
    K = expand_kv(rope_apply(project_heads(x, Wk, bk, nkv, h), c_, s_), g)
    Kh = expand_kv(rope_apply(project_heads(x, Wk_hat, bk, nkv, h), c_, s_), g)
    if mu is not None:
        mu2 = mu.reshape(1, -1).to(dt)
        Qm = rope_apply(project_heads(mu2, Wq, bq, nh, h).expand(T, nh, h), c_, s_).transpose(0, 1)
        Qmh = rope_apply(project_heads(mu2, Wq_hat, bq, nh, h).expand(T, nh, h), c_, s_).transpose(0, 1)

    mask = torch.ones(T, T, dtype=torch.bool, device=dev).triu(1)
    rv = (torch.arange(T, device=dev) >= exclude_first).to(dt)
    zero = torch.zeros((), device=dev, dtype=dt)
    ninf = float("-inf")

    keys = ["kl_full", "kl_q_only", "kl_k_only", "proxy_full"]
    if mu is not None:
        keys += ["kl_mq", "kl_qf", "proxy_mq"]
    sums = {k: torch.zeros(nh, device=dev, dtype=dt) for k in keys}

    for h0 in range(0, nh, head_chunk):
        sl = slice(h0, min(nh, h0 + head_chunk))
        L = s * torch.matmul(Q[sl], K[sl].transpose(1, 2))
        logp = torch.log_softmax(L.masked_fill(mask, ninf), dim=-1)
        p = logp.exp()

        def kl_of(L2):
            logq = torch.log_softmax(L2.masked_fill(mask, ninf), dim=-1)
            val = torch.where(mask, zero, p * (logp - logq)).sum(-1)
            return (val * rv).sum(-1)

        def proxy_of(e):
            e = e.masked_fill(mask, 0.0)
            eb = (p * e).sum(-1, keepdim=True)
            var = (p * (e - eb) ** 2).sum(-1)
            return 0.5 * (var * rv).sum(-1)

        Lf = s * torch.matmul(Qh[sl], Kh[sl].transpose(1, 2))
        sums["kl_full"][sl] += kl_of(Lf)
        sums["kl_q_only"][sl] += kl_of(s * torch.matmul(Qh[sl], K[sl].transpose(1, 2)))
        sums["kl_k_only"][sl] += kl_of(s * torch.matmul(Q[sl], Kh[sl].transpose(1, 2)))
        eps = Lf - L
        sums["proxy_full"][sl] += proxy_of(eps)
        if mu is not None:
            emq = s * (torch.matmul(Qmh[sl], Kh[sl].transpose(1, 2)) - torch.matmul(Qm[sl], K[sl].transpose(1, 2)))
            sums["kl_mq"][sl] += kl_of(L + emq)
            sums["kl_qf"][sl] += kl_of(L + (eps - emq))
            sums["proxy_mq"][sl] += proxy_of(emq)
        del L, logp, p, Lf, eps

    return sums, float(rv.sum().item())
