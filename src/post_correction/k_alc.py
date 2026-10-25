"""K-pass: attention-aware lattice correction of the key projection (Q fixed).

Same objective as the Q-pass, J = sum_heads 1/2 E_t Var_{p_t}(eps_t), now quadratic in W_k.
With u_t = P_t q_hat_t (current quantized query, rotated) and Z_ts = P_s^T u_t, a flip of
W_k[(g, r), i] by delta changes  eps_ts = delta * s * x_s[i] * Z_ts[r]  for every head j in
KV group g (GQA: one K row serves `group` query heads). Hence, exactly,
    dJ = delta * G + 1/2 delta^2 H,
    G[(g,r), i] = s/N   sum_{j in g} sum_t sum_s w_ts x_s[i] Z_ts[r],      w = p (eps - mean_p eps)
    H[(g,r), i] = s^2/N sum_{j in g} sum_t Var_{p_t}( x_s[i] Z_ts[r] )
G uses B_s = sum_t w_ts u_t and A_s = P_s^T B_s, so G = s/N * A_g^T X (no per-query loop).
H = E1 - E2: E1 = sum_s x_s[i]^2 sum_t p_ts Z_ts[r]^2 is exact and cheap; E2 = sum_t (sum_s p x Z)^2
needs a (d x h) tensor per query, so it is computed on `h_sub_queries_per_seq` queries and
rescaled (unbiased). Interactions between flips are left to the exact acceptance check,
which recomputes eps on the acceptance queries for the proposed batch (per KV group).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Callable, Optional

import torch

from src.attention.geometry import AttnGeometry, expand_kv, project_heads
from src.attention.qk_stats import sample_queries
from src.attention.rope import rope_apply, rope_apply_T, rope_diag_var


@dataclass
class KALCConfig:
    accept: str = "exact"             # exact | none
    max_flip_percent: float = 0.05
    queries_per_seq: int = 256
    h_sub_queries_per_seq: int = 8    # queries per seq used for the variance correction of diag H
    accept_queries_per_seq: int = 32
    exclude_first: int = 1
    batch_init: int = 16              # small batches: the diag-only predictor ignores interactions,
    batch_max: int = 64               # and same-direction K flips overshoot (each K row feeds `group` heads)
    row_cap: int = 1
    max_rounds: int = 200
    refresh_every: int = 5            # exact G refresh period (G is stale in between)
    patience: int = 8
    accept_margin: float = 0.0
    min_residual: float = 0.0
    head_chunk: int = 8
    seed: int = 0


@dataclass
class KSeqCache:
    tq: torch.Tensor          # (n_q,)
    sub: Optional[torch.Tensor]
    Qt: torch.Tensor          # (nh, n_q, h) rotated FP queries (teacher)
    U: torch.Tensor           # (nh, n_q, h) rotated current quantized queries
    kr_fp: torch.Tensor       # (T, n_kv, h) rotated FP keys


@torch.no_grad()
def build_k_cache(x, cos, sin, geom: AttnGeometry, Wq, bq, Wk, bk, Wq_hat, tq, sub=None) -> KSeqCache:
    nh, nkv, h = geom.n_heads, geom.n_kv_heads, geom.head_dim
    xq = x[tq]
    cq, sq = cos[tq][:, None, :], sin[tq][:, None, :]
    Qt = rope_apply(project_heads(xq, Wq, bq, nh, h), cq, sq).transpose(0, 1).contiguous()
    U = rope_apply(project_heads(xq, Wq_hat, bq, nh, h), cq, sq).transpose(0, 1).contiguous()
    kr_fp = rope_apply(project_heads(x, Wk, bk, nkv, h), cos[:, None, :], sin[:, None, :]).contiguous()
    return KSeqCache(tq=tq, sub=sub, Qt=Qt, U=U, kr_fp=kr_fp)


def _half_var(p, eps, mask):
    eb = (p * eps).sum(-1, keepdim=True)
    dv = (eps - eb).masked_fill(mask, 0.0)
    return 0.5 * (p * dv * dv).sum(-1)


@torch.no_grad()
def _teacher_and_eps(kc: KSeqCache, x, cos, sin, Wk_hat, bk, geom: AttnGeometry):
    nkv, h, s, g = geom.n_kv_heads, geom.head_dim, geom.scaling, geom.group
    T = x.shape[0]
    kh = rope_apply(project_heads(x, Wk_hat, bk, nkv, h), cos[:T, None, :], sin[:T, None, :])
    Kh = expand_kv(kh, g)
    Kt = expand_kv(kc.kr_fp, g)
    mask = torch.arange(T, device=x.device)[None, :] > kc.tq[:, None]
    logp = torch.log_softmax((s * torch.matmul(kc.Qt, Kt.transpose(1, 2))).masked_fill(mask, float("-inf")), dim=-1)
    p = logp.exp()
    eps = s * (torch.matmul(kc.U, Kh.transpose(1, 2)) - torch.matmul(kc.Qt, Kt.transpose(1, 2)))
    return p, eps.masked_fill(mask, 0.0), mask


@torch.no_grad()
def k_seq_stats(kc: KSeqCache, x, cos, sin, Wk_hat, bk, geom: AttnGeometry, need_H: bool = False, head_chunk: int = 8) -> dict:
    """Unscaled per-sequence sums: J (nh, n_q); G, H (n_kv*h, d) — multiply by s/N and s^2/N."""
    nh, nkv, h, g = geom.n_heads, geom.n_kv_heads, geom.head_dim, geom.group
    T, d = x.shape
    cosT, sinT = cos[:T], sin[:T]
    p, eps, mask = _teacher_and_eps(kc, x, cos, sin, Wk_hat, bk, geom)
    eb = (p * eps).sum(-1, keepdim=True)
    dv = (eps - eb).masked_fill(mask, 0.0)
    w = p * dv
    out = {"J": 0.5 * (w * dv).sum(-1)}

    B = torch.matmul(w.transpose(1, 2), kc.U)                       # (nh, T, h)
    A = rope_apply_T(B, cosT[None], sinT[None])                     # A_s = P_s^T B_s
    Ag = A.view(nkv, g, T, h).sum(1)                                # sum heads of each KV group
    out["G"] = torch.matmul(Ag.transpose(1, 2), x).reshape(nkv * h, d)

    if need_H:
        h2 = h // 2
        U = kc.U
        PT = p.transpose(1, 2)                                      # (nh, T, n_q)
        e2 = torch.matmul(PT, U * U)
        eab = torch.matmul(PT, U[..., :h2] * U[..., h2:])
        E1 = rope_diag_var(torch.zeros_like(e2), e2, eab, cosT[None], sinT[None])   # sum_t p Z^2
        E1g = E1.view(nkv, g, T, h).sum(1)
        H1 = torch.matmul(E1g.transpose(1, 2), x * x)                # (nkv, h, d)

        n_q = kc.tq.numel()
        sub = kc.sub if kc.sub is not None else torch.arange(n_q, device=x.device)
        ns = sub.numel()
        H2 = torch.zeros(nkv, h, d, device=x.device, dtype=x.dtype)
        cT = cosT[:, :h2].t()[None, None]                           # (1, 1, h2, T)
        sT = sinT[:, :h2].t()[None, None]
        gidx = torch.arange(nh, device=x.device) // g
        for h0 in range(0, nh, head_chunk):
            hs = slice(h0, min(nh, h0 + head_chunk))
            ps = p[hs][:, sub].unsqueeze(2)                          # (hc, ns, 1, T)
            Us = U[hs][:, sub]                                       # (hc, ns, h)
            Cx = torch.matmul(ps * cT, x)                            # (hc, ns, h2, d)
            Sx = torch.matmul(ps * sT, x)
            ua = Us[..., :h2].unsqueeze(-1)
            ub = Us[..., h2:].unsqueeze(-1)
            ma = ua * Cx + ub * Sx                                   # sum_s p x Z  (first half of r)
            mb = ub * Cx - ua * Sx                                   # second half of r
            e2h = torch.cat([(ma * ma).sum(1), (mb * mb).sum(1)], dim=1)   # (hc, h, d)
            H2.index_add_(0, gidx[hs], e2h)
            del ps, Us, Cx, Sx, ma, mb, e2h
        out["H"] = (H1 - (n_q / ns) * H2).reshape(nkv * h, d)
    return out


class KAcceptSet:
    """Exact J change of a K batch on the acceptance queries (eps recomputed, p fixed)."""

    def __init__(self, caches, x_fn, cos, sin, geom: AttnGeometry, Wk_hat, bk):
        self.caches = caches
        self.x_fn = x_fn
        self.cos, self.sin = cos, sin
        self.geom = geom
        self.p, self.eps, self.mask, self.Jq = [], [], [], []
        N = 0
        for i, kc in enumerate(caches):
            p, eps, mask = _teacher_and_eps(kc, x_fn(i), cos, sin, Wk_hat, bk, geom)
            self.p.append(p)
            self.eps.append(eps)
            self.mask.append(mask)
            self.Jq.append(_half_var(p, eps, mask))
            N += kc.tq.numel()
        self.N = N
        self.J = sum(j.sum(1) for j in self.Jq) / N

    @torch.no_grad()
    def delta(self, dW):
        g = self.geom
        nkv, h, s = g.n_kv_heads, g.head_dim, g.scaling
        dJ = torch.zeros(g.n_heads, device=dW.device, dtype=dW.dtype)
        new = []
        for i, kc in enumerate(self.caches):
            x = self.x_fn(i)
            T = x.shape[0]
            dk = (x @ dW.t()).view(T, nkv, h)
            dK = expand_kv(rope_apply(dk, self.cos[:T, None, :], self.sin[:T, None, :]), g.group)
            e2 = self.eps[i] + (s * torch.matmul(kc.U, dK.transpose(1, 2))).masked_fill(self.mask[i], 0.0)
            J2 = _half_var(self.p[i], e2, self.mask[i])
            dJ += (J2 - self.Jq[i]).sum(1)
            new.append((e2, J2))
        return dJ / self.N, new

    @torch.no_grad()
    def apply(self, new, accept_heads):
        m3 = accept_heads[:, None, None]
        m2 = accept_heads[:, None]
        for i, (e2, J2) in enumerate(new):
            self.eps[i] = torch.where(m3, e2, self.eps[i])
            self.Jq[i] = torch.where(m2, J2, self.Jq[i])
        self.J = sum(j.sum(1) for j in self.Jq) / self.N


def select_batch(score, row_count, budget_row: int, bsize, n_groups: int, rows_per_group: int, row_cap: int):
    """Per-row top-`row_cap`, then per-group top-`bsize[g]` most negative finite scores."""
    out, d = score.shape
    dev = score.device
    inf_val = float("inf")
    c_cap = max(1, min(row_cap, d))
    v1, i1 = torch.topk(score, c_cap, dim=1, largest=False)
    rem = (budget_row - row_count).clamp_min(0)
    rank_c = torch.arange(c_cap, device=dev)[None, :]
    v1 = torch.where(rank_c < rem[:, None], v1, torch.full_like(v1, inf_val))
    vh = v1.reshape(n_groups, rows_per_group * c_cap)
    ih = i1.reshape(n_groups, rows_per_group * c_cap)
    kk = int(min(int(bsize.max().item()), rows_per_group * c_cap))
    v2, i2 = torch.topk(vh, kk, dim=1, largest=False)
    sel = torch.isfinite(v2) & (torch.arange(kk, device=dev)[None, :] < bsize[:, None])
    rows = (i2 // c_cap) + torch.arange(n_groups, device=dev)[:, None] * rows_per_group
    cols = ih.gather(1, i2)
    return sel, rows, cols


class KLatticeCorrector:
    def __init__(self, config: Optional[KALCConfig] = None):
        self.config = config or KALCConfig()
        self.last_accept_queries: list = []
        self._opt = []

    def _pass(self, opt_x, W, bk, geom, cos, sin, need_H: bool, out: int, d: int, dev, dt):
        s = geom.scaling
        Gs = torch.zeros(out, d, device=dev, dtype=dt)
        Hs = torch.zeros(out, d, device=dev, dtype=dt) if need_H else None
        Jsum = torch.zeros(geom.n_heads, device=dev, dtype=dt)
        N = 0
        for i, kc in enumerate(self._opt):
            x = opt_x(i).to(dev, dt)
            st = k_seq_stats(kc, x, cos, sin, W, bk, geom, need_H=need_H, head_chunk=self.config.head_chunk)
            Gs += st["G"]
            if need_H:
                Hs += st["H"]
            Jsum += st["J"].sum(1)
            N += kc.tq.numel()
            del st, x
        G = s * Gs / N
        H = (s * s * Hs / N).clamp_min(0.0) if need_H else None
        return G, H, Jsum / N

    @torch.no_grad()
    def correct(
        self,
        *,
        geom: AttnGeometry,
        cos: torch.Tensor,
        sin: torch.Tensor,
        opt_x: Callable[[int], torch.Tensor],
        n_opt: int,
        acc_x: Optional[Callable[[int], torch.Tensor]],
        n_acc: int,
        Wq_fp: torch.Tensor,
        bq: Optional[torch.Tensor],
        Wk_fp: torch.Tensor,
        bk: Optional[torch.Tensor],
        Wq_hat: torch.Tensor,
        lattice,
    ) -> dict:
        cfg = self.config
        if cfg.accept not in ("exact", "none"):
            raise ValueError(f"unknown accept rule {cfg.accept}")
        if cfg.accept == "exact" and (acc_x is None or n_acc <= 0):
            raise ValueError("accept='exact' needs acceptance sequences")
        t_start = time.time()
        dev, dt = lattice.codes.device, lattice.codes.dtype
        is_cuda = dev.type == "cuda"
        if is_cuda:
            torch.cuda.reset_peak_memory_stats(dev)
        nh, nkv, h, grp = geom.n_heads, geom.n_kv_heads, geom.head_dim, geom.group
        out, d = lattice.codes.shape
        if out != nkv * h:
            raise ValueError("lattice rows do not match n_kv_heads * head_dim")

        Wq_fp, Wk_fp, Wq_hat = Wq_fp.to(dev, dt), Wk_fp.to(dev, dt), Wq_hat.to(dev, dt)
        bq = None if bq is None else bq.to(dev, dt)
        bk = None if bk is None else bk.to(dev, dt)
        cos, sin = cos.to(dev, dt), sin.to(dev, dt)
        xo = lambda i: opt_x(i).to(dev, dt)  # noqa: E731

        # same query positions as the Q-pass for the same seed
        self._opt = []
        for i in range(n_opt):
            x = xo(i)
            T = x.shape[0]
            tq, sub = sample_queries(T, cfg.queries_per_seq, None, cfg.seed * 1_000_003 + i, cfg.exclude_first)
            if cfg.h_sub_queries_per_seq > 0:
                g_ = torch.Generator().manual_seed(int(cfg.seed * 7_919 + 17 + i))
                sub = torch.randperm(tq.numel(), generator=g_)[: min(cfg.h_sub_queries_per_seq, tq.numel())].sort().values
            self._opt.append(build_k_cache(
                x, cos[:T], sin[:T], geom, Wq_fp, bq, Wk_fp, bk, Wq_hat, tq.to(dev), None if sub is None else sub.to(dev)
            ))
            del x

        W0 = lattice.weight()
        G, H, J_opt0 = self._pass(xo, W0, bk, geom, cos, sin, True, out, d, dev, dt)

        acc = None
        self.last_accept_queries = []
        if cfg.accept == "exact":
            xa = lambda i: acc_x(i).to(dev, dt)  # noqa: E731
            acc_caches = []
            for i in range(n_acc):
                x = xa(i)
                T = x.shape[0]
                tq, _ = sample_queries(T, cfg.accept_queries_per_seq, None, cfg.seed * 1_000_003 + 500_009 + i, cfg.exclude_first)
                tq = tq.to(dev)
                self.last_accept_queries.append(tq.detach().cpu())
                acc_caches.append(build_k_cache(x, cos[:T], sin[:T], geom, Wq_fp, bq, Wk_fp, bk, Wq_hat, tq))
                del x
            acc = KAcceptSet(acc_caches, xa, cos, sin, geom, W0, bk)
        J_acc0 = acc.J.clone() if acc is not None else None
        t_stats = time.time() - t_start

        flip_dir = torch.sign(lattice.pre - lattice.codes)
        flip_dir[flip_dir == 0] = 1.0
        target = lattice.codes + flip_dir
        valid = (target >= lattice.min_int) & (target <= lattice.max_int)
        if cfg.min_residual > 0:
            valid &= (lattice.pre - lattice.codes).abs() >= cfg.min_residual
        step = flip_dir * lattice.scale_eff
        half_curv = 0.5 * step * step * H
        budget_row = int(cfg.max_flip_percent * d)

        used = torch.zeros_like(valid)
        if getattr(lattice, "codes_ref", None) is not None:
            used = lattice.codes != lattice.codes_ref
        blocked = torch.zeros_like(valid)
        row_count = used.sum(1).long()
        row_count0 = row_count.clone()
        bsize = torch.full((nkv,), max(1, cfg.batch_init), dtype=torch.long, device=dev)
        rejects = torch.zeros(nkv, dtype=torch.long, device=dev)
        finished = torch.zeros(nkv, dtype=torch.bool, device=dev)
        W_cur = W0.clone()
        zeros_step = torch.zeros_like(step)
        inf_val = float("inf")

        hist = {"pred": [], "acc": [], "accepted": [], "nsel": []}
        rounds = since_refresh = n_refresh = 0
        just_refreshed = False
        for rnd in range(cfg.max_rounds if budget_row > 0 else 0):
            rounds = rnd + 1
            if cfg.refresh_every > 0 and since_refresh >= cfg.refresh_every:
                G, _, _ = self._pass(xo, W_cur, bk, geom, cos, sin, False, out, d, dev, dt)
                since_refresh = 0
                n_refresh += 1

            rows_ok = (row_count < budget_row) & (~finished).repeat_interleave(h)
            avail = valid & ~used & ~blocked & rows_ok[:, None]
            score = step * G + half_curv
            score = torch.where(avail & (score < 0), score, torch.full_like(score, inf_val))
            sel, rows, cols = select_batch(score, row_count, budget_row, bsize, nkv, h, cfg.row_cap)
            nsel = sel.sum(1)
            if int(nsel.sum().item()) == 0:
                if just_refreshed or cfg.refresh_every <= 0:
                    break
                G, _, _ = self._pass(xo, W_cur, bk, geom, cos, sin, False, out, d, dev, dt)
                since_refresh = 0
                n_refresh += 1
                just_refreshed = True
                continue
            just_refreshed = False

            bmask = torch.zeros_like(valid)
            bmask[rows[sel], cols[sel]] = True
            dW = torch.where(bmask, step, zeros_step)
            pred = (dW * G + 0.5 * dW * dW * H).view(nkv, h, d).sum(dim=(1, 2))
            active = nsel > 0
            if acc is not None:
                dJh, new = acc.delta(dW)
                dJg = dJh.view(nkv, grp).sum(1)
                Jg = acc.J.view(nkv, grp).sum(1)
                accept_g = active & (pred < 0) & (dJg < -cfg.accept_margin * Jg.clamp_min(0.0))
            else:
                dJg, new = None, None
                accept_g = active & (pred < 0)

            acc_rows = accept_g.repeat_interleave(h)
            amask = bmask & acc_rows[:, None]
            if bool(amask.any()):
                lattice.codes += torch.where(amask, flip_dir, zeros_step)
                W_cur += torch.where(amask, step, zeros_step)
                used |= amask
                row_count += amask.sum(1)
                if acc is not None:
                    acc.apply(new, accept_g.repeat_interleave(grp))
            del new

            rej = active & ~accept_g
            single = rej & (nsel == 1)
            if bool(single.any()):
                m1 = sel & single[:, None]
                blocked[rows[m1], cols[m1]] = True
            rejects = torch.where(accept_g, torch.zeros_like(rejects), rejects + single.long())
            bsize = torch.where(accept_g, (bsize * 2).clamp_max(max(1, cfg.batch_max)), bsize)
            bsize = torch.where(rej & ~single, (nsel // 2).clamp_min(1), bsize)
            finished |= rejects >= cfg.patience

            hist["pred"] += pred[active].tolist()
            hist["accepted"] += accept_g[active].tolist()
            hist["nsel"] += nsel[active].tolist()
            if dJg is not None:
                hist["acc"] += dJg[active].tolist()
            since_refresh += 1
            if bool(finished.all()):
                break

        _, _, J_opt1 = self._pass(xo, W_cur, bk, geom, cos, sin, False, out, d, dev, dt)
        flips_group = (row_count - row_count0).view(nkv, h).sum(1)
        log = {
            "pass": "K",
            "config": asdict(cfg),
            "rounds": rounds,
            "refreshes": n_refresh,
            "budget_per_row": budget_row,
            "flips_total": int(flips_group.sum().item()),
            "flips_per_group": flips_group.tolist(),
            "J_opt_before": J_opt0.tolist(),
            "J_opt_after": J_opt1.tolist(),
            "J_acc_before": None if J_acc0 is None else J_acc0.tolist(),
            "J_acc_after": None if acc is None else acc.J.tolist(),
            "history": hist,
            "time_stats_s": t_stats,
            "time_total_s": time.time() - t_start,
            "peak_mem_gb": (torch.cuda.max_memory_allocated(dev) / 1e9) if is_cuda else None,
        }
        self._opt = []
        del acc, G, H
        if is_cuda:
            torch.cuda.empty_cache()
        return log
