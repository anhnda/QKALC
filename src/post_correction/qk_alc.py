"""Attention-aware lattice correction of the query projection (Q-pass, K fixed).

Method (see README, section "Phương pháp"):
  objective   J = 1/2 E_t Var_{p_t}(eps_t)   (second-order proxy of softmax KL, per-query centring)
  scoring     exact single-flip change  delta*G + 1/2 delta^2 H  (G, H from many queries)
  coupling    stratified Kronecker curvature  s^2 sum_k pi_k M_k (x) Cbar_k, diagonally
              re-calibrated to H while staying PSD (D K D + residual diagonal)
  acceptance  exact batch change on an independent acceptance set via per-query cache
              {x_t, c_t, C_t};  accept a head's batch iff predicted < 0 and exact < -margin
  refresh     G recomputed exactly every `refresh_every` rounds
Every move is a complementary flip (to the other adjacent level of pre_round), at most one per
weight, with a per-output-row budget; codes stay in range, so the integer format is unchanged.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Callable, Optional

import torch

from src.attention.geometry import AttnGeometry
from src.attention.qk_stats import build_seq_cache, compute_seq_stats, query_mean


@dataclass
class QKALCConfig:
    mode: str = "full"                # full | meanquery  (error query = real x_t, or mu)
    accept: str = "exact"             # exact | none
    max_flip_percent: float = 0.05    # per output row, fraction of in_features
    queries_per_seq: int = 256        # sampled queries per optimisation sequence
    cov_queries_per_seq: int = 32     # subset of those used for full C_t (Kronecker Cbar)
    accept_queries_per_seq: int = 32  # queries per acceptance sequence (full C_t cached)
    exclude_first: int = 1            # query positions < this are never sampled (row 0 has KL = 0)
    strata: int = 3                   # entropy strata for the Kronecker curvature (1 = plain K-FAC)
    entropy_bins: int = 64
    damping: float = 0.01             # relative damping of M_k
    batch_init: int = 64              # flips per head per round (adaptive)
    batch_max: int = 512
    row_cap: int = 4                  # max flips per output row per round
    max_rounds: int = 200
    refresh_every: int = 10           # exact G refresh period (rounds); <= 0 disables
    patience: int = 8                 # single-flip rejections before a head is finished
    accept_margin: float = 0.0        # accept iff exact dJ_A < -margin * J_A
    min_residual: float = 0.0         # only weights with |pre_round - code| >= this are candidates
    head_chunk: int = 8
    audit_every: int = 0              # every N rounds also compute the exact opt-set dJ of the batch
    seed: int = 0


@dataclass
class LatticeHandle:
    """Mutable view of a quantized q_proj lattice. W_eff = (codes - zp) * scale_eff."""

    codes: torch.Tensor
    pre: torch.Tensor
    zp: torch.Tensor
    scale_eff: torch.Tensor
    min_int: int
    max_int: int

    def weight(self) -> torch.Tensor:
        return (self.codes - self.zp) * self.scale_eff

    @classmethod
    def from_lattice_state(cls, lat) -> "LatticeHandle":
        st = lat.state
        d = st.in_features
        return cls(
            codes=st.integer_weights[:, :d].clone(),
            pre=st.pre_round[:, :d].clone(),
            zp=st.zero_point[:, :d].clone(),
            scale_eff=st.scale[:, :d] / lat.in_scales[None, :],
            min_int=st.min_int,
            max_int=st.max_int,
        )

    def write_back(self, lat):
        d = lat.state.in_features
        lat.state.integer_weights[:, :d] = self.codes


class KronCurvature:
    """H_tilde = D K D + diag(H_res),  K = s2 * sum_k pi_k blockdiag_heads(Cbar_k[j]) (x) M_k.

    D = sqrt(H / diag K) where diag K > 0, so diag(H_tilde) == H exactly and H_tilde is PSD.
    Coordinates with diag K == 0 keep their exact diagonal through H_res and no coupling.
    """

    def __init__(self, Ms, Cbars, pis, s2: float, H: torch.Tensor, n_heads: int, head_dim: int, eps: float = 1e-30):
        self.Ms = list(Ms)
        self.Cbars = list(Cbars)
        self.pis = [float(p) for p in pis]
        self.s2 = float(s2)
        self.nh = n_heads
        self.h = head_dim
        diagK = torch.zeros_like(H)
        for M, C, pi in zip(self.Ms, self.Cbars, self.pis):
            diagK += pi * torch.outer(torch.diagonal(C, dim1=-2, dim2=-1).reshape(-1), torch.diagonal(M))
        diagK *= self.s2
        pos = diagK > eps
        Hc = H.clamp_min(0.0)
        self.D = torch.where(pos, torch.sqrt(Hc / diagK.clamp_min(eps)), torch.zeros_like(H))
        self.Hres = torch.where(pos, torch.zeros_like(H), Hc)

    @torch.no_grad()
    def matvec(self, dW: torch.Tensor) -> torch.Tensor:
        out, d = dW.shape
        nh, h = self.nh, self.h
        res = self.Hres * dW
        if not self.Ms:
            return res
        Z = self.D * dW
        rows = (Z != 0).any(dim=1)
        if not bool(rows.any()):
            return res
        heads = rows.view(nh, h).any(dim=1)
        hidx = heads.nonzero().flatten()
        Zr = Z[rows]
        acc = torch.zeros(hidx.numel(), h, d, device=dW.device, dtype=dW.dtype)
        for M, C, pi in zip(self.Ms, self.Cbars, self.pis):
            T_ = torch.zeros_like(Z)
            T_[rows] = Zr @ M
            acc += pi * torch.bmm(C[hidx], T_.view(nh, h, d)[hidx])
        full = torch.zeros(nh, h, d, device=dW.device, dtype=dW.dtype)
        full[hidx] = acc
        return res + self.s2 * self.D * full.view(out, d)

    def dense(self) -> torch.Tensor:
        """Dense (out*d)^2 matrix, flatten index o*d + i. Only for tiny tests."""
        out, d = self.D.shape
        nh, h = self.nh, self.h
        n = out * d
        Kd = torch.zeros(n, n, dtype=self.D.dtype, device=self.D.device)
        for M, C, pi in zip(self.Ms, self.Cbars, self.pis):
            for j in range(nh):
                sl = slice(j * h * d, (j + 1) * h * d)
                Kd[sl, sl] += pi * torch.kron(C[j], M)
        Kd *= self.s2
        Dv = self.D.reshape(-1)
        return Dv[:, None] * Kd * Dv[None, :] + torch.diag(self.Hres.reshape(-1))


class AcceptCache:
    """Exact batch change of J on the acceptance queries: per query x_t, c_t, C_t."""

    def __init__(self, X, c, C, J, scaling: float, n_heads: int, head_dim: int):
        self.X = X            # (n, d)
        self.c = c            # (n, nh, h)
        self.C = C            # (n, nh, h, h)
        self.J = J            # (nh,) mean 1/2 Var
        self.s = float(scaling)
        self.nh = n_heads
        self.h = head_dim

    @torch.no_grad()
    def delta(self, dW: torch.Tensor):
        n = self.X.shape[0]
        a = (self.X @ dW.t()).view(n, self.nh, self.h)
        Ca = torch.einsum("nhab,nhb->nha", self.C, a)
        dJ = (self.s * (a * self.c).sum(-1) + 0.5 * self.s * self.s * (a * Ca).sum(-1)).mean(0)
        return dJ, Ca

    @torch.no_grad()
    def apply(self, Ca: torch.Tensor, dJ: torch.Tensor, accept_h: torch.Tensor):
        m = accept_h.to(self.c.dtype)
        self.c += self.s * Ca * m[None, :, None]
        self.J += dJ * m


class QKLatticeCorrector:
    def __init__(self, config: Optional[QKALCConfig] = None):
        self.config = config or QKALCConfig()
        self.last_accept_queries: list = []
        self._opt = []
        self._Xe = None

    # ------------------------------------------------------------------ helpers
    def _sample(self, T: int, n: int, n_sub: Optional[int], seed: int):
        g = torch.Generator().manual_seed(int(seed))
        lo = min(max(self.config.exclude_first, 0), T - 1)
        cand = T - lo
        n = max(1, min(int(n), cand))
        tq = (torch.randperm(cand, generator=g)[:n] + lo).sort().values
        sub = None
        if n_sub is not None and n_sub > 0:
            sub = torch.randperm(n, generator=g)[: min(int(n_sub), n)].sort().values
        return tq, sub

    def _opt_pass(self, W, geom: AttnGeometry, bq, out: int, need_c: bool):
        nh, s = geom.n_heads, geom.scaling
        Jsum = torch.zeros(nh, device=W.device, dtype=W.dtype)
        N = 0
        c_rows = []
        for sc in self._opt:
            st = compute_seq_stats(sc, W, bq, geom, head_chunk=self.config.head_chunk)
            nq = sc.tq.numel()
            Jsum += st["J"].sum(1)
            N += nq
            if need_c:
                c_rows.append(st["c"].transpose(0, 1).reshape(nq, out))
            del st
        G = None
        if need_c:
            G = s * (torch.cat(c_rows, 0).t() @ self._Xe) / N
        return G, Jsum / max(N, 1)

    def _initial_stats(self, geom: AttnGeometry, W0, bq, T_max: int, out: int, d: int):
        cfg = self.config
        nh, h, s = geom.n_heads, geom.head_dim, geom.scaling
        dev, dt = W0.device, W0.dtype
        nb = max(1, cfg.entropy_bins)
        edges = torch.linspace(0.0, math.log(max(T_max, 2)) + 1e-4, nb + 1, device=dev, dtype=dt)
        inner = edges[1:-1].contiguous()
        Csum = torch.zeros(nb, nh, h, h, device=dev, dtype=dt)
        Ccnt = torch.zeros(nb, device=dev, dtype=dt)
        c_rows, v_rows, xe_rows, ent_rows = [], [], [], []
        Jsum = torch.zeros(nh, device=dev, dtype=dt)
        N = 0
        for sc in self._opt:
            st = compute_seq_stats(
                sc, W0, bq, geom, need_v=True, need_C=sc.sub is not None, need_ent=True, head_chunk=cfg.head_chunk
            )
            nq = sc.tq.numel()
            c_rows.append(st["c"].transpose(0, 1).reshape(nq, out))
            v_rows.append(st["v"].transpose(0, 1).reshape(nq, out))
            xe_rows.append(sc.xe.expand(nq, d) if sc.xe.shape[0] != nq else sc.xe)
            ent_rows.append(st["ent"])
            Jsum += st["J"].sum(1)
            N += nq
            if "C" in st:
                b = torch.bucketize(st["ent"][sc.sub], inner)
                Csum.index_add_(0, b, st["C"])
                Ccnt.index_add_(0, b, torch.ones(b.numel(), device=dev, dtype=dt))
            del st

        c_all = torch.cat(c_rows, 0)
        v_all = torch.cat(v_rows, 0)
        Xe = torch.cat(xe_rows, 0).contiguous()
        ent = torch.cat(ent_rows, 0)
        del c_rows, v_rows, xe_rows
        G = s * (c_all.t() @ Xe) / N
        H = (s * s) * (v_all.t() @ (Xe * Xe)) / N
        del c_all, v_all
        self._Xe = Xe

        K = max(1, cfg.strata)
        qbin = torch.bucketize(ent, inner)
        if K > 1:
            qs = torch.linspace(0.0, 1.0, K + 1, device=dev, dtype=dt)[1:-1]
            thr = torch.quantile(ent, qs).contiguous()
            centers = 0.5 * (edges[:-1] + edges[1:])
            bin_str = torch.bucketize(centers, thr)
        else:
            bin_str = torch.zeros(nb, dtype=torch.long, device=dev)
        q_str = bin_str[qbin]
        Cglob = Csum.sum(0) / Ccnt.sum().clamp_min(1.0)

        eye = torch.eye(d, device=dev, dtype=dt)
        Ms, Cbs, pis, strata_info = [], [], [], []
        for k in range(K):
            mk = q_str == k
            nk = int(mk.sum().item())
            if nk == 0:
                continue
            Xk = Xe[mk]
            M = Xk.t() @ Xk / nk
            if cfg.damping > 0:
                M += cfg.damping * torch.diagonal(M).mean().clamp_min(1e-12) * eye
            bmask_k = bin_str == k
            ck = float(Ccnt[bmask_k].sum().item())
            Cb = Csum[bmask_k].sum(0) / ck if ck > 0 else Cglob
            Ms.append(M)
            Cbs.append(Cb)
            pis.append(nk / N)
            ent_k = ent[mk]
            strata_info.append(
                {"pi": nk / N, "n_cov_queries": ck, "ent_min": float(ent_k.min()), "ent_max": float(ent_k.max())}
            )
            del Xk
        del eye
        curv = KronCurvature(Ms, Cbs, pis, s * s, H, nh, h)
        return G, H, curv, Jsum / N, strata_info

    def _build_accept(self, acc_x, n_acc, geom, cos, sin, Wq_fp, bq, Wk_fp, bk, Wk_hat, W0):
        cfg = self.config
        nh, h, s = geom.n_heads, geom.head_dim, geom.scaling
        dev, dt = W0.device, W0.dtype
        Xs, cs, Cs = [], [], []
        Jsum = torch.zeros(nh, device=dev, dtype=dt)
        N = 0
        self.last_accept_queries = []
        for i in range(n_acc):
            x = acc_x(i).to(dev, dt)
            T = x.shape[0]
            tq, _ = self._sample(T, cfg.accept_queries_per_seq, None, cfg.seed * 1_000_003 + 500_009 + i)
            tq = tq.to(dev)
            self.last_accept_queries.append(tq.detach().cpu())
            sc = build_seq_cache(x, cos[:T], sin[:T], geom, Wq_fp, bq, Wk_fp, bk, Wk_hat, tq, None, mu=None)
            st = compute_seq_stats(sc, W0, bq, geom, need_C=True, head_chunk=cfg.head_chunk)
            Xs.append(sc.xe)
            cs.append(st["c"].transpose(0, 1).contiguous())
            Cs.append(st["C"])
            Jsum += st["J"].sum(1)
            N += tq.numel()
            del sc, st, x
        return AcceptCache(torch.cat(Xs, 0), torch.cat(cs, 0), torch.cat(Cs, 0), Jsum / N, s, nh, h)

    # ------------------------------------------------------------------ main
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
        Wk_hat: torch.Tensor,
        lattice: LatticeHandle,
        mu: Optional[torch.Tensor] = None,
    ) -> dict:
        cfg = self.config
        if cfg.mode not in ("full", "meanquery"):
            raise ValueError(f"unknown mode {cfg.mode}")
        if cfg.accept not in ("exact", "none"):
            raise ValueError(f"unknown accept rule {cfg.accept}")
        if cfg.accept == "exact" and (acc_x is None or n_acc <= 0):
            raise ValueError("accept='exact' needs acceptance sequences")
        if n_opt <= 0:
            raise ValueError("need at least one optimisation sequence")

        t_start = time.time()
        dev, dt = lattice.codes.device, lattice.codes.dtype
        is_cuda = dev.type == "cuda"
        if is_cuda:
            torch.cuda.reset_peak_memory_stats(dev)
        nh, h = geom.n_heads, geom.head_dim
        out, d = lattice.codes.shape
        if out != nh * h:
            raise ValueError("lattice rows do not match n_heads * head_dim")

        Wq_fp, Wk_fp, Wk_hat = Wq_fp.to(dev, dt), Wk_fp.to(dev, dt), Wk_hat.to(dev, dt)
        bq = None if bq is None else bq.to(dev, dt)
        bk = None if bk is None else bk.to(dev, dt)
        cos, sin = cos.to(dev, dt), sin.to(dev, dt)

        if mu is None:
            mu = query_mean(lambda i: opt_x(i).to(dev, dt), n_opt, cfg.exclude_first)
        mu = mu.to(dev, dt)
        mu_err = mu if cfg.mode == "meanquery" else None

        W0 = lattice.weight()

        # ---- optimisation caches
        self._opt = []
        T_max = 2
        for i in range(n_opt):
            x = opt_x(i).to(dev, dt)
            T = x.shape[0]
            T_max = max(T_max, T)
            tq, sub = self._sample(T, cfg.queries_per_seq, cfg.cov_queries_per_seq, cfg.seed * 1_000_003 + i)
            self._opt.append(
                build_seq_cache(
                    x, cos[:T], sin[:T], geom, Wq_fp, bq, Wk_fp, bk, Wk_hat,
                    tq.to(dev), None if sub is None else sub.to(dev), mu=mu_err,
                )
            )
            del x

        G, H, curv, J_opt0, strata_info = self._initial_stats(geom, W0, bq, T_max, out, d)
        acc = self._build_accept(acc_x, n_acc, geom, cos, sin, Wq_fp, bq, Wk_fp, bk, Wk_hat, W0) if cfg.accept == "exact" else None
        J_acc0 = acc.J.clone() if acc is not None else None
        t_stats = time.time() - t_start

        # ---- lattice moves
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
        blocked = torch.zeros_like(valid)
        row_count = torch.zeros(out, dtype=torch.long, device=dev)
        bsize = torch.full((nh,), max(1, cfg.batch_init), dtype=torch.long, device=dev)
        rejects = torch.zeros(nh, dtype=torch.long, device=dev)
        finished = torch.zeros(nh, dtype=torch.bool, device=dev)
        W_cur = W0.clone()
        G_cur = G.clone()
        zeros_like_step = torch.zeros_like(step)
        inf_val = float("inf")
        ar_nh = torch.arange(nh, device=dev)
        c_cap = max(1, min(cfg.row_cap, d))
        rank_c = torch.arange(c_cap, device=dev)[None, :]

        hist = {"pred": [], "acc": [], "accepted": [], "nsel": [], "audit_pred": [], "audit_exact_opt": [], "audit_acc": []}
        rounds = 0
        since_refresh = 0
        n_refresh = 0
        just_refreshed = False

        for rnd in range(cfg.max_rounds if budget_row > 0 else 0):
            rounds = rnd + 1
            if cfg.refresh_every > 0 and since_refresh >= cfg.refresh_every:
                G_cur, _ = self._opt_pass(W_cur, geom, bq, out, need_c=True)
                since_refresh = 0
                n_refresh += 1

            rows_ok = (row_count < budget_row) & (~finished).repeat_interleave(h)
            avail = valid & ~used & ~blocked & rows_ok[:, None]
            score = step * G_cur + half_curv
            score = torch.where(avail & (score < 0), score, torch.full_like(score, inf_val))

            v1, i1 = torch.topk(score, c_cap, dim=1, largest=False)
            rem = (budget_row - row_count).clamp_min(0)
            v1 = torch.where(rank_c < rem[:, None], v1, torch.full_like(v1, inf_val))
            vh = v1.reshape(nh, h * c_cap)
            ih = i1.reshape(nh, h * c_cap)
            kk = int(min(int(bsize.max().item()), h * c_cap))
            v2, i2 = torch.topk(vh, kk, dim=1, largest=False)
            sel = torch.isfinite(v2) & (torch.arange(kk, device=dev)[None, :] < bsize[:, None])
            nsel = sel.sum(1)

            if int(nsel.sum().item()) == 0:
                if just_refreshed or cfg.refresh_every <= 0:
                    break
                G_cur, _ = self._opt_pass(W_cur, geom, bq, out, need_c=True)
                since_refresh = 0
                n_refresh += 1
                just_refreshed = True
                continue
            just_refreshed = False

            rows = (i2 // c_cap) + ar_nh[:, None] * h
            cols = ih.gather(1, i2)
            bmask = torch.zeros_like(valid)
            bmask[rows[sel], cols[sel]] = True
            dW = torch.where(bmask, step, zeros_like_step)
            dG = curv.matvec(dW)
            pred = (dW * G_cur + 0.5 * dW * dG).view(nh, h, d).sum(dim=(1, 2))
            active = nsel > 0

            if acc is not None:
                dJA, Ca = acc.delta(dW)
                thr = -cfg.accept_margin * acc.J.clamp_min(0.0)
                accept_h = active & (pred < 0) & (dJA < thr)
            else:
                dJA, Ca = None, None
                accept_h = active & (pred < 0)

            if cfg.audit_every > 0 and rnd % cfg.audit_every == 0:
                _, Jb = self._opt_pass(W_cur, geom, bq, out, need_c=False)
                _, Ja = self._opt_pass(W_cur + dW, geom, bq, out, need_c=False)
                hist["audit_pred"] += pred[active].tolist()
                hist["audit_exact_opt"] += (Ja - Jb)[active].tolist()
                if dJA is not None:
                    hist["audit_acc"] += dJA[active].tolist()

            acc_rows = accept_h.repeat_interleave(h)
            amask = bmask & acc_rows[:, None]
            if bool(amask.any()):
                lattice.codes += torch.where(amask, flip_dir, zeros_like_step)
                W_cur += torch.where(amask, step, zeros_like_step)
                used |= amask
                row_count += amask.sum(1)
                G_cur += dG * acc_rows[:, None].to(dt)
                if acc is not None:
                    acc.apply(Ca, dJA, accept_h)

            rej = active & ~accept_h
            single = rej & (nsel == 1)
            if bool(single.any()):
                m1 = sel & single[:, None]
                blocked[rows[m1], cols[m1]] = True
            rejects = torch.where(accept_h, torch.zeros_like(rejects), rejects + single.long())
            bsize = torch.where(accept_h, (bsize * 2).clamp_max(max(1, cfg.batch_max)), bsize)
            bsize = torch.where(rej & ~single, (nsel // 2).clamp_min(1), bsize)
            finished |= rejects >= cfg.patience

            hist["pred"] += pred[active].tolist()
            hist["accepted"] += accept_h[active].tolist()
            hist["nsel"] += nsel[active].tolist()
            if dJA is not None:
                hist["acc"] += dJA[active].tolist()
            since_refresh += 1
            if bool(finished.all()):
                break

        _, J_opt1 = self._opt_pass(W_cur, geom, bq, out, need_c=False)
        flips_head = row_count.view(nh, h).sum(1)
        log = {
            "config": asdict(cfg),
            "rounds": rounds,
            "refreshes": n_refresh,
            "budget_per_row": budget_row,
            "flips_total": int(flips_head.sum().item()),
            "flips_per_head": flips_head.tolist(),
            "J_opt_before": J_opt0.tolist(),
            "J_opt_after": J_opt1.tolist(),
            "J_acc_before": None if J_acc0 is None else J_acc0.tolist(),
            "J_acc_after": None if acc is None else acc.J.tolist(),
            "strata": strata_info,
            "history": hist,
            "time_stats_s": t_stats,
            "time_total_s": time.time() - t_start,
            "peak_mem_gb": (torch.cuda.max_memory_allocated(dev) / 1e9) if is_cuda else None,
        }
        self._opt = []
        self._Xe = None
        del acc, curv, G, H, G_cur
        if is_cuda:
            torch.cuda.empty_cache()
        return log
