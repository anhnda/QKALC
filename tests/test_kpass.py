"""Exactness tests for the K-pass (CPU, float64)."""

from __future__ import annotations

import torch

from src.attention.qk_stats import build_seq_cache, compute_seq_stats
from src.post_correction.k_alc import KALCConfig, KAcceptSet, KLatticeCorrector, build_k_cache, k_seq_stats
from src.post_correction.qk_alc import LatticeHandle, QKALCConfig, QKLatticeCorrector
from src.quantization.base import LatticeState, groupwise_quantize
from tests.test_qk_math import DT, make_problem


def kcaches(pb, tq, Wq_hat, sub=None):
    return [
        build_k_cache(x, pb["cos"], pb["sin"], pb["geom"], pb["Wq"], pb["bq"], pb["Wk"], pb["bk"], Wq_hat, tq, sub)
        for x in pb["xs"]
    ]


def JK(pb, caches, Wk_hat):
    tot, N = 0.0, 0
    for kc, x in zip(caches, pb["xs"]):
        tot = tot + k_seq_stats(kc, x, pb["cos"], pb["sin"], Wk_hat, pb["bk"], pb["geom"])["J"].sum(1)
        N += kc.tq.numel()
    return tot / N


def test_k_objective_matches_q_side_implementation():
    pb = make_problem(seed=12)
    tq = torch.tensor([1, 2, 5, 6, 9, 10])
    Wk2 = pb["Wk_hat"] + 0.03 * torch.randn(pb["Wk_hat"].shape, generator=pb["gen"], dtype=DT)
    Jk = JK(pb, kcaches(pb, tq, pb["Wq_hat"]), Wk2)
    tot, N = 0.0, 0
    for x in pb["xs"]:
        sc = build_seq_cache(x, pb["cos"], pb["sin"], pb["geom"], pb["Wq"], pb["bq"], pb["Wk"], pb["bk"], Wk2, tq)
        tot = tot + compute_seq_stats(sc, pb["Wq_hat"], pb["bq"], pb["geom"])["J"].sum(1)
        N += tq.numel()
    assert torch.allclose(Jk, tot / N, atol=1e-13)


def test_k_single_flip_change_is_exact():
    pb = make_problem(seed=13)
    geom = pb["geom"]
    s, h, grp = geom.scaling, geom.head_dim, geom.group
    tq = torch.arange(1, pb["T"])
    caches = kcaches(pb, tq, pb["Wq_hat"], sub=None)  # sub=None -> variance part on all queries (exact)
    W = pb["Wk_hat"].clone()
    Gs, Hs, N = 0.0, 0.0, 0
    for kc, x in zip(caches, pb["xs"]):
        st = k_seq_stats(kc, x, pb["cos"], pb["sin"], W, pb["bk"], geom, need_H=True, head_chunk=3)
        Gs, Hs, N = Gs + st["G"], Hs + st["H"], N + kc.tq.numel()
    G, H = s * Gs / N, s * s * Hs / N
    J0 = JK(pb, caches, W)
    g = torch.Generator().manual_seed(5)
    for _ in range(6):
        o = int(torch.randint(0, W.shape[0], (1,), generator=g))
        i = int(torch.randint(0, W.shape[1], (1,), generator=g))
        delta = float(torch.randn(1, generator=g, dtype=DT)) * 0.3
        W1 = W.clone()
        W1[o, i] += delta
        dJ = JK(pb, caches, W1) - J0
        pred = delta * G[o, i] + 0.5 * delta * delta * H[o, i]
        assert torch.allclose(dJ.sum(), pred, atol=1e-12, rtol=1e-9)
        kv = o // h
        heads = torch.arange(geom.n_heads) // grp == kv
        assert torch.allclose(dJ[~heads], torch.zeros_like(dJ[~heads]), atol=1e-14)


def test_k_diag_hessian_subset_is_nonnegative_and_consistent():
    pb = make_problem(seed=17)
    geom = pb["geom"]
    tq = torch.arange(1, pb["T"])
    sub = torch.tensor([0, 4, 7])
    x = pb["xs"][0]
    kc_sub = kcaches(pb, tq, pb["Wq_hat"], sub=sub)[0]
    H_sub = k_seq_stats(kc_sub, x, pb["cos"], pb["sin"], pb["Wk_hat"], pb["bk"], geom, need_H=True, head_chunk=2)["H"]
    assert H_sub.min() > -1e-12
    # subset estimate == (n_q / n_sub) * exact H restricted to those queries
    kc_only = kcaches(pb, tq[sub], pb["Wq_hat"], sub=None)[0]
    H_only = k_seq_stats(kc_only, x, pb["cos"], pb["sin"], pb["Wk_hat"], pb["bk"], geom, need_H=True)["H"]
    assert torch.allclose(H_sub, (tq.numel() / sub.numel()) * H_only, atol=1e-12)


def test_k_accept_set_is_exact_and_updates():
    pb = make_problem(seed=14)
    geom = pb["geom"]
    h, grp = geom.head_dim, geom.group
    tq = torch.tensor([1, 3, 4, 8, 10])
    caches = kcaches(pb, tq, pb["Wq_hat"])
    W = pb["Wk_hat"].clone()
    acc = KAcceptSet(caches, lambda i: pb["xs"][i], pb["cos"], pb["sin"], geom, W, pb["bk"])
    assert torch.allclose(acc.J, JK(pb, caches, W), atol=1e-13)

    g = torch.Generator().manual_seed(6)
    dW = torch.zeros_like(W)
    ro = torch.randint(0, W.shape[0], (7,), generator=g)
    co = torch.randint(0, W.shape[1], (7,), generator=g)
    dW[ro, co] = torch.randn(7, generator=g, dtype=DT) * 0.2
    dJ, new = acc.delta(dW)
    assert torch.allclose(dJ, JK(pb, caches, W + dW) - JK(pb, caches, W), atol=1e-12, rtol=1e-9)

    accept_g = torch.tensor([True, False])  # 2 KV groups in make_problem
    acc.apply(new, accept_g.repeat_interleave(grp))
    W_new = W + dW * accept_g.repeat_interleave(h)[:, None].to(DT)
    assert torch.allclose(acc.J, JK(pb, caches, W_new), atol=1e-12)


def _lattice_from(W, d):
    st = groupwise_quantize(W, bits=3, group_size=5)
    for f in ("float_weights", "pre_round", "integer_weights", "scale", "zero_point"):
        setattr(st, f, getattr(st, f).to(DT))
    return LatticeState(st, torch.ones(d, dtype=DT))


def test_k_corrector_end_to_end_invariants():
    pb = make_problem(seed=15, noise=0.0)
    geom = pb["geom"]
    d = pb["d"]
    g = torch.Generator().manual_seed(31)
    xs_opt = [torch.randn(pb["T"], d, generator=g, dtype=DT) for _ in range(3)]
    xs_acc = [torch.randn(pb["T"], d, generator=g, dtype=DT) for _ in range(2)]
    handle = LatticeHandle.from_lattice_state(_lattice_from(pb["Wk"], d))
    codes0 = handle.codes.clone()
    Wq_hat = pb["Wq"] + 0.05 * torch.randn(pb["Wq"].shape, generator=g, dtype=DT)
    cfg = KALCConfig(
        accept="exact", max_flip_percent=0.5, queries_per_seq=8, h_sub_queries_per_seq=3, accept_queries_per_seq=8,
        batch_init=2, batch_max=8, row_cap=2, max_rounds=30, refresh_every=2, patience=3, head_chunk=2, seed=0,
    )
    corr = KLatticeCorrector(cfg)
    log = corr.correct(
        geom=geom, cos=pb["cos"], sin=pb["sin"], opt_x=lambda i: xs_opt[i], n_opt=3,
        acc_x=lambda i: xs_acc[i], n_acc=2, Wq_fp=pb["Wq"], bq=pb["bq"], Wk_fp=pb["Wk"], bk=pb["bk"],
        Wq_hat=Wq_hat, lattice=handle,
    )
    diff = handle.codes - codes0
    fd = torch.sign(handle.pre - codes0)
    fd[fd == 0] = 1
    ch = diff != 0
    assert torch.equal(diff[ch], fd[ch])
    assert handle.codes.min() >= 0 and handle.codes.max() <= 7
    assert int(ch.sum(1).max()) <= int(cfg.max_flip_percent * d)
    assert log["flips_total"] == int(ch.sum())
    Jb = torch.tensor(log["J_acc_before"], dtype=DT)
    Ja = torch.tensor(log["J_acc_after"], dtype=DT)
    assert Ja.sum() <= Jb.sum() + 1e-12
    # incremental acceptance J equals a from-scratch Q-side computation with the corrected K
    Wk_new = handle.weight()
    tot, N = 0.0, 0
    for x, tq in zip(xs_acc, corr.last_accept_queries):
        sc = build_seq_cache(x, pb["cos"], pb["sin"], geom, pb["Wq"], pb["bq"], pb["Wk"], pb["bk"], Wk_new, tq)
        tot = tot + compute_seq_stats(sc, Wq_hat, pb["bq"], geom)["J"].sum(1)
        N += tq.numel()
    assert torch.allclose(tot / N, Ja, atol=1e-11)


def test_repeated_q_pass_respects_budget_and_single_flip():
    pb = make_problem(seed=16, noise=0.0)
    d = pb["d"]
    g = torch.Generator().manual_seed(41)
    xs = [torch.randn(pb["T"], d, generator=g, dtype=DT) for _ in range(4)]
    handle = LatticeHandle.from_lattice_state(_lattice_from(pb["Wq"], d))
    cfg = QKALCConfig(
        accept="exact", max_flip_percent=0.3, queries_per_seq=8, cov_queries_per_seq=4, accept_queries_per_seq=8,
        strata=1, entropy_bins=4, batch_init=2, batch_max=4, row_cap=1, max_rounds=15, refresh_every=3, head_chunk=4,
    )
    corr = QKLatticeCorrector(cfg)
    for _ in range(2):
        corr.correct(
            geom=pb["geom"], cos=pb["cos"], sin=pb["sin"], opt_x=lambda i: xs[i], n_opt=2,
            acc_x=lambda i: xs[2 + i], n_acc=2, Wq_fp=pb["Wq"], bq=pb["bq"], Wk_fp=pb["Wk"], bk=pb["bk"],
            Wk_hat=pb["Wk_hat"], lattice=handle,
        )
    diff = handle.codes - handle.codes_ref
    assert diff.abs().max() <= 1
    assert int((diff != 0).sum(1).max()) <= int(cfg.max_flip_percent * d)
