"""Exactness tests for the Q-pass statistics, curvature and corrector (CPU, float64, tiny sizes).

    python -m pytest tests/ -v
"""

from __future__ import annotations

import torch

from src.attention.geometry import AttnGeometry, project_heads
from src.attention.qk_diag import attention_kl_diagnostics
from src.attention.qk_stats import build_seq_cache, compute_seq_stats
from src.attention.rope import rope_apply, rope_apply_T, rope_diag_var
from src.pipeline.sequential import parse_layer_spec
from src.post_correction.qk_alc import AcceptCache, KronCurvature, LatticeHandle, QKALCConfig, QKLatticeCorrector
from src.quantization.base import LatticeState, groupwise_quantize

DT = torch.float64


# ----------------------------------------------------------------------------- fixtures
def rope_tables(T, h, base=10000.0, factor=1.0):
    inv = 1.0 / (base ** (torch.arange(0, h, 2, dtype=DT) / h))
    f = torch.outer(torch.arange(T, dtype=DT), inv)
    emb = torch.cat([f, f], -1)
    return emb.cos() * factor, emb.sin() * factor


def rot_matrix(cos_t, sin_t):
    """Explicit P_t with P_t v == rope_apply(v) (HF rotate_half layout)."""
    h = cos_t.numel()
    h2 = h // 2
    P = torch.zeros(h, h, dtype=DT)
    for a in range(h2):
        b = a + h2
        c, s = cos_t[a], sin_t[a]
        P[a, a], P[a, b] = c, -s
        P[b, a], P[b, b] = s, c
    return P


def make_problem(seed=0, T=11, d=10, nh=4, nkv=2, h=6, noise=0.08, n_seq=2):
    g = torch.Generator().manual_seed(seed)

    def r(*shape):
        return torch.randn(*shape, generator=g, dtype=DT)

    geom = AttnGeometry(nh, nkv, h, h ** -0.5)
    xs = [r(T, d) for _ in range(n_seq)]
    Wq, bq = r(nh * h, d) * 0.5, r(nh * h) * 0.1
    Wk, bk = r(nkv * h, d) * 0.5, r(nkv * h) * 0.1
    Wq_hat = Wq + noise * r(nh * h, d)
    Wk_hat = Wk + noise * r(nkv * h, d)
    cos, sin = rope_tables(T, h, factor=1.1)  # non-unit attention factor on purpose
    return dict(geom=geom, xs=xs, Wq=Wq, bq=bq, Wk=Wk, bk=bk, Wq_hat=Wq_hat, Wk_hat=Wk_hat, cos=cos, sin=sin, T=T, d=d, gen=g)


def caches_for(pb, tq, sub=None, mu=None):
    return [
        build_seq_cache(x, pb["cos"], pb["sin"], pb["geom"], pb["Wq"], pb["bq"], pb["Wk"], pb["bk"], pb["Wk_hat"], tq, sub, mu=mu)
        for x in pb["xs"]
    ]


def J_mean(caches, W, pb):
    tot, N = 0.0, 0
    for sc in caches:
        st = compute_seq_stats(sc, W, pb["bq"], pb["geom"])
        tot = tot + st["J"].sum(1)
        N += sc.tq.numel()
    return tot / N


def G_H(caches, W, pb):
    geom = pb["geom"]
    out = geom.q_out
    c_rows, v_rows, xe = [], [], []
    for sc in caches:
        st = compute_seq_stats(sc, W, pb["bq"], geom, need_v=True)
        nq = sc.tq.numel()
        c_rows.append(st["c"].transpose(0, 1).reshape(nq, out))
        v_rows.append(st["v"].transpose(0, 1).reshape(nq, out))
        xe.append(sc.xe.expand(nq, -1))
    c, v, X = torch.cat(c_rows), torch.cat(v_rows), torch.cat(xe)
    s = geom.scaling
    N = X.shape[0]
    return s * c.t() @ X / N, s * s * v.t() @ (X * X) / N


def brute_J(pb, x, tq, W_hat, mu=None):
    """Independent implementation with explicit rotation matrices and python loops."""
    geom = pb["geom"]
    nh, h, s, grp = geom.n_heads, geom.head_dim, geom.scaling, geom.group
    T = x.shape[0]
    q = project_heads(x, pb["Wq"], pb["bq"], nh, h)
    k = project_heads(x, pb["Wk"], pb["bk"], geom.n_kv_heads, h)
    kh = project_heads(x, pb["Wk_hat"], pb["bk"], geom.n_kv_heads, h)
    xe = x if mu is None else mu.reshape(1, -1).expand(T, -1)
    qe = project_heads(xe, pb["Wq"], pb["bq"], nh, h)
    qeh = project_heads(xe, W_hat, pb["bq"], nh, h)
    P = [rot_matrix(pb["cos"][t], pb["sin"][t]) for t in range(T)]
    J = torch.zeros(nh, len(tq), dtype=DT)
    for j in range(nh):
        kv = j // grp
        for n, t in enumerate(tq.tolist()):
            logits = torch.stack([s * (P[t] @ q[t, j]) @ (P[u] @ k[u, kv]) for u in range(t + 1)])
            p = torch.softmax(logits, 0)
            eps = torch.stack([
                s * ((P[t] @ qeh[t, j]) @ (P[u] @ kh[u, kv]) - (P[t] @ qe[t, j]) @ (P[u] @ k[u, kv]))
                for u in range(t + 1)
            ])
            J[j, n] = 0.5 * ((p * eps * eps).sum() - (p * eps).sum() ** 2)
    return J


# ----------------------------------------------------------------------------- tests
def test_rope_identities():
    g = torch.Generator().manual_seed(1)
    h = 8
    cos, sin = rope_tables(5, h, factor=0.9)
    for t in range(5):
        P = rot_matrix(cos[t], sin[t])
        v = torch.randn(h, generator=g, dtype=DT)
        assert torch.allclose(rope_apply(v, cos[t], sin[t]), P @ v)
        assert torch.allclose(rope_apply_T(v, cos[t], sin[t]), P.t() @ v)
        # rope_diag_var == diag(P^T Cov(u) P) from weighted samples
        u = torch.randn(7, h, generator=g, dtype=DT)
        p = torch.softmax(torch.randn(7, generator=g, dtype=DT), 0)
        m = p @ u
        e2 = p @ (u * u)
        eab = p @ (u[:, : h // 2] * u[:, h // 2:])
        V = (u - m).t() @ torch.diag(p) @ (u - m)
        assert torch.allclose(rope_diag_var(m, e2, eab, cos[t], sin[t]), torch.diagonal(P.t() @ V @ P))


def test_J_matches_bruteforce_full_and_meanquery():
    pb = make_problem(seed=2)
    tq = torch.tensor([1, 3, 4, 7, 10])
    mu = pb["xs"][0][1:].mean(0)
    for mode_mu in (None, mu):
        caches = caches_for(pb, tq, mu=mode_mu)
        for sc, x in zip(caches, pb["xs"]):
            J = compute_seq_stats(sc, pb["Wq_hat"], pb["bq"], pb["geom"])["J"]
            Jb = brute_J(pb, x, tq, pb["Wq_hat"], mu=mode_mu)
            assert torch.allclose(J, Jb, atol=1e-12, rtol=1e-9)


def test_v_equals_diag_C():
    pb = make_problem(seed=3)
    tq = torch.arange(1, pb["T"])
    sc = caches_for(pb, tq)[0]
    st = compute_seq_stats(sc, pb["Wq_hat"], pb["bq"], pb["geom"], need_v=True, need_C=True, head_chunk=3)
    v = st["v"].transpose(0, 1)                                   # (n_q, nh, h)
    dC = torch.diagonal(st["C"], dim1=-2, dim2=-1)                # (n_q, nh, h)
    assert torch.allclose(v, dC, atol=1e-12)


def test_single_flip_change_is_exact():
    pb = make_problem(seed=4)
    geom = pb["geom"]
    h = geom.head_dim
    tq = torch.arange(1, pb["T"])
    for mu in (None, pb["xs"][1][1:].mean(0)):
        caches = caches_for(pb, tq, mu=mu)
        W = pb["Wq_hat"].clone()
        G, H = G_H(caches, W, pb)
        J0 = J_mean(caches, W, pb)
        g = torch.Generator().manual_seed(9)
        for _ in range(6):
            o = int(torch.randint(0, geom.q_out, (1,), generator=g))
            i = int(torch.randint(0, pb["d"], (1,), generator=g))
            delta = float(torch.randn(1, generator=g, dtype=DT)) * 0.3
            W1 = W.clone()
            W1[o, i] += delta
            dJ = J_mean(caches, W1, pb) - J0
            head = o // h
            pred = delta * G[o, i] + 0.5 * delta * delta * H[o, i]
            assert torch.allclose(dJ[head], pred, atol=1e-12, rtol=1e-9)
            others = torch.ones(geom.n_heads, dtype=torch.bool)
            others[head] = False
            assert torch.allclose(dJ[others], torch.zeros_like(dJ[others]), atol=1e-14)


def test_accept_cache_batch_is_exact_and_updates():
    pb = make_problem(seed=5)
    geom = pb["geom"]
    h, nh = geom.head_dim, geom.n_heads
    tq = torch.arange(1, pb["T"])
    caches = caches_for(pb, tq)
    W = pb["Wq_hat"].clone()

    def build_cache(Wc):
        Xs, cs, Cs, Js, N = [], [], [], 0.0, 0
        for sc in caches:
            st = compute_seq_stats(sc, Wc, pb["bq"], geom, need_C=True)
            Xs.append(sc.xe)
            cs.append(st["c"].transpose(0, 1))
            Cs.append(st["C"])
            Js = Js + st["J"].sum(1)
            N += sc.tq.numel()
        return AcceptCache(torch.cat(Xs), torch.cat(cs), torch.cat(Cs), Js / N, geom.scaling, nh, h)

    acc = build_cache(W)
    assert torch.allclose(acc.J, J_mean(caches, W, pb), atol=1e-13)

    g = torch.Generator().manual_seed(11)
    dW = torch.zeros_like(W)
    idx_o = torch.randint(0, geom.q_out, (9,), generator=g)
    idx_i = torch.randint(0, pb["d"], (9,), generator=g)
    dW[idx_o, idx_i] = torch.randn(9, generator=g, dtype=DT) * 0.2

    dJ, Ca = acc.delta(dW)
    true = J_mean(caches, W + dW, pb) - J_mean(caches, W, pb)
    assert torch.allclose(dJ, true, atol=1e-12, rtol=1e-9)

    accept_h = torch.zeros(nh, dtype=torch.bool)
    accept_h[0] = True
    accept_h[2] = True
    acc.apply(Ca, dJ, accept_h)
    W_new = W + dW * accept_h.repeat_interleave(h)[:, None].to(DT)
    ref = build_cache(W_new)
    assert torch.allclose(acc.J, ref.J, atol=1e-12)
    assert torch.allclose(acc.c, ref.c, atol=1e-12)


def test_kron_curvature_psd_diag_and_matvec():
    g = torch.Generator().manual_seed(7)
    nh, h, d = 2, 2, 3
    out = nh * h

    def psd(n, lead=()):
        A = torch.randn(*lead, n, n, generator=g, dtype=DT)
        return A @ A.transpose(-1, -2) + 1e-3 * torch.eye(n, dtype=DT)

    Ms = [psd(d), psd(d)]
    Cs = [psd(h, (nh,)), psd(h, (nh,))]
    H = torch.rand(out, d, generator=g, dtype=DT) * 3
    H[0, 0] = 0.0
    kc = KronCurvature(Ms, Cs, [0.3, 0.7], 0.25, H, nh, h)
    Hd = kc.dense()
    assert torch.linalg.eigvalsh(Hd).min() > -1e-10
    assert torch.allclose(torch.diagonal(Hd), H.reshape(-1), atol=1e-12)
    dW = torch.randn(out, d, generator=g, dtype=DT)
    dW[1] = 0.0
    assert torch.allclose(kc.matvec(dW).reshape(-1), Hd @ dW.reshape(-1), atol=1e-10)
    # no strata -> pure exact diagonal
    kc0 = KronCurvature([], [], [], 0.25, H, nh, h)
    assert torch.allclose(kc0.matvec(dW), H * dW)


def test_kron_is_exact_when_attention_covariance_is_constant():
    """If C_t is identical for all queries, K=1 Kronecker (no damping) equals the true Hessian."""
    g = torch.Generator().manual_seed(8)
    nh, h, d, n = 2, 2, 3, 5
    X = torch.randn(n, d, generator=g, dtype=DT)
    A = torch.randn(nh, h, h, generator=g, dtype=DT)
    C = A @ A.transpose(-1, -2)
    s = 0.7
    M = X.t() @ X / n
    H_true = torch.zeros(nh * h * d, nh * h * d, dtype=DT)
    for j in range(nh):
        sl = slice(j * h * d, (j + 1) * h * d)
        H_true[sl, sl] = s * s * torch.kron(C[j], M)
    Hdiag = torch.diagonal(H_true).reshape(nh * h, d)
    kc = KronCurvature([M], [C], [1.0], s * s, Hdiag, nh, h)
    assert torch.allclose(kc.dense(), H_true, atol=1e-12)


def test_groupwise_quantize_lattice():
    g = torch.Generator().manual_seed(3)
    W = torch.randn(6, 10, generator=g, dtype=torch.float32)
    st = groupwise_quantize(W, bits=3, group_size=5)
    assert st.integer_weights.min() >= 0 and st.integer_weights.max() <= 7
    assert ((st.pre_round - st.integer_weights).abs() <= 0.5 + 1e-5).all()
    err = (st.dequantize()[:, :10] - W).abs().max()
    assert err <= st.scale.max() * 0.5 + 1e-5


def _run_corrector(pb, cfg, n_opt=3, n_acc=2, seed=21):
    geom = pb["geom"]
    g = torch.Generator().manual_seed(seed)
    xs_opt = [torch.randn(pb["T"], pb["d"], generator=g, dtype=DT) for _ in range(n_opt)]
    xs_acc = [torch.randn(pb["T"], pb["d"], generator=g, dtype=DT) for _ in range(n_acc)]
    st = groupwise_quantize(pb["Wq"], bits=3, group_size=5)
    for f in ("float_weights", "pre_round", "integer_weights", "scale", "zero_point"):
        setattr(st, f, getattr(st, f).to(DT))
    lat = LatticeState(st, torch.ones(pb["d"], dtype=DT))
    handle = LatticeHandle.from_lattice_state(lat)
    codes0 = handle.codes.clone()
    corr = QKLatticeCorrector(cfg)
    log = corr.correct(
        geom=geom, cos=pb["cos"], sin=pb["sin"],
        opt_x=lambda i: xs_opt[i], n_opt=n_opt,
        acc_x=(lambda i: xs_acc[i]) if cfg.accept == "exact" else None, n_acc=n_acc if cfg.accept == "exact" else 0,
        Wq_fp=pb["Wq"], bq=pb["bq"], Wk_fp=pb["Wk"], bk=pb["bk"], Wk_hat=pb["Wk_hat"], lattice=handle,
    )
    return corr, handle, codes0, log, xs_acc


def test_corrector_end_to_end_invariants():
    pb = make_problem(seed=6, noise=0.0)
    cfg = QKALCConfig(
        mode="full", accept="exact", max_flip_percent=0.5, queries_per_seq=8, cov_queries_per_seq=4,
        accept_queries_per_seq=8, strata=2, entropy_bins=8, batch_init=2, batch_max=8, row_cap=2,
        max_rounds=40, refresh_every=3, patience=3, head_chunk=2, audit_every=4, seed=0,
    )
    corr, handle, codes0, log, xs_acc = _run_corrector(pb, cfg)
    geom = pb["geom"]
    d = pb["d"]

    # lattice invariants
    assert handle.codes.min() >= handle.min_int and handle.codes.max() <= handle.max_int
    diff = handle.codes - codes0
    fd = torch.sign(handle.pre - codes0)
    fd[fd == 0] = 1
    changed = diff != 0
    assert torch.equal(diff[changed], fd[changed])
    assert int(changed.sum(1).max()) <= int(cfg.max_flip_percent * d)
    assert log["flips_total"] == int(changed.sum())

    # exact acceptance => J on the acceptance set never increases, and the incremental value is exact
    Jb = torch.tensor(log["J_acc_before"], dtype=DT)
    Ja = torch.tensor(log["J_acc_after"], dtype=DT)
    assert (Ja <= Jb + 1e-12).all()
    W_new = handle.weight()
    tot, N = 0.0, 0
    for x, tq in zip(xs_acc, corr.last_accept_queries):
        sc = build_seq_cache(x, pb["cos"], pb["sin"], geom, pb["Wq"], pb["bq"], pb["Wk"], pb["bk"], pb["Wk_hat"], tq)
        tot = tot + compute_seq_stats(sc, W_new, pb["bq"], geom)["J"].sum(1)
        N += tq.numel()
    assert torch.allclose(tot / N, Ja, atol=1e-11)


def test_corrector_meanquery_without_acceptance_runs():
    pb = make_problem(seed=7, noise=0.0)
    cfg = QKALCConfig(
        mode="meanquery", accept="none", max_flip_percent=0.3, queries_per_seq=6, cov_queries_per_seq=3,
        strata=1, entropy_bins=4, batch_init=2, batch_max=4, row_cap=1, max_rounds=10, refresh_every=2,
        head_chunk=4, seed=1,
    )
    _, handle, codes0, log, _ = _run_corrector(pb, cfg)
    assert log["J_acc_after"] is None
    assert ((handle.codes - codes0).abs() <= 1).all()


def test_diagnostics_zero_when_unquantized():
    pb = make_problem(seed=8)
    x = pb["xs"][0]
    sums, cnt = attention_kl_diagnostics(
        x, pb["cos"], pb["sin"], pb["geom"], pb["Wq"], pb["bq"], pb["Wk"], pb["bk"], pb["Wq"], pb["Wk"],
        mu=x[1:].mean(0), head_chunk=3,
    )
    assert cnt == pb["T"] - 1
    for k, v in sums.items():
        assert torch.allclose(v, torch.zeros_like(v), atol=1e-12), k


def test_parse_layer_spec():
    assert parse_layer_spec("all", 4) is None
    assert parse_layer_spec("0-2,5,9", 6) == {0, 1, 2, 5}
