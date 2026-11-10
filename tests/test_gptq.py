"""GPTQ on the explicit lattice (CPU, tiny sizes).

    python -m pytest tests/test_gptq.py -v
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.post_correction.clc import CLCConfig, CLCCorrection
from src.quantization.base import ActStats, groupwise_quantize, quantize_linear
from src.quantization.gptq import GPTQConfig, gptq_quantize

DT = torch.float64


def make_data(out=24, d=64, n=512, seed=0, correlated=True, dtype=DT, mean_shift=0.0):
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(out, d, generator=g, dtype=DT)
    Z = torch.randn(n, d, generator=g, dtype=DT)
    if correlated:
        A = torch.randn(d, d, generator=g, dtype=DT) / d ** 0.5 + torch.eye(d, dtype=DT)
        X = Z @ A
    else:
        X = Z * torch.linspace(0.2, 3.0, d, dtype=DT)
    X = X + mean_shift * torch.randn(d, generator=g, dtype=DT)
    H = X.t() @ X / n
    return W.to(dtype), X.to(dtype), H.to(dtype)


def out_err(X, W, W_hat):
    return float(((X @ (W - W_hat).t()) ** 2).mean())


def deq(st):
    return st.dequantize()[:, : st.in_features]


VARIANTS = [
    # (act_order, static_groups, blocksize)
    (False, False, 16),
    (True, False, 16),
    (True, True, 16),
    (False, False, 7),   # blocks not aligned with groups -> straddling dynamic groups
    (True, False, 7),
]


# ----------------------------------------------------------------------------- exactness
@pytest.mark.parametrize("act_order,static", [(False, False), (True, True)])
def test_diagonal_hessian_reduces_to_rtn(act_order, static):
    """With uncorrelated inputs there is nothing to compensate: GPTQ == RTN, bit for bit."""
    W, X, _ = make_data(correlated=False, dtype=torch.float32)
    H = torch.diag((X * X).mean(0))
    cfg = GPTQConfig(act_order=act_order, static_groups=static, blocksize=16, dtype=torch.float32)
    st, _ = gptq_quantize(W, H, bits=3, group_size=16, cfg=cfg)
    ref = groupwise_quantize(W, bits=3, group_size=16)
    assert torch.equal(st.integer_weights, ref.integer_weights)
    assert torch.equal(st.scale, ref.scale)
    assert torch.equal(st.zero_point, ref.zero_point)
    assert torch.allclose(st.pre_round, ref.pre_round)


def naive_obs(W, H, bits, damp, order):
    """Reference: quantize columns one by one in `order`, exact OBS update of the remaining ones
    with the inverse of the (damped) Hessian restricted to the remaining columns. Per-row params."""
    W = W.clone()
    d = W.shape[1]
    Hd = H + damp * torch.diagonal(H).mean() * torch.eye(d, dtype=H.dtype)
    max_int = 2 ** bits - 1
    s = ((W.max(1).values - W.min(1).values) / max_int).clamp(min=1e-8)
    z = torch.round(-W.min(1).values / s).clamp(0, max_int)
    codes = torch.zeros_like(W)
    rem = list(order)
    while rem:
        j = rem[0]
        Hinv = torch.linalg.inv(Hd[rem][:, rem])
        q = torch.round(W[:, j] / s + z).clamp(0, max_int)
        e = (W[:, j] - (q - z) * s) / Hinv[0, 0]
        W[:, rem] -= e[:, None] * Hinv[0][None, :]
        codes[:, j] = q
        rem = rem[1:]
    return codes


@pytest.mark.parametrize("act_order", [False, True])
def test_matches_naive_obs_reference(act_order):
    W, X, H = make_data(out=16, d=32, n=256, seed=3)
    cfg = GPTQConfig(act_order=act_order, blocksize=8, damp=0.01, dtype=DT)
    st, _ = gptq_quantize(W, H, bits=3, group_size=0, cfg=cfg)
    order = torch.argsort(torch.diagonal(H), descending=True).tolist() if act_order else list(range(32))
    ref = naive_obs(W, H, 3, 0.01, order)
    mismatch = (st.integer_weights != ref).double().mean().item()
    assert mismatch < 5e-3, mismatch


@pytest.mark.parametrize("act_order,static", [(False, False), (True, False), (True, True)])
def test_lazy_batch_blocksize_invariance(act_order, static):
    W, X, H = make_data(seed=5)
    codes = []
    for bs in (1, 7, 16, 64):
        cfg = GPTQConfig(act_order=act_order, static_groups=static, blocksize=bs, dtype=DT)
        st, _ = gptq_quantize(W, H, bits=3, group_size=16, cfg=cfg)
        codes.append(st.integer_weights)
    for c in codes[1:]:
        assert (c != codes[0]).double().mean().item() < 5e-3


# ----------------------------------------------------------------------------- quality / invariants
@pytest.mark.parametrize("act_order,static,bs", VARIANTS)
def test_gptq_beats_rtn_on_correlated_inputs(act_order, static, bs):
    W, X, H = make_data(seed=1)
    cfg = GPTQConfig(act_order=act_order, static_groups=static, blocksize=bs, dtype=DT)
    st, info = gptq_quantize(W, H, bits=3, group_size=16, cfg=cfg)
    rtn = groupwise_quantize(W, bits=3, group_size=16)
    e_g = out_err(X, W, deq(st))
    e_r = out_err(X, W, rtn.dequantize()[:, :64].to(DT))
    assert e_g < e_r
    assert info["gptq_loss"] > 0 and info["gptq_dead"] == 0


@pytest.mark.parametrize("act_order,static,bs", VARIANTS)
def test_lattice_invariants(act_order, static, bs):
    W, X, H = make_data(seed=2)
    gs, bits = 16, 3
    cfg = GPTQConfig(act_order=act_order, static_groups=static, blocksize=bs, dtype=DT)
    st, _ = gptq_quantize(W, H, bits=bits, group_size=gs, cfg=cfg)
    c = st.integer_weights
    assert st.in_features == st.padded_in_features == W.shape[1]
    assert torch.equal(st.float_weights, W)                         # ORIGINAL W (CLC mean-shift target)
    assert torch.equal(c, torch.round(c)) and c.min() >= 0 and c.max() <= 2 ** bits - 1
    assert torch.equal(c, torch.round(st.pre_round).clamp(0, 2 ** bits - 1))   # nearest level of pre_round
    assert torch.equal(st.zero_point, torch.round(st.zero_point))
    if static or not act_order:
        # contiguous groups in the original column order
        sg = st.scale.view(W.shape[0], -1, gs)
        assert torch.equal(sg, sg[:, :, :1].expand_as(sg))
    else:
        # dynamic act-order groups: at most n_groups distinct (scale, zp) per row
        for r in range(W.shape[0]):
            pairs = torch.stack([st.scale[r], st.zero_point[r]], 1).unique(dim=0)
            assert pairs.shape[0] <= W.shape[1] // gs


def test_dead_channels_keep_weights_and_are_rtn_rounded():
    W, X, _ = make_data(seed=4)
    X[:, 5] = 0.0
    X[:, 17] = 0.0
    H = X.t() @ X / X.shape[0]
    st, info = gptq_quantize(W, H, bits=4, group_size=16, cfg=GPTQConfig(dtype=DT, act_order=False))
    assert info["gptq_dead"] == 2
    assert torch.equal(st.float_weights[:, 5], W[:, 5])
    assert torch.isfinite(st.pre_round).all()


def test_non_divisible_in_features():
    W, X, H = make_data(d=50, seed=6)
    for ao in (False, True):
        st, _ = gptq_quantize(W, H, bits=3, group_size=16, cfg=GPTQConfig(dtype=DT, act_order=ao, blocksize=16))
        assert st.integer_weights.shape == W.shape
        assert torch.isfinite(deq(st)).all()


# ----------------------------------------------------------------------------- integration
def test_actstats_second_moment():
    g = torch.Generator().manual_seed(0)
    x1 = torch.randn(2, 10, 8, generator=g)
    x2 = torch.randn(3, 4, 8, generator=g)
    a = ActStats(hessian=True)
    a.update(x1)
    a.update(x2)
    X = torch.cat([x1.reshape(-1, 8), x2.reshape(-1, 8)])
    assert a.n == X.shape[0]
    assert torch.allclose(a.second_moment(), X.t() @ X / X.shape[0], atol=1e-5)
    with pytest.raises(RuntimeError):
        ActStats().second_moment()


@pytest.mark.parametrize("with_clc", [False, True])
def test_quantize_linear_gptq_and_clc(with_clc):
    W, X, _ = make_data(seed=7, dtype=torch.float32, mean_shift=1.0)
    m = nn.Linear(W.shape[1], W.shape[0], bias=False)
    m.weight.data.copy_(W)
    act = ActStats(hessian=True)
    act.update(X)
    clc = CLCCorrection(CLCConfig(max_flip_percent=0.25)) if with_clc else None
    lat, info = quantize_linear(m, "gptq", 3, 16, act=act, clc=clc, keep_state=True,
                                gptq_cfg=GPTQConfig(act_order=True))
    assert torch.allclose(m.weight.data, lat.effective_weight().to(m.weight.dtype))
    assert torch.equal(lat.in_scales, torch.ones(W.shape[1]))
    assert "gptq_loss" in info
    if with_clc:
        # CLC must reduce the true per-row output mean shift mu^T (W - W_hat)
        base, _ = gptq_quantize(W, act.second_moment(), 3, 16, GPTQConfig(act_order=True))
        mu = clc.prepare_activation_means(act.mean())
        before = ((W - deq(base)) @ mu).abs()
        after = ((W - lat.effective_weight()) @ mu).abs()
        assert info["clc_flips"] > 0
        assert bool((after <= before + 1e-5).all())
        assert after.mean() < before.mean()
