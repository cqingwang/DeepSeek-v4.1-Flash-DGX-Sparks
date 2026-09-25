"""GPU check for adapter/hc_fused.py: the one-walk hc mix stats must equal the stock split-K +
reduce/sinkhorn bit for bit (torch.equal on pre, post and comb) at every row count, plus the
drift guard and the dispatch. Needs the engine image and a GPU:
  docker run --rm --gpus all -v $PWD:/ds41 -w /ds41 -e PYTHONPATH=/ds41/adapter \
      --entrypoint python3 <image> tests/test_hc_fused.py
"""
import os
import types

os.environ["DSV41_HC_FUSED"] = "1"
import torch  # noqa: E402

import hc_fused  # noqa: E402

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(min(1.0, 2e9 / torch.cuda.get_device_properties(0).total_memory))

K, MIX, HC, ITERS, RMS_EPS, HC_EPS = 20480, 24, 4, 20, 1e-20, 1e-6
ROWS = list(range(1, 97)) + [127, 128, 255, 256, 511, 512, 513, 1000, 1024, 2047, 2048,
                              2049, 3000, 4095, 4096, 4097, 8192]


def _mod():
    from sglang.kernels.ops.layernorm import mhc
    return mhc


def _weights(seed):
    """Checkpoint-like: fn ~ N(0, 0.0225) fp32 (full mantissa), base ~ +-3, scale ~ 0.01-0.06."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    fn = torch.randn(MIX, K, device="cuda", generator=g) * 0.0225
    base = torch.randn(MIX, device="cuda", generator=g) * 3
    scale = torch.rand(3, device="cuda", generator=g) * 0.05 + 0.01
    return fn, scale, base


def _x(m, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, K, device="cuda", generator=g) * (0.1 + 4 * (seed % 3))
    x[:, :64] *= 300                     # a few large channels
    if m > 2:
        x[1] = 0                         # an all-zero row
        x[2, ::7] = -0.0
    return x.to(torch.bfloat16)


def test_drift_guard():
    mod = _mod()
    hc_fused.check_engine(mod)
    fake = types.SimpleNamespace(**{n: getattr(mod, n) for n in dir(mod) if not n.startswith("__")})

    def hc_mix_stats_sinkhorn(*a):      # a different body under the same name
        return None
    fake.hc_mix_stats_sinkhorn = hc_mix_stats_sinkhorn
    try:
        hc_fused.check_engine(fake)
    except RuntimeError:
        return
    raise AssertionError("drifted engine accepted")


def test_bit_exact_all_rows():
    mod = _mod()
    for i, m in enumerate(ROWS):
        fn, scale, base = _weights(i % 5)
        x = _x(m, i)
        args = (x, fn, scale, base, HC, ITERS, RMS_EPS, HC_EPS)
        assert hc_fused.eligible(x, fn, HC, min_rows=1)
        ref = mod.hc_mix_stats_sinkhorn(*args)
        got = hc_fused.fused_mix_stats_sinkhorn(mod, *args)
        for name, a, b in zip(("pre", "post", "comb"), ref, got):
            if not torch.equal(a, b):
                raise AssertionError(f"rows {m}: {name} differs, max abs {(a - b).abs().max().item():.3e}")
    # strided rows (a view of a wider buffer) and the real checkpoint when it is mounted
    wide = _x(600, 99)
    fn, scale, base = _weights(7)
    view = torch.empty(600, K + 128, dtype=torch.bfloat16, device="cuda")[:, :K]
    view.copy_(wide)
    args = (view, fn, scale, base, HC, ITERS, RMS_EPS, HC_EPS)
    assert all(torch.equal(a, b) for a, b in zip(mod.hc_mix_stats_sinkhorn(*args),
                                                 hc_fused.fused_mix_stats_sinkhorn(mod, *args)))


def test_real_checkpoint_weights():
    root = "/models/deepseek-ai/DeepSeek-V4.1-Flash"
    if not os.path.exists(root):
        return
    import json
    from safetensors import safe_open
    mod = _mod()
    idx = json.load(open(f"{root}/model.safetensors.index.json"))["weight_map"]
    names = sorted(k for k in idx if k.endswith("hc_attn_fn") or k.endswith("hc_ffn_fn"))[:4]
    for j, name in enumerate(names):
        p = name[:-2]
        with safe_open(f"{root}/{idx[name]}", "pt") as f:
            fn, scale, base = (f.get_tensor(p + s).float().cuda() for s in ("fn", "scale", "base"))
        for m in (6, 96, 4096):
            args = (_x(m, 50 + j), fn, scale, base, HC, ITERS, RMS_EPS, HC_EPS)
            ref = mod.hc_mix_stats_sinkhorn(*args)
            got = hc_fused.fused_mix_stats_sinkhorn(mod, *args)
            assert all(torch.equal(a, b) for a, b in zip(ref, got)), (name, m)


def test_dispatch():
    calls = []
    mod = types.SimpleNamespace(**{n: getattr(_mod(), n) for n in dir(_mod()) if not n.startswith("__")})
    stock = mod.hc_mix_stats_sinkhorn

    def counting(*a):
        calls.append(a[0].shape[0])
        return stock(*a)
    mod.hc_mix_stats_sinkhorn = counting
    hc_fused._state.update(checked=False, disabled=False)
    guard = hc_fused.check_engine
    hc_fused.check_engine = lambda m: None                 # the counting stub is not the audited source
    try:
        hc_fused.install(mod)
    finally:
        hc_fused.check_engine = guard
    fn, scale, base = _weights(1)
    small = _x(96, 3)
    mod.hc_mix_stats_sinkhorn(small, fn, scale, base, HC, ITERS, RMS_EPS, HC_EPS)
    assert calls == [96]                                   # below MIN_ROWS: stock
    big = _x(hc_fused.MIN_ROWS, 4)
    mod.hc_mix_stats_sinkhorn(big, fn, scale, base, HC, ITERS, RMS_EPS, HC_EPS)
    assert calls == [96, hc_fused.MIN_ROWS] and hc_fused._state["checked"]   # first: self-check
    assert not hc_fused._state["disabled"]
    mod.hc_mix_stats_sinkhorn(big, fn, scale, base, HC, ITERS, RMS_EPS, HC_EPS)
    assert calls == [96, hc_fused.MIN_ROWS]                # then fused only
    mod.hc_mix_stats_sinkhorn(big.float(), fn, scale, base, HC, ITERS, RMS_EPS, HC_EPS)
    assert calls[-1] == hc_fused.MIN_ROWS and len(calls) == 3   # fp32 activations: stock


if __name__ == "__main__":
    test_drift_guard()
    test_bit_exact_all_rows()
    test_real_checkpoint_weights()
    test_dispatch()
    print(f"test_hc_fused: ok ({len(ROWS)} row counts bit-exact)")
