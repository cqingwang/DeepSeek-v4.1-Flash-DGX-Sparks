"""CPU checks for adapter/wo_a_input_fp8.py (DSV41_WO_A_INPUT_FP8): the rounding of wo_a's input o
against DeepSeek's reference act_quant(o, 32, "ue8m0", float8_e8m0fnu, inplace=True)
(deepseek-ai/DeepSeek-V4.1-Flash inference/kernel.py; called from model.py since revision 4dd75a8b44).

`hf_act_quant_inplace` below transcribes the reference tilelang kernel step by step (amax floored at
1e-4, fast_round_scale from the fp32 bits, x / s, clamp +-448, cast to e4m3, times s, back to bf16);
`e4m3_rne` is an independent nearest-ties-to-even rounding onto the enumerated e4m3 grid, so a wrong
cast (truncation, ties away, flushed subnormals) fails too. Needs torch only.

With Triton importable AND (a GPU, or TRITON_INTERPRET=1 on CPU), the standalone kernel and the fused
QX routes of adapter/wo_a_w8.py are run as well and must match bit for bit. On a GPU run it twice, once
with DSV41_WO_A_INPUT_FP8_EMULATE=1 (the fp32 emulation of the e4m3 rounding) and once without (the
hardware conversion); without a GPU the interpreter checks the emulation:
    TRITON_INTERPRET=1 PYTHONPATH=adapter python3 tests/test_wo_a_input_fp8.py
"""
import os
import sys
import types
from pathlib import Path

os.environ["DSV41_WO_A_INPUT_FP8"] = "1"
if os.environ.get("TRITON_INTERPRET") == "1":
    # the interpreter's fp8 conversion does not round ties to even; check the fp32 emulation there
    os.environ["DSV41_WO_A_INPUT_FP8_EMULATE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "adapter"))
import torch  # noqa: E402

import wo_a_input_fp8 as q8  # noqa: E402

BF16 = torch.bfloat16


# ------------------------------------------------------------------------------------------
# reference transcriptions
# ------------------------------------------------------------------------------------------
def _f32(x):
    return torch.tensor(x, dtype=torch.float32)


def hf_fast_round_scale(amax):
    """kernel.py fast_round_scale(amax, fp8_max_inv) = fast_pow2(fast_log2_ceil(amax * fp8_max_inv))."""
    x = amax * _f32(1 / 448.0)                                 # fp8_max_inv = 1 / fp8_max (fp32 literal)
    bits_x = x.view(torch.int32)
    exp_x = (bits_x >> 23) & 0xFF
    man_bits = bits_x & ((1 << 23) - 1)
    log2c = exp_x - 127 + (man_bits != 0).to(torch.int32)      # fast_log2_ceil
    return ((log2c + 127) << 23).view(torch.float32)            # fast_pow2


def hf_act_quant_inplace(x, block=32):
    """act_quant_kernel(..., round_scale=True, inplace=True) on rows of `block` along the last dim."""
    xl = x.reshape(-1, block)
    amax = torch.maximum(xl.float().abs().amax(dim=1), _f32(1e-4))
    s = hf_fast_round_scale(amax)[:, None]
    v = torch.clamp(xl.float() / s, -448.0, 448.0)
    y = (v.to(torch.float8_e4m3fn).float() * s).to(x.dtype)
    return y.reshape(x.shape)


def _e4m3_grid():
    vals = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    vals = vals[torch.isfinite(vals)]
    return torch.unique(vals.abs())                             # +0 .. 448, sorted (-0 folded into +0)


def e4m3_rne(v):
    """v fp32 within +-448 -> nearest e4m3 value, ties to the even mantissa (enumerated grid)."""
    grid = _e4m3_grid()
    codes = torch.arange(len(grid))                             # consecutive codes: parity = mantissa LSB
    a = v.abs().double()
    hi = torch.searchsorted(grid.double(), a).clamp(max=len(grid) - 1)
    lo = (hi - 1).clamp(min=0)
    dlo, dhi = a - grid[lo].double(), grid[hi].double() - a
    pick_hi = (dhi < dlo) | ((dhi == dlo) & (codes[hi] % 2 == 0))
    r = torch.where(pick_hi, grid[hi], grid[lo]).float()
    return torch.where(torch.signbit(v), -r, r)                 # signed zero kept, like the cast


def independent_qdq(x, block=32):
    """Same rounding, every step re-derived: scale from frexp, e4m3 from the grid."""
    xl = x.reshape(-1, block).float()
    amax = xl.abs().amax(dim=1).clamp_min(_f32(1e-4)) * _f32(1 / 448.0)
    m, e = torch.frexp(amax)                                    # amax = m * 2^e, m in [0.5, 1)
    e = torch.where(m == 0.5, e - 1, e)                         # exact power of two: ceil(log2) = e - 1
    s = torch.ldexp(torch.ones_like(amax), e)[:, None]
    y = e4m3_rne((xl / s).clamp(-448, 448)) * s
    return y.to(x.dtype).reshape(x.shape)


# ------------------------------------------------------------------------------------------
# inputs
# ------------------------------------------------------------------------------------------
def attention_like(t=7, groups=2, d=4096, seed=0):
    g = torch.Generator().manual_seed(seed)
    o = torch.randn(t, groups, d, generator=g) * torch.exp(torch.randn(t, groups, d // 32, 1, generator=g) * 2).repeat_interleave(32, -2).squeeze(-1)
    return o.to(BF16)


def edge_blocks():
    """[n, 32] bf16 rows hitting the corners of the reference."""
    rows = [
        torch.zeros(32),                                        # all zero: floor 1e-4, stays 0
        torch.full((32,), 1e-6),                                # amax below the floor
        torch.tensor([1e-6, 1e-9, -3e-9, 5e-10] + [0.0] * 28),   # the floor decides which round to 0
        torch.full((32,), -0.0),                                # signed zero
        torch.linspace(-448, 448, 32),                          # amax exactly 448: scale 1
        torch.linspace(-448 * 2.0 ** -10, 448 * 2.0 ** -10, 32),  # amax exactly 448 * 2^-10
        torch.linspace(-224, 224, 32),                          # amax exactly 224
        torch.linspace(-449, 449, 32),
        torch.linspace(-1e4, 3e4, 32),                          # huge
        torch.cat([torch.tensor([448.0]), torch.linspace(-1, 1, 31) * 2.0 ** -12]),  # e4m3 subnormals
        torch.tensor([1.0] + [0.0] * 31) * 2.0 ** -20,          # tiny lone value
        torch.tensor([255.0, 76.0, 72.0, 80.0, 1.0 + 1 / 16, 1.0 + 3 / 16] + [0.0] * 26),  # ties when amax <= 448
    ]
    return torch.stack(rows).to(BF16)


# ------------------------------------------------------------------------------------------
# tests (torch only)
# ------------------------------------------------------------------------------------------
def test_e4m3_grid_and_cast():
    grid = _e4m3_grid()
    assert len(grid) == 127 and grid[-1] == 448 and grid[1] == 2.0 ** -9
    v = torch.linspace(-448, 448, 200_001)
    v = torch.cat([v, (grid[1:] + grid[:-1]) / 2, -(grid[1:] + grid[:-1]) / 2])   # every tie
    assert torch.equal(v.to(torch.float8_e4m3fn).float(), e4m3_rne(v)), "torch e4m3 cast is not RNE"


def test_matches_hf_transcription_and_independent():
    cases = [attention_like(t, seed=t) for t in (1, 3, 8, 37)] + [edge_blocks()]
    for x in cases:
        x = x.reshape(-1, x.shape[-1])
        want = hf_act_quant_inplace(x)
        assert torch.equal(q8.reference_qdq(x).view(torch.int16), want.view(torch.int16))
        assert torch.equal(independent_qdq(x).view(torch.int16), want.view(torch.int16))


def test_known_answers():
    # amax 1 -> scale 2^ceil(log2(1/448)) = 2^-8; bf16(0.3) = 0.30078125 -> 77 -> e4m3 80 -> 0.3125
    x = torch.tensor([1.0, 0.3, 76 / 256, 72 / 256, -100 / 256] + [0.0] * 27).to(BF16)
    y = q8.reference_qdq(x[None]).float()[0]
    assert y[0] == 1.0 and y[1] == 0.3125
    assert y[2] == 80 / 256              # tie 76 between 72 (odd mantissa) and 80 (even) -> 80
    assert y[3] == 72 / 256              # on the grid
    assert y[4] == -96 / 256             # tie 100 between 96 (even) and 104 -> 96
    # below the floor: scale 2^-22; 1e-6 * 2^22 = 4.19 -> 4.0
    z = q8.reference_qdq(torch.full((1, 32), 1e-6).to(BF16)).float()
    assert torch.all(z == 4.0 * 2.0 ** -22)
    # amax exactly 448 * 2^-10: 448 * fp32(1/448) rounds to 1.0, scale 2^-10, values kept
    w = torch.linspace(-448, 448, 32).round().to(BF16)[None] * 2.0 ** -10
    assert torch.equal(q8.reference_qdq(w.to(BF16)), hf_act_quant_inplace(w.to(BF16)))


def test_changes_o_and_blocks_are_32_along_last_dim():
    o = attention_like(5)
    r = q8.reference_qdq(o)
    rel = ((r.float() - o.float()).norm() / o.float().norm()).item()
    assert 0.003 < rel < 0.08, rel       # e4m3 has 3 mantissa bits: the step is not a no-op
    # a value that only moves block j's amax changes block j and nothing else
    p = o.clone()
    p[2, 1, 64 + 5] = 1e3
    d = (q8.reference_qdq(p) != r).nonzero().tolist()
    assert d and all(t == 2 and g == 1 and 64 <= c < 96 for t, g, c in d), d


def test_tp_split_is_rank_local():
    """TP1 view [T, 8 groups, 4096] vs each TP4 rank's 2 groups, and the flat [T, 32768] row."""
    full = attention_like(4, groups=8, seed=3)
    want = q8.reference_qdq(full)
    assert torch.equal(want, q8.reference_qdq(full.reshape(4, -1)).view_as(full))
    for rank in range(4):
        part = full[:, 2 * rank:2 * rank + 2].contiguous()
        assert torch.equal(q8.reference_qdq(part), want[:, 2 * rank:2 * rank + 2])
    # the reference's head layout: 64 heads x 512 -> 8 groups of 4096; blocks never cross a head
    assert 512 % q8.BLOCK == 0 and 4096 % q8.BLOCK == 0


def test_idempotent():
    for x in (attention_like(9, seed=7), edge_blocks()):
        once = q8.reference_qdq(x)
        assert torch.equal(q8.reference_qdq(once), once)


def test_round_o_inplace_strided_cpu():
    base = attention_like(6, groups=3)                         # o = base[:, 1:3]: stride(0) 12288
    o = base[:, 1:]
    want = q8.reference_qdq(o.contiguous())
    keep = base[:, 0].clone()
    assert q8.round_o_(o) is o
    assert torch.equal(o, want) and torch.equal(base[:, 0], keep)
    for bad in (torch.zeros(2, 2, 40, dtype=BF16), torch.zeros(2, 64, dtype=BF16), torch.zeros(2, 2, 64)):
        try:
            q8.round_o_(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad.shape} {bad.dtype}")
    q8.round_o_(torch.zeros(0, 2, 4096, dtype=BF16))


def test_install_wraps_dispatch_and_order():
    seen = {}

    def stock(o, wo_a, is_decode, **kw):
        seen["o"] = o.clone()
        return "stock"

    mod = types.SimpleNamespace(_apply_wo_a_bf16_matmul=stock)
    q8.install_model(mod)
    o = attention_like(3)
    raw = o.clone()
    assert mod._apply_wo_a_bf16_matmul(o, None, is_decode=False, is_target_verify=True) == "stock"
    assert torch.equal(seen["o"], q8.reference_qdq(raw))
    assert getattr(mod._apply_wo_a_bf16_matmul, "_dsv41_wo_a_input_fp8", False)
    wrapped = mod._apply_wo_a_bf16_matmul
    q8.install_model(mod)                                       # idempotent
    assert mod._apply_wo_a_bf16_matmul is wrapped
    for bad in (types.SimpleNamespace(), types.SimpleNamespace(_apply_wo_a_bf16_matmul=stock, _dsv41_wo_a_mid=True)):
        try:
            q8.install_model(bad)
        except RuntimeError:
            continue
        raise AssertionError("drift / wrong order accepted")
    # the draft must hold the wrapped dispatch (it imports it by name from deepseek_v4)
    v4 = types.SimpleNamespace(_apply_wo_a_bf16_matmul=wrapped)
    sys.modules["sglang.srt.models.deepseek_v4"] = v4
    try:
        q8.install_dspark(types.SimpleNamespace(_apply_wo_a_bf16_matmul=wrapped))
        try:
            q8.install_dspark(types.SimpleNamespace(_apply_wo_a_bf16_matmul=stock))
        except RuntimeError:
            pass
        else:
            raise AssertionError("draft holding the unwrapped dispatch accepted")
    finally:
        del sys.modules["sglang.srt.models.deepseek_v4"]


def test_w8_routes_round_exactly_once():
    """adapter/wo_a_w8.py's mid/drop wrapper (needs Triton to import; kernels replaced by recorders):
    its own routes get the raw o and QX on, every fall-through reaches the stock dispatch rounded,
    and the wrong install order is refused."""
    import wo_a_w8 as w8

    calls = []
    saved = {k: getattr(w8, k) for k in ("ENABLED", "MID", "DROP", "QX", "_partial_w8", "wo_a_mid",
                                          "dequant_into_scratch")}
    saved_state = (dict(w8._TWINS), set(w8._DROPPED), dict(w8._REFS))
    try:
        w8.ENABLED, w8.MID, w8.DROP, w8.QX = True, True, True, True
        w8.wo_a_mid = lambda o, twin, qx=False: calls.append(("mid", qx, o.clone())) or "mid"
        w8._partial_w8 = lambda o, twin, m, qx=False: calls.append(("partial", qx, o.clone())) or "p"
        w8.dequant_into_scratch = lambda twin, device: "scratch"
        w8._REFS["quantize_partial"] = lambda p: (p,)

        def stock(o, wo_a, *a, **kw):
            calls.append(("stock", wo_a, o.clone()))
            return "stock"

        try:
            w8.install_mid(types.SimpleNamespace(_apply_wo_a_bf16_matmul=stock))
        except RuntimeError:
            pass
        else:
            raise AssertionError("wo_a_w8 mid installed without the rounding wrapper under it")
        mod = types.SimpleNamespace(_apply_wo_a_bf16_matmul=stock, Mxfp8SwizzledInput=lambda *x: "swizzled")
        q8.install_model(mod)
        w8.install_mid(mod)
        call = mod._apply_wo_a_bf16_matmul
        wo, dropped = torch.zeros(4), torch.zeros(4)
        w8._TWINS[wo.data_ptr()] = w8._TWINS[dropped.data_ptr()] = ("twin",)
        verify = dict(is_decode=False, is_target_verify=True, is_prefill=False)

        def check(o_rows, weight, kw, route, qx):
            o = attention_like(o_rows, seed=o_rows)
            raw = o.clone()
            calls.clear()
            out = call(o, weight, **kw)
            name, flag, seen = calls[-1]
            assert name == route, (name, route)
            if route == "stock":
                assert torch.equal(seen, q8.reference_qdq(raw)), "fall-through not rounded"
            else:
                assert flag is qx and torch.equal(seen, raw), "fused route got a pre-rounded o"
                assert torch.equal(o, raw)
            return out, flag

        check(12, wo, verify, "mid", True)                                    # MID verify rows
        check(200, wo, verify, "stock", None)                                 # above MID: stock bmm
        check(12, wo, dict(verify, is_target_verify=False, is_prefill=True), "stock", None)  # prefill
        check(6, wo, verify, "stock", None)                                   # 2..8 via stock small (pre-rounded)
        w8._DROPPED.add(dropped.data_ptr())
        out, _ = check(6, dropped, dict(verify, fuse_mxfp8_quant=True), "partial", True)   # DROP 2..8
        assert out == "swizzled"
        check(40, dropped, verify, "mid", True)                               # DROP + MID
        out, flag = check(1, dropped, dict(verify, is_decode=True, is_target_verify=False), "stock", None)
        assert flag == "scratch"                                              # DROP 1 row: stock on scratch
    finally:
        for k, v in saved.items():
            setattr(w8, k, v)
        w8._TWINS.clear(); w8._TWINS.update(saved_state[0])
        w8._DROPPED.clear(); w8._DROPPED.update(saved_state[1])
        w8._REFS.clear(); w8._REFS.update(saved_state[2])


# ------------------------------------------------------------------------------------------
# Triton (GPU, or TRITON_INTERPRET=1)
# ------------------------------------------------------------------------------------------
def _triton_device():
    if q8.triton is None:
        return None
    if torch.cuda.is_available():
        return "cuda"
    return "cpu" if os.environ.get("TRITON_INTERPRET") == "1" else None


def test_triton_standalone_kernel(dev):
    run = q8._qdq_inplace_kernel
    for x in (attention_like(5, seed=11), attention_like(2, groups=2, d=96, seed=12),
              edge_blocks().reshape(1, 1, -1).repeat(3, 2, 1)):
        base = torch.cat([x, x[:, :1]], dim=1).to(dev)          # strided view: o = base[:, :-1]
        o = base[:, :-1]
        want = q8.reference_qdq(o.cpu().contiguous())
        t, g, d = o.shape
        nb = 32 if d % 1024 == 0 else 1
        run[(t, g, d // (nb * 32))](o, o.stride(0), o.stride(1), NB=nb, num_warps=4)
        assert torch.equal(o.cpu().view(torch.int16), want.view(torch.int16)), "standalone kernel"
        assert torch.equal(base[:, -1].cpu(), x[:, 0])


def test_triton_fused_w8_routes(dev):
    import wo_a_w8
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_wo_a_w8 import checkpoint_like

    twin = tuple(t.to(dev) for t in wo_a_w8.make_twin(checkpoint_like()))
    for m in (2, 6, 8):
        o = attention_like(m, seed=m).to(dev)
        rounded = q8.reference_qdq(o.cpu()).to(dev)
        fused = wo_a_w8._partial_w8(o, twin, m, qx=True)
        plain = wo_a_w8._partial_w8(rounded, twin, m)
        assert torch.equal(fused, plain), f"partial M={m}"
    if dev == "cpu":
        # the interpreter cannot run _wo_a_mid_kernel at all (its K loop ranges over program ids)
        print("  mid route: GPU only")
        return
    for m in (9, 24, 40, 100):
        o = attention_like(m, seed=m).to(dev)
        rounded = q8.reference_qdq(o.cpu()).to(dev)
        assert torch.equal(wo_a_w8.wo_a_mid(o, twin, qx=True), wo_a_w8.wo_a_mid(rounded, twin)), f"mid M={m}"


if __name__ == "__main__":
    test_e4m3_grid_and_cast()
    test_matches_hf_transcription_and_independent()
    test_known_answers()
    test_changes_o_and_blocks_are_32_along_last_dim()
    test_tp_split_is_rank_local()
    test_idempotent()
    test_round_o_inplace_strided_cpu()
    test_install_wraps_dispatch_and_order()
    if q8.triton is not None:
        test_w8_routes_round_exactly_once()
    dev = _triton_device()
    if dev is None:
        print("test_wo_a_input_fp8: ok (Triton checks skipped: no GPU and TRITON_INTERPRET unset)")
    else:
        test_triton_standalone_kernel(dev)
        test_triton_fused_w8_routes(dev)
        mode = "native e4m3" if q8._E4M3_NATIVE else "emulated e4m3"
        print(f"test_wo_a_input_fp8: ok (Triton checks on {dev}, {mode})")
