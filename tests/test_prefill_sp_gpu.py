"""GPU check for adapter/prefill_sp.py on one GPU (engine image, <= 2 GB):

  docker run --rm --gpus all --network none -v $PWD:/ds41 -w /ds41 -e PYTHONPATH=/ds41/adapter \
      --entrypoint python3 <image> tests/test_prefill_sp_gpu.py [drift] [kernels] [emulate]

drift    the pinned engine sources hash as expected in this image (incl. the tail helpers), a
         drifted one is refused, and SP refuses an hc_prefill_fused that is not SP-aware.
kernels  every per-row kernel stage 1 runs on a shard gives, on rows [r*M/4, (r+1)*M/4), the bits
         of the full-M call when the gates key on the full row count (sp_logical_rows), at
         M = 4096 / 2048 / 1500 / 128: hc combine(+norm) incl. the first-layer norm, hc mix stats
         (hc_fused on and off), hc_post, the fused Engram gate, the Engram wkv MXFP8 GEMM
         (b12x, as production routes it, and FlashInfer cutlass). A shard-keyed control shows
         which of them would change bits without the helper.
emulate  four TP ranks as threads (baton-passing, so one issues GPU work at a time) on a fake
         process group; one full-row layer (the engine's real forward_hc_pre_from_prev and hc
         kernels, a real Engram gate and MXFP8 wkv, stand-in attention/MoE whose wo_b / expert
         outputs are per-rank partials) followed by one bounded-replay tail layer, through
         prefill_sp's loop copy. stage-1 EXACT must equal the unsharded reference bit for bit on
         every rank, stage-1 fast must equal P0 (comm) bit for bit.
"""
import os
import sys
import threading
import types
from contextlib import contextmanager, nullcontext

os.environ.setdefault("DSV41_PREFILL_SP", "1")
os.environ.setdefault("DSV41_HC_FUSED", "1")
os.environ.pop("DSV41_HC_PREFILL_FUSED", None)
import torch  # noqa: E402

import prefill_sp as sp  # noqa: E402

torch.cuda.set_per_process_memory_fraction(min(1.0, 2e9 / torch.cuda.get_device_properties(0).total_memory))
DEV = "cuda"
BF = torch.bfloat16
H, HC, MIX, ITERS = 5120, 4, 24, 20
W = 4


def _mods():
    from sglang.kernels.ops.layernorm import mhc
    from sglang.srt.models import deepseek_v4 as dm
    return mhc, dm


_INSTALLED = {}


def setup():
    """hc_fused as production has it, b12x MXFP8 routing, prefill_sp's modules + hc wrappers."""
    if _INSTALLED:
        return _INSTALLED
    mhc, dm = _mods()
    # no runtime context in a bare container: the one query mhc_post makes (prefill CP interleave?)
    # answers as production does (no CP)
    mhc.is_dsa_prefill_cp_interleave = lambda: False
    import hc_fused
    hc_fused.install(mhc)
    import mxfp8_b12x
    from sglang.srt.layers.quantization import fp8_utils
    _INSTALLED["mxfp8_cutlass"] = fp8_utils.flashinfer_mxfp8_blockscaled_linear
    mxfp8_b12x.install(fp8_utils)
    sp._load_modules(dm)
    L = dm.DeepseekV4DecoderLayer
    _INSTALLED.update(
        mhc=mhc, dm=dm, hc_fused=hc_fused, fp8_utils=fp8_utils,
        combine=sp._make_hc_combine(L._hc_combine), post=sp._make_hc_post(L.hc_post),
        stats=L._hc_mix_stats, fwd=L.forward_hc_pre_from_prev, stream=L._get_hc_stats_stream)
    return _INSTALLED


def _norm(seed):
    from sglang.srt.layers.layernorm import RMSNorm
    n = RMSNorm(H, eps=1e-6).to(DEV)
    g = torch.Generator(device=DEV).manual_seed(seed)
    with torch.no_grad():
        n.weight.copy_((1 + 0.1 * torch.randn(H, device=DEV, generator=g)).to(n.weight.dtype))
    n.weight.data = n.weight.data.to(BF)
    return n


def _hc_params(seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    fn = torch.randn(MIX, HC * H, device=DEV, generator=g) * 0.0225
    base = torch.randn(MIX, device=DEV, generator=g) * 3
    scale = torch.rand(3, device=DEV, generator=g) * 0.05 + 0.01
    return fn, scale, base


def _residual(m, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    r = torch.randn(m, HC, H, device=DEV, generator=g, dtype=BF) * 0.5
    r[:, :, :64] *= 40
    return r


class _Fp8Linear:
    """The MXFP8 dense linear the Engram wkv runs (fp8 weight, ue8m0 block scales, swizzled)."""

    def __init__(self, n, k, seed, routed=True):
        from flashinfer import block_scale_interleave
        g = torch.Generator(device=DEV).manual_seed(seed)
        self.w = (torch.randn(n, k, device=DEV, generator=g) * 2).to(torch.float8_e4m3fn)
        su8 = (127 + torch.randint(-6, 1, (n, k // 32), device=DEV, generator=g)).to(torch.uint8)
        self.sf = block_scale_interleave(su8.contiguous()).contiguous()
        self.routed = routed        # True: production routing (mxfp8_b12x -> b12x)
        self.calls = []

    def __call__(self, x):
        self.calls.append(x.shape[0])
        s = setup()
        f = s["fp8_utils"].flashinfer_mxfp8_blockscaled_linear if self.routed else s["mxfp8_cutlass"]
        return f(x, self.w, self.sf, backend="cutlass"), None


class Layer:
    """DeepseekV4DecoderLayer's hc methods (real engine code, SP-wrapped where stage 1 wraps them)
    on a stand-in carrying the attributes they read."""
    hc_mult, hc_sinkhorn_iters, rms_norm_eps, hc_eps = HC, ITERS, 1e-6, 1e-6
    hc_pre_from_prev_sublayer = True
    hc_stats_stream = None
    dsa_enable_prefill_cp = False
    use_fused_mhc_post_pre = False
    config = types.SimpleNamespace(model_type="deepseek_v41", vision_n_layers=0, image_token_id=7)

    def __init__(self, seed, layer_id=0, attn=None, mlp=None, engram=None):
        self.layer_id = layer_id
        self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base = _hc_params(seed)
        self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base = _hc_params(seed + 50)
        self.input_layernorm, self.post_attention_layernorm = _norm(seed + 1), _norm(seed + 2)
        self.self_attn, self.mlp, self.engram = attn, mlp, engram

    def _run_moe_ffn_dp_sync(self, hidden_states, forward_batch, *, input_ids, input_ids_global):
        # the TP-only branch of the engine's method (no DP gather / CP / a2a)
        with FORWARD.scoped(mlp_reduce_scatter=False):
            return self.mlp(hidden_states, forward_batch, input_ids=input_ids,
                            input_ids_global=input_ids_global, skip_shared_experts=False)


def _bind_layer_methods():
    s = setup()
    Layer._hc_combine = s["combine"]
    Layer.hc_post = s["post"]
    Layer._hc_mix_stats = s["stats"]
    Layer.forward_hc_pre_from_prev = s["fwd"]
    Layer._get_hc_stats_stream = s["stream"]


@contextmanager
def _as_shard(m, r):
    prev = sp._ctx.plan
    sp._ctx.plan = sp._Plan(m, W, r, None, True, False)
    try:
        yield
    finally:
        sp._ctx.plan = prev


# ============================================================================================
def test_drift():
    s = setup()
    got = sp.engine_hashes()
    bad = {k: v for k, v in got.items() if sp._EXPECTED[k] != v}
    assert not bad, bad
    sp._check_tail_engine()
    fp8_old, sp.FP8 = sp.FP8, True
    try:
        sp.check_engine()                     # incl. the stage-2 attention sources
    finally:
        sp.FP8 = fp8_old
    saved = dict(sp._EXPECTED)
    sp._EXPECTED["v4.MQALayer.forward"] = "0000000000000000"
    try:
        sp.check_engine()
        raise AssertionError("drift accepted")
    except RuntimeError as exc:
        assert "drifted" in str(exc)
    finally:
        sp._EXPECTED.clear()
        sp._EXPECTED.update(saved)
    os.environ["DSV41_HC_PREFILL_FUSED"] = "1"
    try:
        import hc_prefill_fused
        aware = getattr(hc_prefill_fused, "SP_LOGICAL_ROWS", False)
        try:
            sp._check_hc_prefill_fused()
            refused = False
        except RuntimeError:
            refused = True
        assert refused != bool(aware), (aware, refused)
    finally:
        os.environ.pop("DSV41_HC_PREFILL_FUSED", None)
    print(f"  drift: {len(got)} engine sources + 3 tail helpers match; a drifted hash is refused; "
          f"hc_prefill_fused SP-aware={bool(aware)} -> {'accepted' if aware else 'refused'}", flush=True)


# ============================================================================================
def _kernels_at(m, layer, lin_b12x, lin_cut, seed):
    """{kernel: (full, [shard r outputs with the helper], [shard r outputs without])}."""
    from sglang.srt.layers.engram import engram_gate
    s = setup()
    R = _residual(m, seed)
    x_out = torch.randn(m, H, device=DEV, dtype=BF) * 0.3
    kv = torch.randn(m, (HC + 1) * H, device=DEV, dtype=BF) * 0.5
    qw = (1 + 0.1 * torch.randn(HC, H, device=DEV)).to(BF)
    kw = (1 + 0.1 * torch.randn(HC, H, device=DEV)).to(BF)
    emb = torch.randn(m, 6144, device=DEV, dtype=BF) * 0.2
    lay = layer

    def stats(t):
        return lay._hc_mix_stats(t, lay.hc_attn_fn, lay.hc_attn_scale, lay.hc_attn_base, None)

    pre, post, comb = stats(R)
    fns = {
        "combine0 (norm of stream 0)": lambda sl: lay._hc_combine(R[sl], None, lay.input_layernorm),
        "combine+norm": lambda sl: lay._hc_combine(R[sl], pre[sl], lay.input_layernorm),
        "mix stats (hc_fused)": lambda sl: stats(R[sl]),
        "mix stats (stock split-K)": lambda sl: _stock_stats(lambda: stats(R[sl])),
        "hc_post": lambda sl: lay.hc_post(x_out[sl], R[sl], post[sl], comb[sl]),
        "engram gate": lambda sl: engram_gate(R[sl].contiguous(), kv[sl], qw, kw, 1e-6, 1e-6),
        "wkv MXFP8 b12x": lambda sl: lin_b12x(emb[sl])[0],
        "wkv MXFP8 cutlass": lambda sl: lin_cut(emb[sl])[0],
    }
    out = {}
    for name, fn in fns.items():
        full = fn(slice(None))
        ok = raw_ok = True
        for r in range(W):
            lo, hi = sp.shard_range(m, W, r)
            want = _slice(full, lo, hi)
            with _as_shard(m, r):
                ok = ok and _eq(want, fn(slice(lo, hi)))
            raw_ok = raw_ok and _eq(want, fn(slice(lo, hi)))
        out[name] = (ok, raw_ok)
        del full
    return out


def _stock_stats(f):
    hf = setup()["hc_fused"]
    prev = hf._state["disabled"]
    hf._state["disabled"] = True
    try:
        return f()
    finally:
        hf._state["disabled"] = prev


def _eq(a, b):
    if isinstance(a, (tuple, list)):
        return all(_eq(u, v) for u, v in zip(a, b))
    return torch.equal(a, b)


def _slice(a, lo, hi):
    if isinstance(a, (tuple, list)):
        return tuple(_slice(u, lo, hi) for u in a)
    return a[lo:hi]


def test_kernels():
    _bind_layer_methods()
    lay = Layer(11)
    lin_b12x = _Fp8Linear(25600, 6144, 3)                 # production routing: b12x
    lin_cut = _Fp8Linear.__new__(_Fp8Linear)              # SGLang's own FlashInfer cutlass,
    lin_cut.w, lin_cut.sf, lin_cut.routed, lin_cut.calls = lin_b12x.w, lin_b12x.sf, False, []
    fails = []
    for m in (4096, 2048, 1500, 128):
        res = _kernels_at(m, lay, lin_b12x, lin_cut, seed=m)
        line = []
        for name, (ok, raw_ok) in res.items():
            if not ok:
                fails.append((m, name))
            line.append(f"{name}: {'OK' if ok else 'DIFF'}{'' if raw_ok else ' (shard-keyed: DIFF)'}")
        print(f"  kernels M={m} (shard {m // W}): " + "; ".join(line), flush=True)
        del res
        torch.cuda.empty_cache()
    assert not [f for f in fails if not f[1].startswith("wkv")], fails
    if fails:
        print(f"  NOTE: wkv shard GEMM not row-invariant at {fails}: the runtime self-check falls back "
              "to the full-M GEMM there (exact either way)", flush=True)


# ============================================================================================
# four ranks on one GPU
# ============================================================================================
class ThreadGroup:
    """Collectives across rank threads. Baton passing: a thread holds the lock except while it
    waits at a barrier, so exactly one rank issues GPU work at a time (stream order = issue
    order, and no concurrent JIT compiles)."""

    def __init__(self, world, rs_order="ring"):
        self.world_size, self.unique_name, self.device_group = world, "tp", None
        self._bar = threading.Barrier(world)
        self._lock = threading.Lock()
        self._slots = [None] * world
        self._tl = threading.local()
        self.rs_order = rs_order

    @property
    def rank_in_group(self):
        return self._tl.rank

    def _wait(self):
        self._lock.release()
        try:
            self._bar.wait()
        finally:
            self._lock.acquire()

    def _exchange(self, t):
        # no copy: every rank's tensor stays untouched until the second barrier
        r = self.rank_in_group
        self._slots[r] = t
        self._wait()
        vals = list(self._slots)
        self._wait()
        return vals

    def all_gather_into_tensor(self, out, x):
        vals = self._exchange(x.contiguous())
        dst = out.view(self.world_size, -1)
        for i, v in enumerate(vals):
            dst[i].copy_(v.reshape(-1))

    def all_reduce(self, x):
        return rank_order_sum(self._exchange(x))

    def reduce_scatter_tensor(self, out, x):
        vals = self._exchange(x.contiguous())
        r, s = self.rank_in_group, out.shape[0]
        parts = [v[r * s:(r + 1) * s] for v in vals]
        order = list(range(self.world_size)) if self.rs_order == "same" else \
            [(r + 1 + k) % self.world_size for k in range(self.world_size)]
        out.copy_(rank_order_sum([parts[o] for o in order]))

    def run(self, fn):
        out, errs = [None] * self.world_size, []

        def body(r):
            self._tl.rank = r
            self._lock.acquire()
            try:
                out[r] = fn(r)
            except BaseException as exc:  # noqa: BLE001
                errs.append(exc)
                self._bar.abort()
            finally:
                self._lock.release()

        ts = [threading.Thread(target=body, args=(r,)) for r in range(self.world_size)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        if errs:
            raise errs[0]
        return out


def rank_order_sum(parts):
    acc = parts[0].clone()
    for p in parts[1:]:
        acc = acc + p
    return acc


class Serial:
    """The reference 'group': one thread computes every rank's partial and reduces in rank order."""
    world_size = W


_fwd = threading.local()


class _Forward:
    @property
    def mlp_reduce_scatter(self):
        return getattr(_fwd, "mrs", False)

    @contextmanager
    def scoped(self, mlp_reduce_scatter=False):
        prev = getattr(_fwd, "mrs", False)
        _fwd.mrs = mlp_reduce_scatter
        try:
            yield
        finally:
            _fwd.mrs = prev


FORWARD = _Forward()


class RowParallel(torch.nn.Module):
    def __init__(self, group, w):
        super().__init__()
        self.group, self.w = group, w
        self.reduce_results, self.tp_size = True, W

    def partial(self, r, h):
        return (h @ self.w[r]).to(BF)

    def forward(self, input_, skip_all_reduce=False):
        out = self.partial(self.group.rank_in_group, input_)
        return (out if skip_all_reduce else self.group.all_reduce(out)), None


class Attn(torch.nn.Module):
    """Row-mixing stand-in (a causal running mean), per-rank head scale, wo_b partial."""

    def __init__(self, group, seed):
        super().__init__()
        g = torch.Generator(device=DEV).manual_seed(seed)
        self.group, self.attn_tp_size = group, W
        self.scale = (1 + 0.1 * torch.randn(W, H, device=DEV, generator=g)).to(BF)
        self.wo_b = RowParallel(group, (torch.randn(W, H, 256, device=DEV, generator=g) / 16).to(BF)
                                .transpose(1, 2).contiguous())   # [W, 256, H]: h[:, :256] @ w
        self.rows_seen = []

    def accepts_mxfp8_swizzled_input(self):
        return False

    def maybe_use_decode_attn_tp(self, fb):
        return nullcontext()

    def head(self, r, x):
        h = x[:, :256].float().cumsum(0) / torch.arange(1, x.shape[0] + 1, device=DEV).unsqueeze(1)
        return (h * self.scale[r, :256].float()).to(BF)

    def forward(self, x, positions, forward_batch, x_quant=None):
        assert positions.shape[0] == x.shape[0]
        self.rows_seen.append(x.shape[0])
        if self.group is Serial:
            return rank_order_sum([self.wo_b.partial(r, self.head(r, x)) for r in range(W)])
        o, _ = self.wo_b(self.head(self.group.rank_in_group, x))
        return o


class MoE(torch.nn.Module):
    def __init__(self, group, seed):
        super().__init__()
        g = torch.Generator(device=DEV).manual_seed(seed)
        self.group, self.tp_size = group, W
        self._shared_expert_tp1 = self._enable_a2a_moe = False
        self.table = (0.5 * torch.randn(16, W, H, device=DEV, generator=g)).to(BF)

    def partial(self, r, x, ids):
        return (x * self.table[ids % 16, r]).to(BF)

    def forward(self, hidden_states, forward_batch=None, gemm_output_zero_allocator=None,
                input_ids=None, input_ids_global=None, skip_shared_experts=False):
        assert input_ids.shape[0] == hidden_states.shape[0]
        if self.group is Serial:
            return rank_order_sum([self.partial(r, hidden_states, input_ids) for r in range(W)])
        out = self.partial(self.group.rank_in_group, hidden_states, input_ids)
        return out if FORWARD.mlp_reduce_scatter else self.group.all_reduce(out)


class Embed:
    def __init__(self, group, seed, vocab=1009, cols=2):
        g = torch.Generator(device=DEV).manual_seed(seed)
        self.group, self.tp_size, self._shared, self.vocab = group, W, False, vocab
        self.table = (torch.randn(vocab, cols, 256, device=DEV, generator=g) * 0.3).to(BF)

    def owned(self, r, ids):
        lo, hi = self.vocab * r // W, self.vocab * (r + 1) // W
        vals = self.table[ids, torch.arange(ids.shape[1], device=DEV)]
        return vals.masked_fill(~((ids >= lo) & (ids < hi)).unsqueeze(-1), 0)

    def _owned_rows(self, ids):
        return self.owned(self.group.rank_in_group, ids)


class Engram:
    def __init__(self, group, seed):
        from sglang.srt.layers.engram import engram_gate
        self.group, self.layer_hash_index = group, 0
        self.embed = Embed(group, seed)
        self.wkv = _Fp8Linear((HC + 1) * H, 512, seed + 1)
        g = torch.Generator(device=DEV).manual_seed(seed + 2)
        self.q = (1 + 0.1 * torch.randn(HC, H, device=DEV, generator=g)).to(BF)
        self.k = (1 + 0.1 * torch.randn(HC, H, device=DEV, generator=g)).to(BF)
        self._gate = engram_gate

    def apply_gate(self, x, kv):
        return self._gate(x, kv, self.q, self.k, 1e-6, 1e-6)

    def __call__(self, x, ids, forward_batch=None, cp_all_tokens=False):
        if self.group is Serial:
            emb = rank_order_sum([self.embed.owned(r, ids) for r in range(W)])
        else:
            emb = self.group.all_reduce(self.embed._owned_rows(ids))
        kv, _ = self.wkv(emb.flatten(-2))
        return self.apply_gate(x, kv)


class Model:
    """Layer 0: Engram + one full-row layer. Layer 1: bounded-replay tail layer."""

    def __init__(self, group, seed):
        self.pp_group = types.SimpleNamespace(world_size=1)
        self.hidden_size, self.hc_mult, self.hc_pre_from_prev_sublayer = H, HC, True
        self.start_layer, self.end_layer, self.late_layer_start = 0, 2, 1
        self.config = Layer.config
        self.layers = [Layer(seed + 10 * i, i, Attn(group, seed + 10 * i + 3),
                             MoE(group, seed + 10 * i + 4),
                             Engram(group, seed + 99) if i == 0 else None) for i in range(2)]
        self.dspark_layers_to_capture = [0, 1]

    def engram_hasher(self, input_ids, fb):
        return ((input_ids[:, None, None] * torch.tensor([[31, 17]], device=DEV) + 5) % 1009)

    def _check_late_layer_tail_readers(self, fb):
        pass


def _tail(lens):
    from sglang.srt.layers.attention.deepseek_v4_backend import LateLayerTail
    idx, pos = [], 0
    for n in lens:
        k = min(128, n)
        idx.extend(range(pos + n - k, pos + n))
        pos += n
    t = torch.tensor(idx, dtype=torch.int64, device=DEV)
    ext = [min(128, n) for n in lens]
    return LateLayerTail(token_indices=t, positions=t.clone(),
                         extend_seq_lens=torch.tensor(ext, device=DEV), extend_seq_lens_cpu=ext,
                         swa_out_cache_loc=t.clone(),
                         contiguous_start=(pos - ext[0]) if len(lens) == 1 else None)


class _Backend:
    def __init__(self, tail):
        self.tail_forward_metadata = types.SimpleNamespace(late_layer_tail=tail)

    def enter_late_layer_tail(self, fb):
        return "saved"

    def exit_late_layer_tail(self, saved, fb):
        assert saved == "saved"


Attn.forward = sp._make_attn_forward(Attn.forward)        # as install() wraps MQALayer.forward
MoE.forward = sp._make_moe_forward(MoE.forward)           # ... and DeepseekV2MoE.forward
sp._mega_moe = lambda moe, x: False                       # stand-in MoE is not a mega-MoE

FB = types.SimpleNamespace(forward_mode=types.SimpleNamespace(
    is_extend_without_speculative=lambda: True, is_decode=lambda: False,
    is_target_verify=lambda: False))


def _engine_ns(group, backend):
    dm = setup()["dm"]
    ns = types.SimpleNamespace(**{k: getattr(dm, k) for k in dir(dm) if not k.startswith("__")})
    ns.get_attn_backend = lambda: backend
    ns.check_cuda_graph_backend = lambda *a: False
    ns.get_global_expert_distribution_recorder = lambda: types.SimpleNamespace(
        with_current_layer=lambda i: nullcontext())
    ns.get_forward = lambda: FORWARD
    ns.is_in_breakable_cuda_graph = lambda: False
    ns.is_cp_active = lambda fb: False
    ns.get_attn_tp_context = lambda: types.SimpleNamespace(input_scattered=False)
    ns.get_tp_group = lambda: group
    ns.get_parallel = lambda: types.SimpleNamespace(attn_dp_size=1, tp_size=W, attn_tp_size=W)
    ns.get_moe_a2a_backend = lambda: types.SimpleNamespace(is_none=lambda: True)
    ns.MQALayer = Attn
    return ns


def _reference(model, R0, ids, pos, tail):
    """The engine's stock loop semantics on one 'rank' (all rows, partials reduced in rank order)."""
    hash_ids = model.engram_hasher(ids, FB)
    hs, prev_pre, aux = R0.clone(), None, []
    for i in range(model.end_layer):
        if i == model.late_layer_start:
            hs, prev_pre, ids, pos = tail.rows(hs), tail.rows(prev_pre), tail.rows(ids), tail.positions
            hash_ids = tail.rows(hash_ids)
        lay = model.layers[i]
        if lay.engram is not None:
            hs = lay.engram(hs, hash_ids[:, 0], FB)
        if i in model.dspark_layers_to_capture:
            aux.append((tail.rows(hs) if i < model.late_layer_start else hs).mean(dim=1))
        hs, prev_pre = lay.forward_hc_pre_from_prev(
            positions=pos, hidden_states=hs, input_ids=ids, forward_batch=FB, input_ids_global=ids,
            prev_pre=prev_pre, precomputed_attn=None, next_norm=None, next_input=[])
    return hs, prev_pre, aux


def _emulate(m, lens, seed=5, with_comm=True):
    _bind_layer_methods()
    tail = _tail(lens)
    g = torch.Generator(device=DEV).manual_seed(seed)
    R0 = torch.randn(m, HC, H, device=DEV, generator=g, dtype=BF) * 0.5
    ids = torch.randint(0, 5000, (m,), device=DEV, generator=g)
    pos = torch.arange(m, device=DEV)

    ref_model = Model(Serial, seed)
    sp._M["v4"] = _engine_ns(None, _Backend(tail))
    ref = _reference(ref_model, R0, ids, pos, tail)
    del ref_model
    torch.cuda.empty_cache()

    def arm(mode, exact, rs_order="ring"):
        group = ThreadGroup(W, rs_order)
        model = Model(group, seed)
        sp._M["v4"] = _engine_ns(group, _Backend(tail))
        sp._M["v2"] = types.SimpleNamespace(DeepseekV2MoE=MoE)
        sp._M["linear"] = types.SimpleNamespace(RowParallelLinear=RowParallel)
        sp._STATIC["checked"] = False
        sp._WKV_OK.clear()
        old = sp.MODE, sp.EXACT

        def body(r):
            aux = []
            p = sp._plan(model, R0, FB)
            assert p is not None and p.rank == r
            # R0 itself (not a per-rank clone: 4 x 168 MB); the loop never writes its input
            hs, pre, _ = sp._sp_forward_layers(model, p, pos, R0, FB, ids, ids, True, aux)
            return hs, pre, aux

        sp.MODE, sp.EXACT = mode, exact
        try:
            out = group.run(body)
        finally:
            sp.MODE, sp.EXACT = old
        wkv = dict(sp._WKV_OK)
        rows_seen = model.layers[0].self_attn.rows_seen
        del model
        torch.cuda.empty_cache()
        return out, wkv, rows_seen

    r0_sum = R0.view(torch.int16).sum().item()
    exact, wkv_e, seen = arm("shard", True)
    fast, wkv_f, _ = arm("shard", False)
    # P0 keeps every row on every rank: 4 ranks of it fit the 2 GB budget only at ~2k rows
    comm = arm("comm", False)[0] if with_comm else None
    assert R0.view(torch.int16).sum().item() == r0_sum, "the loop wrote into its input"
    same = lambda a, b: _eq(a[0], b[0]) and _eq(a[1], b[1]) and _eq(tuple(a[2]), tuple(b[2]))  # noqa: E731
    ok_exact = [same(exact[r], ref) for r in range(W)]
    ok_fast = [same(fast[r], comm[r]) for r in range(W)] if comm is not None else "n/a (memory)"
    diff = (fast[0][0].float() - ref[0].float()).abs()
    rel = (diff / ref[0].float().abs().clamp_min(1e-3)).max().item()
    print(f"  emulate M={m} tails={lens if len(lens) <= 4 else f'{len(lens)} requests'}: "
          f"exact==reference per rank {ok_exact}; fast==P0 per rank {ok_fast}; "
          f"fast vs reference: {int((diff > 0).sum())}/{diff.numel()} elements differ, max rel {rel:.2e}; "
          f"wkv shard ON {wkv_e}; attention saw rows {sorted(set(seen))}", flush=True)
    assert all(ok_exact) and (comm is None or all(ok_fast)), (ok_exact, ok_fast)


def test_emulate():
    sp._agree_min = lambda p, local: bool(min(int(v) for v in p.group._exchange(
        torch.tensor([1 if local else 0], device=DEV))))
    for m, lens, comm in ((4096, [1000, 50, 2046, 1000], False), (4096, [4096], False),
                          (2052, [513, 1, 1025, 513], True), (2048, [2048], True)):
        _emulate(m, lens, with_comm=comm)


def _debug_arm(m, lens, seed=5):
    """Stage-1 exact with DSV41_PREFILL_SP_DEBUG=compare,fp on the emulated 4 ranks."""
    _bind_layer_methods()
    s = setup()
    Layer._hc_mix_stats = sp._make_hc_mix_stats(s["stats"])
    Layer.forward_hc_pre_from_prev = sp._make_layer_forward(s["fwd"])
    tail = _tail(lens)
    g = torch.Generator(device=DEV).manual_seed(seed)
    R0 = torch.randn(m, HC, H, device=DEV, generator=g, dtype=BF) * 0.5
    ids = torch.randint(0, 5000, (m,), device=DEV, generator=g)
    pos = torch.arange(m, device=DEV)
    group = ThreadGroup(W)
    model = Model(group, seed)
    sp._M["v4"] = _engine_ns(group, _Backend(tail))
    sp._M["v2"] = types.SimpleNamespace(DeepseekV2MoE=MoE)
    sp._M["linear"] = types.SimpleNamespace(RowParallelLinear=RowParallel)
    sp._STATIC["checked"] = False
    sp._WKV_OK.clear()
    old = sp.MODE, sp.EXACT, set(sp.DEBUG)
    sp.DEBUG.clear()
    sp.DEBUG.update({"compare", "fp"})

    def body(r):
        p = sp._plan(model, R0, FB)
        d = sp._Dbg(1, m, p, r)
        sp._ctx.dbg = d
        try:
            out = sp._sp_forward_layers(model, p, pos, R0, FB, ids, ids, True, [])
        finally:
            sp._ctx.dbg = None
        sp._debug_end(d, out)
        return d.checked, list(d.bad)

    sp.MODE, sp.EXACT = "shard", True
    try:
        return group.run(body)
    finally:
        sp.MODE, sp.EXACT = old[0], old[1]
        sp.DEBUG.clear()
        sp.DEBUG.update(old[2])
        _bind_layer_methods()
        del model
        torch.cuda.empty_cache()


def test_debug():
    """4 ranks x a gathered [M, 4, 5120] reference per compared op: M=2052 fits the 2 GB cap."""
    sp._agree_min = lambda p, local: bool(min(int(v) for v in p.group._exchange(
        torch.tensor([1 if local else 0], device=DEV))))
    m, lens = 2052, [513, 1, 1025, 513]
    res = _debug_arm(m, lens)
    assert all(c == 12 and not bad for c, bad in res), res   # 1 full-row layer: 6 hc + wo_b + moe + 2 engram + 2 tail
    print(f"  debug compare, clean: {[c for c, _ in res]} ops checked per rank, 0 mismatches", flush=True)
    # fault: the shard takes the combine branch with a combine kernel that is off by one ulp
    good_gate, good_mhc = sp._combine_norm_fused, sp._M["mhc"]
    real = good_mhc.hc_combine
    sp._combine_norm_fused = lambda layer, x, pre, norm, rows: rows == x.shape[0]
    sp._M["mhc"] = types.SimpleNamespace(hc_combine=lambda x, pre, hc, dt: (
        lambda y: y.view(torch.int16).add_(1).view(dt))(real(x, pre, hc, dt)))
    try:
        res = _debug_arm(m, lens)
    finally:
        sp._combine_norm_fused, sp._M["mhc"] = good_gate, good_mhc
    ops = sorted({b[1] for _, bad in res for b in bad})
    assert ops and all(o.startswith("hc_combine") for o in ops) and all(bad for _, bad in res), ops
    print(f"  debug compare, injected 1-ulp combine on the shard: flagged {ops} on every rank "
          f"(first at layer {res[0][1][0][0]})", flush=True)



# ============================================================================================
# stage 2: MXFP8 attention-input gathers
# ============================================================================================
def test_fp8_quant():
    """Shard-wise MXFP8 (linear scales) + gather + 128x4 swizzle == the stock full-M quantization,
    and the MXFP8 linear on the pre-quantized tuple == on the bf16 rows (b12x and cutlass)."""
    s = setup()
    lin = _Fp8Linear(1792, H, 21)
    for m in (4096, 3152, 2052, 2048, 1500, 128):
        x = torch.randn(m, H, device=DEV, dtype=BF) * 0.5
        x[:, :64] *= 40
        rq, rs = sp._mxfp8_quantize(x, True)
        lq, ls = sp._mxfp8_quantize(x, False)
        parts_q, parts_s = [], []
        for r in range(W):
            lo, hi = sp.shard_range(m, W, r)
            q, sf = sp._mxfp8_quantize(x[lo:hi].contiguous(), False)
            parts_q.append(q.view(torch.uint8))
            parts_s.append(sf.view(torch.uint8).reshape(-1)[: (hi - lo) * (H // 32)].view(hi - lo, H // 32))
        qg, sg = torch.cat(parts_q), torch.cat(parts_s)
        sw = sp.swizzle_128x4(sg)
        data_ok = torch.equal(qg, rq.view(torch.uint8))
        lin_ok = torch.equal(sg, ls.view(torch.uint8).reshape(-1)[: m * (H // 32)].view(m, H // 32))
        sw_ok = torch.equal(sw, rs.view(torch.uint8).reshape(-1))
        xq = sp._mxfp8_input_cls()(qg.view(rq.dtype), sw.view(rs.dtype).view(rs.shape))
        f_b12x = s["fp8_utils"].flashinfer_mxfp8_blockscaled_linear
        f_cut = s["mxfp8_cutlass"]
        cons = {}
        for name, f in (("b12x", f_b12x), ("cutlass", f_cut)):
            ref = f(x, lin.w, lin.sf, backend="cutlass")
            got = f(xq[0], lin.w, lin.sf, input_scale=xq[1], backend="cutlass")
            cons[name] = torch.equal(ref, got)
        print(f"  fp8 quant M={m} (shard {m // W}): data {data_ok}, linear scales {lin_ok}, swizzled "
              f"scales {sw_ok} (stock shapes q {tuple(rq.shape)} {rq.dtype}, sf {tuple(rs.shape)} {rs.dtype}, "
              f"linear sf {tuple(ls.shape)}); linear on pre-quantized == bf16: {cons}", flush=True)
        assert data_ok and sw_ok and all(cons.values()), (m, data_ok, lin_ok, sw_ok, cons)


class Attn8(Attn):
    """Attention stand-in that consumes its input the way MQALayer's prefill does on a layer without
    compressor / indexer: wqkv_a (MXFP8 linear) on x_quant if given, else on the bf16 x."""

    def __init__(self, group, seed, source=False):
        super().__init__(group, seed)
        self.compressor = object() if source else None      # a source layer reads bf16 x
        self.indexer = None
        self.fuse_wqa_wkv = True
        lin = _Fp8Linear(1792, H, seed + 7)

        def wqkv_a(xin):
            f = setup()["fp8_utils"].flashinfer_mxfp8_blockscaled_linear
            if isinstance(xin, tuple):
                return f(xin[0], lin.w, lin.sf, input_scale=xin[1], backend="cutlass"), None
            return f(xin, lin.w, lin.sf, backend="cutlass"), None

        self.wqkv_a = wqkv_a
        self.fp8_inputs = 0

    def accepts_mxfp8_swizzled_input(self):
        return True

    def head(self, r, x):
        return super().head(r, x)

    def forward(self, x, positions, forward_batch, x_quant=None):
        assert positions.shape[0] == x.shape[0]
        self.rows_seen.append(x.shape[0])
        if x_quant is not None:
            self.fp8_inputs += 1
        if self.compressor is not None:                     # reads the bf16 rows themselves
            assert x_quant is None and not torch.isnan(x[:1]).any()
        qkv = self.wqkv_a(x_quant if x_quant is not None else x)[0]
        h = qkv[:, :256].contiguous()
        if self.group is Serial:
            return rank_order_sum([self.wo_b.partial(r, self.head(r, h)) for r in range(W)])
        o, _ = self.wo_b(self.head(self.group.rank_in_group, h))
        return o


Attn8.forward = sp._make_attn_forward(Attn8.forward)


class Model8(Model):
    """Layer 0: Engram + fp8-eligible attention. Layer 1: a 'source' layer (bf16). Layer 2: tail."""

    def __init__(self, group, seed):
        self.pp_group = types.SimpleNamespace(world_size=1)
        self.hidden_size, self.hc_mult, self.hc_pre_from_prev_sublayer = H, HC, True
        self.start_layer, self.end_layer, self.late_layer_start = 0, 3, 2
        self.config = Layer.config
        self.layers = [Layer(seed + 10 * i, i, Attn8(group, seed + 10 * i + 3, source=(i == 1)),
                             MoE(group, seed + 10 * i + 4),
                             Engram(group, seed + 99) if i == 0 else None) for i in range(3)]
        self.dspark_layers_to_capture = [0]


def _arm8(m, lens, *, fp8, exact, debug=False, seed=9):
    _bind_layer_methods()
    s = setup()
    if debug:
        Layer._hc_mix_stats = sp._make_hc_mix_stats(s["stats"])
        Layer.forward_hc_pre_from_prev = sp._make_layer_forward(s["fwd"])
    tail = _tail(lens)
    g = torch.Generator(device=DEV).manual_seed(seed)
    R0 = torch.randn(m, HC, H, device=DEV, generator=g, dtype=BF) * 0.5
    ids = torch.randint(0, 5000, (m,), device=DEV, generator=g)
    pos = torch.arange(m, device=DEV)
    group = ThreadGroup(W)
    model = Model8(group, seed)
    sp._M["v4"] = _engine_ns(group, _Backend(tail))
    sp._M["v4"].MQALayer = Attn8
    sp._M["v2"] = types.SimpleNamespace(DeepseekV2MoE=MoE)
    sp._M["linear"] = types.SimpleNamespace(RowParallelLinear=RowParallel)
    sp._STATIC["checked"] = False
    sp._WKV_OK.clear()
    sp._FP8_M.clear()
    old = sp.MODE, sp.EXACT, sp.FP8, set(sp.DEBUG)
    sp.DEBUG.clear()
    if debug:
        sp.DEBUG.update({"compare", "fp"})

    def body(r):
        p = sp._plan(model, R0, FB)
        d = sp._Dbg(1, m, p, r) if debug else None
        sp._ctx.dbg = d
        aux = []
        try:
            hs, pre, _ = sp._sp_forward_layers(model, p, pos, R0, FB, ids, ids, True, aux)
        finally:
            sp._ctx.dbg = None
        if d is not None:
            sp._debug_end(d, (hs, pre))
        return hs, pre, aux, (d.checked, list(d.bad)) if d is not None else None

    sp.MODE, sp.EXACT, sp.FP8 = "shard", exact, fp8
    try:
        out = group.run(body)
    finally:
        sp.MODE, sp.EXACT, sp.FP8 = old[0], old[1], old[2]
        sp.DEBUG.clear()
        sp.DEBUG.update(old[3])
        _bind_layer_methods()
    info = ([l.self_attn.fp8_inputs for l in model.layers], dict(sp._FP8_M))
    del model
    torch.cuda.empty_cache()
    return out, info


def test_fp8_emulate(only_debug=False):
    sp._agree_min = lambda p, local: bool(min(int(v) for v in p.group._exchange(
        torch.tensor([1 if local else 0], device=DEV))))
    same = lambda a, b: _eq(a[0], b[0]) and _eq(a[1], b[1]) and _eq(tuple(a[2]), tuple(b[2]))  # noqa: E731
    for m, lens in () if only_debug else ((4096, [1000, 50, 2046, 1000]), (3152, [3152]),
                                          (2052, [513, 1, 1025, 513])):
        for exact in (True, False):
            s1, _ = _arm8(m, lens, fp8=False, exact=exact)
            s2, info = _arm8(m, lens, fp8=True, exact=exact)
            ok = [same(s1[r], s2[r]) for r in range(W)]
            print(f"  fp8 emulate M={m} {'exact' if exact else 'fast '}: stage2 == stage1 per rank {ok}; "
                  f"x_quant calls per layer (x4 ranks) {info[0]}; check {info[1]}", flush=True)
            assert all(ok), ok
            assert info[0] == [W, 0, 0] and all(v[0] for v in info[1].values()), info
    # debug gathers a full [M, 4, 5120] reference per compared op on 4 ranks: 2052 rows fit 2 GB
    res, info = _arm8(2052, [513, 1, 1025, 513], fp8=True, exact=True, debug=True)
    dbg = [r[3] for r in res]
    assert all(c > 0 and not bad for c, bad in dbg), dbg
    print(f"  fp8 debug compare: {[c for c, _ in dbg]} ops checked per rank (incl. attn x mxfp8 "
          f"data/scales and wqkv_a), 0 mismatches", flush=True)


if __name__ == "__main__":
    which = sys.argv[1:] or ["drift", "kernels", "emulate", "debug", "fp8"]
    print(f"torch {torch.__version__}, {torch.cuda.get_device_name(0)}", flush=True)
    if "drift" in which:
        test_drift()
    if "kernels" in which:
        test_kernels()
    if "emulate" in which:
        test_emulate()
    if "debug" in which:
        test_debug()
    if "fp8debug" in which:
        test_fp8_emulate(only_debug=True)
    if "fp8" in which:
        test_fp8_quant()
        test_fp8_emulate()
    print(f"test_prefill_sp_gpu: ok (peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB)")
