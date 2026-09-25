"""CPU check for adapter/prefill_sp.py: the row partition, the tail-row gathers, and the whole
sharded layer loop against the engine's stock loop, with four TP ranks emulated by threads over
a fake process group (all-gather = concatenation, all-reduce = rank-order bf16 sum, reduce-scatter
= a different summation order, like NCCL's).

The model is a toy with the engine's structure: residual [M, 4, D], a row-mixing "attention"
whose wo_b all-reduces unless told to skip, a hash-routed "MoE" that honours mlp_reduce_scatter,
an Engram (owned-row lookup + all-reduce, wkv, gate), bounded replay (late layers on the tail
rows), DSpark aux capture and the vision `where`. Its hc combine picks one of two numerically
different reductions by row count (as the engine's >= 4096 fused norm does), keyed on
sp_logical_rows(): a gate keyed on the shard size would change bits, and the test shows it.

Checks, for several chunk sizes (4096, 2052 = odd shards, 2048 with a full-row tail, 2050 = not
divisible, below MIN_ROWS) and tail layouts (one request, requests straddling shard boundaries,
many short requests, no bounded replay):
  * stage 1 EXACT == stock, bit for bit, on every rank;
  * stage 1 fast == P0 (comm), bit for bit (same reduce-scatters on the same tensors);
  * P0 differs from stock only through the summation order (and matches it with an exact RS);
  * the wkv self-check falls back to the full-M GEMM when the shard GEMM is not bit-exact.

  PYTHONPATH=adapter python tests/test_prefill_sp.py
"""
import os
import threading
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

os.environ.setdefault("DSV41_PREFILL_SP", "1")
import torch  # noqa: E402

import prefill_sp as sp  # noqa: E402

W = 4
D = 64
BF = torch.bfloat16


# ------------------------------------------------------------------------------------------
# fake process group over threads
# ------------------------------------------------------------------------------------------
class ThreadGroup:
    def __init__(self, world, rs_order="ring"):
        self.world_size = world
        self.unique_name = "tp"
        self.device_group = None
        self._bar = threading.Barrier(world)
        self._slots = [None] * world
        self._tl = threading.local()
        self.rs_order = rs_order
        self.calls = {"ag": 0, "ar": 0, "rs": 0}

    @property
    def rank_in_group(self):
        return self._tl.rank

    def bind(self, rank):
        self._tl.rank = rank

    def _exchange(self, t):
        r = self.rank_in_group
        self._slots[r] = t.clone()
        self._bar.wait()
        vals = list(self._slots)
        self._bar.wait()
        return vals

    def all_gather_into_tensor(self, out, x):
        vals = self._exchange(x.contiguous())
        if self.rank_in_group == 0:
            self.calls["ag"] += 1
        out.view(-1).copy_(torch.cat([v.reshape(-1) for v in vals]))

    def all_reduce(self, x):
        vals = self._exchange(x)
        if self.rank_in_group == 0:
            self.calls["ar"] += 1
        acc = vals[0].clone()
        for v in vals[1:]:
            acc = acc + v                               # rounds to the dtype at every hop
        return acc

    def reduce_scatter_tensor(self, out, x):
        vals = self._exchange(x.contiguous())
        r, s = self.rank_in_group, out.shape[0]
        if r == 0:
            self.calls["rs"] += 1
        parts = [v[r * s:(r + 1) * s] for v in vals]
        order = list(range(self.world_size)) if self.rs_order == "same" else \
            [(r + 1 + k) % self.world_size for k in range(self.world_size)]
        acc = parts[order[0]].clone()
        for o in order[1:]:
            acc = acc + parts[o]
        out.copy_(acc)


def run_ranks(group, fn):
    out, errs = [None] * group.world_size, []

    def body(r):
        group.bind(r)
        try:
            out[r] = fn(r)
        except BaseException as exc:  # noqa: BLE001
            errs.append((r, exc))
            group._bar.abort()

    ts = [threading.Thread(target=body, args=(r,)) for r in range(group.world_size)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    if errs:
        raise errs[0][1]
    return out


# ------------------------------------------------------------------------------------------
# fake engine
# ------------------------------------------------------------------------------------------
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
GATE = {"logical": True}


def hc_rows(x):
    return sp.sp_logical_rows(x) if GATE["logical"] else x.shape[0]


class RowParallel(torch.nn.Module):
    def __init__(self, group, weights):
        super().__init__()
        self.group, self.w = group, weights
        self.reduce_results, self.tp_size = True, group.world_size

    def forward(self, input_, skip_all_reduce=False):
        out = (input_.float() @ self.w[self.group.rank_in_group]).to(BF)
        if not skip_all_reduce:
            out = self.group.all_reduce(out)
        return out, None


class FakeAttn(torch.nn.Module):
    def __init__(self, group, g):
        super().__init__()
        self.group = group
        self.scale = torch.randn(group.world_size, D, generator=g)
        self.wo_b = RowParallel(group, torch.randn(group.world_size, D, D, generator=g) / D ** 0.5)
        self.attn_tp_size = group.world_size
        self.seen_rows = []

    def forward(self, x, positions, forward_batch, x_quant=None):
        assert positions.shape[0] == x.shape[0], "attention needs the positions of every row"
        self.seen_rows.append(x.shape[0])
        # causal row mixing: attention needs every earlier row
        h = x.float().cumsum(0) / torch.arange(1, x.shape[0] + 1).unsqueeze(1)
        h = (h * self.scale[self.group.rank_in_group]).to(BF)
        o, _ = self.wo_b(h)
        return o

    def maybe_use_decode_attn_tp(self, fb):
        return nullcontext()


class FakeMoE(torch.nn.Module):
    def __init__(self, group, g):
        super().__init__()
        self.group = group
        self.table = torch.randn(16, group.world_size, D, generator=g)
        self.tp_size = group.world_size
        self._shared_expert_tp1 = False
        self._enable_a2a_moe = False

    def forward(self, hidden_states, forward_batch=None, gemm_output_zero_allocator=None,
                input_ids=None, input_ids_global=None, skip_shared_experts=False):
        assert input_ids.shape[0] == hidden_states.shape[0], "hash routing needs every row's id"
        w = self.table[input_ids % 16, self.group.rank_in_group]
        out = (hidden_states.float() * torch.tanh(w)).to(BF)
        if not FORWARD.mlp_reduce_scatter:
            out = self.group.all_reduce(out)
        return out


class FakeEmbed:
    def __init__(self, group, g, vocab, cols, dim):
        self.group, self.tp_size, self._shared = group, group.world_size, False
        self.table = torch.randn(vocab, cols, dim, generator=g).to(BF)
        self.vocab = vocab

    def _owned_rows(self, ids):
        r = self.group.rank_in_group
        lo, hi = self.vocab * r // self.tp_size, self.vocab * (r + 1) // self.tp_size
        owned = (ids >= lo) & (ids < hi)                                 # [T, cols]
        vals = self.table[ids, torch.arange(ids.shape[1])]              # [T, cols, dim]
        return vals.masked_fill(~owned.unsqueeze(-1), 0)


class FakeEngram:
    def __init__(self, group, g, hash_index, m_dependent=False):
        self.group, self.layer_hash_index = group, hash_index
        self.embed = FakeEmbed(group, g, 97, 3, 8)
        self.wkv_w = torch.randn(24, D, generator=g) / 5
        self.m_dependent = m_dependent
        self.wkv_calls = []

    def wkv(self, t):
        self.wkv_calls.append(t.shape[0])
        out = (t.float() @ self.wkv_w)
        if self.m_dependent and t.shape[0] >= 2048:   # a GEMM whose tactic changes with M
            out = out * (1 + 2 ** -7)
        return out.to(BF), None

    def apply_gate(self, x, kv):
        return (x.float() + torch.sigmoid(kv.float()).unsqueeze(1)).to(x.dtype)

    def __call__(self, x, ids, forward_batch=None, cp_all_tokens=False):
        emb = self.group.all_reduce(self.embed._owned_rows(ids))
        kv, _ = self.wkv(emb.flatten(-2))
        return self.apply_gate(x, kv)


class FakeLayer:
    def __init__(self, group, g, layer_id, engram=None):
        self.layer_id = layer_id
        self.self_attn = FakeAttn(group, g)
        self.mlp = FakeMoE(group, g)
        self.engram = engram
        self.fn = torch.randn(2, 4, D, generator=g)
        self.norm_w = torch.randn(D, generator=g)
        self.dsa_enable_prefill_cp = False
        self.use_fused_mhc_post_pre = False

    def _hc_combine(self, R, pre):
        if pre is None:
            return R[:, 0, :].contiguous()
        p = pre.float()
        if hc_rows(R) >= 4096:          # "fused" reduction order
            y = ((R[:, 0].float() * p[:, 0:1] + R[:, 1].float() * p[:, 1:2])
                 + (R[:, 2].float() * p[:, 2:3] + R[:, 3].float() * p[:, 3:4]))
        else:                           # "unfused": sequential, rounded before the norm
            y = R[:, 0].float() * p[:, 0:1]
            for c in (1, 2, 3):
                y = y + R[:, c].float() * p[:, c:c + 1]
            y = y.to(BF).float()
        y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + 1e-6) * self.norm_w
        return y.to(BF)

    def _stats(self, R, which):
        mix = (R.float() * self.fn[which]).sum(-1)                         # [N, 4]
        pre = torch.sigmoid(mix)
        post = 2 * torch.sigmoid(mix.flip(-1))
        comb = torch.softmax(mix.unsqueeze(-1) * mix.unsqueeze(-2) / 8, dim=-1)
        return pre, post, comb

    def hc_post(self, x, R, post, comb):
        out = post.unsqueeze(-1) * x.float().unsqueeze(1) + (comb.unsqueeze(-1) * R.float().unsqueeze(2)).sum(1)
        return out.to(BF)

    def forward_hc_pre_from_prev(self, positions, hidden_states, input_ids, forward_batch,
                                 input_ids_global, prev_pre, precomputed_attn=None,
                                 next_norm=None, next_input=None):
        R = hidden_states
        x = self._hc_combine(R, prev_pre)
        x = self.self_attn(x=x, positions=positions, forward_batch=forward_batch, x_quant=None)
        attn_pre, post, comb = self._stats(R, 0)
        R = self.hc_post(x, R, post, comb)
        x = self._hc_combine(R, attn_pre)
        with FORWARD.scoped(mlp_reduce_scatter=False):
            x = self.mlp(x, forward_batch, input_ids=input_ids, input_ids_global=input_ids_global,
                         skip_shared_experts=False)
        ffn_pre, post, comb = self._stats(R, 1)
        R = self.hc_post(x, R, post, comb)
        return R, ffn_pre


class Tail:
    def __init__(self, lens, window=128, pad_rows=0):
        starts, idx = [], []
        pos = 0
        for n in lens:
            k = min(window, n)
            idx.extend(range(pos + n - k, pos + n))
            pos += n
        self.token_indices = torch.tensor(idx, dtype=torch.int64)
        self.contiguous_start = (pos - min(window, lens[0])) if len(lens) == 1 else None
        self.positions = self.token_indices.clone()
        self.pad_rows = pad_rows
        self.cp_metadata = None

    def real_rows(self, t):
        return t[self.contiguous_start:] if self.contiguous_start is not None else t[self.token_indices]

    def rows(self, t):
        r = self.real_rows(t)
        if self.pad_rows:
            r = torch.cat([r, r.new_zeros((self.pad_rows, *r.shape[1:]))])
        return r


class Backend:
    def __init__(self, tail):
        self.tail_forward_metadata = SimpleNamespace(late_layer_tail=tail)
        self.entered = 0

    def enter_late_layer_tail(self, fb):
        self.entered += 1
        return "saved"

    def exit_late_layer_tail(self, saved, fb):
        assert saved == "saved"


class FakeModel:
    def __init__(self, group, seed, n_layers=6, late=4, vision=False, m_dependent_wkv=False):
        g = torch.Generator().manual_seed(seed)
        self.pp_group = SimpleNamespace(world_size=1)
        self.hidden_size, self.hc_mult, self.hc_pre_from_prev_sublayer = D, 4, True
        self.start_layer, self.end_layer, self.late_layer_start = 0, n_layers, late
        self.config = SimpleNamespace(model_type="deepseek_v41", vision_n_layers=1 if vision else 0,
                                      image_token_id=7)
        eng = {1: FakeEngram(group, g, 0, m_dependent_wkv), 3: FakeEngram(group, g, 1, m_dependent_wkv)}
        self.layers = [FakeLayer(group, g, i, eng.get(i)) for i in range(n_layers)]
        self.hash_a = torch.randint(1, 97, (2, 3), generator=g)
        self.dspark_layers_to_capture = [2, 5]

    def engram_hasher(self, input_ids, fb):
        return (input_ids[:, None, None] * self.hash_a + torch.arange(3)) % 97

    def _check_late_layer_tail_readers(self, fb):
        pass


def fake_engine(group, backend):
    return SimpleNamespace(
        get_attn_backend=lambda: backend,
        check_cuda_graph_backend=lambda *a: False,
        Phase=SimpleNamespace(PREFILL="prefill"),
        Backend=SimpleNamespace(TC_PIECEWISE="pcg"),
        get_global_expert_distribution_recorder=lambda: SimpleNamespace(
            with_current_layer=lambda i: nullcontext()),
        get_forward=lambda: FORWARD,
        is_in_breakable_cuda_graph=lambda: False,
        is_cp_active=lambda fb: False,
        get_attn_tp_context=lambda: SimpleNamespace(input_scattered=False),
        get_tp_group=lambda: group,
        get_parallel=lambda: SimpleNamespace(attn_dp_size=1, tp_size=group.world_size,
                                             attn_tp_size=group.world_size),
        get_moe_a2a_backend=lambda: SimpleNamespace(is_none=lambda: True),
        get_platform=lambda: SimpleNamespace(is_sm100=False),
        envs=SimpleNamespace(SGLANG_ENABLE_DETERMINISTIC_INFERENCE=SimpleNamespace(get=lambda: False)),
        MQALayer=FakeAttn,
    )


FakeAttn.forward = sp._make_attn_forward(FakeAttn.forward)
FakeMoE.forward = sp._make_moe_forward(FakeMoE.forward)
FakeLayer.forward_hc_pre_from_prev = sp._make_layer_forward(FakeLayer.forward_hc_pre_from_prev)
sp._REQUIRE_CUDA = False


def _agree_min(p, local):
    vals = p.group._exchange(torch.tensor([1 if local else 0]))
    return bool(min(int(v) for v in vals))


sp._agree_min = _agree_min


# ------------------------------------------------------------------------------------------
# reference: the engine's stock loop, same fake model
# ------------------------------------------------------------------------------------------
def stock_forward_layers(self, positions, hidden_states, forward_batch, input_ids, input_ids_global,
                         capture_dspark, aux_out):
    hash_ids = self.engram_hasher(input_ids, forward_batch)
    tail = None
    if self.late_layer_start is not None:
        backend = sp._M["v4"].get_attn_backend()
        tail = backend.tail_forward_metadata.late_layer_tail
    prev_pre = None
    saved = None
    for i in range(self.start_layer, self.end_layer):
        if tail is not None and i == self.late_layer_start:
            saved = backend.enter_late_layer_tail(forward_batch)
            hidden_states, prev_pre, input_ids, input_ids_global = (
                tail.rows(hidden_states), tail.rows(prev_pre), tail.rows(input_ids),
                tail.rows(input_ids_global))
            positions = tail.positions
            hash_ids = tail.rows(hash_ids)
        engram = self.layers[i].engram
        if engram is not None:
            before = hidden_states
            hidden_states = engram(hidden_states, hash_ids[:, engram.layer_hash_index], forward_batch)
            if self.config.vision_n_layers > 0:
                hidden_states = torch.where((input_ids == self.config.image_token_id)[:, None, None],
                                            before, hidden_states)
        if capture_dspark and i in self.dspark_layers_to_capture:
            aux = hidden_states
            if tail is not None and i < self.late_layer_start:
                aux = tail.rows(aux)
            aux_out.append(aux.mean(dim=1))
        hidden_states, prev_pre = self.layers[i].forward_hc_pre_from_prev(
            positions=positions, hidden_states=hidden_states, input_ids=input_ids,
            forward_batch=forward_batch, input_ids_global=input_ids_global, prev_pre=prev_pre)
    if saved is not None:
        backend.exit_late_layer_tail(saved, forward_batch)
        return hidden_states, prev_pre, tail
    return hidden_states, prev_pre, None


FB = SimpleNamespace(forward_mode=SimpleNamespace(is_extend_without_speculative=lambda: True))


def run(mode, rows, lens, *, exact=False, late=4, vision=False, rs_order="ring", m_dep=False, seed=1,
        debug=False):
    """Every rank's (hidden, pre, aux list) for one forward in `mode` (stock | comm | shard)."""
    group = ThreadGroup(W, rs_order)
    model = FakeModel(group, seed, late=late, vision=vision, m_dependent_wkv=m_dep)
    tail = Tail(lens) if late is not None else None
    backend = Backend(tail)
    sp._M.clear()
    sp._M.update(v4=fake_engine(group, backend), v2=SimpleNamespace(DeepseekV2MoE=FakeMoE))
    sp._STATIC["checked"] = False
    sp._STATIC["tail_checked"] = True         # the engine's tail helpers are hash-checked on GPU
    sp._WKV_OK.clear()
    g = torch.Generator().manual_seed(seed + 100)
    R0 = torch.randn(rows, 4, D, generator=g).to(BF)
    ids = torch.randint(0, 40, (rows,), generator=g)
    pos = torch.arange(rows)
    old = sp.MODE, sp.EXACT

    def body(r):
        aux = []
        if mode == "stock":
            out = stock_forward_layers(model, pos, R0.clone(), FB, ids, ids, True, aux)
            return out[0], out[1], aux, None
        p = sp._plan(model, R0, FB)
        if p is None:
            out = stock_forward_layers(model, pos, R0.clone(), FB, ids, ids, True, aux)
            return out[0], out[1], aux, None
        d = sp._Dbg(1, rows, p, r) if debug else None
        sp._ctx.dbg = d
        try:
            out = sp._sp_forward_layers(model, p, pos, R0.clone(), FB, ids, ids, True, aux)
        finally:
            sp._ctx.dbg = None
        if d is not None:
            sp._debug_end(d, out)
            return out[0], out[1], aux, (p, d)
        return out[0], out[1], aux, p

    sp.MODE, sp.EXACT = ("off" if mode == "stock" else mode), exact
    try:
        res = run_ranks(group, body)
    finally:
        sp.MODE, sp.EXACT = old
    return res, model, group


def same(a, b):
    ha, pa, xa, _ = a
    hb, pb, xb, _ = b
    return (torch.equal(ha, hb) and torch.equal(pa, pb) and len(xa) == len(xb)
            and all(torch.equal(u, v) for u, v in zip(xa, xb)))


def main():
    torch.set_num_threads(1)
    # --- partition -------------------------------------------------------------------------
    for rows in (4096, 2048, 2052, 1500, 128, 65536):
        if not sp.eligible_rows(rows, W, 128):
            continue
        cover = []
        for r in range(W):
            lo, hi = sp.shard_range(rows, W, r)
            cover.extend(range(lo, hi))
        assert cover == list(range(rows)), rows
    assert not sp.eligible_rows(2050, W, 1024) and not sp.eligible_rows(2044, W, 2048)
    assert sp.eligible_rows(2048, W, 2048) and not sp.eligible_rows(4096, 1, 2048)
    idx = torch.tensor([0, 511, 512, 1023, 1024, 4095])
    o, l = sp.owner_local(idx, 1024)
    assert o.tolist() == [0, 0, 0, 0, 1, 3] and l.tolist() == [0, 511, 512, 1023, 0, 1023]
    assert sp.tail_plan(4096, 4, 128) == "owners" and sp.tail_plan(2048, 4, 512) == "full"

    # --- tail-row gathers against tail.rows(full) ---------------------------------------------
    for rows, lens, pad in ((4096, [4096], 0), (4096, [1000, 50, 2046, 1000], 0),
                            (2052, [513, 1, 1025, 513], 0), (2048, [64] * 32, 0),
                            (4096, [4000, 96], 3), (8192, [2047, 2049, 4096], 0)):
        group = ThreadGroup(W)
        tail = Tail(lens, pad_rows=pad)
        full = torch.randn(rows, 4, D).to(BF)
        pre = torch.rand(rows, 4)

        def body(r, rows=rows, tail=tail, full=full, pre=pre, group=group):
            p = sp._Plan(rows, W, r, group, True, False)
            return sp._tail_rows(p, tail, full[p.lo:p.hi]), sp._tail_rows(p, tail, pre[p.lo:p.hi])

        for got_h, got_p in run_ranks(group, body):
            assert torch.equal(got_h, tail.rows(full)) and torch.equal(got_p, tail.rows(pre)), (rows, lens)

    # --- whole loop ----------------------------------------------------------------------------
    MIN = sp.MIN_ROWS
    sp.MIN_ROWS = 1024
    try:
        cases = [
            ("one request", 4096, [4096], dict()),
            ("straddling requests", 4096, [1000, 50, 2046, 1000], dict()),
            ("odd shards + vision", 2052, [513, 1, 1025, 513], dict(vision=True)),
            ("tail == all rows", 2048, [64] * 32, dict()),
            ("no bounded replay", 4096, [4096], dict(late=None)),
            ("not divisible (stock)", 2050, [2050], dict()),
            ("below MIN_ROWS (stock)", 1020, [1020], dict()),
        ]
        for name, rows, lens, kw in cases:
            stock, _, _ = run("stock", rows, lens, **kw)
            exact, m_ex, g_ex = run("shard", rows, lens, exact=True, **kw)
            fast, m_fast, g_fast = run("shard", rows, lens, **kw)
            comm, _, g_comm = run("comm", rows, lens, **kw)
            comm_same, _, _ = run("comm", rows, lens, rs_order="same", **kw)
            planned = exact[0][3] is not None
            for r in range(W):
                assert same(exact[r], stock[r]), (name, "exact != stock", r)
                assert same(fast[r], comm[r]), (name, "fast != P0", r)
                assert same(comm_same[r], stock[r]), (name, "P0 with a same-order RS != stock", r)
            if planned:
                assert not same(fast[0], stock[0]), (name, "the RS order should show")
                attn = m_fast.layers[0].self_attn
                assert set(attn.seen_rows) == {rows}, (name, attn.seen_rows)
                late = kw.get("late", 4)
                tail_ars = 0 if late is None else 2 * (len(m_fast.layers) - late)   # stock late layers
                assert g_fast.calls["ar"] == tail_ars == g_comm.calls["ar"], (name, g_fast.calls)
                eng_shard = m_fast.layers[1].engram.wkv_calls
                assert rows // W in eng_shard, (name, eng_shard)
            else:
                assert same(fast[0], stock[0]) and same(comm[0], stock[0]), name
            print(f"  {name:24s} rows={rows}: exact==stock, fast==P0"
                  f"{'' if planned else ' (stock path)'}; collectives fast {g_fast.calls}")

        # gates keyed on the shard size instead of the chunk: bits change (the test is sensitive)
        GATE["logical"] = False
        try:
            wrong, _, _ = run("shard", 4096, [4096], exact=True)
        finally:
            GATE["logical"] = True
        stock, _, _ = run("stock", 4096, [4096])
        assert not same(wrong[0], stock[0]), "a shard-keyed gate should change the result"

        # wkv whose result depends on M: the self-check must fall back and stay exact
        stock, _, _ = run("stock", 4096, [1000, 3096], m_dep=True)
        exact, m_ex, _ = run("shard", 4096, [1000, 3096], exact=True, m_dep=True)
        assert all(same(exact[r], stock[r]) for r in range(W)), "wkv fallback not exact"
        assert sp._WKV_OK and not any(sp._WKV_OK.values()), sp._WKV_OK
        stock, _, _ = run("stock", 4096, [1000, 3096])
        exact, _, _ = run("shard", 4096, [1000, 3096], exact=True)
        assert all(same(exact[r], stock[r]) for r in range(W))
        assert sp._WKV_OK and all(sp._WKV_OK.values()), sp._WKV_OK
    finally:
        sp.MIN_ROWS = MIN
    # debug compare + fingerprints: on (exact and fast), results unchanged, nothing flagged
    old_dbg = set(sp.DEBUG)
    sp.DEBUG.clear()
    sp.DEBUG.update({"compare", "fp"})
    try:
        stock, _, _ = run("stock", 4096, [1000, 50, 2046, 1000])
        for exact_ in (True, False):
            dbg, _, _ = run("shard", 4096, [1000, 50, 2046, 1000], exact=exact_, debug=True)
            ref, _, _ = run("shard", 4096, [1000, 50, 2046, 1000], exact=exact_)
            for r in range(W):
                d = dbg[r][3][1]
                assert same(dbg[r][:3] + (None,), ref[r][:3] + (None,)), "debug changed the result"
                assert d.checked >= 10 and not d.bad, (d.checked, d.bad)
                assert {k for _, k, _ in d.fps} == {"Rin", "ain", "aout", "min", "mout", "Rout", "pre"}
                assert {layer for layer, _, _ in d.fps} == set(range(6)), d.fps[:3]
            if exact_:
                assert all(same(dbg[r][:3] + (None,), stock[r]) for r in range(W))
    finally:
        sp.DEBUG.clear()
        sp.DEBUG.update(old_dbg)
    print("test_prefill_sp: ok")


if __name__ == "__main__":
    main()
