"""CPU checks of DSV41_INDEXER_NOSYNC (adapter/indexer_chunked_v3.py): the eager prefill indexer's slot ids
without host syncs are the stock ids, and the whole indexer output is unchanged. No GPU, no sglang import.

Runs tests/test_indexer_chunked_v3.py first (stock == chunked, imported as the harness), then:
- slot_ids_nosync == the stock per-request loop (k_slots values + dtype, starts) on 400 random batches:
  1-17 requests in a random permutation of a 64-row req_to_token pool of random slots, lc 0..300 (zeros
  included), compress ratios 1 / 2 / 4 / 128, req_pool_indices longer than the batch (padding);
- the full indexer (page / raw indices, published and tail-only candidate masks, consumed masks, the SG18 split at
  world 1 / 2 / 4) is identical with the knob on, off, and in the stock function lifted from the backend, at mixed
  request lengths with permuted pool rows and random slots;
- no host read of device data: on the meta device (any .item() / data-dependent shape raises) the nosync builder
  runs and the stock loop raises, which is the per-request sync the knob removes;
- the knob is read per call under the armed in-boot A/B (adapter/ab_variant.py RUNTIME key), once otherwise.

usage: cd tests && PYTHONPATH=../adapter python3 test_indexer_nosync.py /path/to/deepseek_v4_backend.py
"""
import os
import random
import sys

import torch

os.environ.pop("DSV41_INDEXER_NOSYNC", None)
import test_indexer_chunked_v3 as base  # noqa: E402  (runs the v3 equivalence suite; same argv)
import indexer_chunked_v3 as V  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(("[OK]   " if cond else "[FAIL] ") + name + (f" -- {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


def stock_slots(req_to_token, req_pool_indices, lc_per_req, ratio):
    """The verbatim stock loop (indexer_chunked_v3 / deepseek_v4_backend)."""
    device = req_to_token.device
    slot_chunks, starts, start = [], [], 0
    for r, lc in enumerate(lc_per_req):
        starts.append(start)
        if lc == 0:
            continue
        j = torch.arange(lc, device=device)
        slot_chunks.append(req_to_token[req_pool_indices[r], j * ratio].to(torch.int64) // ratio)
        start += lc
    return (torch.cat(slot_chunks) if slot_chunks else None), starts


# ---- 1. the slot-id builder ----------------------------------------------------------------------------------
rng = random.Random(20261026)
bad = 0
for trial in range(400):
    n = rng.randint(1, 17)
    ratio = rng.choice((1, 2, 4, 4, 4, 128))
    lc = [rng.choice((0, 1, rng.randint(1, 40), rng.randint(1, 300))) for _ in range(n)]
    if trial % 7 == 0:
        lc = [0] * n
    width = max(1, max(lc) * ratio + 1)
    pool = torch.randint(0, 1 << 30, (64, width), dtype=torch.int32)
    rows = torch.randperm(64)[: n + rng.randint(0, 3)].to(torch.int64)
    want, want_starts = stock_slots(pool, rows, lc, ratio)
    got, got_starts = V.slot_ids_nosync(pool, rows, lc, ratio, pool.device)
    same = (got_starts == want_starts) and (
        (want is None and got is None) or
        (want is not None and got is not None and got.dtype == want.dtype and torch.equal(got, want)))
    bad += not same
check("slot ids: nosync == stock loop on 400 random batches (values, dtype, starts, all-zero batches)", bad == 0,
      f"{bad} differ")
h = V.h2d([3, 0, 5], torch.int32, torch.device("cpu"))
check("h2d on CPU == torch.tensor", h.dtype == torch.int32 and h.tolist() == [3, 0, 5])


# ---- 2. the whole indexer, knob on / off / stock --------------------------------------------------------------
def run(mode, budget, perm_seed, **kw):
    self, layer, x, q_lora, pos, fb, q_lens, q_lens_cpu = base.build(**kw)
    g = torch.Generator().manual_seed(perm_seed)
    nreq = len(kw["seq_lens"])
    width = self.req_to_token.shape[1]
    self.req_to_token = torch.randint(0, 1 << 20, (48, width), generator=g, dtype=torch.int32)
    fb.req_pool_indices = torch.randperm(48, generator=g)[:nreq].to(torch.int32)
    if mode == "stock":
        import types
        self._publish_or_consume_candidates = types.MethodType(base.stock_publish, self)
        base.StockBackend._low_ratio_index_topk_dense(self, layer, x, q_lora, pos, fb, q_lens, q_lens_cpu)
    else:
        os.environ["DSV41_INDEXER_NOSYNC"] = "1" if mode == "nosync" else "0"
        f = V.make_index_topk_dense(base.fake_module, base.quantize_fp4_indexer_tensor, budget)
        os.environ.pop("DSV41_INDEXER_NOSYNC", None)
        f(self, layer, x, q_lora, pos, fb, q_lens, q_lens_cpu)
    cm = self.forward_metadata.candidate_metadata
    masks = cm.request_masks if isinstance(cm, base.CandidateMasks) else None
    core = self.forward_metadata.core_metadata
    return core._page.clone(), core._raw.clone(), masks


def same_masks(a, b, tail_lens=None):
    if a is None or b is None:
        return a is None and b is None
    if len(a) != len(b):
        return False
    for i, (m, r) in enumerate(zip(a, b)):
        if tail_lens is not None and r.numel():
            r = r[r.shape[0] - tail_lens[i]:]
        if m.shape != r.shape or not torch.equal(m, r):
            return False
    return True


def full(name, budgets, **kw):
    ok = True
    for seed in (1, 2, 3):
        sp, sr, sm = run("stock", None, seed, **kw)
        for budget in budgets:
            op, orw, om = run("off", budget, seed, **kw)
            np_, nr, nm = run("nosync", budget, seed, **kw)
            tl = kw.get("tail_lens")
            ok &= torch.equal(np_, op) and torch.equal(nr, orw) and same_masks(nm, om)
            ok &= torch.equal(np_, sp) and torch.equal(nr, sr) and (not kw["source"] or same_masks(nm, sm, tl))
    check(f"indexer: {name}: nosync == off == stock (3 pool permutations, budgets {budgets})", ok)


small = [16, 160, 1 << 31]
full("mixed lengths, 5 requests", small, seq_lens=[40, 9, 61, 120, 17], q_lens=[12, 9, 5, 33, 17], ratio=4,
     source=False)
full("lc = 0 and t_len = 0 requests", small, seq_lens=[1, 30, 20, 3], q_lens=[1, 0, 6, 3], ratio=4, source=False)
full("publish masks, ratio 1", small, seq_lens=[40, 25, 70], q_lens=[10, 7, 30], ratio=1, source=True)
full("publish tail-only masks", small, seq_lens=[40, 25, 33], q_lens=[10, 7, 9], ratio=1, source=True,
     tail_lens=[3, 7, 1])
full("consume candidate masks", small, seq_lens=[40, 25], q_lens=[10, 7], ratio=1, source=False,
     consume_masks=[torch.rand(10, 40) > 0.4, torch.rand(7, 25) > 0.4])
full("16 requests (a c16 prefill batch)", [1 << 31], seq_lens=[28 + 3 * i for i in range(16)],
     q_lens=[28 + 3 * i for i in range(16)], ratio=4, source=False)
for world in (1, 2, 4):
    base.SPLIT[0] = base.make_split(world)
    full(f"SG18 split world={world}, ragged", [64, 1 << 31], seq_lens=[300, 129, 700], q_lens=[140, 129, 260],
         ratio=1, source=False)
    full(f"SG18 split world={world}, publish tail-only", [64, 1 << 31], seq_lens=[300, 260, 500],
         q_lens=[140, 130, 200], ratio=1, source=True, tail_lens=[3, 130, 7])
base.SPLIT[0] = None

# negative control: the full-indexer comparison catches slot ids taken from the wrong pool row
real_builder = V.slot_ids_nosync


def wrong_row(req_to_token, req_pool_indices, lc_per_req, ratio, device):
    return real_builder(req_to_token, req_pool_indices.flip(0), lc_per_req, ratio, device)


V.slot_ids_nosync = wrong_row
n_before = len(FAILS)
import contextlib, io  # noqa: E401,E402
with contextlib.redirect_stdout(io.StringIO()):
    full("neg", [1 << 31], seq_lens=[40, 9, 61], q_lens=[12, 9, 5], ratio=4, source=False)
caught = len(FAILS) > n_before
del FAILS[n_before:]
V.slot_ids_nosync = real_builder
check("negative control: slot ids from the wrong pool rows are caught", caught)

# ---- 3. no host read of device data (meta device) -------------------------------------------------------------
meta = torch.device("meta")
pool = torch.empty(64, 4096, dtype=torch.int32, device=meta)
rows = torch.empty(5, dtype=torch.int64, device=meta)
try:
    k, starts = V.slot_ids_nosync(pool, rows, [10, 0, 7, 33, 1], 4, meta)
    ks = torch.repeat_interleave(V.h2d(starts, torch.int32, meta), torch.empty(5, dtype=torch.int64, device=meta),
                                 output_size=51)
    nosync_ok = k.shape == (51,) and k.dtype == torch.int64 and starts == [0, 10, 10, 17, 50] and ks.shape == (51,)
except Exception as exc:  # noqa: BLE001
    nosync_ok = f"{type(exc).__name__}: {exc}"
try:
    stock_slots(pool, rows, [10, 0, 7, 33, 1], 4)
    stock_raises = False
except Exception:  # noqa: BLE001
    stock_raises = True
check("no host sync: the nosync builder runs on the meta device; the stock loop cannot (its .item() per request)",
      nosync_ok is True and stock_raises, f"nosync={nosync_ok} stock_raises={stock_raises}")

# ---- 4. the knob under the in-boot A/B --------------------------------------------------------------------------
cur = {"v": "0"}
saved = V._ab
V._ab = type("AB", (), {"ACTIVE": True, "KNOWN": ("DSV41_INDEXER_NOSYNC",),
                        "env": staticmethod(lambda name, default=None: cur["v"])})
reader = V._nosync_reader()
first = reader()
cur["v"] = "1"
second = reader()
V._ab = type("AB", (), {"ACTIVE": False, "KNOWN": ()})
os.environ["DSV41_INDEXER_NOSYNC"] = "1"
fixed = V._nosync_reader()
os.environ["DSV41_INDEXER_NOSYNC"] = "0"
check("knob: per call under the armed A/B (0 then 1), fixed at build time otherwise",
      first is False and second is True and fixed() is True)
# standalone and A/B readers parse every value the same way
agree = True
for raw in ("0", "off", "OFF", "false", "False", "no", "NO", "", " 0 ", "1", "on", "true", "yes"):
    V._ab = type("AB", (), {"ACTIVE": True, "KNOWN": ("DSV41_INDEXER_NOSYNC",),
                            "env": staticmethod(lambda name, default=None, r=raw: r)})
    ab_val = V._nosync_reader()()
    V._ab = type("AB", (), {"ACTIVE": False, "KNOWN": ()})
    os.environ["DSV41_INDEXER_NOSYNC"] = raw
    alone = V._nosync_reader()()
    want = raw.strip().lower() not in ("", "0", "off", "false", "no")
    agree &= ab_val is want and alone is want
check("knob: standalone and A/B readers agree (no / OFF / False are off in both)", agree)
os.environ.pop("DSV41_INDEXER_NOSYNC", None)
os.environ.pop("DSV41_INDEXER_NOSYNC", None)
V._ab = saved
try:
    import ab_variant
    check("knob: DSV41_INDEXER_NOSYNC is an ab_variant RUNTIME key (never unioned)",
          "DSV41_INDEXER_NOSYNC" in ab_variant.RUNTIME and "DSV41_INDEXER_NOSYNC" not in ab_variant.UNION_BOOL)
except ImportError:
    print("[SKIP] ab_variant not on the path")

print("ALL PASS" if not FAILS else f"FAILED: {len(FAILS)}: {FAILS}")
sys.exit(1 if FAILS else 0)
