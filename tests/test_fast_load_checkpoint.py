"""In-image check of adapter/fast_load.py against the real checkpoint (not part of the build).

Run on a node with the weights mounted, CPU only:
  docker run --rm -v <model dir>:/models/DeepSeek-V4.1-Flash:ro \
    -v $PWD/tests/test_fast_load_checkpoint.py:/t.py:ro \
    --entrypoint /opt/sglang/bin/python3 dsv41-4x-spark:canary-roce /t.py

Checks: draft filtering yields exactly the mtp keys with 45 shards skipped; eager tensors are
bitwise equal to the stock safe_open tensors for every dtype; the target set covers ~155 GB per
EP rank, and at EP1 (moe_tp 4) the same ~155 GB read / ~83 GB kept with every TP slice bitwise equal
to the stock tensor's narrow; RSS returns to baseline after a shard's tensors are released.
"""
import os, sys, json, time, torch
os.environ["DSV41_FAST_LOAD"] = "1"
sys.path.insert(0, "/opt/dsv41/adapter")
import fast_load, safetensors
from safetensors import safe_open as raw_open
M = "/models/DeepSeek-V4.1-Flash"
files = sorted(f"{M}/{f}" for f in os.listdir(M) if f.endswith(".safetensors"))
idx = json.load(open(f"{M}/model.safetensors.index.json"))["weight_map"]
mtp = {k for k in idx if k.startswith("mtp.")}
n = fast_load._n_routed_experts(files[0]); assert n == 384, n
for ep in ((0, 2), (1, 2)):
    tot = sum(b - a for f in files for a, b in fast_load.needed_ranges(f, "target", ep, n))
    print(f"target ep={ep}: {tot/1e9:.1f} GB eager"); assert 150e9 < tot < 160e9
# EP1 (moe_tp 4): each rank reads its w1/w3 row block and whole w2 rows of all 384 experts
for tp in ((0, 4), (3, 4)):
    plans = {f: fast_load.read_plan(f, "target", (0, 1), n, tp) for f in files}
    tot = sum(b - a for f in files for a, b in fast_load.plan_ranges(f, plans[f]))
    kept = sum(fast_load.spec_bytes(fast_load._parse_header(f)[1][k], sp)[1] for f in files for k, sp in plans[f].items())
    print(f"target ep1 moe_tp={tp}: {tot/1e9:.1f} GB read, {kept/1e9:.1f} GB kept"); assert 150e9 < tot < 160e9 and 80e9 < kept < 86e9
from sglang.srt.model_loader import weight_utils
fast_load.install_weight_utils(weight_utils)
# draft phase through the real loader
fast_load._state["phase"] = "draft"
# Compare tensor by tensor and drop them: never hold a whole draft (7.9 GB) or shard in memory,
# the buffers may be pinned and the node may be serving.
t = time.time(); seen = set(); dtypes = {}; raw = {}
def raw_tensor(name):
    f = raw.get(idx[name])
    if f is None:
        f = raw[idx[name]] = raw_open(f"{M}/{idx[name]}", framework="pt", device="cpu")
    return f.get_tensor(name)
for name, e in weight_utils.buffered_multi_thread_safetensors_weights_iterator(files, max_workers=1):
    r = raw_tensor(name)
    assert e.dtype == r.dtype and e.shape == r.shape, name
    assert torch.equal(e.view(-1).view(torch.uint8), r.view(-1).view(torch.uint8)), name
    dtypes[str(e.dtype)] = dtypes.get(str(e.dtype), 0) + 1
    seen.add(name); del e, r
print(f"draft iteration: {len(seen)} tensors in {time.time()-t:.1f}s; skipped={fast_load._state['skipped']}")
assert seen == mtp, (len(seen), len(mtp))
print("draft tensors bitwise equal:", dtypes)
# target phase on one shard for EP rank 1 with a fake layout
fast_load._state["phase"] = "target"
fast_load._ep_info = lambda: (1, 2)
with safetensors.safe_open(files[7], framework="pt", device="cpu") as f, raw_open(files[7], framework="pt", device="cpu") as g:
    keys = list(f.keys()); need = set(fast_load.needed_names(files[7], "target", (1, 2), n))
    ok = 0
    for k in keys:
        e = f.get_tensor(k); r = g.get_tensor(k)
        assert e.dtype == r.dtype and e.shape == r.shape and torch.equal(e.view(-1).view(torch.uint8), r.view(-1).view(torch.uint8)), k
        ok += 1; del e, r
print(f"target shard: {ok} tensors ({len(need)} eager) bitwise equal to mmap")
# EP1 TP slicing on the same shard for moe_tp rank 1 of 4: the narrow FusedMoE takes
# (shape[dim] // 4 at rank 1) returns bytes equal to the stock tensor's narrow
fast_load._ep_info = lambda: (0, 1)
fast_load._moe_tp_info = lambda: (1, 4)
Slice = fast_load._tp_slice_cls(); sl = 0
with safetensors.safe_open(files[7], framework="pt", device="cpu") as f, raw_open(files[7], framework="pt", device="cpu") as g:
    for k in f.keys():
        e = f.get_tensor(k); r = g.get_tensor(k)
        if isinstance(e, Slice):
            d = fast_load._TP_SLICE_DIM[k.rsplit(".", 2)[1]]; m = r.shape[d] // 4
            a, b = e.narrow(d, m, m), r.narrow(d, m, m)
            assert a.dtype == b.dtype and a.shape == b.shape and torch.equal(
                a.contiguous().view(-1).view(torch.uint8), b.contiguous().view(-1).view(torch.uint8)), k
            try:
                e.narrow(d, 0, m); raise AssertionError(f"{k}: out-of-slice narrow did not raise")
            except RuntimeError:
                pass
            sl += 1
        else:
            assert torch.equal(e.view(-1).view(torch.uint8), r.view(-1).view(torch.uint8)), k
        del e, r
print(f"target shard at EP1 moe_tp 1/4: {sl} expert tensors TP-sliced, slices bitwise equal"); assert sl > 0
fast_load._ep_info = lambda: (1, 2)
print("OK")
# memory: eager tensors of a whole shard must be released, not retained by malloc arenas
import gc
def rss():
    for l in open("/proc/self/status"):
        if l.startswith("VmRSS"): return int(l.split()[1]) // 1024
r0 = rss()
with safetensors.safe_open(files[9], framework="pt", device="cpu") as f:
    held = [f.get_tensor(k) for k in fast_load.needed_names(files[9], "target", (1, 2), n)[:300]]
r1 = rss(); del held; gc.collect(); fast_load._release_all(); r2 = rss()
print(f"RSS MB: before {r0}, holding 300 tensors {r1}, after release {r2}")
assert r1 - r0 > 500 and r2 - r0 < 300, (r0, r1, r2)
print("OK3")
