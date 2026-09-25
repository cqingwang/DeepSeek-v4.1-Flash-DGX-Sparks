"""fast_load slabs: a shard's eager tensors live as views in a few large host buffers (CPU check).

Runs without CUDA (the slabs are anonymous mmaps then; the packing, views and lifetimes are the
same code as the pinned path). Checks, on synthetic shards with the checkpoint's expert shapes:
- ``slab_layout``: every eager tensor placed once, in the loader's (sorted) order within a slab, page-aligned,
  non-overlapping, within the slab size; tensors bigger than a slab get their own buffer.
- bytes: every rank at EP1 (moe_tp 4) and EP2 (moe_tp 2), target and draft, through the wrapped
  ``safe_open``, equal to the stock ``safe_open`` path (reusing test_fast_load_tp_slice.run_rank),
  for a small slab (many slabs per shard, big tensors standalone), the default and SLAB_MB=0.
- allocation count: one host buffer per slab (+ one per oversized tensor) instead of one per tensor.
- lifetimes: every slab is gone once the tensors are dropped; unrequested futures do not pin a slab
  after ``__exit__``; a view the model kept (plain or TP-sliced stand-in) is reported by name by
  ``_release_all`` and its slab frees when it is dropped.
- pacing + bounded memory: driving the shards like sglang's buffered iterator (max_workers=1) into
  a paced, slow copy pool keeps the budget on tensor bytes and the live slab bytes within the
  loader window + budget + 2 slabs.
- delayed small copies: the first (small) tensor of every slab copies slowly while its siblings
  finish at once. Each slow copy keeps its whole slab alive; the slab charge keeps the live slabs
  within window + budget + 2 slabs, where the old per-tensor charge let every slab of the load
  stay alive at once (shown with the old charge swapped in).
- escape after a load: through the real ``load_weights`` wrappers, a view kept past the target load
  is an error naming the tensor (a clean load returns normally, a load that raised keeps its own
  exception); after the draft load it is a warning.
- memlog: ``DSV41_FAST_LOAD_MEMLOG=1`` logs one line per phase with the allocation counts.
"""
import collections
import concurrent.futures
import gc
import itertools
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_fast_load_tp_slice as tps  # noqa: E402  (sets DSV41_FAST_LOAD=1, imports fast_load)

import safetensors  # noqa: E402
import safetensors.torch as st  # noqa: E402
import torch  # noqa: E402

fl = tps.fl
MB = 1 << 20


def set_slab(mb):
    if mb is None:
        os.environ.pop("DSV41_FAST_LOAD_SLAB_MB", None)
    else:
        os.environ["DSV41_FAST_LOAD_SLAB_MB"] = str(mb)


def all_slabs_expired():
    gc.collect()
    with fl._lock:
        refs = list(fl._slab_refs)
    return [r for r in refs if r[0] is not None and not r[0].expired()]


def check_layout(files):
    for path in files:
        base, header = fl._parse_header(path)
        for plan in (fl.read_plan(path, "target", (0, 1), 4, (1, 4)), dict.fromkeys(fl.needed_names(path, "target", (0, 2), 4)),
                     dict.fromkeys(fl.needed_names(path, "draft"))):
            for slab in (1 * MB, 4 * MB, 256 * MB):
                lay = fl.slab_layout(header, plan, slab)
                order = [m[0] for _, ms in lay for m in ms]
                want = [k for k in sorted(plan) if fl.spec_bytes(header[k], plan[k])[1] > 0]
                assert sorted(order) == want and len(order) == len(set(order)), (path, slab)
                assert all([m[0] for m in ms] == sorted(m[0] for m in ms) for _, ms in lay), (path, slab)
                for nbytes, ms in lay:
                    end = 0
                    for name, off, kept in ms:
                        assert off % fl._SLAB_ALIGN == 0 and off >= end, (name, off, end)
                        assert kept == fl.spec_bytes(header[name], plan[name])[1]
                        end = off + kept
                    assert end <= nbytes
                    assert nbytes <= slab or len(ms) == 1, (nbytes, slab, len(ms))
                    if nbytes > slab:
                        assert ms[0][2] > slab
    assert fl.slab_bytes() == 256 * MB
    for v, want in (("0", 0), ("300", 256 * MB), ("1", MB), ("0.1", MB), ("junk", 256 * MB), ("4096", 4096 * MB)):
        os.environ["DSV41_FAST_LOAD_SLAB_MB"] = v
        assert fl.slab_bytes() == want, (v, fl.slab_bytes())
    set_slab(None)
    print("slab_layout: sorted, page-aligned, non-overlapping, bounded; SLAB_MB parsing rounds to 2^k")


def count_eager(files, phase, ep, tp):
    """Tensors with bytes that the plan reads for one rank (= host buffers without slabs)."""
    n = 0
    for f in files:
        _, header = fl._parse_header(f)
        plan = fl.read_plan(f, phase, ep if phase == "target" else None, tps.N_TARGET,
                            fl.tp_slicing(ep, tp, "auto") if phase == "target" or ep[1] == 1 else None)
        if phase == "target" and fl.tp_slicing(ep, tp, "auto") is None:
            plan = dict.fromkeys(fl.needed_names(f, phase, ep, tps.N_TARGET))
        n += sum(1 for k, sp in plan.items() if fl.spec_bytes(header[k], sp)[1] > 0)
    return n


def check_bytes_and_counts(files):
    rows = []
    for slab_mb in (4, None, 0):
        set_slab(slab_mb)
        for r in range(4):
            for ep, tp in (((0, 1), (r, 4)), ((r // 2, 2), (r % 2, 2))):
                for phase in ("target", "draft"):
                    before = fl._state["host_allocs"], fl._state["slabs"]
                    s = tps.run_rank(files, phase, ep, tp, None)   # asserts bytes == stock safe_open
                    allocs = fl._state["host_allocs"] - before[0]
                    slabs = fl._state["slabs"] - before[1]
                    left = all_slabs_expired()
                    assert not left, (slab_mb, r, ep, phase, [x[3][:2] for x in left])
                    n_eager = count_eager(files, phase, ep, tp)
                    if slab_mb == 0:
                        assert slabs == 0 and allocs == n_eager, (allocs, n_eager)
                    else:
                        assert slabs == allocs < n_eager, (slabs, allocs, n_eager)
                    if r == 0:
                        rows.append((slab_mb, "EP1" if ep[1] == 1 else "EP2", phase, n_eager, allocs))
                    fl._escaped_slabs()          # forget this run's slabs
    set_slab(None)
    for slab_mb, lay, phase, n_eager, allocs in rows:
        print(f"  SLAB_MB={slab_mb if slab_mb is not None else '256 (default)':>13} {lay} {phase:6}: "
              f"{n_eager:3d} eager tensors -> {allocs:3d} host buffers")
    print("bytes: EP1 + EP2, 4 ranks, target + draft, SLAB_MB 4 / 256 / 0: equal to stock safe_open; "
          "no slab alive after the tensors are dropped")


def check_lifetimes(files):
    set_slab(4)
    fl._ep_info, fl._moe_tp_info, fl._state["phase"] = (lambda: (0, 1)), (lambda: (1, 4)), "target"
    os.environ.pop("DSV41_FAST_LOAD_TP_SLICE", None)
    fl._escaped_slabs()
    # 1) only one tensor requested: __exit__ must drop the other futures, so after the kept tensor
    #    goes, nothing of the shard stays alive
    with safetensors.safe_open(files[0], framework="pt", device="cpu") as f:
        assert isinstance(f, fl._EagerShard)
        one = f.get_tensor("layers.0.attn.wq_a.weight")
        handle = f
    assert not handle._futures, "unrequested futures survived __exit__"
    time.sleep(0.05)
    alive = all_slabs_expired()
    assert len(alive) == 1, len(alive)                 # only the slab holding `one`
    del one
    assert not all_slabs_expired()
    del handle
    fl._escaped_slabs()
    # 2) escaped views are reported by name, and free when dropped
    with safetensors.safe_open(files[0], framework="pt", device="cpu") as f:
        got = {k: f.get_tensor(k) for k in f.keys()}
    kept_plain = got["layers.0.attn.wo_a.weight"]
    kept_slice = got["layers.0.ffn.experts.1.w1.scale"]          # TP-sliced stand-in
    kept_narrow = got["layers.0.ffn.experts.0.w3.weight"].narrow(0, 576, 576)   # the view FusedMoE copies from
    assert isinstance(kept_slice, fl._tp_slice_cls())
    del got
    live, live_bytes, rss, escaped = fl._release_all()
    names = sorted({n for _, _, ns in escaped for n in ns})
    assert names == sorted(["layers.0.attn.wo_a.weight", "layers.0.ffn.experts.1.w1.scale",
                            "layers.0.ffn.experts.0.w3.weight"]), names
    assert 1 <= len(escaped) <= 3 and live >= len(escaped), (escaped, live)
    print(f"escape guard: {len(escaped)} slab(s) kept alive by 3 held views reported by name: {names}")
    # the weak references were handed to the report; re-register the survivors through the storage
    from torch.multiprocessing.reductions import StorageWeakRef
    refs = [StorageWeakRef(t.untyped_storage()) for t in (kept_plain, kept_narrow, kept_slice._dsv41_part)]
    del kept_plain, kept_slice, kept_narrow
    gc.collect()
    assert all(r.expired() for r in refs)
    mm = [o for o in gc.get_objects() if isinstance(o, __import__("mmap").mmap) and not o.closed]
    assert not mm, len(mm)
    set_slab(None)
    print("lifetimes: __exit__ drops unrequested futures; slabs free with their last view; nothing left mapped")


def build_many(d, n_shards=8):
    """n_shards target shards with two routed experts each (layer i), plus one draft shard."""
    files = []
    for i in range(n_shards):
        sd = tps.experts(f"layers.{i}", range(2 * (i % 2), 2 * (i % 2) + 2))
        sd[f"layers.{i}.attn.wq_a.weight"] = tps.rnd((96, 128), torch.bfloat16)
        sd[f"layers.{i}.ffn.gate.weight"] = tps.rnd((4, 5120), torch.bfloat16)
        p = os.path.join(d, f"model-{i + 1:05d}-of-{n_shards:05d}.safetensors")
        st.save_file(sd, p)
        files.append(p)
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump({"text_config": {"n_routed_experts": tps.N_TARGET}}, f)
    return files


def buffered_iterator(files, max_workers=1):
    """sglang weight_utils.buffered_multi_thread_safetensors_weights_iterator (image 2026-09-18)."""
    def _load_file(st_file):
        with safetensors.safe_open(st_file, framework="pt", device="cpu") as f:
            return {k: f.get_tensor(k) for k in f.keys()}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        it = iter(files)
        pending = collections.deque((p, ex.submit(_load_file, p)) for p in itertools.islice(it, max_workers + 1))
        while pending:
            _, fut = pending.popleft()
            state_dict = fut.result()
            del fut
            nxt = next(it, None)
            if nxt is not None:
                pending.append((nxt, ex.submit(_load_file, nxt)))
            for name in sorted(state_dict.keys()):
                yield name, state_dict[name]


def check_paced_bound(d):
    files = build_many(d)
    slab = 4 * MB
    set_slab(4)
    fl._ep_info, fl._moe_tp_info, fl._state["phase"] = (lambda: (0, 1)), (lambda: (2, 4)), "target"
    fl._escaped_slabs()
    fl._state["slab_peak_live"] = fl._state["slab_peak_bytes"] = 0
    shard_kept = []
    for p in files:
        _, h = fl._parse_header(p)
        lay = fl.slab_layout(h, fl.read_plan(p, "target", (0, 1), tps.N_TARGET, (2, 4)), slab)
        shard_kept.append(sum(n for n, _ in lay))
    one = 5120 * 288                                   # a w2 slice, the largest single copy here
    budget = 3 * one
    os.environ["DSV41_FAST_LOAD_INFLIGHT_GB"] = repr(budget / 2**30)

    def orig(*, executor, futures, use_async, func, func_args=(), func_kwargs=None):
        futures.append(executor.submit(func, *func_args, **(func_kwargs or {})))

    mod = types.SimpleNamespace(maybe_executor_submit=orig)
    fl.install_deepseek_v4(mod)
    lock, cur, peak = threading.Lock(), [0], [0]
    param = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
    Slice = fl._tp_slice_cls()

    def copy(p, t):
        n = t._dsv41_resident_nbytes if isinstance(t, Slice) else t.numel() * t.element_size()
        with lock:
            cur[0] += n
            peak[0] = max(peak[0], cur[0])
        src = t.narrow(t._dsv41_dim, t._dsv41_start, t._dsv41_part.shape[t._dsv41_dim]) if isinstance(t, Slice) else t
        src.contiguous().view(torch.uint8).sum()      # touch the bytes like the H2D copy would
        time.sleep(0.002)
        with lock:
            cur[0] -= n

    futures = []
    with concurrent.futures.ThreadPoolExecutor(24) as ex:
        for name, t in buffered_iterator(files):
            mod.maybe_executor_submit(executor=ex, futures=futures, use_async=True, func=copy, func_args=(param, t))
            del t
        for fu in concurrent.futures.as_completed(futures):
            fu.result()
    del futures
    peak_live = fl._state["slab_peak_bytes"]
    window = max(sum(shard_kept[i:i + 3]) for i in range(len(shard_kept)))
    assert peak[0] <= budget, (peak[0], budget)
    assert peak_live <= window + budget + 2 * slab, (peak_live, window, budget)
    assert not all_slabs_expired()
    fl._escaped_slabs()
    set_slab(None)
    print(f"paced load over {len(files)} shards: copies in flight peak {peak[0] / 1e6:.1f} MB <= budget "
          f"{budget / 1e6:.1f} MB; live slabs peak {peak_live / 1e6:.1f} MB <= 3-shard window {window / 1e6:.1f} MB "
          f"+ budget + 2 slabs; all slabs freed")


def build_small(d, n_shards=8, per_shard=64):
    """n_shards shards of per_shard 64 KiB non-expert tensors (16 per 1 MiB slab, 4 slabs a shard)."""
    files = []
    for i in range(n_shards):
        sd = {f"layers.{i}.attn.t{j:03d}.weight": tps.rnd((128, 256), torch.bfloat16) for j in range(per_shard)}
        p = os.path.join(d, f"model-{i + 1:05d}-of-{n_shards:05d}.safetensors")
        st.save_file(sd, p)
        files.append(p)
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump({"text_config": {"n_routed_experts": tps.N_TARGET}}, f)
    return files


def delayed_load(files, slow, budget, gate=None, delay=0.15):
    """Paced load where the tensors in ``slow`` copy late: after ``gate`` is set (the old charge,
    which never blocks here) or after ``delay`` seconds. Returns (peak live slab bytes, seconds)."""
    fl._escaped_slabs()
    fl._state["slab_peak_live"] = fl._state["slab_peak_bytes"] = 0
    os.environ["DSV41_FAST_LOAD_INFLIGHT_GB"] = repr(budget / 2**30)

    def orig(*, executor, futures, use_async, func, func_args=(), func_kwargs=None):
        futures.append(executor.submit(func, *func_args, **(func_kwargs or {})))

    mod = types.SimpleNamespace(maybe_executor_submit=orig)
    fl.install_deepseek_v4(mod)
    param = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    def copy(p, t, name):
        if name in slow:
            if gate is not None:
                assert gate.wait(timeout=30), "gate never opened"
            else:
                time.sleep(delay)
        t.view(torch.uint8).sum()

    t0 = time.time()
    futures = []
    with concurrent.futures.ThreadPoolExecutor(64) as ex:
        for name, t in buffered_iterator(files):
            mod.maybe_executor_submit(executor=ex, futures=futures, use_async=True, func=copy,
                                      func_args=(param, t, name))
            del t
        if gate is not None:
            gate.set()
        for fu in concurrent.futures.as_completed(futures):
            fu.result()
    del futures
    peak = fl._state["slab_peak_bytes"]
    assert not all_slabs_expired()
    fl._escaped_slabs()
    return peak, time.time() - t0


def check_delayed_copies(d):
    files = build_small(d)
    slab = 1 * MB
    set_slab(1)
    fl._ep_info, fl._moe_tp_info, fl._state["phase"] = (lambda: (0, 1)), (lambda: (0, 4)), "target"
    slow, shard_kept, n_slabs = set(), [], 0
    for p in files:
        _, h = fl._parse_header(p)
        lay = fl.slab_layout(h, fl.read_plan(p, "target", (0, 1), tps.N_TARGET, (0, 4)), slab)
        assert len(lay) == 4 and all(len(ms) == 16 for _, ms in lay), [len(ms) for _, ms in lay]
        slow |= {ms[0][0] for _, ms in lay}          # the first 64 KiB tensor of every slab
        shard_kept.append(sum(n for n, _ in lay))
        n_slabs += len(lay)
    budget = 3 * slab
    window = max(sum(shard_kept[i:i + 3]) for i in range(len(shard_kept)))
    bound = window + budget + 2 * slab
    peak, secs = delayed_load(files, slow, budget)
    assert peak <= bound, (peak, bound)
    # the same load with the old charge (tensor bytes only): nothing stops the loader, so every slab
    # stays alive behind its slow copy
    new_charge = fl._charge
    fl._charge = lambda args: (None, fl._tensor_bytes(args))
    try:
        old_peak, _ = delayed_load(files, slow, budget, gate=threading.Event())
    finally:
        fl._charge = new_charge
    assert old_peak > bound, (old_peak, bound)
    set_slab(None)
    print(f"delayed small copies over {n_slabs} slabs of {slab >> 20} MiB: live slabs peak {peak / MB:.0f} MiB <= "
          f"window {window / MB:.0f} + budget {budget / MB:.0f} + 2 slabs = {bound / MB:.0f} MiB ({secs:.1f} s); "
          f"per-tensor charge: {old_peak / MB:.0f} MiB")


def check_escape_raises(files):
    set_slab(4)
    fl._ep_info, fl._moe_tp_info, fl._state["phase"] = (lambda: (0, 1)), (lambda: (1, 4)), "target"
    fl._escaped_slabs()
    held = []
    draft_file = next(p for p in files if any(k.startswith("mtp.") for k in fl._parse_header(p)[1]))

    def load(path, keep, fail=False):
        with safetensors.safe_open(path, framework="pt", device="cpu") as f:
            got = {k: f.get_tensor(k) for k in f.keys()}
        if keep:
            held.append(got[keep])
        del got
        if fail:
            raise ValueError("load failed")
        return "loaded"

    class Target:
        def load_weights(self, weights, keep=None, fail=False):
            return load(files[0], keep, fail)

    class Draft:
        def load_weights(self, weights, keep=None):
            return load(draft_file, keep)

    fl._wrap_target_load(types.SimpleNamespace(DeepseekV4ForCausalLM=Target))
    fl.install_dspark(types.SimpleNamespace(DeepseekV4ForCausalLMDSpark=Draft))
    assert Target().load_weights(None) == "loaded"
    try:
        Target().load_weights(None, keep="layers.0.attn.wo_a.weight")
        raise AssertionError("an escaped slab after the target load did not raise")
    except RuntimeError as exc:
        assert "still alive after the target load" in str(exc) and "layers.0.attn.wo_a.weight" in str(exc), exc
    held.clear()
    try:
        Target().load_weights(None, keep="layers.0.attn.wo_a.weight", fail=True)
        raise AssertionError("the load's own exception was lost")
    except ValueError:
        pass
    held.clear()
    mtp = sorted(k for k in fl._parse_header(draft_file)[1] if k.startswith("mtp."))[0]
    assert Draft().load_weights(None, keep=mtp) == "loaded" and fl._state["phase"] == "target"
    held.clear()
    gc.collect()
    assert not all_slabs_expired()
    set_slab(None)
    print("escape after load: target raises naming the tensor, a failed load keeps its exception, draft warns")


def check_memlog(files):
    """DSV41_FAST_LOAD_MEMLOG=1: one line per phase end with the counters and the allocation counts."""
    import logging

    recs = []

    class H(logging.Handler):
        def emit(self, record):
            recs.append(record.getMessage())

    h = H()
    fl.logger.addHandler(h)
    os.environ["DSV41_FAST_LOAD_MEMLOG"] = "1"
    try:
        fl._ep_info, fl._moe_tp_info, fl._state["phase"] = (lambda: (0, 1)), (lambda: (3, 4)), "target"
        for k in ("host_allocs", "host_alloc_bytes", "slabs", "slab_peak_live", "slab_peak_bytes"):
            fl._state[k] = 0
        fl._mark_load_start()
        with safetensors.safe_open(files[1], framework="pt", device="cpu") as f:
            got = {k: f.get_tensor(k) for k in f.keys()}
        del got
        fl._log_phase("target")
    finally:
        os.environ.pop("DSV41_FAST_LOAD_MEMLOG", None)
        fl.logger.removeHandler(h)
    line = next(r for r in recs if r.startswith("DSV41 fast load memlog: phase=target"))
    summary = next(r for r in recs if "host buffers:" in r)
    for key in ("'pinned_requests': 1", "'slabs': 1", "'slabs_alive_after_release': 0", "'gpu_procs_mib'",
                "'torch_host_alloc'", "'slab_mb': 256"):
        assert key in line, (key, line)
    if os.path.exists("/proc/meminfo"):
        keys = ["'MemTotal'", "'MemAvailable'", "'unaccounted_gib'"] if fl._cuda_mem()[0] is not None else \
            ["'MemTotal'", "'MemAvailable'"]
        if "SecPageTables" in open("/proc/meminfo").read():
            keys.append("'SecPageTables'")
        for key in keys:
            assert key in line, (key, line)
    assert fl._state["host_allocs"] == 0, "phase counters must reset"
    print("memlog: " + line[:160] + " ...")
    print("summary: " + summary[summary.index("host buffers"):][:120])


def main():
    d = tempfile.mkdtemp(prefix="fl_slab_")
    try:
        files = tps.build(d)
        fl.install_weight_utils(types.SimpleNamespace(safetensors=None))
        assert fl._n_routed_experts(files[0]) == tps.N_TARGET
        check_layout(files)
        check_bytes_and_counts(files)
        check_lifetimes(files)
        d2 = os.path.join(d, "many")
        os.makedirs(d2)
        check_paced_bound(d2)
        d3 = os.path.join(d, "small")
        os.makedirs(d3)
        check_delayed_copies(d3)
        check_escape_raises(files)
        check_memlog(files)
        fl._release_all()
        print("fast_load slab OK")
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    main()
