"""GB10 check of DSV41_INDEXER_NOSYNC's slot-id builder (adapter/indexer_chunked_v3.py), one idle GPU, no engine.

1. equality: slot_ids_nosync == the stock per-request loop on CUDA tensors (300 random batches, permuted pool rows,
   lc 0..2000, ratios 1 / 2 / 4 / 128), and the pinned non-blocking h2d == torch.tensor(..., device=cuda);
2. no host sync, by PyTorch's own detector: under torch.cuda.set_sync_debug_mode("error") the nosync builder plus
   the request-start copy run, while the stock loop raises (control: the detector is live);
4. queued-copy stress: 400 builds alternating two streams with varying lengths, each queued behind a
   sleep kernel, host staging tensors dropped at once and same-sized pinned blocks churned between builds so the
   caching host allocator would hand a still-in-flight block out again if the copy did not hold it; results are
   compared only after the queue drained;
5. memory: peak device allocation of one build at the largest contract (1 x 65536 and 16 x 8192 compressed slots)
   and the host allocator's pinned statistics (recorded, informational);
3. host cost behind a queued GPU kernel (torch.cuda._sleep ~5 ms ahead, as during a prefill): stock waits for the
   queue once per request, nosync returns at host speed; bs 1 / 4 / 16 at the speedbench prompt sizes (lc 5..30) and
   one long-context request (lc 32768). Pure host timing, no profiler.
Memory: < 64 MiB device, pinned staging of a few hundred bytes. Exit 0 = PASS, 1 = FAIL.
usage: PYTHONPATH=/opt/dsv41/adapter python3 gpu_test_indexer_nosync.py [--out F.json]
       TT_CPU_SMOKE=1 ... runs part 1 on the CPU (no sync detector, no timing)
"""
import argparse
import json
import random
import statistics
import sys
import time

import torch

import indexer_chunked_v3 as V


def stock_slots(req_to_token, req_pool_indices, lc_per_req, ratio):
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


def stock_build(pool, rows, lc, ratio):
    k, starts = stock_slots(pool, rows, lc, ratio)
    return k, torch.tensor(starts, dtype=torch.int32, device=pool.device)


def nosync_build(pool, rows, lc, ratio):
    k, starts = V.slot_ids_nosync(pool, rows, lc, ratio, pool.device)
    return k, V.h2d(starts, torch.int32, pool.device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--smoke", action="store_true", help="CPU only (also TT_CPU_SMOKE=1)")
    a = ap.parse_args()
    import os
    smoke = a.smoke or os.environ.get("TT_CPU_SMOKE") == "1"
    dev = torch.device("cpu" if smoke else "cuda")
    res = {"device": str(dev), "checks": {}}
    rng = random.Random(7)
    pool = torch.randint(0, 1 << 30, (128, 8192 * 4 + 1), dtype=torch.int32, device=dev)   # 16.8 MB
    bad = 0
    for trial in range(300):
        n = rng.randint(1, 16)
        ratio = rng.choice((1, 2, 4, 4, 128))
        lc = [rng.choice((0, rng.randint(1, 30), rng.randint(1, 2000))) for _ in range(n)]
        lc = [min(x, (pool.shape[1] - 1) // ratio) for x in lc]
        rows = torch.randperm(128, device=dev)[:n].to(torch.int64)
        (k0, s0), (k1, s1) = stock_build(pool, rows, lc, ratio), nosync_build(pool, rows, lc, ratio)
        same = torch.equal(s0, s1) and ((k0 is None and k1 is None) or (k0 is not None and k1 is not None and
                                                                     k0.dtype == k1.dtype and torch.equal(k0, k1)))
        bad += not same
    res["checks"]["equal_300"] = bad == 0
    if not smoke:
        lc, rows = [12, 0, 30, 7], torch.tensor([5, 9, 77, 3], dtype=torch.int64, device=dev)
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            nosync_build(pool, rows, lc, 4)
            res["checks"]["nosync_no_sync"] = True
        except RuntimeError as exc:
            res["checks"]["nosync_no_sync"] = False
            res["nosync_sync_error"] = str(exc)[:300]
        try:
            stock_build(pool, rows, lc, 4)
            res["checks"]["stock_detected"] = False          # the detector would be dead
        except RuntimeError:
            res["checks"]["stock_detected"] = True
        finally:
            torch.cuda.set_sync_debug_mode("default")
        torch.cuda.synchronize()
        # 4. queued-copy stress
        streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        jobs, outs = [], []
        rngs = random.Random(11)
        for it in range(400):
            n = rngs.randint(1, 16)
            lc = [rngs.choice((0, rngs.randint(1, 64), rngs.randint(1, 3000))) for _ in range(n)]
            rows = torch.randperm(128)[:n].to(torch.int64)
            jobs.append((rows.clone(), list(lc)))
            st = streams[it % 2]
            with torch.cuda.stream(st):
                torch.cuda._sleep(200_000)
                k, sdev = nosync_build(pool, rows.to(dev, non_blocking=False), lc, 4)
                outs.append((k, sdev))
            churn = [torch.empty(len(lc) * 2 + 1, dtype=torch.int64, pin_memory=True) for _ in range(4)]
            for c in churn:
                c.fill_(-7)            # writes to blocks of the staging size while copies may be in flight
            del churn
        torch.cuda.synchronize()
        pool_cpu = pool.cpu()
        bad = 0
        for (rows, lc), (k, sdev) in zip(jobs, outs):
            k0, s0 = stock_slots(pool_cpu, rows, lc, 4)
            ok = sdev.cpu().tolist() == s0 and ((k0 is None and k is None) or
                                                 (k0 is not None and k is not None and torch.equal(k.cpu(), k0)))
            bad += not ok
        res["checks"]["stress_equal"] = bad == 0
        res["stress_bad"] = bad
        # 5. memory at the largest contract
        big = torch.randint(0, 1 << 30, (17, 65536 * 4 + 1), dtype=torch.int32, device=dev)
        mem = {}
        for name, lcs in (("1x65536", [65536]), ("16x8192", [8192] * 16)):
            rows = torch.arange(len(lcs), dtype=torch.int64, device=dev)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            k, sdev = nosync_build(big, rows, lcs, 4)
            torch.cuda.synchronize()
            mem[name] = {"peak_extra_mib": round((torch.cuda.max_memory_allocated() - base) / 2**20, 2),
                         "result_mib": round(k.numel() * k.element_size() / 2**20, 2)}
            del k, sdev
        hs = getattr(torch.cuda, "host_memory_stats", None)
        if hs is not None:
            try:
                st = hs()
                mem["host_pinned"] = {k_: v for k_, v in st.items() if "peak" in k_ or "allocations" in k_}
            except Exception as exc:  # noqa: BLE001
                mem["host_pinned"] = f"unavailable: {exc!r}"
        res["memory"] = mem
        del big
        timing = {}
        cases = {"bs1_short": [8], "bs4_short": [5, 8, 7, 6], "bs16_short": [5 + i % 6 for i in range(16)],
                 "bs1_long32k": [32768 // 4]}
        for name, lcs in cases.items():
            rows = torch.randperm(128, device=dev)[:len(lcs)].to(torch.int64)
            out = {}
            for label, fn in (("stock", stock_build), ("nosync", nosync_build)):
                xs = []
                for it in range(25):
                    torch.cuda.synchronize()
                    torch.cuda._sleep(5_000_000)          # a few ms of queued GPU work ahead of the host
                    t0 = time.perf_counter_ns()
                    fn(pool, rows, lcs, 4)
                    xs.append(time.perf_counter_ns() - t0)
                    torch.cuda.synchronize()
                out[label] = {"p50_us": round(statistics.median(xs[5:]) / 1e3, 1),
                              "min_us": round(min(xs[5:]) / 1e3, 1)}
            timing[name] = out
        res["host_us_behind_queue"] = timing
    ok = all(res["checks"].values())
    # a CPU smoke is never a CUDA receipt: it says SMOKE, and a qualification requires PASS + cuda
    res["verdict"] = ("SMOKE" if smoke else "PASS") if ok else "FAIL"
    print(json.dumps(res, indent=1))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
