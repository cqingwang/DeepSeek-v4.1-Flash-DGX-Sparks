"""GPU qualification + timing of adapter/cert_head.py on ONE Spark with the fleet STOPPED (no engine, no NCCL).

Run inside the serving image so Triton / torch / cuBLAS match production (container is left exited, never removed):

  docker run --name cert-qual-$(date +%H%M%S) --gpus all --ipc=host --network none --entrypoint python3 \
    -v ~/models/deepseek-ai/DeepSeek-V4.1-Flash:/model:ro -v "$PWD":/work \
    -e PYTHONPATH=/work/adapter "$IMAGE" /work/tests/gpu_test_cert_ds.py --model /model

Per rank shard (head.weight rows [r * V/tp, (r + 1) * V/tp), exactly what ParallelLMHead holds at TP4):
  1 kernels   which cuBLAS kernel torch.matmul picks per M (profiler)
  2 qualify   full mode (flag 0): the exact kernel's bf16 logits vs torch.matmul(x, W.T) (the SGLang stock op), bit for
              bit, every logit, every M in --ms, families normlike / massive / peaked / neartie
  3 certify   cert mode (flag 1): lo/hi soundness of the kernel's screen interval against the stock logits
              (hi >= l everywhere, T <= max l), masked argmax + value == stock, computed tiles bit-identical,
              tiles computed per step
First rank only:
  4 timing    CUDA-graph replay time: stock matmul vs cert mode vs full mode, per M
Prints DSV41_CERT_HEAD_M built ONLY from the M that passed 2 and 3 on every tested rank, and OVERALL PASS/FAIL.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "adapter"))
os.environ.setdefault("DSV41_CERT_HEAD", "1")

import cert_head as ch  # noqa: E402

FAILS = []


def verdict(ok, name, detail=""):
    tag = "INFO" if ok is None else ("PASS" if ok else "FAIL")
    print(f"{tag} {name} {detail}", flush=True)
    if ok is False:
        FAILS.append(name)


def load(model_dir, key, r0=None, r1=None):
    from safetensors import safe_open
    idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
    with safe_open(os.path.join(model_dir, idx[key]), framework="pt") as f:
        sl = f.get_slice(key)
        t = sl[r0:r1] if r0 is not None else sl[:]
    return t


def acts(kind, M, W, gamma, gen):
    V, K = W.shape
    n = torch.randn(M, K, generator=gen, device="cuda")
    if kind == "massive":
        n[:, torch.randint(0, K, (3,), generator=gen, device="cuda")] *= 30.0
    n = n / n.pow(2).mean(-1, keepdim=True).sqrt()
    x = n * gamma[None, :]
    if kind in ("peaked", "neartie"):
        t = torch.randint(0, V, (M,), generator=gen, device="cuda")
        w = W[t].float()
        a = torch.empty(M, 1, device="cuda").uniform_(0.15, 0.45, generator=gen)
        x = x + a * w / w.norm(dim=1, keepdim=True) * x.norm(dim=1, keepdim=True)
        if kind == "neartie":
            t2 = (t + 1 + torch.randint(0, V - 1, (M,), generator=gen, device="cuda")) % V
            w2 = W[t2].float()
            x = x + a * w2 / w2.norm(dim=1, keepdim=True) * x.norm(dim=1, keepdim=True)
    return x.to(torch.bfloat16).contiguous()


def stock(x, W):
    return torch.matmul(x.to(W.dtype), W.T)       # LogitsProcessor._compute_lm_head, plain bf16 branch


def sec_kernels(W, gamma, gen, Ms):
    try:
        from torch.profiler import ProfilerActivity, profile
        for M in Ms:
            x = acts("normlike", M, W, gamma, gen)
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                stock(x, W)
                torch.cuda.synchronize()
            names = sorted({e.name for e in prof.events() if e.device_type.name == "CUDA"})
            verdict(None, f"M={M:3d} stock kernels:", "; ".join(n[:110] for n in names))
    except Exception as e:  # noqa: BLE001
        verdict(None, "profiler unavailable", repr(e))


def run_cert(st, bufs, x, flag):
    bufs.flag.fill_(flag)
    bufs.flag_host = flag
    bufs.need.zero_()
    dbg = {}
    out = ch.certified_logits(st, bufs, x, debug=dbg)
    torch.cuda.synchronize()
    return out, dbg, int(bufs.need.item())


def sec_rank(st, bufs, W, gamma, gen, Ms, reps, rank):
    print(f"\n== [rank {rank}] 2 qualify (full mode) + 3 certify (cert mode) ==", flush=True)
    fams = ("normlike", "massive", "peaked", "neartie")
    ok_m = set()
    nt = (st.V + ch.XBV - 1) // ch.XBV
    for M in Ms:
        full_bad = full_n = 0
        snd_bad = arg_bad = comp_bad = 0
        tiles = []
        for rep in range(reps):
            x = acts(fams[rep % len(fams)], M, W, gamma, gen)
            ref = stock(x, W)
            full, _, need_f = run_cert(st, bufs, x, 0)
            full_bad += int((full != ref).sum())
            full_n += ref.numel()
            if need_f != nt:
                full_bad += 1
            cert, dbg, need = run_cert(st, bufs, x, 1)
            tiles.append(need)
            r = ref.float()
            hi = dbg["hi"].float()
            T = dbg["thr"]
            snd_bad += int((hi < r).sum()) + int((T > r.max(dim=1).values).sum())
            c = cert.float()
            ia, ib = r.argmax(dim=1), c.argmax(dim=1)
            arg_bad += int((ia != ib).sum()) + int((r.gather(1, ia[:, None]) != c.gather(1, ib[:, None])).sum())
            comp = c != float("-inf")
            comp_bad += int(((c != r) & comp).sum())
        ok = full_bad == 0 and snd_bad == 0 and arg_bad == 0 and comp_bad == 0
        if ok:
            ok_m.add(M)
        tiles.sort()
        verdict(ok, f"[rank {rank}] M={M:3d}", f"full-mode logits differing {full_bad}/{full_n}; interval violations "
                f"{snd_bad}; argmax/value mismatches {arg_bad}; computed-logit mismatches {comp_bad}; tiles computed "
                f"median {tiles[len(tiles) // 2]} max {tiles[-1]} of {nt}")
    # min_new_tokens stop rule: the argmax of 3 rows declared a stop id; exact with and without the -inf mask
    sbad = 0
    for M in Ms[:4]:
        for rep in range(4):
            x = acts(("peaked", "neartie", "normlike", "massive")[rep], M, W, gamma, gen)
            ref = stock(x, W).float()
            stops = sorted(set(int(t) for t in ref.argmax(1)[:3].tolist()) | {1, 2})
            bufs.stop.fill_(-1)
            bufs.stop[:len(stops)] = torch.tensor(stops, dtype=torch.int32, device="cuda")
            cert, _, _ = run_cert(st, bufs, x, 1)
            for masked in (False, True):
                a, c = ref.clone(), cert.float()
                if masked:
                    a[:, stops] = float("-inf")
                    c[:, stops] = float("-inf")
                ia, ic = a.argmax(1), c.argmax(1)
                sbad += int((ia != ic).sum()) + int((a.gather(1, ia[:, None]) != c.gather(1, ic[:, None])).sum())
    bufs.stop.fill_(-1)
    verdict(sbad == 0, f"[rank {rank}] min_new_tokens stop rule", f"violations {sbad}")
    # non-finite row: every tile computed, the finite rows still exact
    M = Ms[0]
    x = acts("normlike", M, W, gamma, gen)
    x[1, 7] = float("inf")
    ref = stock(x, W)
    cert, dbg, need = run_cert(st, bufs, x, 1)
    fin = torch.isfinite(ref).all(dim=1)
    same = bool(((cert == ref) | (cert.isnan() & ref.isnan()))[fin].all())
    verdict(need == nt and same, f"[rank {rank}] non-finite row", f"tiles {need}/{nt}, finite rows bit-identical {same}")
    return ok_m


def graph_time(fn, iters=200):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(20):
        g.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        g.replay()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters


def sec_timing(st, bufs, W, gamma, gen, Ms):
    print("\n== 4 timing (CUDA graph replay, weights stream from DRAM) ==", flush=True)
    for M in Ms:
        x = acts("peaked", M, W, gamma, gen)
        t_stock = graph_time(lambda: stock(x, W))
        bufs.flag.fill_(1)
        t_cert = graph_time(lambda: ch.certified_logits(st, bufs, x))
        bufs.flag.fill_(0)
        t_full = graph_time(lambda: ch.certified_logits(st, bufs, x))
        bufs.flag.fill_(1)
        t_stock2 = graph_time(lambda: stock(x, W))
        ts = min(t_stock, t_stock2)
        verdict(None, f"M={M:3d}", f"stock {t_stock:.3f}/{t_stock2:.3f} ms, certified {t_cert:.3f} ms "
                f"(saving {ts - t_cert:+.3f} ms), full mode {t_full:.3f} ms ({t_full - ts:+.3f} ms vs stock)")


def sec_exact_tiles(st, bufs, W, gamma, gen):
    """Full-mode (T>0 steps) speed of other exact-kernel tilings; each is also checked bit-identical at M=6/24."""
    print("\n== 5 exact-kernel tilings (full mode, M=6 / 24) ==", flush=True)
    keep = (ch.XBV, ch.XBK, ch.XWARPS, ch.XSTAGES)
    for bv, bk, wp, sg in ((16, 128, 4, 3), (16, 128, 4, 4), (16, 64, 4, 4), (32, 128, 4, 3), (32, 64, 4, 4),
                           (64, 64, 4, 3), (64, 128, 8, 3)):
        ch.XBV, ch.XBK, ch.XWARPS, ch.XSTAGES = bv, bk, wp, sg
        try:
            res = []
            for M in (6, 24):
                x = acts("normlike", M, W, gamma, gen)
                ref = stock(x, W)
                full, _, _ = run_cert(st, bufs, x, 0)
                diff = int((full != ref).sum())
                bufs.flag.fill_(0)
                t = graph_time(lambda: ch.certified_logits(st, bufs, x))
                res.append(f"M={M} {t:.3f} ms diff {diff}")
            verdict(None, f"BV={bv} BK={bk} warps={wp} stages={sg}", "; ".join(res))
        except Exception as e:  # noqa: BLE001
            verdict(None, f"BV={bv} BK={bk} warps={wp} stages={sg}", f"failed: {e!r}"[:200])
    ch.XBV, ch.XBK, ch.XWARPS, ch.XSTAGES = keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--ranks", default="0,1,2,3")
    ap.add_argument("--ms", default="6,12,18,24,30,36,42,48,60,72,84,96")
    ap.add_argument("--timing-ms", default="6,12,24,48,96")
    ap.add_argument("--reps", type=int, default=8)
    a = ap.parse_args()
    Ms = [int(v) for v in a.ms.split(",")]
    verdict(None, "torch", f"{torch.__version__} cuda {torch.version.cuda} {torch.cuda.get_device_name()} "
            f"bf16 reduced-precision reduction {torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}")
    gamma = load(a.model, "norm.weight").float().cuda()
    full_v = None
    per_rank = {}
    gen = torch.Generator(device="cuda").manual_seed(4321)
    for rank in (int(v) for v in a.ranks.split(",")):
        if full_v is None:
            from safetensors import safe_open
            idx = json.load(open(os.path.join(a.model, "model.safetensors.index.json")))["weight_map"]
            with safe_open(os.path.join(a.model, idx["head.weight"]), framework="pt") as f:
                shp = f.get_slice("head.weight").get_shape()
            full_v = shp[0]
            verdict(None, "head.weight", f"shape {shp}")
        vs = full_v // a.tp
        t0 = time.time()
        W = load(a.model, "head.weight", rank * vs, (rank + 1) * vs).to(torch.bfloat16).cuda().contiguous()
        st = ch.build_tables(W)
        bufs = ch.Buffers(W.device, ch.MAX_M)
        verdict(None, f"[rank {rank}] tables", f"{st.V} x {st.K}, eps {st.eps * 100:.3f} %, unsafe rows {st.n_unsafe}, "
                f"{time.time() - t0:.1f} s")
        if rank == 0:
            sec_kernels(W, gamma, gen, sorted({6, 12, 24, 48, 96} & set(Ms)))
        per_rank[rank] = sec_rank(st, bufs, W, gamma, gen, Ms, a.reps, rank)
        if rank == 0:
            sec_timing(st, bufs, W, gamma, gen, [int(v) for v in a.timing_ms.split(",")])
            sec_exact_tiles(st, bufs, W, gamma, gen)
        del W, st, bufs
        torch.cuda.empty_cache()
    ok = set(Ms)
    for s in per_rank.values():
        ok &= s
    line = ",".join(str(m) for m in sorted(ok))
    print(f"\nDSV41_CERT_HEAD_M={line}", flush=True)
    print("OVERALL", "PASS" if not FAILS and ok else "FAIL", f"({len(FAILS)} failed checks)", flush=True)


if __name__ == "__main__":
    main()
