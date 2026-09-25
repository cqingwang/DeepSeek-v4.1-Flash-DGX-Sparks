# Decode tok/s on TP3: what to borrow from knapcio's TP4 profile, and what else to do

Date: 2026-09-24. Audited: [knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4)
at `4d8f4c0` (README, CHANGELOG, `docs/session-20260919.md`, `docs/upstream-watch.md`, every adapter,
the vendored `runtime/b12x`), this repository at `25379c2` plus the uncommitted working tree, the live
TP3 fleet (boot of 2026-09-24 02:12 local), `logs/profile-2026-09-10/REPORT.md` and the journals of
spark1/spark3. Nothing was restarted or benchmarked for this document: the fleet was serving a
dashboard's prefill sweep the whole time, and every number below is either theirs (labelled
"TP4, knapcio"), ours from REPORT.md, or read from the current boot log.

## 1. Ranked list

| # | Action | Expected on TP3 | Effort | Risk |
|---|---|---|---|---|
| 1 | **Re-pack the Engram shards on all three nodes** (`./start.sh pack`). They are missing: the head's `~/dsv41-engram/` is empty, spark2 holds two TP2 shards (`r1of2`), spark3 has no directory. Every rank logs `packed=False`, 2 reads per lookup, and the workers read the checkpoint over NFS from the head. | REPORT §6/§8 measured this state at ~11 ms/step of rank-0 all-reduce wait; boot 2 → 3 (packing + b12x) went 120 → 98 ms/step. **≈ +10 % decode, and the head stops serving NFS during decode.** | 30 min, no code, no boot beyond the one you need anyway | none (spark2 needs its 101 GB of TP2 shards deleted first: 61 GB free) |
| 2 | **Quiet spark3.** It runs a desktop (`graphical.target`, gdm active), 13 other containers, sparkDash at 23 % CPU, and has 4.0–4.4 GB `MemAvailable` against spark2's 7.7 GB. It is the rank that ran out of memory tonight (journal 01:33:54, `NVRM ... NV_ERR_NO_MEMORY`) and, in REPORT §1, the rank whose spikes set the step. | Removes the straggler and the wedge risk; 2–5 % by REPORT's rank skew; the dashboard's 2-second GPU polling alone cost knapcio 0.6 ms/step. | 30 min | low |
| 3 | **Turn on the adapters already copied from knapcio, one switch per boot, at `MEM_FRACTION_STATIC=0.95`** (order in §4.1). | Greedy step ≈ −9 to −10 ms of 82 (**≈ +12 %**): wo_a fp8 twin ≈ −6.5 ms, draft LM head fp8 ≈ −0.9 ms, Engram prefetch ≈ −1.5 to −2.5 ms; sampled chat another +2–4 % accepted tokens (block verification, draft temperature). | 1 boot each, ≈ 13 min per boot | tonight's boot enabled all of them at once at 0.98 and spark3 ran out of memory; the switches themselves are lossless by construction |
| 4 | **`DSPARK_BLOCK_SIZE=5` with `DSV41_VERIFY_CAP=conf:0.1`** (after 3). | Code +10–25 % (knapcio: k=3 costs −16 to −26 % on code on two fleets), prose 0 to +5 % (their cap: prose c1 66.0 → 69.7, c4 +6 %). The cap is what makes k=5 safe on prose, which is why k=3 was chosen here. | 2 boots (k=5 alone, then with the cap) | none: greedy output identical, sampled goes through block verification |
| 5 | **Port `adapter/autotune_keep.py`** (88 lines, one hook). Our boot log shows the same `per-rank caches disagree, discarding them` twice per boot. | Ends the boot-to-boot tactic lottery FlashInfer plays under EP (knapcio: several % between boots, 26 re-tunes → 0) and removes two autotune passes per boot. | 20 min | none |
| 6 | **Unlock the GPU clocks.** All three nodes run `nvidia-clock-lock.service` = `nvidia-smi -lgc 0,2200`; live SM clock 2177–2190 MHz, default applications clock 2418. | knapcio measured the 2200 cap at −1.5 % prose, −0.7 % code, −1.5 to −5 % prefill. | one command per node, no boot | thermal/power: whoever installed the lock had a reason; check `nvidia-smi -q -d TEMPERATURE,PERFORMANCE` under load |
| 7 | **A/B `EP_SIZE=1`** (tensor-sharded experts instead of expert-parallel). | knapcio: EP4 → EP2 halved the per-layer straggler wait (NCCL 16.5 → 10.2 ms/step). At TP3 the measurable skew is ~2 ms (rank 0 NCCL 15.6 vs rank 2 13.4 ms), so ≤ 3 %. Bytes per rank are identical either way. | 1 boot | the MXFP4 runner at intermediate 768 is untested here |
| 8 | **Engram row cache in the GB10 display reservation** (`DSV41_ENGRAM_DRM_NODE`, needs `row_store.cpp` + `engram_backend.py` from knapcio, `nvidia-drm modeset=1`, headless, reboot). | ~1.8 GiB per node outside `MemAvailable` for one layer's row cache; on their traffic hit rate 67–99 %. Speed-neutral on TP4 once the prefetch is on; on TP3 it is the only way to have a row cache at all. | host change on 3 nodes + reboot | driver 580.x only (ours is 580.159.03, theirs 580.173.02) |
| 9 | **RoCEnante (b12x one-shot RDMA all-reduce) for the TP all-reduces.** | ~88 SUM all-reduces per step at 70–100 µs median (REPORT §1); b12x measured 65 → 24 µs per 48 KB all-reduce on 4 Sparks. Ceiling ≈ −4 to −6 ms/step (**5–7 %**), larger than their TP4 gain because NCCL is 16 % of our step versus 1 % of theirs. | high: the published overlay stripes every peer across both HCAs with same-index HCAs, which cannot connect on our pairwise triangle; §6.6 has the patch shape | hang risk (b12x #313), a new transport inside the graphs |
| 10 | **Canary image** (upstream `dsv4.1` branch at `f80c91a4b`, `Dockerfile.canary`). | TP4, knapcio: prose c1 51.6 → 55.4 (+7 %), c8 +11 %, code c16 +41 %; the branch's mHC/metadata kernels target the "~2000 tiny kernels, 10 ms" line of our budget. | high: new image, every TP3 hook (`tp3_pad`, `mxfp8_b12x` scale repair, `flash_mla_sm120`) re-validated | padded vocab under the branch's sharded argmax; sampled decode −5 % vs greedy on that image |

Items 1, 2, 5 and 6 need no engine boot beyond the next one. Items 3 and 4 are eight boots at
~13 min each. Items 7–10 are separate projects.

## 2. Where a TP3 decode step goes

From REPORT §12 (boot 6, 2026-09-10; k=5, packed shards, b12x, NCCL trimmed; 82 ms mean, rank 0):

| Component | ms | Notes |
|---|---:|---|
| b12x MXFP8 dense GEMMs | ~17 | at M=6, ~79 µs per call |
| `wo_a` bf16 einsum (`cutlass_80_wmma`) | ~11 | 4 groups × 1024 × 4096 bf16 = 32 MiB per layer, 43 layers = 1.34 GiB per step at ~135 GB/s |
| LM head bf16 | ~2 | 43,136 padded vocab rows per rank |
| MoE grouped MXFP4 GEMM | ~27 | bandwidth-bound, scales with the verify rows (6 at k=5) |
| NCCL, true cost | 13–16 | 104 collectives across 3 nodes, 70–100 µs median per LL all-reduce |
| ~2000 small kernels | ~10 | attention 2, hyper-connection mixers 2, quant/norm/MoE aux |
| Engram host callbacks | ~1 | with packed shards; 2.7 ms unpacked |
| CPU gaps | ~4 | |

Today's boot is not that state: k=3 (4 verify rows), shards unpacked, KV pin clamped (`available_bytes=1.04 GB`,
`max_total_num_tokens=491520` against the 750,000 pin), head `MemAvailable` 2.9 GB. The last two hours
of `Decode batch` lines at one running request give a median `gen throughput` of 38 tok/s (p90 46.5,
peak 50.0) at a mean accept length of 2.94 out of a maximum of 4 (k=3). The README's 37.9 tok/s prose
c1 figure matches.

For scale, knapcio's TP4 production step is 41–47 ms on prose at k=5 with `conf:0.1`, prose c1 69.1,
code c1 113.8. Their base image at the same k=5 was 51.6 / 96.7, so their adapters and profile are worth
+34 % / +18 % on TP4. Not all of that transfers: two of their biggest levers are memory (16 request
slots, 4 GiB row cache) and one is a TP4-only shape (§5).

## 3. What knapcio's repository is, and what is already here

It is a downstream fork of this repository (credits Mia for the launcher, Engram store, b12x routing,
memory model), tuned for four Sparks with a switched RoCE fabric, and it keeps every change as an
env-gated import hook in `adapter/sitecustomize.py`, the same pattern as ours. Production stack as of
2026-09-23 night, with their measured deltas:

| Their lever | Measured by them (TP4) | On TP3 |
|---|---|---|
| `EP_SIZE=2` instead of 4 | NCCL 16.5 → 10.2 ms/step | EP=1 is our analogue, §6.5 |
| `DSV41_CACHE_GIB=4`, 16 ways | 67–99 % Engram hit rate | no memory for it; §4.2 item 8 |
| `MAX_RUNNING_REQUESTS=16` | adds a c16 tier, +64 % aggregate over c8 | 16 graphs cost 5.6 GB; we have 4 |
| `DSPARK_BLOCK_SIZE=5` | k=3: prose 33.8 vs 44.6, code 80.4 vs 102.7 | with the verify cap, yes |
| `DSV41_SHARED_PAD_K=1` | −0.9 ms/step | not needed: shared expert K is 768 at TP3 (2304/3), already a multiple of 128; the boot log has no b12x rejection |
| `DSV41_INDEXER_CHUNKED=1` + chunk 4096 (+ prefill TP split) | prefill 128k 3499 → 4364; decoders during a long prefill 5.7 → 7.8 tok/s | prefill; secondary |
| `DSV41_ENGRAM_PREFETCH=1` | step 51.7 → 49.5 ms, prose c1 57–59 → 60–61 | copied, off |
| `DSV41_WO_A_W8=1` (+ `MID`, `DROP`) | 43 layers 3.43 → 2.20 ms; step 52.9 → 51.7; MID c4 82.1 → 78.3 ms verify step; DROP frees 722 MB/rank | copied and adapted for TP3, off |
| `DSV41_DRAFT_HEAD_FP8=1` | 1405 → 721 µs; step 51.7 → 50.9 | copied, off |
| `DSV41_DRAFT_TAU=0.8` | +1.2 % accepted tokens (sampled) | copied, off |
| `DSV41_BLOCK_VERIFY=1` | +1.8 % prose / +2.5 % coding-with-thinking accepted (sampled) | copied, off |
| `DSV41_FOLDED_FENCE=1` | closes the sglang#40919 race | copied, off |
| `DSV41_VERIFY_CAP=conf:0.1` | prose c1 66.0 → 69.7, c4 +6 %, sampled thinking +5 %; fixed caps lose 8–40 % | copied, off |
| `DSV41_AUTOTUNE_KEEP=1` | 26 re-tunes per boot → 0 | not copied |
| RoCEnante (`Dockerfile.canary-roce`) | +3 % prose c1, +8 % structured, c16 +4.5 % | not copied, needs a triangle patch |
| canary image (`dsv4.1` at `f80c91a4b`) | prose c1 +7 %, code c16 +41 % over base | not copied |
| `DSV41_FAST_LOAD=1` | boot 343 → 124 s, KV pool −3 to −13 % | not wanted here (§5) |
| display-reservation row cache | +1.8 GiB headroom per node | not copied |
| `--sleep-on-idle`, `--enable-cache-report`, `--min-free-slots-delay 1` | idle CPU 47 → 14 %; no decode change | ours already / n.a. below 8 slots |

**Already in the working tree (uncommitted, 2026-09-24 00:52):** `block_verify.py`, `draft_head_fp8.py`,
`draft_tau.py`, `engram_prefetch.py`, `folded_result_fence.py`, `verify_cap.py` byte-identical to knapcio's;
`wo_a_w8.py` adapted for TP3 (the twin is `[G, 1024, 4096]` with G = 4 local o_groups after the 8 → 12 pad,
and an "einsum bridge" because the `dev-dsv41` image has no `wo_a_bf16_small_batch` kernel and dispatches
`wo_a` from `_apply_wo_a_bf16_matmul`); the matching tests; `sitecustomize.py`, `start.sh` and the
`Dockerfile` hooks. `start.sh` forces every switch off on TP3 with the comment that turning them on "made
spark3 drop its weights at FlashInfer autotune". The journal agrees on the what but not on the why: the
head container started at 01:32:16, spark3's kernel logged `NV_ERR_NO_MEMORY` from
`kgrctxAllocCtxBuffers` at 01:33:54 (a GPU context allocation failing for lack of memory, 98 s into the
weight load), spark3 was rebooted at 01:49, and the `.env` note says the fraction was 0.98 for that boot.
That is memory exhaustion on the node with the least headroom, with every switch on and 2 % of RAM left
for everything outside the static pool. It does not implicate any single adapter. §4.1 gives the order
that separates them.

## 4. Borrow list, item by item

### 4.1 Already copied: what each does on TP3, and the order to enable them

Each is a `DSV41_*` variable forwarded by `start.sh` to all three containers; each refuses to arm if the
engine symbol it patches has drifted, and `./start.sh doctor` prints the switch line. Keep
`MEM_FRACTION_STATIC=0.95` (the working value; 0.98 is what tonight's boot ran). One switch per boot,
smoke + a 4-prompt greedy batch + one sampled prompt after each, and `Decode batch` lines from an idle
fleet (`#running-req` must equal the concurrency being measured).

1. **`DSV41_FOLDED_FENCE=1`.** Correctness only: clones the six `AcceptOuts` tensors on the folded
   (all-greedy) verify path before the overlapped D2H copy. Free. Start here because it touches no weights.
2. **`DSV41_DRAFT_TAU=0.8`.** Scales the draft proposal temperature for sampled requests; the same tensor
   drives proposal and acceptance, so the output distribution is unchanged. +1.2 % accepted tokens at
   T=1 / top_p 0.95 on their traffic. Invisible on greedy benches.
3. **`DSV41_BLOCK_VERIFY=1`.** Block verification (Sun et al., ICLR 2025) for sampled rows: exact output
   distribution, +1.8 % (prose) / +2.5 % (coding with thinking) accepted tokens per step. Invisible on
   greedy benches. Together with 2 this is the +2–4 % real-chat gain.
4. **`DSV41_DRAFT_HEAD_FP8=1`.** The draft's block logits come from an fp8 copy of the shared LM head
   (target keeps bf16). At TP3 the head is 43,136 × 5120 bf16 = 442 MB per rank read once per step
   (≈ 1.9 ms), so the fp8 copy saves ≈ 0.9 ms/step. Memory: +221 MB per rank for the copy.
5. **`DSV41_WO_A_W8=1`** alone (MID and DROP off). After load it builds an fp8 twin (e4m3 + one exponent
   per 32×32 block) of each bf16 `wo_a` from the checkpoint's own fp8 bytes and keeps it only when the
   dequantized twin equals the bf16 weight bit for bit; the 2–8-row verify/draft path then runs the
   Triton kernel on the twin. This is the biggest single item on TP3 because our `wo_a` is twice
   theirs: 4 groups per rank instead of 2 (the o_groups 8 → 12 pad), so 1.34 GiB per step at ~135 GB/s
   ≈ 11 ms (REPORT §12). Half the bytes at their measured 164 GB/s ≈ 4.3 ms: **≈ −6.5 ms/step**.
   Rank 2's four groups are all zero, so its twins are exact trivially. Memory during the check:
   +16 MiB per layer of twin next to the bf16 copy (+0.7 GB per rank until DROP).
6. **`DSV41_WO_A_W8_MID=1`.** The 9–192-row kernel for verify/draft rows at c2–c4 (their verify step
   c4 82.1 → 78.3 ms, c8 120.0 → 118.1 ms). Our bridge only takes it during decode/verify/draft-extend
   forwards, so short prefills stay on einsum. Worth +3–5 % per stream at c4.
7. **`DSV41_ENGRAM_PREFETCH=1`.** Forks a side stream right after `EngramHasher.forward`, runs the id
   copy, the host row lookup and the row copies for both Engram layers there, and joins at each layer's
   gather; inside the CUDA graph that is a fork/join with a host node. Their step 51.7 → 49.5 ms. Ours:
   the two callbacks are ~1 ms/step packed (2.7 ms unpacked); layer 1's lookup has ~1 ms of GPU work to
   hide behind and layer 14's ~15 ms, so expect **−1.5 to −2.5 ms/step**. Pinned staging buffers are
   allocated at the first forward, before capture. `DSV41_ENGRAM_PREFETCH_CHECK=1` re-runs the old path
   and counts differing gathers (they saw 0).
8. **`DSV41_WO_A_W8_DROP=1`** last. After a second bit-identical check, the bf16 `wo_a` becomes a
   zero-stride view of a one-element tag and every call is served from the twin; prefill dequantizes
   into a fresh 32 MiB buffer per call. On TP3 this releases 43 × 32 MiB = **1.34 GiB per rank**, which
   on the head is more than the entire current KV budget (1.04 GB). Enable it alone on its own boot and
   read the `DSV4 memory calculation ... available_bytes=` line: that is the one adapter that moves the
   KV pool, and the one the `start.sh` comment blames. If the boot is clean, the freed memory can go to
   the pin (it is clamped today) or stay as head headroom, which REPORT §6 and the 2026-09-10 session
   showed is worth more than any kernel when it drops below ~1 GB.

Then, with all of the above on:

9. **`DSPARK_BLOCK_SIZE=5`** on its own boot (baseline the prose/code c1 and c4 numbers against k=3: the
   README's "22.3 → 25.4 prose at k=3" predates the packed shards, b12x and the fp8 kernels, and knapcio's
   same measurement went the other way). The boot log's `DSpark gamma mismatch ... draft config
   block_size=5` line says the draft was trained for 5.
10. **`DSV41_VERIFY_CAP=conf:0.1`** with k=5. Per request and step, only the leading drafts whose running
    product of the draft confidence head's survival stays ≥ 0.1 are verified; the dead rows' router ids
    are overwritten with the anchor row's (one in-graph Triton launch per layer, 38 µs/step for 43
    layers), so they add no expert reads, and acceptance is capped through the engine's own cutoff. Exact.
    Their gain is 5–6 % on prose at c1/c4 and +5 % on sampled thinking traffic; ours should be at least
    that, because the MoE is 33 % of our step versus ~25 % of theirs and a dead row on TP3 costs more
    expert bytes. `DSV41_VERIFY_CAP_LOG=<file>` records confidences per step on rank 0 (costs a sync) if
    the threshold needs re-fitting for TP3; they found per-position thresholds worth only +0.4 % over the
    single 0.1. Note `DSV41_VERIFY_CAP_STRIDE` defaults to `DSPARK_BLOCK_SIZE + 1`, so it follows k.

Expected sum, greedy prose c1: 82 → ≈ 72 ms/step from items 4–7 (+12–14 %), plus whatever k=5 + cap
gives on top; code c1 +10–25 % from k=5 alone. Sampled chat: +2–4 % more from 2–3.

### 4.2 Not yet copied, worth taking

- **`adapter/autotune_keep.py`** (`DSV41_AUTOTUNE_KEEP=1`). Hooks
  `sglang.srt.model_executor.runner.flashinfer_autotune`; our image has the same `_autotune_cache_digest`
  / `flashinfer_autotune_context` / `flashinfer_autotune_cache_path` symbols (checked in the running
  container). Replaces the per-rank byte digest of the cache (which can never agree under EP, so the
  cache is deleted every boot and the MoE tactics re-drawn from timing noise, sglang#40320) with the
  load decision plus a launch fingerprint sidecar. Copy the file, add the hook and the module name to
  the finder list in `sitecustomize.py`, forward the variable in `start.sh`. Note their `_VOLATILE`
  list already excludes the verify-cap/draft switches from the fingerprint so A/B boots share tactics.
- **`DSPARK_BLOCK_SIZE=5` + `conf:0.1`**: above.
- **`row_store.cpp` / `engram_backend.py` display-reservation backing** (item 8 in §1): 46 lines of
  C++ (`row_store_next_cache_on_drm`, a DRM dumb buffer on `/dev/dri/card0`) and 11 lines of Python;
  the diff against ours is otherwise empty. Only useful once the host is headless with
  `nvidia-drm modeset=1 fbdev=0`; going headless alone raised their KV pool by ~1 M tokens.
- **Measurement practice** (`docs/session-20260919.md`, README "Measurement notes"): bench from a
  worker, discard the first two points after a boot (cold row cache, FlashInfer tuning during the first
  run), check `#running-req` for foreign traffic, quote `usage.completion_tokens / elapsed` or the
  `gen throughput` field, never SSE chunk counts, and set the dashboard's GPU poll to 10 s
  (`POLL_INTERVAL_GPU=10000`, `POLL_INTERVAL_BANDWIDTH=10000`). Their `scripts/qeval.py` (75 auto-scored
  tasks, McNemar pairing) is the quality gate to run after the kernel switches; it executes
  model-generated Python, so run it from a worker.
- **`runtime/roce_tp4_adapt.py` + `runtime/sglang-rocenante.patch`** are the starting point for §6.6, not
  usable as they are.

### 4.3 Their own dead ends, so we do not repeat them

Draft fine-tune on own traffic (+3.4 % held-out, 0 % on new prompts), multi-pass drafting (+2–4 %
accepted for +9 % step per pass), expert-sharing routing (changes routing, step +7 %), n-gram lookup
(+1 % even as an oracle), relaxed acceptance (not exact, +20 % accepted only), split-K MXFP8 for the
small-M projections (1.5–3× slower than b12x), hyper-connection kernel variants (23–28 µs whatever the
layout), Markov W2 unsharded (+0.6 ms/step), parallel Engram misses in `row_store.cpp` (nothing; the
side-stream prefetch is what removed the gap), RoCE all-gather port (all-gathers are 0.56 % of their
step), fixed verify caps (−8 to −40 %), per-request adaptive block size (+4 % code at best), chunk 1024
on TP4 (prefill −34 to −38 % and decoders behind a prefill got worse), sglang#39704 (±2 % below c32),
newer `dsv4.1` heads (SM121 loses the candidate indexer after #39671), NVFP4 experts (no bandwidth saved
on GB10, +16 GiB), LuZ-0.1.7 fused kernels (few %, not bit-exact, version-bound), DSpark SPS table
(crashes the Engram path on this model, which closes the "profile the SPS table" item in our own
REPORT §2).

## 5. Not worth borrowing on TP3, and why

| Their item | Why not here |
|---|---|
| `MAX_RUNNING_REQUESTS=16` | 16 decode graphs cost them ~5.6 GB; the head has ~3 GB free while serving |
| `DSV41_CACHE_GIB=4` | the 2026-09-10 session measured the last GiB of head RAM at 2.4× throughput (12.4 → 30.1 tok/s) when the row cache was turned off; only the display-reservation variant fits |
| `DSV41_SHARED_PAD_K` | TP4-specific: K = 2304/4 = 576 is not a multiple of 128; at TP3 K = 768 is, and the boot log shows no b12x rejection |
| `DSV41_FAST_LOAD` | reads the checkpoint eagerly into pinned host RAM; costs 3–13 % of the KV pool, and our pool is already clamped by the head's 1.04 GB budget |
| `EP_SIZE=2` literally | 3 is not divisible by 2; the analogue is `EP_SIZE=1` (§6.5) |
| `CHUNKED_PREFILL_SIZE=4096` | our long-prompt peak scales with chunk × prefix (docs/chunked-prefill-memory.md); 1024 is what fits 208k on the head |
| `--min-free-slots-delay 1` | the admission delayer only engages at ≥ 8 slots |
| switchless ring, `NCCL_SWITCHLESS_RING_ONLY` | four-node ring only; our three nodes are a full triangle and boot on the defaults |

## 6. Beyond borrowing: what this audit found on the TP3 fleet

### 6.1 The Engram shards are gone (the largest item on the list)

Current boot, rank 0, one minute of stats: `Engram layer=1 lookups=43128 hit_rate=0.0% reads=86256
cache=0.0GiB ... packed=False`; the same on TP1 and TP2. Two reads per lookup means weight and scale are
read from the checkpoint tensors, and on the workers the checkpoint is the NFS export from spark1. On
disk: `~/dsv41-engram/` on the head is empty (mtime 2026-09-17 22:09:49), `~/dsv41-3x-spark/engram/` on
spark2 and spark3 are empty with the same mtime, and spark2's `~/dsv41-engram/` holds
`engram-l1-r1of2.bin` / `engram-l14-r1of2.bin` from a TP2 experiment on 2026-09-12 (101 GB). The loader
looks for `engram-l{layer}-r{rank}of{tp}.bin` under `DSV41_PACKED_DIR`, so nothing matches.

REPORT §6 traced exactly this state to ~11 ms/step of rank-0 waiting at the all-reduce that follows
each Engram gather (35–43 % of its NCCL time), and §8 recorded the fix: step 120 → 98 ms together with
b12x, Engram wait 11 → 0.5 ms. The 2026-09-10 session also noted that `./start.sh build`'s rsync
`--delete` used to sweep the workers' shards; the README now says `engram/` is excluded, but something on
2026-09-17 at 22:09 (the night of the memory-guard trip in `logs/oom-guard.log`) emptied all three
directories at once. Whatever it was, `./start.sh pack` rewrites them (~63 GiB per node, ~10 min per
node, `.partial` then rename, safe while serving; picked up at the next boot). spark2 has 61 GB free
against 63 GiB needed: delete its two TP2 shards first. Confirm with `packed=True` in every rank's
`Exact nvme Engram` line.

### 6.2 GPU clocks are locked at 2200 MHz on every node

`nvidia-clock-lock.service` (`nvidia-smi -lgc 0,2200`) is active on spark1/2/3; the SM clock reads
2177–2190 MHz under load against a default applications clock of 2418 and a maximum of 3003. knapcio
measured the same 2200 cap on a bandwidth-bound step at −1.5 % prose decode, −0.7 % code, −1.5 to −5 %
prefill. `nvidia-smi -rgc` on each node restores the default without a boot; keep it only if
temperatures or power under a sustained bench say otherwise (they reported 57–59 °C uncapped at
2509–2554 MHz).

### 6.3 spark3 is the weak rank

`graphical.target` with gdm active, 13 non-serving containers (sparkDash at 23 % CPU, autoclip ×5,
redis, postgres, `local-llm`, model-trader ×2, slate, nexusui), load average 12 in the five minutes after
its reboot, 4.0–4.4 GB `MemAvailable` against spark2's 7.7 GB. REPORT §1 already named spark3's spikes as
what sets the step (the slowest rank does), and tonight it was the rank that ran out of memory. Moving
sparkDash and the rest to spark2 (idle apart from the worker) or to a fourth machine, and
`systemctl set-default multi-user.target` on spark3, is the cheapest straggler fix on the list. If the
dashboard stays, raise its GPU/bandwidth poll interval to 10 s; it also polls `/server_info`, `/v1/loads`,
`/model_info` and `/v1/models` on the head every 2–3 s.

### 6.4 What tonight's failure was, and how to run the retry

Timeline from the journals: adapters copied 00:52; head container started 01:32:16 with every switch on
and (per the `.env` note) `MEM_FRACTION_STATIC=0.98`; spark3 kernel `NV_ERR_NO_MEMORY` from
`kgrctxAllocCtxBuffers` at 01:33:54; spark3 clean-rebooted 01:49–01:50; head restarted 01:52 and again
02:12 with the switches off. 0.98 leaves 2.4 GB per node for the CUDA context, the tokenizer/detokenizer
on the head, NCCL's 139 MB, the row-store threads and the desktop on spark3; the adapters add ~0.9 GB per
rank transiently (wo_a twins next to the bf16 copies until DROP, the fp8 draft head, prefetch staging).
The `start.sh` comment's mechanism (DROP's one-element stand-in while the autotune target-verify
forward runs the prefetch and the remap) is possible but unproven; the memory arithmetic is sufficient
on its own. Retry at 0.95, one switch per boot in the §4.1 order, DROP last, spark3 quiet first.

### 6.5 `EP_SIZE=1`: the TP3 analogue of their EP2

At EP=3 each rank holds 128 of the 384 routed experts at full width and runs only the tokens routed to
them; the per-layer straggler is the rank with the most routed work. At EP=1 with TP=3 each rank holds
all 384 experts at one third of the intermediate width (768 of 2304): the same bytes per rank, every
rank does identical work, no straggler, and still one post-MoE all-reduce (REPORT §4 established that
EP=3 without an all-to-all backend already does "input replicated, local experts, one all-reduce").
knapcio's EP4 → EP2 took 6 ms/step off their NCCL time for that reason. Our measurable skew is smaller
(rank 0 waits ~2 ms more than rank 2 spends), so the upside is ≤ 3 %; the cost is more, smaller grouped
GEMM problems per layer (up to 36 distinct experts at 768 wide instead of ~11 at 2304). One boot answers
it. `tp3_pad` pads the draft's 128 experts to 129 keyed on TP, not EP, so it behaves as today.

### 6.6 RoCEnante on a three-node triangle: what would have to change

The b12x runtime (`runtime/b12x/b12x/comm/roce/`) opens every HCA listed in `B12X_ROCE_HCA`, creates one
QP per (HCA, peer), and in `connect_qp` connects HCA *h* of rank *i* to HCA *h* of rank *j* using the
peer's GID at the same index; `post_op` then stripes every peer's payload across all HCAs and writes one
completion flag per (peer, HCA), and the GPU kernel waits for every flag. That assumes a switch where
each rail reaches every peer. Our fabric is `rocep1s0f1` 10.0.22.1 → spark2, `rocep1s0f0` 10.0.23.1 →
spark3, spark2 ↔ spark3 on 10.0.33.0/24: each HCA reaches exactly one peer, and the same-index pairing
is not even guaranteed to be the cabled pair. The 2026-09-14 session's note that "each peer gets its own
dedicated rail" is the right topology but not what the code does.

The patch is contained: (1) at `roce_create`/connect time, choose for every peer the local HCA whose GID
shares the peer's subnet (`/24` on the IPv4-mapped GID) and the peer's HCA on that link, instead of
index-equal pairing; (2) in `post_op`, send one stripe per peer on that HCA and write both flag words for
that peer over the same QP, so the kernel's wait loop is untouched; (3) drop the `len(hca_names) != 2`
warning path in the overlay. Then rhys101's SG17 overlay (214-line `sglang-rocenante.patch`, plus
knapcio's `roce_tp4_adapt.py` relaxing world size to 4/8) needs world size 3 allowed and
`SGLANG_ROCE_MAX_SIZE` sized for the 4-slot step (4 × 6 × 5120 × 2 B = 240 KB, inside the 512 KiB
default). Ceiling on our profile: ~88 SUM all-reduces per step (40 `wo_b`, 40 post-MoE, 2 Engram, 6 in
the draft) at 70–100 µs → ~25–30 µs each, i.e. −4 to −6 ms of an 82 ms step; the 6 all-gathers and 8
broadcasts stay on NCCL. That is a 5–7 % lever that costs a new transport inside the CUDA graphs and
the open b12x #313 wedge report; the result-boundary health check in the overlay is the mitigation and
`SGLANG_ROCE_ALLREDUCE=0` the escape hatch.

### 6.7 Head memory and the KV pin

`available_bytes=1.04 GB` this boot, so the 750,000-token pin was silently clamped to 491,520 (the same
silent clamp REPORT §3 describes). Two things on this list give the head memory back without touching
kernels: DROP (1.34 GiB per rank of bf16 `wo_a`, §4.1 item 8) and the display-reservation row cache
(1.8 GiB per node outside `MemAvailable`). Spend it as headroom first: the 2026-09-10 session measured
the head's last GiB at 2.4× throughput, and REPORT §14 put the decode-side cost of a bigger pool at
zero. A bigger pin only matters for long contexts.

### 6.8 Things that look like levers and are not

- **Uneven head split (32/16/16 local heads, no padding).** The SM120 sparse-MLA kernel only exists for
  8/16/32/64/128 local heads, and the step is set by the rank with the most heads, which would still be
  32. The padding costs ranks 0 and 1 nothing extra on the critical path; the true cost of TP3 is 32
  heads per rank instead of 21.3, and only a 24-head kernel would recover it.
- **Skipping rank 2's all-zero projections.** Rank 2 would finish earlier and wait; the step is ranks
  0/1. Giving rank 2 more experts to use that slack is out: the experts are ~90 GB of each rank's
  ~104 GB.
- **Markov W2 at TP3.** 43,136 × 256 bf16 = 22 MB per rank against the 25.2 MB L2; their pricing shows
  the read collapsing to DRAM speed once it no longer fits. Worth ~0.3–0.5 ms/step if it is falling out
  here; not worth a change until the bigger items are in.
- **`SGLANG_SPEC_TP_SYNC=all`.** 13 tiny syncs, ~1 ms/step (REPORT §1). Not worth the correctness risk.

## 7. Boot plan and measurement protocol

Before any boot (no engine restart needed): delete spark2's TP2 shards, `./start.sh pack` on all
nodes, quiet spark3 (or at least the dashboard's poll interval), decide on the clock lock, port
`autotune_keep`.

| Boot | Change | Record |
|---|---|---|
| A | packed shards, k=3, everything else as today | `packed=True` ×3 ranks; `available_bytes`; prose/code c1 and c4 via `benchmarks/decode_window.py` or sparkDash from spark2; 40-step profile (`scripts/profile/profile_decode.sh`) for the new budget |
| B | + `DSV41_FOLDED_FENCE=1 DSV41_DRAFT_TAU=0.8 DSV41_BLOCK_VERIFY=1 DSV41_AUTOTUNE_KEEP=1` (no weight-touching switches) | boot clean; greedy text identical to A; sampled c1 tok/s at T=1 / top_p 0.95 vs A |
| C | + `DSV41_DRAFT_HEAD_FP8=1` | step probe; acceptance unchanged |
| D | + `DSV41_WO_A_W8=1` | `[wo_a_w8]` lines: 43 of 43 exact twins on every rank; step −6 ms expected |
| E | + `DSV41_WO_A_W8_MID=1` | c2/c4 per-stream |
| F | + `DSV41_ENGRAM_PREFETCH=1` (`_CHECK=1` for this boot only) | `differing gathers: 0`; step −2 ms expected |
| G | + `DSV41_WO_A_W8_DROP=1` | `available_bytes` up by ~1.3 GB; greedy text unchanged after the first request |
| H | `DSPARK_BLOCK_SIZE=5` | prose and code c1/c4 vs G |
| I | + `DSV41_VERIFY_CAP=conf:0.1` | prose c1/c4, sampled thinking, `spec_verify_ct`; qeval from a worker |
| J | `EP_SIZE=1` (against I) | step, NCCL share, `#running-req` clean |

Rules that their and our measurements both insist on: idle fleet and `#running-req` checked in the log;
first two points after a boot discarded; the same sparkDash/`decode_window.py` prompts every time;
greedy for kernel A/Bs (sampling costs ~5 % and adds noise), sampled T=1 / top_p 0.95 for the
draft-side switches; `usage.completion_tokens / elapsed` or `gen throughput`, never SSE chunks; the
memory guard armed for anything that changes memory; and no long prompts in the boot tests.
