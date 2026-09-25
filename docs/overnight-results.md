# Overnight TP3 optimization — results

2026-09-24, 03:11-11:11 IDT, three DGX Sparks (GB10), TP_SIZE=3 throughout. Plan:
[merged-new.md](merged-new.md). Running log with every experiment: [overnight-progress.md](overnight-progress.md).
Raw evidence (per-request JSONL, summaries, acceptance joins, boot logs, env files, profiles):
[results/overnight-20260924-0311/](results/overnight-20260924-0311/).

**Result: the fleet serves a configuration that is 17-40 % faster single-stream and 19 % faster at C4
than the one it started with, validated on two boots with no measured quality regression.**
Biggest levers: TP3/EP1 (+8-12 %), the `wo_a` FP8 twin with DROP/MID, k=5 with the confidence cap
(code), and Engram prefetch. Two operational defects were found and fixed on the way: rank 2 was running
a different SGLang build from ranks 0/1, and the packed Engram shards were missing on every rank.

| workload (5 reps, 512 tokens, from spark2) | original (b0) | final F1 (fresh tune) | final F2 (cache reuse) | change b0 -> F2 |
|---|---|---|---|---|
| C1 prose, greedy (tok/s) | 29.28 | 34.29 | **34.15** | **+16.6 %** |
| C1 prose2, greedy (added at E2; E2 value in first column) | (29.56) | 32.27 | **32.56** | (+10.1 % vs E2) |
| C1 code, greedy | 44.65 | 62.69 | **62.24** | **+39.4 %** |
| C1 sampled chat (T=0.7) | 28.87 | 36.57 | **37.20** | **+28.9 %** |
| C4 aggregate, 2 prose + 2 code | 63.78 | 76.03 | **75.79** | **+18.8 %** |
| C4 per-stream decode | 19.46 | 24.52 | 24.31 | +24.9 % |
| TTFT, C1 (52-token prompt) | 0.28-0.30 s | 0.25-0.29 s | 0.25-0.29 s | unchanged |
| bs=1 step, prose / code | 73.2 / 74.0 ms | 58.7 / 68.5 ms | 59.0 / 69.7 ms | -14 / -4 ms (k 3 -> 5) |
| acceptance, prose / code | 2.17 / 3.27 | 2.05 / 4.00 | 2.04 / 3.98 | code +22 % (k=5) |
| effective KV pool (tokens) | 491,520 | 658,176 | 749,824 (pin) | see memory section |

Quality on the final configuration (F1 and F2, plus E8 with identical settings): 8/8 checkable tasks,
8/8 under 4-way concurrency, greedy 256-token repeat 1 distinct of 3, needles found at 28.7k and 95.5k
(F1) and at 190.9k prompt tokens (E8) with prose/code/sampled requests running alongside. No
NaN/inf/OOM/traceback lines in any final-configuration log. With autotune reuse, the greedy code text
was byte-identical between F1 and F2.

The fleet is left serving the final configuration from the default `.env` via `./start.sh` (F3,
rebooted ~09:47 after the E11 extra experiment; same settings and tactics as F2).

## How it was measured

- `scripts/overnight/bench.sh LABEL 5 512` runs [benchmarks/overnight_bench.py](../benchmarks/overnight_bench.py)
  **from spark2** against the head: one discarded warm-up per workload and one C4 warm-up wave, then
  5 repetitions per workload, 512 max tokens, thinking off. Workloads: greedy prose (lighthouse essay),
  greedy prose2 (fishing-village story, added at E2), greedy code (LRU cache + tests), sampled chat
  (T=0.7, top_p=0.95); C4 = four concurrent requests (2 prose + 2 code) x 5 waves.
- Tokens are `usage.completion_tokens`; never SSE event counts. C1 decode tok/s =
  (tokens - 1) / (last content delta - first content delta); C4 aggregate = total completion tokens /
  wave wall time. Acceptance and step time come from the head's `Decode batch` lines inside each phase
  window ([scripts/overnight/accept.py](../scripts/overnight/accept.py): step = accept_len x bs /
  gen throughput).
- `scripts/overnight/quality.sh LABEL [needle_k,...]` runs [benchmarks/overnight_quality.py](../benchmarks/overnight_quality.py):
  8 checkable tasks (arithmetic, JSON schema, generated `is_prime` executed against 10 cases, sorting,
  counting, facts), the same 8 fired 4 at a time, a greedy repeat check, and needle retrieval.
- Every experiment is its own env file (`.env` + the listed changes) in `results/.../envs/`, booted with
  `ENV_FILE=<file> ./start.sh serve`. `start.sh` sources the env file with `set -a`, so command-line
  overrides of variables that `.env` defines are silently ignored; use an env file.
- Foreign traffic: none during measurements (last foreign chat request 02:29). sparkDash on spark3
  polls the API every 2 s (5 GETs, 0.06 % CPU); pausing it was refused by the session's safety policy,
  so it is a constant background in every number here.
- **Cross-boot noise.** Until autotune retention worked (E7), FlashInfer re-drew its tactics every boot,
  which changes the numerics, hence the greedy text, hence acceptance: a few % of C1 tok/s between boots
  of one configuration. bs=1 step time is the steadier signal and is reported alongside. From E7 on,
  arms that differ only in volatile switches reuse one tactic set, and greedy code output was
  byte-identical across boots.

## Starting state and what was repaired

| issue found | action |
|---|---|
| Packed Engram shards absent on all ranks (`packed=False`) | spark2's old TP2 shards (101.4 GB) moved to **spark1:`/home/mia/archive/dsv41-engram-tp2-spark2/`** (SHA-256 verified both sides, `SHA256SUMS` there; pointer at spark2:`/home/zurih/dsv41-engram/MOVED.txt`); TP3 shards packed on all three ranks; `packed=True` for layers 1 and 14 on ranks 0/1/2 after restart |
| **Rank 2 ran a different SGLang build** (spark3's `lmsysorg/sglang:dev-dsv41` = 381b27ff / da64c5cbb, 40 source files different from ranks 0/1 incl. `deepseek_v4.py` and the DSpark worker) | spark3's images kept as `lmsysorg/sglang:dev-dsv41-da64c5cbb-20260911` and `dsv41-3x-spark:mixed-da64c5cbb-20260924`; the head's base (37939c26) loaded onto spark3; overlay rebuilt everywhere |
| Autotune caches discarded twice per boot | `adapter/autotune_keep.py` ported, plus two fixes (below) |

## Accepted changes (in order of application)

| step | change | evidence |
|---|---|---|
| B1 | packed Engram + identical images on all ranks | C4 +5 %, code +2 %; Engram waiting gone from the profile |
| E2 | `DSV41_WO_A_W8=1`, `DSV41_WO_A_W8_DROP=1`, `DSV41_WO_A_W8_MID=1` (target layers only) | bs=1 step 72 -> 68-69 ms, bs=4 131.5 -> 125.6 ms; KV pool back to the 750k pin (DROP frees 1280 MB/rank) |
| E4 | `DSPARK_BLOCK_SIZE=5` + `DSV41_VERIFY_CAP=conf:0.1` + `DSV41_BLOCK_VERIFY=1` | code +9.6 %, prose/sampled/C4 neutral vs k=3 |
| E5 | `DSV41_DRAFT_HEAD_FP8=1` | step -0.1 to -1.3 ms on all C1 workloads, +210 MiB |
| E6b | `DSV41_ENGRAM_PREFETCH=1` (check-mode pass first: 0 differing gathers on all ranks) | step -2.3 to -2.9 ms everywhere, C1 +3.7 to +4.2 % |
| E8 | `EP_SIZE=1` (TP3/EP1) | C1 +7.8 to +12 %, C4 +9.7 %; NCCL 12.9 -> 5.6 ms/step |
| infra | `DSV41_AUTOTUNE_KEEP=1` (fixed port) | all three ranks `reused` both caches on E7, E9 and the final repeat boot; greedy output reproducible across boots |

## Rejected or neutral

| experiment | result |
|---|---|
| E1: `wo_a` W8 twin without DROP | does not boot at fraction 0.95: +688 MiB leaves "no GPU memory for the KV cache ... minimum viable 0.9503" |
| E1b: W8 + DROP as shipped | booted but acceptance 1.0 (C1 ~14 tok/s): dev-dsv41's draft runs its own einsum on `self.wo_a.weight`, which DROP had replaced by a zero tag. **Fixed in the adapter** (draft left on bf16 in bridge mode) |
| E1c: W8 + DROP without MID | C4 -8 %: 16-row verifies fall to per-call dequantize; MID fixes it (E2) |
| E3: k=5 without the cap | code +11 %, prose -8 to -16 %, C4 -10 % |
| E7: `DSV41_DRAFT_TAU=0.8` | sampled +1.2 % pooled over 15 runs, inside noise; not a demonstrated win, left off |
| E9: `conf:0.2` instead of 0.1 | mixed within noise, C4 -2.8 % |
| E10: k=3 without cap on EP1 | code -10.5 %, prose2 -3.3 %, prose/C4 equal: k=5 + `conf:0.1` stays |
| E11: draft `wo_a` einsum on the fp8 twin (`DSV41_WO_A_W8_DRAFT=1`, new, default off) | C1 -0.8 to -1.3 %, C4 +1.3 % on the same tactics; step -1.5 to +0.2 ms, prose acceptance dipped; left off |

## Adapter fixes made tonight

- `adapter/wo_a_w8.py`: in einsum-bridge mode (dev-dsv41) the draft's `wo_a` is never dropped (the draft
  never reaches the shared dispatch, so DROP broke it) and by default not twinned; the new
  `DSV41_WO_A_W8_DRAFT=1` (default off, forwarded by `start.sh`, volatile for autotune) routes the
  draft's own einsum to a twin instead (E11, neutral).
- `adapter/autotune_keep.py` (ported from knapcio 4d8f4c0):
  1. deletes a rank's cache when its sidecar does not match the launch. Otherwise, when no rank has a
     matching sidecar, every digest is "", the stock gate sees agreement, and FlashInfer still loads each
     rank's stale file (the cache key omits block size, graph sizes and adapter switches);
  2. excludes `SGLANG_RUN_ID` (a per-boot timestamp the launcher sets) from the fingerprint; with it,
     no sidecar could ever match and nothing was reused;
  3. volatile list extended with `DSV41_DRAFT_HEAD_FP8`, `DSV41_ENGRAM_PREFETCH(_CHECK)`.
  Tests extended for (1) and (2).
- `start.sh`: forwards `DSV41_AUTOTUNE_KEEP` and `DSV41_ENGRAM_PREFETCH_CHECK` on head and workers, and
  `SGLANG_DSPARK_FOLDED_SAMPLING` only when set (unset stays the engine's AUTO, which chose the folded
  sampler on every boot tonight).

## Exact final configuration

`.env` (installed 08:50; the previous file is `state/env.pre-overnight-20260924`). Changed keys only;
everything else is unchanged (TP_SIZE=3, NNODES=3, MEM_FRACTION_STATIC / HEAD_MEM_FRACTION_STATIC 0.95,
MAX_TOTAL_TOKENS 750000, MAX_RUNNING_REQUESTS 4, CHUNKED_PREFILL_SIZE 1024, DSV41_CACHE_GIB 0,
DSV41_MXFP8_BACKEND b12x, SGLANG_FLASHINFER_MOE_FUSED_FINALIZE 0, PYTORCH_CUDA_ALLOC_CONF
expandable_segments:False, CONTEXT_LENGTH 262144, NCCL settings as before):

```
EP_SIZE=1
DSV41_WO_A_W8=1
DSV41_WO_A_W8_MID=1
DSV41_WO_A_W8_DROP=1
DSV41_DRAFT_HEAD_FP8=1
DSV41_VERIFY_CAP=conf:0.1
DSV41_BLOCK_VERIFY=1
DSV41_ENGRAM_PREFETCH=1
DSPARK_BLOCK_SIZE=5
DSV41_AUTOTUNE_KEEP=1
```

Images: `dsv41-3x-spark:local` rebuilt from this working tree on all three nodes on the common base
`lmsysorg/sglang:dev-dsv41@sha256:3dbc3130…` (public on Docker Hub; arm64 image id 37939c26c0ba…,
2026-09-10, SGLang 0.0.0.dev0,
FlashInfer 0.6.18). Packed Engram: head `~/dsv41-engram/engram-l{1,14}-r0of3.bin`, workers
`/home/zurih/dsv41-3x-spark/engram/engram-l{1,14}-r{1,2}of3.bin` (31.5 GiB each).
Engine-side facts on the final boots: folded DSpark sampling active (AUTO), verify cap captured for
M=6/12/18/24, `packed=True` on all ranks, autotune caches `reused` on the repeat boot.

## Reproducible commands

```bash
# build (all three nodes) and serve the final configuration
./start.sh build
./start.sh                                   # = serve, reads .env
# benchmark from spark2 and join acceptance/step from the head log
RESULTS_DIR=docs/results/<dir> scripts/overnight/bench.sh <label> 5 512
# only some C1 workloads / phases
PHASES=c1 C1_WORKLOADS=chat_sampled RESULTS_DIR=… scripts/overnight/bench.sh <label> 10 512
# quality gate incl. needles (k tokens)
RESULTS_DIR=… scripts/overnight/quality.sh <label> 30,100,200
# an experiment arm = .env plus changes
RESULTS_DIR=… BASE_ENV=.env scripts/overnight/mkenv.sh <name> KEY=VAL …
ENV_FILE=$RESULTS_DIR/envs/<name>.env ./start.sh serve
# decode profile of rank 0 (send one request from a worker while it waits), analyse on spark2
scripts/profile/profile_decode.sh 45 /tmp/prof
```

## Memory tradeoffs

| item | per-rank effect |
|---|---|
| `wo_a` DROP (40 target layers) | bf16 copies released: -1280 MB; fp8 twins kept: +640 MB + scales. **Net about -640 MB** vs the original. The draft's 3 `wo_a` stay bf16 (96 MB). Twins without DROP (+688 MiB) do not fit at 0.95. |
| draft LM head fp8 twin | +210 MiB (43136 x 5120 e4m3 + scales); bf16 head kept (the target uses it) |
| Engram prefetch | side stream only; no measurable change |
| EP1 vs EP3 | same expert bytes per rank; KV calculation 929k (E8) vs 957k (E5, EP3): within boot variance |
| KV pool (effective) | original boot 491,520; B1 226,816; W8+DROP boots at the 750k pin (749,824); final boots 658,176 (F1) and 749,824 (F2). The engine sizes from the minimum free memory across ranks, which varies by ~0.5 GB between boots; below the pin the pool shrinks rather than failing. 190,886-token needle passed on E8. |
| host headroom while serving | head ~2 GB `MemAvailable`, spark2 ~7 GB, spark3 ~3-4 GB (unchanged from the original); head PSI "some" avg10 peaked at 1.33 during a 190k-token prefill |

## Rollback

- **Whole campaign:** `cp state/env.pre-overnight-20260924 .env && ./start.sh`. The rebuilt image keeps
  every adapter off unless its variable is set, so the old `.env` reproduces the old serving flags on
  converged images with packed Engram. (Deliberately not restoring the mixed-build fleet: the original
  spark3 overlay is kept as `dsv41-3x-spark:mixed-da64c5cbb-20260924` if it is ever needed.)
- **Single feature:** set it back in `.env` and `./start.sh`: `EP_SIZE=3`; `DSPARK_BLOCK_SIZE=3` with
  `DSV41_VERIFY_CAP=0`; `DSV41_WO_A_W8=0` (turns MID/DROP off too); `DSV41_DRAFT_HEAD_FP8=0`;
  `DSV41_ENGRAM_PREFETCH=0`; `DSV41_AUTOTUNE_KEEP=0` (stock discard-and-retune).
- **Stale autotune caches:** delete `~/.cache/sglang/flashinfer/autotune/0.6.18/sm121/*/rank_*.{json,launch}`
  on all three nodes; the next boot retunes.
- **Packed Engram** needs no rollback (serving falls back to the checkpoint if the files are absent).
  The old TP2 rank-1 shards are at spark1:`/home/mia/archive/dsv41-engram-tp2-spark2/` (`SHA256SUMS`).

## Remaining opportunities (ranked by the final profile)

1. **b12x dense MXFP8 GEMMs, 17.4 ms/step** (234 calls): now the largest fixed cost next to MoE.
   Shape-specific tile tuning at M=6-24, or fusing the MXFP8 activation quantize (0.7 ms) into the
   producer.
2. **MoE grouped GEMM, 21.1 ms/step** under EP1: narrower 768-wide shards; worth a tactic sweep now
   that caches persist.
3. **Draft `wo_a` on bf16, 2.4 ms/step**: tried as E11 (`DSV41_WO_A_W8_DRAFT=1`, einsum proxy);
   neutral. A profile of that switch should check the dispatch and why draft acceptance moved.
4. **NCCL all-reduce 5.6 ms/step**: the triangle-aware RoCEnante port remains the transport lever
   (deferred tonight: needs a protocol proof and standalone three-rank validation).
5. **Draft temperature 0.8**: +1.2 % sampled, not significant in 15 runs; a longer sampled run would
   settle it (it is exact and volatile, so it can be A/B'd on one tactic set).
6. **KV pool determinism**: pin at ~650k, or drop page cache before boot, so the pool is the same
   every boot (it ranged 658k-750k tonight).
7. Not attempted (scope or time): clock unlock (prohibited tonight), DRM-backed row cache (host
   change), the newer da64c5cbb/canary image as a whole-fleet upgrade, more request slots / C8,
   SPS/STS tables, folded-result fence, per-position cap thresholds, k=4 + cap.

## After the campaign: long-prompt crash and chunk size (2026-09-24 afternoon)

In normal use of the final configuration, a single new prompt of at least ~200k tokens ran spark1
out of memory during prefill (1024-token chunks; first `NV_ERR_NO_MEMORY` at 13:23, host hung until
a hard reset at 14:05). The KV cache was only ~31 % full: the per-chunk indexer transient was the
peak. The campaign's long-context check had stopped at 191k, so the range 191k-262k was never
validated on this stack, and no memory guard was running.

Change: `CHUNKED_PREFILL_SIZE=768` (default in `.env.example`). The next boot got the 750k KV pin
with a 1.24M budget and the head at ~5.2 GB `MemAvailable` (was ~2 GB). Not yet measured: the
largest prompt that is safe at 768, and whether EP1 or the larger KV cache lowered the ceiling
compared with the pre-campaign configuration (208k verified there, 256k failed). Until that is
measured, run `scripts/verify/memguard.py` on the head for long-context work.
