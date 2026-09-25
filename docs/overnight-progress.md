# Overnight TP3 campaign — progress log

Started 2026-09-24 03:11 IDT (00:11 UTC). Window ends ~11:11 IDT; validation/restore reserve from ~10:25.
Raw evidence: [results/overnight-20260924-0311/](results/overnight-20260924-0311/).
Benchmark: `RESULTS_DIR=… scripts/overnight/bench.sh LABEL 5 512` — runs
[benchmarks/overnight_bench.py](../benchmarks/overnight_bench.py) from spark2 against the head,
5 reps per workload after a discarded warm-up, 512 max tokens, thinking off. Tokens from
`usage.completion_tokens`; decode tok/s = (tokens-1)/(last delta - first delta). Acceptance and
step time joined from the head's `Decode batch` lines (step = accept_len x bs / gen throughput).

Constant background during every measurement: sparkDash on spark3 polls the head API (5 GETs
every 2 s, 0.06 % CPU). Pausing it was not permitted by the session's safety policy, so it stays on.

## 0. Starting state (03:11-03:25)

- Configs, container env/mounts for all three ranks, adapter hashes, git diff: saved in the results dir.
- Running: TP3/EP3, DSpark k=3, 4 slots, chunk 1024, KV pin 750k (effective pool 491,520), fraction
  0.95, b12x MXFP8, cache 0 GiB, adapters off, **packed=False** on all ranks (engram dirs empty).
- **Mixed SGLang builds across ranks.** head and spark2 run base `lmsysorg/sglang:dev-dsv41` =
  `37939c26` (2026-09-10; public on Docker Hub as `lmsysorg/sglang@sha256:3dbc3130…`, whose amd64 member is
  the 0xSero pin; its internal labels say `local/sglang:dev`, version 0.0.0.dev0); spark3's tag of the same name points at
  `381b27ff` (2026-09-11, `da64c5cbb`). 40 SGLang source files differ, among them
  `models/deepseek_v4.py`, `deepseek_v4_dspark.py`, `dspark_worker_v2.py`, MoE layer/topk, fp8
  quantization and the memory pool; spark3 also has 9 extra modules (e.g. `wo_a_bf16_small_batch.py`,
  `all_reduce_fusion.py`). Rank 2 therefore runs a different model implementation than ranks 0/1.
  Likely contributors: per-rank autotune-cache disagreement, the `wo_a_w8` adapter selecting different
  paths per rank (it keys on the presence of the newer kernel module), and greedy nondeterminism.
- Disk: head 378 GB free, spark2 61 GB free (needs 67.6 GB for its TP3 shards), spark3 392 GB free.
  spark2 holds 95 GB of old TP2 shards at `/home/zurih/dsv41-engram` (outside the worker mount).

### b0 — original configuration (mixed builds, unpacked)

| workload | decode tok/s median (min-max) | TTFT | accept len | step ms |
|---|---|---|---|---|
| C1 prose | 29.28 (27.68-29.51) | 0.30 s | 2.17 | 73.2 |
| C1 code | 44.65 (42.39-44.79) | 0.30 s | 3.27 | 74.0 |
| C1 sampled chat | 28.87 (26.65-29.72) | 0.28 s | 2.23 | 77.8 |
| C4 aggregate (2 prose + 2 code) | 63.78 (60.18-67.14), per-stream 19.46 | | 2.50 | 133.8 (bs 4) |

Greedy prose produced 4 distinct outputs out of 5, greedy code 3 of 5: not deterministic at T=0.

## 1. Packed Engram restored, rank images converged (03:25-03:48)

- Fleet drained at 03:24 (last foreign chat request 02:29).
- **spark2 disk:** old TP2 shards `engram-l1-r1of2.bin` + `engram-l14-r1of2.bin` (101,379,729,008 bytes)
  copied to **spark1:`/home/mia/archive/dsv41-engram-tp2-spark2/`**, SHA-256 identical on both sides
  (`SHA256SUMS` stored there), then removed from spark2 (`/home/zurih/dsv41-engram/MOVED.txt` points to
  the new location). spark2 free space 61 -> 156 GB.
- **Packing:** `pack_engram.py` for ranks 0/1/2 into the real mounts (head `~/dsv41-engram`, workers
  `/home/zurih/dsv41-3x-spark/engram`); 92-111 s per 31.5 GiB layer. Free afterwards: head ~310 GB,
  spark2 93 GB, spark3 329 GB.
- **Image convergence:** spark3's images were preserved as `lmsysorg/sglang:dev-dsv41-da64c5cbb-20260911`
  and `dsv41-3x-spark:mixed-da64c5cbb-20260924`; the head's `lmsysorg/sglang:dev-dsv41` (37939c26) was
  `docker save | docker load`ed onto spark3 and the overlay rebuilt on all three nodes (identical
  `deepseek_v4.py` hash everywhere).
- **Verified after restart:** `packed=True` for layers 1 and 14 on ranks 0, 1 and 2 (six checks).
- The autotune "per-rank caches disagree" message still appears twice per boot with identical images:
  it is the EP-structural problem, not the version mismatch.
- KV pool on this boot: 226,816 tokens (was 491,520). The head had 1.35 GB less free after the draft
  load than the previous boot (the boot started with ~86 GB of page cache from archiving/packing).
  Needles at 96k still pass; tracked per boot below.

### B1 — converged + packed (new baseline)

| workload | decode tok/s median (min-max) | TTFT | accept | step ms |
|---|---|---|---|---|
| C1 prose | 29.40 (29.19-29.48) | 0.29 s | 2.12 | 72.1 |
| C1 code | 45.68 (44.12-46.01) | 0.29 s | 3.45 | 72.9 |
| C1 sampled chat | 28.73 (28.19-30.99) | 0.26 s | 2.16 | 73.1 |
| C4 aggregate | 66.88 (66.23-70.15), per-stream 20.69 | | 2.46 | 131.5 |

Quality gate: 8/8 tasks, 8/8 under 4-way concurrency (all identical to C1 outputs), greedy repeat
(256 tok) 1 distinct of 3, needle at 28.7k and 95.5k prompt tokens found (17 s / 68 s).
Greedy code 1 distinct of 5 (was 3 in b0); greedy 512-token prose still 5 distinct of 5.

Packing gave +2 % code and +5 % C4, prose unchanged: on this boot Engram is no longer on the
critical path (profile below), but the step was already ~72 ms before.

## 2. Profile of B1 (rank 0, 44 decode steps, C1 prose)

Step wall 71.3 ms median, GPU busy 67.6 ms. Per step:

| kernel | ms/step | calls/step |
|---|---|---|
| MoE grouped GEMM (cutlass GroupProblemShape, MXFP4) | 23.8 | 88 |
| b12x dense block-scaled MXFP8 GEMM | 17.0 | 234 |
| NCCL all-reduce (LL ring) | 12.9 | 92 |
| **bf16 wmma GEMM = `wo_a` einsum** (1/layer, 43+3) | **11.5** | 46 |
| mHC, sparse MLA, router, quant, norms, rest | ~6 | |
| Engram-following NCCL wait | 0.7 (5 % of NCCL) | |

`wo_a` is the largest single-op target, as the plan predicted; Engram waiting is gone.

## 3. Autotune-cache retention ported (adapter/autotune_keep.py)

Copied from knapcio 4d8f4c0 with its test, hooked in `sitecustomize.py`, `DSV41_AUTOTUNE_KEEP`
forwarded on head and worker launches (also `DSV41_ENGRAM_PREFETCH_CHECK`, and
`SGLANG_DSPARK_FOLDED_SAMPLING` only when set, so unset stays the engine's AUTO).
**Fix added:** the engine's cache key omits block size, graph sizes and adapter switches, and when no
rank has a matching sidecar every digest is "" -> the stock gate sees agreement and FlashInfer loads
each rank's stale file. The port now deletes a rank's cache when its sidecar does not match the
launch, so a boot either reuses matched caches on all ranks or retunes from scratch. Test extended.
Enabled from E1 on (its no-sidecar path is equivalent to the stock discard); reuse is checked on a
matched repeat boot.

## 4. `wo_a` FP8 twin (adapter/wo_a_w8.py, einsum bridge on dev-dsv41)

**E1 — W8=1, MID=0, DROP=0: failed at boot (04:13).** The 40 target twins (+3 draft) add ~688 MiB
and the head was already at the margin: `ValueError: Loaded weights leave no GPU memory for the KV
cache under --mem-fraction-static=0.95 ... minimum viable = 0.9503`. Twin-only cannot run at 0.95;
raising the fraction is what took spark3 down at 01:32, so not retried. Log: `boot-e1-fail-head.log`.

**E1b — W8=1, DROP=1 (04:27): booted, but acceptance collapsed to 1.0.** C1 prose/code/chat 14.3/14.1/14.3
tok/s, C4 26.6. The bf16 copies were released (40 target = 1280 MB, 3 draft = 96 MB per rank) and
the KV pool rose to the 750k pin (749,824; memory calculation 841,472 before the pin). Target output
stayed coherent. Cause (source): dev-dsv41's draft attention (`deepseek_v4_dspark.py:354`) runs its
own `torch.einsum` on `self.wo_a.weight` and never reaches `_apply_wo_a_bf16_matmul`, so DROP handed
it the 1-element zero tag: every draft was garbage. The draft twins are also unused on this image.
**Fix:** in bridge mode the adapter now leaves the draft's `wo_a` alone (no twin, no drop).
A profile of E1b validated the target dispatch: `_wo_a_partial_w8` runs 40x/step for 3.47 ms total,
replacing ~10 ms of bf16 wmma einsum on those layers (about -6.5 ms/step), and the draft's three
einsums on zero-stride tags took 3.78 ms.

### E1c — W8=1, DROP=1 on target layers only, MID=0 (05:04)

KV pool 749,824 (pin; calculation 792,576). Head memory-stall time during the whole benchmark 2.8 ms.

| workload | decode tok/s (min-max) | vs B1 | accept | step ms |
|---|---|---|---|---|
| C1 prose | 28.97 (28.82-29.02) | -1.5 % | 1.98 | 67.9 |
| C1 code | 49.58 (49.39-49.85) | **+8.5 %** | 3.45 | 69.3 |
| C1 sampled chat | 31.34 (30.13-32.21) | **+9.1 %** | 2.20 | 68.9 |
| C4 aggregate | 61.34 (59.96-65.40), per-stream 18.85 | -8.3 % | 2.40 | 146.7 |

The bs=1 step fell ~4 ms (72 -> 68-69 ms). Prose tok/s did not follow because its greedy text changed
(the twin's fp32 accumulation order differs from einsum), and this prompt's acceptance moved
2.12 -> 1.98; prose outputs were already non-repeatable in B1. C4 regressed as predicted: at k=3 a
4-request verify is 16 rows, which without MID falls to the per-call dequantize + einsum path.

### E2 — W8 + DROP (target) + MID (05:24) — **promoted**

KV pool 749,824 (pin). Autotune: "tuned and saved" on each boot (no sidecar yet, as designed).

| workload | decode tok/s (min-max) | vs B1 | accept | step ms |
|---|---|---|---|---|
| C1 prose | 28.86 (28.40-29.05) | -1.8 % | 1.98 | 68.3 |
| C1 prose2 (added here) | 29.56 (29.50-29.94) | | 2.00 | 66.5 |
| C1 code | 48.04 (47.94-48.11) | +5.2 % | 3.29 | 68.9 |
| C1 sampled chat | 31.96 (30.76-32.53) | +11.2 % | 2.20 | 68.8 |
| C4 aggregate | 69.64 (68.88-72.72), per-stream 21.72 | +4.1 % | 2.44 | 125.6 |

Quality gate: 8/8, 8/8 concurrent (7 of 8 byte-identical to C1; one differs under batching), greedy
repeat 1 distinct of 3, 28.7k needle found. MID restored C4 (146.7 -> 125.6 ms step; below B1's 131.5).

**Measurement note.** Greedy text is repeatable within a boot for code but differs between boots
(B1, E1c and E2 each produced a different code text, see `text_sha`): FlashInfer re-draws its tactics
every boot, so the numerics and hence the text move, and acceptance moves with the text
(code 3.45 in E1c vs 3.29 in E2 with identical kernels at bs=1). Cross-boot C1 tok/s therefore carries
a few % of text noise; bs=1 step time (72.1 -> 68.3-68.9 ms) is the cleaner signal. Autotune
retention should reduce this on repeated boots of one configuration.

## 5. DSpark block size

### E3 — E2 + `DSPARK_BLOCK_SIZE=5` (05:47)

| workload | decode tok/s (min-max) | vs E2 (k=3) | accept | step ms |
|---|---|---|---|---|
| C1 prose | 26.65 (26.58-26.87) | -7.7 % | 2.12 | 78.6 |
| C1 prose2 | 24.97 (24.64-25.07) | -15.5 % | 1.92 | 77.4 |
| C1 code | 53.40 (53.28-53.47) | **+11.2 %** | 4.19 | 78.8 |
| C1 sampled chat | 29.22 (28.11-30.97) | -8.6 % | 2.35 | 78.9 |
| C4 aggregate | 62.45 (60.92-62.89), per-stream 21.28 | -10.3 % | 2.61 | 123.1 |

k=5 adds ~10 ms to the bs=1 step (68 -> 78.6 ms: 6 verify rows instead of 4, 5 draft rows instead
of 3) and prose acceptance barely moves (1.98-2.00 -> 1.92-2.12), so only code gains. Not promoted
alone; the confidence cap (E4) is the mechanism meant to recover prose at k=5.

## 6. Confidence cap

### E4 — E3 + `DSV41_VERIFY_CAP=conf:0.1` + `DSV41_BLOCK_VERIFY=1` (06:09) — **promoted**

Block verification is enabled as the cap's dependency for sampled rows (it honours the live length;
without it the chain sampler would ignore the cut). Log: cap captured into the verify graphs for
M=6/12/18/24, `confidence present`.

| workload | decode tok/s (min-max) | vs E2 (k=3) | vs E3 (k=5) | accept | step ms |
|---|---|---|---|---|---|
| C1 prose | 30.20 (30.20-30.30) | +4.6 % | +13.3 % | 2.00 | 68.4 |
| C1 prose2 | 28.10 (27.98-28.46) | -4.9 % | +12.5 % | 1.85 | 66.0 |
| C1 code | 52.67 (52.27-52.82) | **+9.6 %** | -1.4 % | 4.15 | 78.8 |
| C1 sampled chat | 31.77 (29.70-32.89) | -0.6 % | +8.7 % | 2.21 | 69.8 |
| C4 aggregate | 68.90 (67.26-70.39), per-stream 22.82 | -1.1 % | +10.3 % | 2.31 | 112.1 |

Quality gate: 8/8, 8/8 concurrent (all 8 byte-identical to C1), greedy repeat 1 distinct of 3,
28.7k needle found. The cap brings prose steps back to the k=3 cost (66-68 ms) while code keeps the
k=5 acceptance (4.15). Prose mean over both prompts 29.15 vs 29.21 (E2): neutral; code +9.6 %;
sampled and C4 neutral. Greedy 512-token code varied within the boot (4 distinct of 5; E3 without
the cap: 2 of 5), prose already varied at k=3.

Autotune: `DSV41_DRAFT_HEAD_FP8`, `DSV41_ENGRAM_PREFETCH(_CHECK)` added to the adapter's volatile list
(they touch no FlashInfer-tuned op), so from E5 on those A/B arms can share tuned tactics.

## 7. Draft LM head FP8

### E5 — E4 + `DSV41_DRAFT_HEAD_FP8=1` (06:33) — **promoted**

Twin of the draft LM head (43136 x 5120 per rank) built at attach; the KV calculation fell
1,072,384 -> 956,928 tokens (~192 MB, matching the ~210 MiB estimate) and the pool stays at the 750k pin.

| workload | decode tok/s (min-max) | vs E4 | accept | step ms (E4) |
|---|---|---|---|---|
| C1 prose | 30.50 (30.24-30.79) | +1.0 % | 2.10 | 68.3 (68.4) |
| C1 prose2 | 27.93 (27.91-28.33) | -0.6 % | 1.82 | 65.0 (66.0) |
| C1 code | 54.33 (54.16-54.62) | +3.2 % | 4.30 | 77.5 (78.8) |
| C1 sampled chat | 32.98 (32.26-34.61) | +3.8 % | 2.34 | 69.1 (69.8) |
| C4 aggregate | 69.47 (68.33-69.57), per-stream 23.29 | +0.8 % | 2.37 | 113.3 (112.1) |

Step time fell on every C1 workload (0.1-1.3 ms), the size expected; target tokens are unaffected by
construction (only the draft's logits use the twin).

## 8. Engram prefetch and an autotune-retention bug

### E6a — E5 + `DSV41_ENGRAM_PREFETCH=1` + `DSV41_ENGRAM_PREFETCH_CHECK=1` (06:54): correctness

After the quality-gate traffic (8/8, 8/8 concurrent and byte-identical, needle 28.7k found) the check
counters read **`[0, 0]` differing gathers on ranks 0, 1 and 2** over ~91k lookups per layer
(prefill incl. the needle, decode, 4-way concurrency). No fallback/error messages. Not timed
(check mode repeats the lookup).

**Autotune retention never reused a cache on this image.** E6a differed from E5 only in volatile
switches, yet every rank logged "tuned and saved". The live scheduler's environment carries
`SGLANG_RUN_ID=sglang-run-<timestamp>-<rand>`, set fresh by the launcher on every boot and matched by
the fingerprint's `SGLANG_` prefix, so no sidecar could ever match. Fixed (excluded, test added);
before the fix the adapter behaved exactly like the stock discard path, so E1-E6a are unaffected.

### E6b — E5 + `DSV41_ENGRAM_PREFETCH=1`, check off (07:09) — **promoted**

| workload | decode tok/s (min-max) | vs E5 | accept | step ms (E5) |
|---|---|---|---|---|
| C1 prose | 31.73 (31.51-31.76) | **+4.0 %** | 2.10 | 65.5 (68.3) |
| C1 prose2 | 29.11 (28.88-29.66) | **+4.2 %** | 1.82 | 62.7 (65.0) |
| C1 code | 56.33 (56.06-56.48) | **+3.7 %** | 4.32 | 75.1 (77.5) |
| C1 sampled chat | 32.50 (31.21-35.01); 10 reps: 32.94 (31.81-34.34) | -1.5 % (noise, sd 1.3) | 2.12-2.17 | 66.2 (69.1) |
| C4 aggregate | 69.27 (68.03-70.08), per-stream 23.46 | -0.3 % | 2.31 | |

The bs=1 step fell 2.3-2.9 ms on every workload at unchanged acceptance: more than the 0.7 ms
rank-0 profile estimate, because the other ranks' Engram waits surface inside rank 0's NCCL time.
The folded draft sampler is active (AUTO chose it: "draft proposal (greedy + sampling) folded into
the draft cuda graph"), which the draft-temperature experiment needs.

## 9. Sampled decoding: draft temperature

### E7 — E6b + `DSV41_DRAFT_TAU=0.8` (07:33) — neutral, not promoted

First boot to **reuse** the autotune caches: all three ranks logged `reused` for both caches and no
"per-rank caches disagree" message (E6b's sidecars, same launch fingerprint; `DRAFT_TAU` is volatile).
With the same tactics, the greedy code text was byte-identical to E6b's (`787b5ff22342656b`) and two of
four prose texts matched: cross-boot greedy output is now reproducible for the first time tonight.
KV pool 697,600 on this boot (calculation below the 750k pin; boot-to-boot head free-memory variance).

| workload | E7 tok/s | E6b tok/s | accept E7 / E6b |
|---|---|---|---|
| C1 sampled chat, 10 reps | 33.02 (31.59-33.83) | 32.94 (31.81-34.34) | 2.15 / 2.17 |
| C1 sampled chat, 5 reps | 33.61 (32.85-36.49) | 32.50 (31.21-35.01) | 2.25 / 2.12 |
| C1 prose / prose2 / code | 31.63 / 29.11 / 56.41 | 31.73 / 29.11 / 56.33 | unchanged |
| C4 aggregate | 70.11 | 69.27 | 2.34 / 2.31 |

Pooled over 15 sampled runs: +1.2 % tok/s, +1.4 % acceptance, the size the offline estimate
predicts but inside run-to-run noise (sd ~0.9-1.3 tok/s). Exact by construction and greedy is
untouched, but it is not a demonstrated win, so it is left out of the final configuration.

## 10. TP3/EP1

### E8 — E6b + `EP_SIZE=1` (07:53) — **promoted (final candidate)**

Boots cleanly (MXFP4 experts tensor-sharded to 768-wide intermediates; new autotune cache keys).
KV calculation 929,024 -> pool at the 750k pin.

| workload | decode tok/s (min-max) | vs E6b (EP3) | accept | step ms (E6b) |
|---|---|---|---|---|
| C1 prose | 34.20 (33.73-34.43) | **+7.8 %** | 2.04 | 58.8 (65.5) |
| C1 prose2 | 32.59 (31.92-32.69) | **+12.0 %** | 1.83 | 57.0 (62.7) |
| C1 code | 62.53 (61.68-62.83) | **+11.0 %** | 3.98 | 68.8 (75.1) |
| C1 sampled chat | 35.82 (35.70-38.41) | **+10.2 %** | 2.21 | 60.3 (66.2) |
| C4 aggregate | 75.96 (74.59-76.86), per-stream 24.44 | **+9.7 %** | 2.36 | |

Quality gate: 8/8, 8/8 concurrent (7 byte-identical), greedy repeat 1 distinct of 3, needle 28.7k.
Long context: needles at **95,541 and 190,886 prompt tokens found** (58.5 s / 136.9 s); prose, code and
sampled requests run concurrently alongside completed normally. No NaN/inf/OOM/traceback lines.
Head PSI during the 190k prefill: some avg10 1.33.

Profile (rank 0, 42 steps, prose): step 56.1 ms median. **NCCL all-reduce 12.9 -> 5.6 ms/step** (the
per-layer straggler wait collapsed: every rank now streams a third of every active expert instead of
whole experts on whichever rank owns them); MoE grouped GEMM 23.8 -> 21.1 ms; b12x dense MXFP8 17.4 ms
(now the largest fixed cost with MoE); `wo_a` twin 3.5 ms; draft `wo_a` einsum 2.4 ms; draft head fp8 1.0 ms.

### E9 — E8 with `DSV41_VERIFY_CAP=conf:0.2` (08:21) — neutral, not promoted

Reused E8's EP1 caches on all ranks (threshold is volatile), so only the cap differs.

| workload | E9 conf:0.2 | E8 conf:0.1 | accept E9 / E8 |
|---|---|---|---|
| C1 prose | 33.93 | 34.20 (-0.8 %) | 1.94 / 2.04 |
| C1 prose2 | 32.66 | 32.59 (+0.2 %) | 1.80 / 1.83 |
| C1 code | 63.30 | 62.53 (+1.2 %) | 4.12 / 3.98 |
| C1 sampled | 37.31 | 35.82 (+4.2 %, sd ~1) | 2.17 / 2.21 |
| C4 aggregate | 73.83 | 75.96 (-2.8 %) | 2.33 / 2.36 |

Mixed, within noise except C4; `conf:0.1` kept.

### E10 — E8 with k=3 and no cap (08:37) — rejected

| workload | E10 (EP1 k=3) | E8 (EP1 k=5 conf:0.1) |
|---|---|---|
| C1 prose | 34.23 | 34.20 |
| C1 prose2 | 31.52 (-3.3 %) | 32.59 |
| C1 code | 55.99 (-10.5 %) | 62.53 |
| C1 sampled | 36.98 (+3.2 %, noise) | 35.82 |
| C4 aggregate | 76.20 | 75.96 |

k=5 + cap keeps its code advantage on EP1 and is otherwise equal. **Final configuration = E8.**
Note: E10 retuned into the same EP1 cache directory (the engine's cache key omits block size), so
E8's tactics were overwritten; the final boot retunes once, and a repeat boot checks reuse.

## 11. Final configuration installed as `.env` (08:50)

Previous `.env` saved to `state/env.pre-overnight-20260924`. Effective settings identical to
`envs/e8-ep1.env`: EP_SIZE=1, DSV41_WO_A_W8/MID/DROP=1, DSV41_DRAFT_HEAD_FP8=1,
DSV41_VERIFY_CAP=conf:0.1, DSV41_BLOCK_VERIFY=1, DSV41_ENGRAM_PREFETCH=1, DSPARK_BLOCK_SIZE=5,
DSV41_AUTOTUNE_KEEP=1; everything else as before (fraction 0.95, pin 750k, 4 slots, chunk 1024).

### F1 — final configuration from `.env`, fresh tuning (08:54)

KV pool 658,176 (calculation below the pin this boot). Autotune "tuned and saved" (E10 had overwritten
the EP1 caches).

| workload | F1 | E8 (same config, other tactics) | accept | step ms |
|---|---|---|---|---|
| C1 prose | 34.29 (33.73-34.52) | 34.20 | 2.05 | 58.7 |
| C1 prose2 | 32.27 (31.85-32.46) | 32.59 | 1.88 | 58.1 |
| C1 code | 62.69 (62.00-62.76) | 62.53 | 4.00 | 68.5 |
| C1 sampled chat | 36.57 (35.78-39.02) | 35.82 | 2.25 | 60.7 |
| C4 aggregate | 76.03 (74.99-76.22), per-stream 24.52 | 75.96 | 2.40 | |

Reproduces E8 within 1-2 % on an independent tuning pass. Quality gate: 8/8, 8/8 concurrent (7
byte-identical), greedy repeat 1 distinct of 3, needles at 28.7k and 95.5k found (15 s / 56 s).
Host memory while serving: spark1 2 GB available (PSI some avg10 0.26), spark2 7 GB, spark3 3 GB.

### F2 — identical repeat boot (09:12) — left serving

All three ranks `reused` both autotune caches (no "disagree"); `packed=True` for both layers on all
three ranks; KV pool 749,824 (pin).

| workload | F2 | F1 | accept | step ms |
|---|---|---|---|---|
| C1 prose | 34.15 (33.86-34.52) | 34.29 | 2.04 | 59.0 |
| C1 prose2 | 32.56 (32.00-32.66) | 32.27 | 1.85 | 57.3 |
| C1 code | 62.24 (61.47-62.85) | 62.69 | 3.98 | 69.7 |
| C1 sampled chat | 37.20 (36.25-37.76) | 36.57 | 2.20 | 60.2 |
| C4 aggregate | 75.79 (73.36-76.60), per-stream 24.31 | 76.03 | 2.34 | |

Quality gate 8/8, 8/8 concurrent, greedy repeat 1 distinct of 3, needle 28.7k found. Greedy code
text identical to F1's. Final report: [overnight-results.md](overnight-results.md).

## 12. Extra experiment in the remaining time: draft `wo_a` on the fp8 twin

### E11 — final + `DSV41_WO_A_W8_DRAFT=1` (09:32) — neutral, rejected

New default-off switch in `adapter/wo_a_w8.py`: in bridge mode the draft's 3 `wo_a` get fp8 twins
(bf16 kept, +48 MB) and the draft module's global `torch` is replaced by a proxy whose `einsum`
routes `"bgd,grd->bgr"` with a twinned weight to the twin kernels (2-8 rows small, 9-192 MID); all
other calls pass through. Volatile for autotune: E11 reused F2's tactics on all ranks.

| workload | E11 | F2 | accept E11 / F2 | step ms E11 / F2 |
|---|---|---|---|---|
| C1 prose | 33.86 | 34.15 (-0.8 %) | 1.93 / 2.04 | 58.4 / 59.0 |
| C1 prose2 | 32.25 | 32.56 (-1.0 %) | 1.73 / 1.85 | 57.5 / 57.3 |
| C1 code | 61.67 | 62.24 (-0.9 %) | 4.12 / 3.98 | 68.2 / 69.7 |
| C1 sampled | 36.73 | 37.20 (-1.3 %) | 2.23 / 2.20 | 60.3 / 60.2 |
| C4 aggregate | 76.80 | 75.79 (+1.3 %) | 2.49 / 2.34 | |

Quality 8/8, 8/8 concurrent, needle found. The step moved -1.5 to +0.2 ms (less than the 2.4 ms the
draft einsum costs) and prose acceptance dipped; net neutral-to-negative on the priority workloads.
Switch left off (default); code kept for a later profile-guided look.
Fleet restored to the final `.env` configuration (F3) at ~09:47.
