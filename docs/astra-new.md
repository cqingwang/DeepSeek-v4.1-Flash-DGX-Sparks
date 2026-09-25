# TP3 decode throughput: what to borrow and what to fix first

Reviewed 2026-09-24. Target: three GB10 DGX Sparks, TP3/EP3, native DeepSeek-V4.1-Flash weights, NVMe Engram, DSpark. The objective is faster generated tokens per second, with single-stream speed and aggregate throughput reported separately.

**Recommendation: restore the packed Engram path first, then test DSpark block size, finish the TP3 `wo_a` port, and evaluate Engram prefetch.** A TP3 RoCEnante port and TP3/EP1 are worthwhile subsequent experiments. Copying the TP4 production environment wholesale would introduce incompatible shapes, topology assumptions, and memory budgets.

This is a source and existing-evidence review. No servers were restarted, no serving settings were changed, and no new GPU throughput measurements were performed. Gains quoted below are either the author's TP4 measurements, historical local TP3 measurements, or explicitly labelled estimates.

The inspected downstream checkout is [knapcio at `4d8f4c0868ed3b394688b65d88cd99a82c237642`](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4/tree/4d8f4c0868ed3b394688b65d88cd99a82c237642), committed September 23. Local HEAD is `25379c22e8ff177c7587b21b09eac33a92efeb54`, with existing uncommitted adapter and launcher changes included in this review. The GitHub-rendered README initially returned older September 18 content; the cloned source and September 23 changelog are the basis for the newer findings.

**The immediate local finding: packed Engram is missing on the head.**

[The latest saved engine log](../logs/dsv41.log) records `packed=False` for layers 1 and 14 at startup and during requests. For example, the September 23 23:33:29 log entries show 80 lookups and 160 reads for each layer. The configured head directory, `/home/mia/dsv41-engram`, is empty at review time. `.env` points that directory at `/engram` and sets `DSV41_PACKED_DIR=/engram`.

[The adapter](../adapter/engram_backend.py) silently falls back to the original checkpoint when the expected packed file is absent. That means two reads per miss; on a worker whose checkpoint is NFS-mounted it also means network storage access. I verified the head directory and saved head log, not the current worker directories or live container mounts. The worker state must be checked before attributing a current throughput loss to NFS.

This matters more than a speculative new tuning flag. In [the historical TP3 profile, sections 6–10](../logs/profile-2026-09-10/REPORT.md), rank 0 waited roughly 11 ms for workers' Engram reads; after packed worker shards were introduced, the Engram-related all-reduce wait fell from roughly 11 ms to 0.5 ms. Other changes were made in that boot, so its overall speedup is not an isolated packing result.

Use the existing `./start.sh pack` workflow during a maintenance window if the files are absent, then verify all six rank/layer files and container mounts. Expected names are `engram-l1-r0of3.bin`, `engram-l14-r0of3.bin`, and the corresponding rank 1 and rank 2 files on their own hosts. A successful serving boot must log `packed=True` for both layers on every rank. TP4 `of4` files cannot substitute for TP3 shards. Packing consumes substantial disk space and I/O; it should not run alongside a throughput benchmark.

**What is already present locally.**

The configured baseline is TP3/EP3, `DSPARK_BLOCK_SIZE=3`, four running requests, a 750,000-token KV pin, 1,024-token prefill chunks, memory fraction 0.95, no Engram row cache, native allocator, b12x MXFP8 routing, and fused MoE finalize disabled. The saved launch log confirms block 3 and CUDA graph batches 1–4. These settings and the [README's historical 37.9 tok/s prose C1 result](../README.md) are not a fresh benchmark of today's boot.

Six local adapters are byte-identical to the inspected downstream revision: `engram_prefetch.py`, `draft_head_fp8.py`, `draft_tau.py`, `block_verify.py`, `verify_cap.py`, and `folded_result_fence.py`. The local `wo_a_w8.py` already contains additional TP3 changes. All performance experiments are currently disabled, with draft temperature multiplier at 1. Copying these six files again would add nothing.

[The launcher](../start.sh) and [.env.example](../.env.example) explicitly record a TP3 autotune/startup failure involving padded rank 2, `wo_a` storage replacement, prefetch, and verify remapping. This establishes a failed combination, not which individual adapter caused it. Test one feature per boot; do not enable the entire downstream set together.

| Candidate | Evidence in the downstream project | TP3 verdict |
|---|---|---|
| Packed node-local Engram | Already part of the original Mia recipe; historical TP3 worker wait reduction | **First action:** restore/verify the existing path |
| DSpark block-size sweep | TP4 code and prose prefer 5 in its later workload probes | **Low implementation effort:** benchmark 2/3/4/5 on TP3; retain 3 until measured |
| `wo_a` FP8 weight reads | TP4 C1 step 52.9 → 51.7 ms; later medium-row path helps C4 | **High priority engineering:** local code exists, but dispatch and startup require TP3 validation |
| Engram side-stream prefetch | Same-image TP4 step 51.69 → 49.49 ms | **High priority controlled trial:** local file is identical; first restore packing |
| Draft-only FP8 LM head | TP4 step 51.7 → 50.9 ms | **Second tier:** extra TP3 memory and acceptance checks |
| Confidence-based verify cap | TP4 prose C1 66.0 → 69.7; C4 125.4 → 133.5 tok/s | **Second tier:** preserve fixed layout, test rank synchronization and TP3 shape handling |
| Draft temperature / block verification | Small increases in accepted tokens per step on sampled workloads | **Second tier:** useful for sampled chat, not a greedy-speed claim |
| Pinned canary SGLang kernels | TP4 base/canary benchmarks improve | **Selective port:** some important kernels explicitly require TP4 shapes |
| RoCEnante small all-reduces | Measured improvement on switched TP4 | **Potentially valuable port:** TP3 integration and triangle routing are unsupported as shipped |
| Engram cache | TP4 uses 4 GiB per host | **Only small-budget experiments:** TP3 cannot inherit that memory allocation |
| Shared-expert K padding | Fixes TP4 K=576 kernel fallback | **Do not enable:** TP3 K=768 already meets 128 alignment |
| EP2 | Important TP4 improvement | **Invalid TP3 group factorization:** investigate EP1 separately |

Performance evidence: [September 23 changelog](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4/blob/4d8f4c0868ed3b394688b65d88cd99a82c237642/CHANGELOG.md), [Engram A/B profile](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4/blob/4d8f4c0868ed3b394688b65d88cd99a82c237642/docs/upstream-watch.md), and [block-size experiments](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4/blob/4d8f4c0868ed3b394688b65d88cd99a82c237642/docs/session-20260919.md). These are workload-specific TP4 observations, not predicted TP3 percentages.

**1. Finish the TP3 `wo_a` path rather than just switching images.**

The historical TP3 trace attributes approximately 11 ms of an 82 ms block-5 step to the bf16 `wo_a` einsum. This makes it a stronger local target than the many sub-millisecond flags. That timing is historical and must be reprofiled at today's block size.

TP3 pads 8 output groups to 12, leaving **four local groups**. The downstream adapter hardcodes two groups in its kernels. The [local adapter](../adapter/wo_a_w8.py) generalizes the group dimension and adds a bridge for the older image's einsum dispatch; preserve those changes.

There is a second trap: in [the pinned canary model code](https://github.com/sgl-project/sglang/blob/f80c91a4b/python/sglang/srt/models/deepseek_v4.py), `_apply_wo_a_bf16_matmul` enters its optimized path only for `o.shape[1:] == (2, 4096)` and weight shape `(2, 1024, 4096)`. TP3's group count is four. Merely replacing the imported small-batch kernels does not bypass this guard. In the local adapter, finding the canary kernel module also selects `_patch_kernels` instead of the older-image einsum bridge. Thus a canary upgrade plus `DSV41_WO_A_W8=1` alone can leave TP3 small-row calls on einsum.

Implementation target: explicit G=4 dispatch for verify/draft rows, retaining a correct fallback for prefill and unsupported shapes. Validate representative M=3/4/5/6 and the C2/C4 shapes, strided activations, padded rank 2, and both target and draft models. Confirm actual kernel execution in a trace. The current CPU test checks that a TP3 weight view has the right shape; that does not establish GPU output or graph-capture correctness.

Start with `DSV41_WO_A_W8=1`, keeping `MID=0`, `DROP=0`, prefetch and verify remapping off. Next test `MID=1`. Consider `DROP=1` only after warmup/autotune and all forwards are covered, since it substitutes a one-element backing allocation for the old weight storage.

Memory arithmetic, assuming 43 eligible G=4 layers: bf16 weights occupy 1,376 MiB; FP8 twins approximately 688 MiB plus scales. Keeping both copies **adds** roughly 688 MiB; replacing bf16 with the twins eventually **saves net** roughly 688 MiB. The gross amount of bf16 storage released is not the net saving. Temporary FP32 conversion allocations also need headroom. Actual eligible-layer counts and memory must be logged.

The exact twin check preserves weight values, but the einsum replacement may change reduction order. Do not extend the downstream TP4 fused-epilogue bitwise claim to the local TP3 bridge without testing. Require finite logits, stable greedy outputs across repeated same-stack runs, numerical comparisons, and task quality checks.

**2. Tune speculative decoding for the actual TP3 workload.**

For one stream, the useful objective is:

`decode tok/s ≈ committed tokens per speculative step / step wall time`

Increasing block size helps only if the extra committed tokens compensate for a more expensive draft and target verification. On TP3 the larger expert shards make that tradeoff different from TP4. Sweep block sizes 2, 3, 4, and 5 separately for prose, code, structured output, and sampled chat, at C1/C2/C4. Retest the winning sizes after improving `wo_a` or communication because their cost balance changes. Do not spend time on 7 until 5 demonstrates an advantage.

The downstream September 19 probes show a substantial gap between prose and code acceptance. That is evidence for measuring workloads separately, not evidence that its chosen block 5 must win here. Keep thinking mode, temperature, top-p, and prompt corpus fixed within each comparison. Disabling thinking can shorten total response time but is not a decode-throughput optimization.

For sampled traffic, the existing local `draft_tau.py` and `block_verify.py` are plausible incremental improvements. Try draft multiplier 0.8 against 1.0, then block verification in isolation. They alter the proposal or verification algorithm; correctness requires acceptance to use the actual proposal distribution. Validate the complete engine path, including mixed greedy/sampled batches, top-p/top-k, and TP rank agreement. Toy-distribution tests are useful but do not certify that integration.

`DSV41_VERIFY_CAP=conf:0.1` is more interesting than a fixed cap: it keeps the rectangular target verify layout, caps committed length, and remaps unused rows to reuse anchor experts. It aims to reduce distinct expert weights read. The local implementation derives its stride from `DSPARK_BLOCK_SIZE` and broadcasts lengths from rank 0, but its demonstrated benefit is TP4/block 5. For sampled rows its intended integration includes block verification. Test those dependencies before reporting a result. Fixed caps were rejected in the downstream measurements; do not assume fewer verified rows automatically means faster output.

Keep SPS/ragged verification separate. The local README suggests profiling an SPS table, but the downstream reports an Engram-path crash with ragged verification. Confidence caps deliberately retain the existing rectangular layout. Do not generate and deploy an SPS table as an immediate speed fix without reproducing compatibility.

**3. Hide Engram misses, then decide whether caching is worthwhile.**

The prefetch adapter starts row lookup after hash IDs are available and joins the side stream at each gather. This can remove exposed I/O latency without a multi-GiB cache. Test the already-present `DSV41_ENGRAM_PREFETCH=1` on a clean, packed-shard baseline, with every other experimental hook disabled.

Use `DSV41_ENGRAM_PREFETCH_CHECK=1` during correctness testing to compare against synchronous lookup, then turn checking off for timing. Also verify that the adapter actually stays enabled: it can log a runtime error and fall back. Warm every captured batch shape and test graph replay, short/long prefill, and concurrent requests. The earlier combined startup failure is a reason for isolation, not proof this feature alone is broken.

The TP4 2.2 ms step saving would be only about 2.8% on an 82 ms step if the same exposed gap existed. This is illustrative arithmetic, not a prediction. Packing, caching, and prefetch attack overlapping I/O cost; their headline gains must not be added.

After prefetch, measure remaining miss stalls before allocating cache. Sweep `DSV41_CACHE_GIB=0/0.25/0.5/1` only within observed memory headroom; use 16 ways when a nonzero cache is tested. The backend accepts fractional GiB and divides the host budget between the two layers. More ways with a zero-size cache buys nothing. Compare real diverse text, warm and cold caches, and head memory pressure. Do not copy TP4's 4 GiB cache into a TP3 baseline with roughly 6 GB historical headroom.

**4. Borrow newer kernels selectively, with pinned dependencies.**

The downstream [canary Dockerfile](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4/blob/4d8f4c0868ed3b394688b65d88cd99a82c237642/Dockerfile.canary) combines SGLang `f80c91a4b`, `sglang-kernel==0.4.7`, `sgl-deep-gemm==0.2.0`, and its corresponding SM120 MLA overlay. It is a useful reproducible comparison point; a moving image tag or latest branch is not an equivalent experiment.

Preserve TP3 padding, positive-scale repair on zero weight shards, the SM120 page-split path, and all local compatibility hooks. Inspect branch dispatch predicates for G=4 and TP3 before expecting mHC, metadata, or projection improvements. Keep fast load, shared-expert padding, RoCEnante, and prefill splitting off during the initial image comparison.

For that pinned branch, downstream uses `SGLANG_DSPARK_FOLDED_SAMPLING=2` and `SGLANG_RUST_BUILD_MODE=never`. Its [folded-result fence](../adapter/folded_result_fence.py) addresses a result-buffer race and should be treated as a compatibility/correctness fix, not a tok/s optimization. Version-match it to the engine.

The local launcher has explicit Docker environment allowlists and currently lacks downstream's `EXTRA_CONTAINER_ENV` and `BUILD_DOCKERFILE` plumbing. New environment variables placed only in `.env` will not necessarily reach a container. Wire each required variable into both head and worker launches, including prefetch check mode and folded-sampling controls, and inspect the effective container settings. Preserve current local edits when porting build support.

The [autotune-cache adapter](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4/blob/4d8f4c0868ed3b394688b65d88cd99a82c237642/adapter/autotune_keep.py) may improve repeatability by retaining valid per-rank EP tactics across identical launches. It is not a demonstrated steady-state speedup or a remedy for the recorded startup failure. Its hooks target the newer engine layout. Extend cache provenance to include image/kernel versions if ported; invalidate after incompatible changes.

**5. Port small-message RoCEnante only after resolving topology.**

Historical TP3 has roughly 13–16 ms of NCCL time per block-5 step after rank skew was repaired. This is a meaningful optimization target, but profiler time inside a collective can include waiting for a slower rank. Profile all ranks before treating that entire number as network latency.

Three source facts determine portability:

1. [The SGLang adaptation](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4/blob/4d8f4c0868ed3b394688b65d88cd99a82c237642/runtime/roce_tp4_adapt.py) accepts only TP4/TP8. It rejects TP3.
2. [The underlying runtime](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4/blob/4d8f4c0868ed3b394688b65d88cd99a82c237642/runtime/b12x/b12x/comm/roce/roce_oneshot.py) declares support for world sizes 2 through 16, including 3. Three ranks are not inherently excluded by its reduction algorithm.
3. [The proxy](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4/blob/4d8f4c0868ed3b394688b65d88cd99a82c237642/runtime/b12x/b12x/comm/roce/_roce_proxy.c) connects every peer on every selected HCA; the kernel waits on peer/HCA completion flags. This assumes reachability that a two-port direct-cable triangle does not supply in the same form as two switched rails.

On a switched three-node fabric, a guarded TP3 integration and transport test is relatively contained. On this repository's triangle, simply changing `(4, 8)` to `(3, 4, 8)` is insufficient: implement per-peer HCA/GID selection and matching send/receive/flag bookkeeping, or prove a supported routed fabric supplies the required connectivity. Selecting a single HCA does not make all three triangle nodes directly reachable through it.

Before engine integration, compare three-rank bf16/fp32 SUM correctness with NCCL in eager and captured/replayed execution; test timeout handling and symmetric fallback. Benchmark the actual TP3 payload histogram, including its block size and four-stream limit. The TP4 2 MiB threshold targets its 16-slot regime, not this one. Account for pinned transport memory and CPU proxy cost on GB10.

Illustration only: saving 5 ms from an 82 ms step at unchanged acceptance gives `82 / 77 - 1 = 6.5%`; saving 8 ms gives 10.8%. Eliminating all 16 ms would give about 24%, an unrealistic upper bound rather than a target. Establish exposed transport cost before forecasting a gain.

**6. TP3-specific opportunities beyond copying the fork.**

**Compare EP3 with EP1 under TP3.** EP2 does not evenly factor a three-rank world. EP1 can instead tensor-shard each routed expert over three ranks: every rank holds a slice of every expert rather than a subset of complete experts. It does not inherently replicate all full expert weights on every node. The checkpoint's 2,304 intermediate dimension becomes 768, which is divisible by 128; that is a promising shape check, not proof the entire backend works.

[SGLang's FusedMoE code](https://github.com/sgl-project/sglang/blob/f80c91a4b/python/sglang/srt/layers/moe/fused_moe_triton/layer.py) divides the intermediate dimension by MoE TP size, and [the group construction](https://github.com/sgl-project/sglang/blob/f80c91a4b/python/sglang/srt/distributed/parallel_state.py) distinguishes MoE TP and EP. Test `TP_SIZE=3`, `EP_SIZE=1` in isolation. It may reduce expert imbalance across ranks, but introduces more, narrower expert GEMMs and different metadata/workspaces. Validate the native MXFP4 runner, draft expert padding/loading, memory, numerical behavior, and kernel efficiency. This is an unmeasured TP3 hypothesis; no speedup should be assigned in advance.

**Investigate removing padded attention work on rank 2.** With 64 heads padded to 96 and 8 groups padded to 12, rank 2's attention output shard is entirely padding. A guarded zero-output path could avoid its padded projection/attention/output work while retaining all required KV/indexer updates and collectives. It must not skip Engram, its real experts, shared computation, or alter collective ordering. Since ranks 0/1 still do real work, this may save energy and memory more than step time; it speeds decode only if it reduces the critical path or prevents memory pressure. Treat a balanced 24-head-per-rank kernel/layout as longer-term development, not an environment toggle: current SM120 legal local head counts exclude 24.

**Reduce rank skew and memory stalls.** Check clocks, thermals, CPU contention, disk latency, and `/proc/pressure/memory` on every rank. The local profile already found worker GEMMs 5–8% slower during competing activity. Keep benchmark generation/analysis off the head, preserve the existing reduced NCCL buffer footprint, and validate b12x actually handles padded rank 2. These can restore lost throughput without changing model numerics. Raising memory fraction or shrinking KV blindly is not a demonstrated decode optimization.

**Separate aggregate throughput from interactive speed.** Once memory is stable, consider 4 → 6 → 8 request slots, measuring graph-capture memory and long-context pressure at each step. At eight or more, evaluate `--min-free-slots-delay 1` so admission does not leave a slot unused. Below eight, the documented admission condition does not apply. More slots can raise aggregate tok/s while lowering each stream's rate; do not promise a C1 improvement. TP4's 16-slot graph allocation is unsuitable as a starting point for TP3.

**Do not borrow these as direct decode improvements.**

| TP4 feature | Reason to exclude or defer |
|---|---|
| `DSV41_SHARED_PAD_K=1` | [The adapter](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4/blob/4d8f4c0868ed3b394688b65d88cd99a82c237642/adapter/shared_pad_k.py) requires shared down-projection `[5120,576]` and raises on another shared-expert shape. TP3 has `[5120,768]`, already aligned. |
| `DSV41_TP_PAD=0` | TP3 needs head/group/vocab/draft padding; TP4 does not. |
| 0.80 memory fraction, 8M KV pin, 16 slots, 4 GiB cache | These assume TP4's much smaller per-rank weight footprint. |
| Fast weight loading | Improves startup, not steady decode, and downstream records a smaller KV pool. Local code already carries a synchronous-copy UMA safeguard. |
| Chunk 4096 / bounded indexer / SG18 split | Primarily prefill and memory work. The bounded indexer may indirectly protect decode under mixed traffic; establish that separately. [SG18's topology gate](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4/blob/4d8f4c0868ed3b394688b65d88cd99a82c237642/adapter/spark_prefill_dense.py) accepts only 4 or 8 ranks. TP3 needs a port and partition-tail tests. |
| `--enable-deepseek-v4-fp4-indexer` | Worth an isolated backend test, but the local metadata adapter already handles ratio-1/2 FP4 indexers. Establish which remaining call sites change before claiming an additional gain. |
| `--sleep-on-idle` / cache usage reporting | Useful operation/client features; no demonstrated steady decode benefit. |
| CPU weight/KV offload | CPU and GPU share physical memory on GB10; moving allocation domains does not create another memory pool. |
| `expandable_segments:True` / fused MoE finalize | The local record has NaN and nondeterminism regressions respectively. Preserve the known working settings. |

**A concrete experiment order.**

| Stage | Change | Evidence required to keep it |
|---|---|---|
| A | Verify/recover packed files and mounts on all ranks | Both layers `packed=True` per rank; measure Engram wait and fresh C1/C4 baseline |
| B | Sweep block 2/3/4/5 with existing image | Best committed-tokens/step divided by step time per workload; quality and memory unchanged |
| C | One-at-a-time trials: G=4 `wo_a` dispatch, then prefetch | Actual accelerated dispatch; clean autotune/capture; correctness checks; repeatable end-to-end gain |
| D | Add medium-row `wo_a`, then cautiously evaluate storage drop | C2/C4 benefit and measured net memory saving; no padded-rank failure |
| E | Draft head, folded sampling/fence, temperature, block verification, confidence cap | Compatible hooks and distributions; sampled/greedy/mixed-batch results; no rank divergence |
| F | Pinned canary comparison and TP3/EP1 | All-rank numerical and memory validation; fresh profiles; retain only measured improvements |
| G | RoCEnante TP3 port; small row-cache and concurrency sweeps | Topology validated independently; bounded extra memory; workload-specific gains |

Take one change at a time initially, then validate the chosen combination. Record image digest, SGLang/kernel commits, adapter hashes, TP/EP, block size, graph sizes, cache budget, packed status, and effective head/worker environment. Restoring previous settings/images and disabling the relevant adapter is the rollback; avoid replacing the whole launcher or adapter directory with the downstream versions.

Use at least five warmed repetitions per condition with identical prompts and token limits. Include diverse English prose, code, structured output, and sampled chat; test C1/C2/C4 and short/long resident context separately. Run a mixed long-prefill/decode workload as its own experiment. Exclude foreign traffic, record TTFT and p95 inter-token latency, and retain raw results. Small gains must exceed measured local variance, not merely the downstream's reported noise band.

For steady decode, use committed-token counts and measured step windows. For end-to-end aggregate throughput, divide total completion tokens by the whole workload wall time. Never count SSE events as tokens: a speculative block can arrive in one event. [The local decode-window script](../benchmarks/decode_window.py) explicitly estimates token timing from character growth; label that estimate and do not make it the sole acceptance metric for a 1–3% change.

The [profiling tools](../scripts/profile/profile_decode.sh) and [step analyzer](../scripts/profile/analyze_steps.py) can separate `wo_a`, routed MoE, communication, and Engram gaps. Capture every rank for the candidate that wins. GPU profiles, finite-logit checks, repeated greedy requests, long-context retrieval, and representative code/JSON/math tests are needed before calling a kernel or sampler port successful.

**Expected outcome:** there is credible room to improve TP3, especially if worker packed shards are also missing, but this review does not establish a new TP3 tok/s number. The strongest evidence-backed engineering target is the remaining `wo_a` path; the strongest immediate operational finding is the absent head packing. DSpark tuning and prefetch are inexpensive follow-ups, while communication and EP layout offer larger but less certain development opportunities.
