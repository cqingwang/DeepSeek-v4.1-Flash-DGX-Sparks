# TP3 decode optimization — merged, cross-checked report

2026-09-24. Combines [Astra](astra-new.md) and [Fable](fable-new.md), with additional read-only checks of the three-node fleet, local adapters, current container source, checkpoint metadata, and the pinned downstream source. Original reports are preserved. No services, weights, clocks, caches, or launch settings were changed, and no inference benchmark was run for this merge.

**Recommended order: recover packed Engram on all ranks, reduce contention and establish a clean baseline, stabilize autotune, then test the TP3 `wo_a` path and Engram prefetch independently.** After that, evaluate speculative block size and confidence caps, sampled-decoding improvements, and draft-head FP8. EP1 and triangle-compatible RoCEnante remain separate experiments. There is no measured new TP3 tok/s result yet.

The downstream revision remains [knapcio `4d8f4c0868ed3b394688b65d88cd99a82c237642`](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4/tree/4d8f4c0868ed3b394688b65d88cd99a82c237642), including its September 23 changes. Local HEAD is `25379c22e8ff177c7587b21b09eac33a92efeb54` plus existing working-tree changes. Read-only fleet checks were taken around 02:55–02:57 Asia/Jerusalem; selected raw outputs are saved in [the cross-check record](results/merged-review-20260924.json). Runtime values are snapshots, not permanent properties of the fleet.

**Where the original reports agree, and what the cross-check changes.**

| Question | Reconciled finding |
|---|---|
| Are packed shards missing? | **Confirmed across all three ranks.** Astra originally verified only the head; Fable's worker observation is now independently confirmed by worker mounts and `packed=False` logs. |
| Does spark2 have room to pack? | **No at the observed free space.** Exact checkpoint arithmetic requires 67,586,019,080 bytes for its two TP3 shards; 64,906,600,448 bytes were available. Additional operating headroom is needed too. |
| Must its old TP2 shards be deleted? | No. They occupy about 101.38 GB but are outside the active worker `/engram` mount. Archive/relocate them, use another local volume, or remove them only if confirmed disposable. The merged plan does not prescribe deletion. |
| Did an adapter cause the failed boot? | spark3's journal confirms `NV_ERR_NO_MEMORY` in graphics-context allocation at 01:33:54. `.env` records a prior 0.98 experiment. This supports a memory-allocation failure, **not a causal diagnosis of a particular adapter or fraction**. Allocator state and simultaneous changes remain confounders. |
| Does `start.sh` force the adapters off? | No. `${VAR:-0}` supplies a default and preserves an explicit nonzero setting. Their effective configured state is off. |
| Can autotune retention work on the current image? | Fable is right: the running container has `_autotune_cache_digest`, `flashinfer_autotune_context`, and `flashinfer_autotune_cache_path` at the expected module path. Astra's compatibility uncertainty can be narrowed to testing the hook and distributed behavior. |
| How much memory does `wo_a` DROP save? | About **688 MiB net versus the original bf16 baseline**, assuming 43 G=4 layers, plus small scale overhead. It releases 1,376 MiB gross while retaining approximately 688 MiB of FP8 twins. Compared with an already-enabled twin-only configuration, DROP can release the full gross amount. |
| Are the adapters risk-free or bitwise lossless on TP3? | No such result has been established. Exact stored weights do not imply identical kernel accumulation; sampler mathematics does not prove integration correctness. TP3 graph capture, padded shards, and sampled paths need testing. |
| Is k=5 plus `conf:0.1` guaranteed to improve prose? | No. TP4 results justify a TP3 experiment, not a safe default or a minimum expected gain. Measure k=5 alone before adding the cap. |
| Can the individual speed estimates be added? | No. Both the denominator and acceptance can change, and packing/cache/prefetch overlap. Fable's +12–14% combined kernel forecast and +10–25% code forecast are hypotheses, not established TP3 outcomes. |
| Can prefetch save 1.5–2.5 ms after a profile shows only ~1 ms exposed Engram time? | Not from that same measured gap alone. Reprofile after packing; do not reuse the downstream's 2.2 ms saving as a TP3 estimate. |
| Does DRM caching create 1.8 GiB of free head memory here? | Not with the current zero-cache baseline. It can provide another cache backing store; it frees ordinary memory only relative to an ordinary cache that otherwise existed. The downstream says boot-time KV sizing is unchanged. |
| Is EP1 bounded to a 3% improvement? | No. Subtracting aggregate NCCL times on two ranks does not establish a strict EP1 bound. EP1 changes routing balance, GEMM shapes, metadata, and overlap. Its net result is unmeasured. |
| Is the proposed RoCEnante triangle patch ready? | No. Per-peer HCA selection is necessary, but Fable's proposal to synthesize both completion flags needs a protocol and memory-ordering proof before implementation is trusted. |

**Verified baseline and immediate repairs.**

The current configuration is TP3/EP3, DSpark block 3, four request slots, 1,024-token prefill chunks, requested KV pin 750,000, memory fraction 0.95, b12x dense routing, no Engram row cache, and experimental adapters off. The [saved boot log](../logs/dsv41.log) reports an actual pool of **491,520 tokens**, `available_bytes=1.04 GB`, and two autotune-cache disagreement events. The requested pin is not the effective capacity.

| Node | Cross-checked state | Implication |
|---|---|---|
| spark1 / rank 0 | `/home/mia/dsv41-engram` is empty and mounted at `/engram`; about 3.07 GiB `MemAvailable` | Head packed path is absent; little room for added weight copies |
| spark2 / rank 1 | Active `/engram` comes from `/home/zurih/dsv41-3x-spark/engram`; logs show unpacked reads. Old `r1of2` files are in `/home/zurih/dsv41-engram`. About 7.55 GiB `MemAvailable` | Old TP2 files are not usable TP3 shards; disk space must be resolved first |
| spark3 / rank 2 | Active `/engram` also comes from the worker checkout; logs show unpacked reads. Desktop/gdm active, 19 non-serving containers in this snapshot, about 4.13 GiB `MemAvailable` | More memory/CPU contention than spark2; investigate which services can be moved or paused |

All three hosts have a clock-lock unit configured with `nvidia-smi -lgc 0,2200`; the worker units were confirmed active. Fable's earlier live clock readings are consistent with this configuration, but no new clock/performance sweep was run here. Container count changed since Fable's snapshot, illustrating why a controlled workload is necessary.

The most concrete action is to recover the existing packed-storage path. [The packer](../scripts/pack_engram.py) writes weight and scale together, using `.partial`, flush/fsync, and rename. Checkpoint header arithmetic gives **62.944 GiB per rank** for both layers combined. Reserve that much plus free-space margin on each destination, accounting for any existing partial files. `./start.sh pack` uses `ENGRAM_DIR` on the head and **`WORKER_ENGRAM_DIR`** on the workers; inspect those actual destinations rather than assuming matching home directories.

At the next controlled maintenance window, pack, restart/reload the engine, and require `packed=True` for layers 1 and 14 on each rank. Rows already attached to the old backing file do not switch merely because new files appeared. Atomic publication prevents a partial shard being opened as complete, but does not eliminate packing's CPU, memory, NFS, and disk load. Do not describe packing as operationally free while the fleet is serving.

Historical [TP3 profiling, sections 6–10](../logs/profile-2026-09-10/REPORT.md) found roughly 11 ms of Engram-related rank-0 waiting before packed worker shards and approximately 0.5 ms afterwards. The same boot also enabled b12x, so the 120 → 98 ms total-step change cannot be attributed entirely to packing. Packing removes Engram checkpoint reads from NFS; it does not promise zero NFS traffic for every other purpose.

Quieting spark3 belongs before kernel comparisons. Inventory workloads with their owner, move or pause only those that are dispensable, and reduce dashboard polling if it contributes measurable overhead. Moving everything to spark2 may simply move the straggler. `systemctl set-default multi-user.target` changes the next boot target; it does not stop the current desktop session by itself. Disabling display services and changing clock limits are separate operational decisions, not steps performed by this document.

**Ranked optimization opportunities.**

| Priority | Candidate | Evidence and expected scope | TP3 work required |
|---|---|---|---|
| 1 | Packed Engram + reduced contention | Verified configuration problems; historical TP3 evidence for substantial lost time | Resolve disk capacity; restore all rank/layer files; establish fresh timings |
| 2 | Autotune-cache retention | Current image discards differing rank caches; downstream preserves valid EP caches | Port hook and env forwarding; verify distributed cache hits and invalidation |
| 3 | `wo_a` FP8 twin path | Historical TP3 `wo_a` is ~11 ms of an 82 ms block-5 step | Validate G=4 dispatch and GPU results on the current image; retain local bridge |
| 4 | Engram prefetch | Downstream same-image TP4 step 51.69 → 49.49 ms | Test alone after packing, with check mode followed by timing without checks |
| 5 | DSpark block size / confidence cap | Downstream TP4 cap improves prose C1 66.0 → 69.7 tok/s | Compare k=3/5 first; isolate cap; extend to 2/4 if useful |
| 6 | Draft FP8 head, draft temperature, block verification | Downstream reports small step/acceptance improvements | Account for extra head memory; confirm folded sampling and exact proposal handling |
| 7 | Small ordinary cache, DRM-backed cache, concurrency | May improve effective throughput or memory use | Spend only measured headroom; DRM needs a TP3 memory-aware port |
| 8 | TP3/EP1 | Plausible reduction in expert imbalance | Validate narrower MXFP4 GEMMs, memory, padding, and acceptance |
| 9 | Triangle-aware RoCEnante | Meaningful historical collective cost | Transport port and standalone three-rank validation before serving integration |
| 10 | Pinned canary image | Downstream TP4 kernel gains | Revalidate every TP3 hook and shape-dependent dispatch; compare separately |

The priority order balances evidence and implementation effort, not a claim that later candidates have smaller possible gains. TP4 figures above come from [the September 23 changelog](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4/blob/4d8f4c0868ed3b394688b65d88cd99a82c237642/CHANGELOG.md) and [the Engram profile](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4/blob/4d8f4c0868ed3b394688b65d88cd99a82c237642/docs/upstream-watch.md).

**Autotune retention: a useful addition from Fable.**

The running image's cache digest hashes the entire per-rank configuration. Its two recorded disagreement events agree with the downstream diagnosis. [The downstream adapter](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4/blob/4d8f4c0868ed3b394688b65d88cd99a82c237642/adapter/autotune_keep.py) instead coordinates whether each rank can load its cache and tracks launch compatibility with a sidecar.

Port the file, its import hook/finder entry, and the variable into **both** Docker launch paths. The required symbols exist on the current image; this is not contingent on a canary upgrade. Test identical relaunches and changed settings. Include image/kernel versions in cache provenance and invalidate incompatible entries. A per-rank hit on one node and retune on another can desynchronize collective timing, so this is not a zero-risk cosmetic patch. Its main proven value is repeatability and startup work; a steady-state decode gain must be measured.

**`wo_a`: strongest kernel target, with corrected memory accounting.**

[The local adapter](../adapter/wo_a_w8.py) generalizes the downstream two-group kernels to TP3's four local groups and supplies an einsum bridge for the current image. Do not overwrite it with the TP4 file. Six other local adapters are already byte-identical to the pinned fork: prefetch, draft head, draft temperature, block verification, confidence cap, and folded-result fence.

Start with W8 enabled alone and `MID=DROP=0`. Confirm the accelerated path is executed for actual target and draft shapes. Then try MID independently. At k=3, C2 verification has eight rows and still falls in the small-row path; C4 has sixteen. At k=5, C2 has twelve rows and reaches MID. Graph padding and draft shape can add other cases. Label results with block size rather than assuming C2 always exercises MID.

For 43 eligible G=4 layers, the memory ledger is:

| State | Approximate `wo_a` weight storage per rank |
|---|---:|
| Original bf16 | 1,376 MiB |
| bf16 plus FP8 twins | 2,064 MiB plus scales |
| FP8 twins after DROP | 688 MiB plus scales |

DROP releases 1,376 MiB relative to the twin-only test, but the final configuration saves **688 MiB net relative to the original baseline**. It may not translate one-for-one into extra KV tokens. Build/check temporaries, allocator retention, graph buffers, and other allocations affect boot-time memory. The draft FP8 head adds approximately 210.6 MiB (220.86 MB) per rank plus scales for a 43,136 × 5,120 shard, with larger temporary conversions during construction.

DROP must come after validating all forward paths because it replaces bf16 storage with a one-element tag and relies on wrappers to handle every use. The older-image bridge's 2–8-row path is selected by shape, without the MID path's speculative-mode guard, so test short prefills too. Exact twin reconstruction guarantees weight values, not identical accumulation versus einsum. GPU numerical and task checks remain necessary.

The pinned canary introduces another issue: [its model dispatch](https://github.com/sgl-project/sglang/blob/f80c91a4b/python/sglang/srt/models/deepseek_v4.py) requires `(2,4096)` activations and `(2,1024,4096)` weights to enter the fast small-row path. TP3 has G=4. Finding the canary kernel module causes the local adapter to choose kernel replacement instead of its old-image bridge; kernel replacement alone does not widen that guard. Carry an explicit G=4 dispatch solution into any canary trial.

Fable's projected ~6.5 ms saving extrapolates bandwidth across different shapes and kernel paths. It is a useful hypothesis to profile, not an expected measurement. The historical 82 ms denominator is from k=5 in September 10's configuration, not today's unpacked k=3 boot.

**Engram prefetch and row caches.**

Test the already-copied prefetch adapter independently after packing. Its side stream starts lookup after hashing and joins at the gather. Warm all graph shapes, compare synchronous and prefetched rows with `DSV41_ENGRAM_PREFETCH_CHECK=1`, inspect fallback/error messages, then disable check mode before timing. Check mode repeats work and may synchronize; it cannot be used to measure the intended speedup.

If meaningful exposed misses remain, a 0.25/0.5 GiB ordinary host cache is a possible experiment once headroom is proven. There is no evidence that DRM is the only conceivable TP3 caching option; equally, no evidence supports copying a 4 GiB cache into today's constrained baseline. Sixteen ways is useful only with a nonzero cache. Compare diverse text and warm/cold states, since repeated-token prompts overstate cache benefit.

Fable's display-reservation lead is worth keeping, but **the copied implementation needs additional TP3 policy**. [Its documentation](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4/blob/4d8f4c0868ed3b394688b65d88cd99a82c237642/docs/display-reserve.md) qualifies the feature on driver 580.173.02, not all 580.x drivers. Source inspection found:

- The Python adapter derives a per-layer budget from `DSV41_CACHE_GIB`, then limits one layer to the DRM pool. At the current value 0, enabling the DRM variable still yields no cache slots.
- Raising the global budget also allocates an ordinary-RAM cache for the other layer. Keys/metadata and staging remain in ordinary memory even for the DRM-backed layer.
- Failed DRM allocation falls back to anonymous ordinary memory. On a constrained node that silently changes the memory budget; port an explicit disable/fail policy before relying on the reservation.
- The claimed ~1.8 GiB headroom is relative to an equivalent existing ordinary cache as it fills, not new free RAM versus cache-off. Boot-time KV allocation is not automatically enlarged.

Add independent per-layer budgets so the non-DRM layer can stay at zero; account for metadata and fallback; include DRM build headers and container access. Host headless/modeset changes and reboot make this a later experiment, with an isolated driver compatibility probe first. No such host changes were made during this review.

**Speculative decoding: maximize committed tokens per unit time.**

Use `committed tokens per speculative step / step time` for C1. A longer draft can increase accepted tokens while reducing tok/s because verification becomes more expensive. Compare k=3 and k=5 on the clean baseline, optionally 2 and 4, and repeat the finalists after kernel changes. Code, prose, structured answers, and thinking/sampled traffic need separate results.

The downstream's [block-size probes](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4/blob/4d8f4c0868ed3b394688b65d88cd99a82c237642/docs/session-20260919.md) favor 5 on its TP4 fleet. The draft-config mismatch log establishes the configured draft size; it does not establish the throughput optimum. Try `conf:0.1` only after measuring k=5 alone. It preserves rectangular verification and remaps dead rows to anchor experts; its sampled path depends on compatible verification, and rank-0 length broadcast must stay consistent.

For sampled traffic, test draft multiplier 0.8 and block verification separately. Their TP4 accepted-token gains are not additive guarantees of real-chat speed. Keep target temperature/top-p/top-k constant, and confirm that acceptance uses the actual proposal distribution. Mixed greedy/sampled batches, capped blocks, and rank agreement are required checks.

**Folded-sampling prerequisite:** the live head has no explicit `SGLANG_DSPARK_FOLDED_SAMPLING` variable. Container source defaults it to AUTO, whose decision depends on minimum free memory across ranks. Therefore unset does not mean disabled, but setting draft temperature does not prove the folded path is active either. Inspect/log that decision; if FORCE=2 is tested, budget its static buffers first. The local launcher does not currently forward this variable or prefetch check mode. Add explicit head/worker forwarding before testing. The folded-result fence is a correctness candidate, not a free throughput boost.

Keep ragged/SPS verification out of the initial sequence: the downstream reports an Engram failure on that path. Confidence caps are a different mechanism. The claimed distributional properties of these adapters require integration checks; avoid the original report's blanket “none” risk labels.

**Later TP3 experiments.**

TP3/EP1 is valid to investigate; TP3/EP2 is not the corresponding even factorization. With EP1, ranks tensor-shard every expert's 2,304-wide intermediate to 768 instead of storing subsets of full experts. Ideal expert weight bytes per rank are comparable, but padding, scales, workspace, metadata, and runtime memory may differ. More, narrower grouped GEMMs can offset reduced expert imbalance; host/rank skew is not eliminated. Test the native MXFP4 path and draft padding. Source basis: [FusedMoE](https://github.com/sgl-project/sglang/blob/f80c91a4b/python/sglang/srt/layers/moe/fused_moe_triton/layer.py) and [parallel groups](https://github.com/sgl-project/sglang/blob/f80c91a4b/python/sglang/srt/distributed/parallel_state.py).

RoCEnante is a real port, not an env change. [The integration](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4/blob/4d8f4c0868ed3b394688b65d88cd99a82c237642/runtime/roce_tp4_adapt.py) accepts only TP4/TP8 even though its vendored runtime allows three ranks. Its [proxy](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4/blob/4d8f4c0868ed3b394688b65d88cd99a82c237642/runtime/b12x/b12x/comm/roce/_roce_proxy.c) connects each peer over every selected HCA and stripes transfers; the GPU waits for those completion flags. A direct-cable triangle needs per-peer local and remote HCA mapping, payload placement, and matching completion semantics. Do not merely enable world size 3 or select one HCA.

Start with eager and graph-replay three-rank tests against NCCL, then timeout/failure and all-rank fallback tests, then actual TP3 payload sizes. The illustrative bf16 hidden tensor at C4/k=5 is 245,760 bytes; k=3 gives 163,840 bytes. Other tensors differ, so route limits must follow a measured histogram. Historical collective timings include waiting for compute; an 88-collective latency multiplication is a hypothesis, not an end-to-end bound. An actual 5 ms reduction from an 82 ms step would yield 6.5% at unchanged acceptance.

A pinned canary trial should use its matching SGLang/kernel/MLA bundle and retain TP3 padding and scale repair. Keep transport, fast load, prefill split, and TP4 shared padding off for the first comparison. The G=4 guard above, vocab padding, and local hook signatures must all pass. A moving latest image would invalidate the comparison.

Skipping rank 2's all-zero attention projections is a lower-priority memory/compute experiment. It helps step time only when that rank or its memory pressure limits progress; otherwise ranks 0/1 still determine latency. Preserve KV/indexer work and collective ordering. Neither “guaranteed faster” nor “can never help” follows from the padding alone. Balanced 24-head kernel work is a larger redesign because the existing SM120 kernel set does not support that local head count.

Clock unlocking should be an isolated A/B after the main fixes, with sustained thermal/power measurements and the persistent unit accounted for. The downstream cap cost was only about 0.7–1.5% for its decode workloads. More request slots similarly need a separate objective: aggregate throughput may rise while per-stream speed falls. Try six/eight only after memory is stable; at eight evaluate `--min-free-slots-delay 1` and graph coverage.

**Excluded from the initial decode plan.**

Keep shared-expert padding off: TP3 K=768 already meets 128 alignment, and the TP4 adapter expects K=576. Preserve TP3 attention/vocab/draft padding. Do not copy TP4's 16 slots, 8M KV pool, 0.80 fraction, or 4 GiB row cache. Fast load is a startup optimization with memory tradeoffs. Bounded indexer and SG18 split are primarily prefill work; SG18 also has a TP4/TP8 gate. Cache reporting and sleep-on-idle are operational improvements, not established steady decode wins. Keep the known-working allocator and MoE-finalize choices. CPU offload does not create another physical RAM pool on GB10.

Defer the downstream's rejected fine-tuning/multi-pass/n-gram/relaxed-acceptance and small-kernel experiments unless a fresh TP3 profile supplies a new reason. Its negative results are useful prioritization evidence, not universal impossibility proofs.

**Controlled rollout and acceptance plan.**

| Stage | Isolated comparison | Required result |
|---|---|---|
| A | Packing, then separately reduced background contention | Six layer/rank `packed=True` checks; fresh C1/C4 timings and all-rank step profile |
| B | Autotune retention; then matched repeat boot | Correct cache hits on every rank; invalidation after incompatible changes; repeatable tactics |
| C | Folded-result fence if version-applicable | Clean capture/replay and correct greedy result handling |
| D | W8 alone; then MID | G=4 kernels actually execute; finite/correct outputs; measured C1/C2/C4 gain |
| E | Prefetch independently | Zero comparison failures; time only with CHECK off; lower exposed I/O gap |
| F | DROP after W8 validation; draft FP8 head in a separate comparison | Net memory ledger, startup peaks, effective KV pool, acceptance and step time |
| G | Block-size finalists; then confidence cap | Separate prose/code results and committed tokens/step; quality and rank consistency |
| H | Folded-sampling decision/override; draft temperature; block verification | One at a time; sampled and mixed-batch correctness and wall-time gain |
| I | EP1, canary, cache/DRM, transport, clocks or more slots | Separate baselines for each; promote only measured workload-specific improvements |

Dependencies override alphabetical order: confidence-cap sampled testing waits for its compatible block-verification integration; forced folded sampling waits for sufficient memory. Initial feature trials should branch from a known-good baseline, then combine only winners. Do not put several sampler switches into the same diagnostic boot as Fable's original boot B did.

Keep memory fraction 0.95 for the initial comparisons, but treat it as an engine budgeting parameter, not a guarantee that exactly 5% remains available. Record actual `MemAvailable`, pressure/stalls, load/capture peaks, and effective KV capacity on every rank. Preserve enough headroom before enabling twins or increasing concurrency. Test long-context behavior after short correctness smokes; avoiding long prompts during the first boot does not validate the advertised context envelope.

Use at least five warmed repetitions per condition, same prompts and completion limits, idle fleet, and matched reasoning/sampling settings. Record median/dispersion, TTFT, inter-token latency, aggregate completion throughput, acceptance, and step time. Discard warmup/autotune runs. Live mixed-traffic `gen throughput` snapshots, including Fable's median 38 tok/s, are useful operational observations but are not controlled baselines.

For end-to-end aggregate throughput, use total completion tokens divided by workload wall time. For steady decode use committed-token windows or measured speculative steps. Never count SSE messages as tokens. [The existing decode-window script](../benchmarks/decode_window.py) estimates token timestamps from character growth; label it as an estimate and avoid using it alone to accept small gains. [The profiler](../scripts/profile/profile_decode.sh) and [step analyzer](../scripts/profile/analyze_steps.py) should verify which cost actually fell.

Record image digest, engine/kernel versions, adapter hashes, actual container variables on every node, graph shapes, packed status, KV pool, and raw benchmark output. Validate numerical behavior, repeated greedy requests, sampled distributions where relevant, representative code/JSON/math tasks, concurrent traffic, and long-context retrieval. Promotion requires a repeatable end-to-end benefit without unexplained quality or memory regression. Disable the specific experiment and restore its saved configuration/image to roll back.
