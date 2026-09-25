# Overnight TP3 optimization goal

Paste the following as one message to start the campaign.

```text
/goal Complete an overnight optimization campaign for DeepSeek-V4.1-Flash on our three DGX Sparks, using docs/merged-new.md as the starting plan. Deliver the fastest configuration validated during this campaign, with reproducible TP3 decode benchmarks and no demonstrated correctness regression.

Work autonomously for up to eight hours from starting. Reserve the final 45 minutes for validation, restoring services, and reporting. Prioritize single-stream prose and code decode tok/s, then C4 aggregate throughput. Keep TP_SIZE=3.

Authorization:
- You may inspect all three nodes, edit this repository, build images, pack Engram shards, and drain/restart the DeepSeek serving containers for experiments.
- Preserve existing uncommitted work, checkpoint weights, credentials, and unrelated services.
- If disk space prevents packing, relocate the old TP2 Engram shards to suitable storage, verify the copies before removing the source copies, and record their new location.
- You may temporarily pause identified dashboard benchmark generators or reduce their polling to obtain uncontaminated measurements; restore their previous state afterwards.
- Do not change drivers, firmware, boot settings, desktop services, or persistent clock settings during this campaign.
- Continue without asking about routine implementation choices. If an action exceeds this scope, defer that action and pursue another useful experiment.

Start with the highest-value work:
1. Recheck current state and save configurations, image identifiers, effective container settings, and a brief baseline.
2. Restore packed TP3 Engram shards on every node. Use the actual head/worker mount paths, resolve disk capacity, and verify packed=True for both layers on all three ranks after restart.
3. Establish clean C1 and C4 baselines and profile the remaining bottlenecks.
4. Port and validate autotune-cache retention if the observed cache-disagreement problem remains.
5. Test the existing TP3 G=4 wo_a FP8 path and Engram prefetch independently. Validate dispatch and correctness; introduce MID and DROP separately, with measured memory headroom.
6. Compare DSpark block sizes, prioritizing k=3 versus k=5, then evaluate confidence caps and sampled-decoding improvements with their dependencies satisfied.
7. Attempt draft-head FP8 or TP3/EP1 if evidence and remaining time justify them. Defer the RoCEnante triangle port, DRM cache, and canary migration unless the higher-value opportunities are exhausted and sufficient recovery time remains.

Experiment discipline:
- Change one factor at a time initially; combine only demonstrated winners.
- Benchmark from a worker, exclude foreign traffic, warm up, and use at least five repetitions for promotion decisions.
- Report prose, code, sampled chat, C1 per-stream, and C4 aggregate results separately.
- Use actual completion/committed-token counts. Do not count SSE events as tokens or treat character-estimated timing as precise.
- Record acceptance, step time, TTFT, memory pressure, effective KV capacity, and errors.
- Check GPU numerical behavior, representative task quality, concurrency, and long-context behavior before promoting a configuration.
- Treat the report’s projected gains as hypotheses. Roll back failures promptly; avoid repeated identical failed boots.

Maintain docs/overnight-progress.md after each experiment and save raw evidence under docs/results/overnight-<timestamp>/.

Finish with docs/overnight-results.md containing baseline versus final results, accepted/rejected changes, reproducible commands, exact final configuration, quality checks, memory tradeoffs, rollback instructions, and remaining opportunities. Leave the fleet serving the best validated configuration, or restore the original working configuration if no candidate passes. Do not stop after planning or the first improvement; continue through the highest-value feasible experiments within the work window.
```
