# Per-request cache reporting

`boot.py` enables SGLang's `--enable-cache-report` for both TP=3 and TP=4.
This exposes cached-token counts in OpenAI-compatible Chat Completions
`usage.prompt_tokens_details.cached_tokens` when the backend reports a hit.
For streaming requests, send `"stream_options": {"include_usage": true}`
and inspect the final usage chunk. Cold/zero-hit details can be absent in
some SGLang versions; the option does not guarantee cache reuse.

Do not use vLLM's `--enable-prompt-tokens-details` here: this recipe launches
`sglang.launch_server`, which has a different CLI. Reporting does not change
cache policy, memory allocation, or enable Prometheus metrics.

Existing servers only pick up the change on their next operator-initiated
restart. `python3 tests/test_cache_reporting.py` checks the unconditional
base argument list using the AST, without importing or executing `boot.py`.
Live cache reuse and the recipe's runtime image still need deployment-time
verification; this test is intentionally CPU-only.
