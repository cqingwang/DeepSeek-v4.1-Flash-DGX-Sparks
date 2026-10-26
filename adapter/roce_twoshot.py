"""Two-shot RoCE all-reduce for wide steps (DSV41_ROCE_TWOSHOT=1 + DSV41_ROCE_TWOSHOT_ROWS=<rows>).

DSV41_ROCE_TWOSHOT=1 makes the SGLang overlay build its TP runtime from b12x.comm.roce_twoshot
(runtime/roce_twoshot_adapt.py) instead of the production b12x.comm.roce: the same one-shot
transport plus the pipelined two-shot reduce-scatter + all-gather of flashinfer-ai/flashinfer#6915
(Luke Alonso). This adapter, installed on that module, supplies the row threshold per call:

  DSV41_ROCE_TWOSHOT_ROWS=0 (default)   every all-reduce stays one-shot
  DSV41_ROCE_TWOSHOT_ROWS=48            an all-reduce of a [rows >= 48, ...] tensor whose size fits
                                        pipe_bytes (B12X_ROCE_PIPE_BYTES, default SGLANG_ROCE_MAX_SIZE)
                                        takes the two-shot path

The value is read through ab_variant.env(), so DSV41_AB_V<i>="DSV41_ROCE_TWOSHOT_ROWS=..." gives each
in-boot A/B graph set its own threshold (captured per variant; eager calls follow the runtime
variant, which every rank switches on the same scheduler iteration). The decision depends only on
the tensor's shape and replicated configuration, never on a pointer or node state.

Numerics: the row owner sums the four partials in FP32 in rank order and rounds once, exactly as
the one-shot kernel does on every rank; the all-gather is a byte copy. Outputs are bit-identical
to the one-shot (tests/test_roce_twoshot.py models it; on the fleet, T=0 outputs with top-20 logprobs were
byte-identical to the one-shot production line on 75 qeval prompts).
"""
import logging
import os

logger = logging.getLogger(__name__)

MODULE = "b12x.comm.roce_twoshot.roce_oneshot"
KEY = "DSV41_ROCE_TWOSHOT_ROWS"


def enabled(environ=os.environ):
    return environ.get("DSV41_ROCE_TWOSHOT", "0").strip() == "1"


def _reader():
    try:
        from ab_variant import env
    except ImportError:  # the adapter dir without ab_variant (unit tests)
        env = os.environ.get

    def min_rows():
        raw = (env(KEY, "0") or "0").strip()
        value = int(raw)
        if value < 0:
            raise ValueError(f"{KEY}={raw}: must be >= 0 (0 = off)")
        return value

    return min_rows


def install(module):
    cls = module.RoceOneshotAllReduce
    if getattr(cls, "_dsv41_twoshot", False):
        return
    cls._dsv41_twoshot = True
    fn = _reader()
    fn()  # fail at install, not inside a capture, on a malformed value
    cls.pipe_min_rows_fn = staticmethod(fn)
    logger.warning("DSV41_ROCE_TWOSHOT: two-shot all-reduce armed, %s=%s at install (per call "
                   "through ab_variant.env)", KEY, os.environ.get(KEY, "0"))
