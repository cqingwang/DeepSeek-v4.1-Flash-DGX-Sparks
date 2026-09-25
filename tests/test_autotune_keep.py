"""CPU check for adapter/autotune_keep.py: per-rank caches with disjoint shape entries digest alike
when every rank's sidecar matches its launch, and differently (-> dropped) when one does not."""
import contextlib
import json
import os
import tempfile
import types
from pathlib import Path

os.environ["DSV41_AUTOTUNE_KEEP"] = "1"
import autotune_keep as ak  # noqa: E402


def fake_module(paths, contents):
    mod = types.SimpleNamespace()
    mod._autotune_cache_digest = lambda p, env: "stock"
    seen = {}

    @contextlib.contextmanager
    def ctx(model_runner, **kw):
        seen[model_runner] = paths[model_runner].is_file()   # what tuning started from
        yield
        paths[model_runner].write_text(contents[model_runner])   # tuning saves its cache

    mod.seen = seen

    mod.flashinfer_autotune_context = ctx
    mod.flashinfer_autotune_cache_path = lambda mr: paths[mr]
    return mod


def main():
    d = Path(tempfile.mkdtemp())
    meta = {"flashinfer": "0.6.18", "arch": "sm121"}
    paths = {}
    contents = {}
    for rank, shapes in ((0, ["(192, 2304, 320)"]), (1, ["(64, 2304, 320)"])):
        p = d / f"rank_tp{rank}_pp0_dp0.json"
        contents[rank] = json.dumps({"_metadata": meta, **{s: [16, 36] for s in shapes}})
        p.write_text(contents[rank])
        paths[rank] = p
    mod = fake_module(paths, contents)
    ak.install(mod)
    env = {"cuda": "13.0"}
    assert mod._autotune_cache_digest(paths[0], env) == "", "no sidecar yet: must not be kept"
    for rank in (0, 1):                                  # a tuning boot writes the sidecars
        with mod.flashinfer_autotune_context(rank):
            pass
    assert mod.seen == {0: False, 1: False}, "a cache without this launch's sidecar must not be loaded"
    for rank in (0, 1):                                  # identical relaunch: kept and loaded
        with mod.flashinfer_autotune_context(rank):
            pass
    assert mod.seen == {0: True, 1: True}, "a cache with a matching sidecar must be kept"
    d0, d1 = mod._autotune_cache_digest(paths[0], env), mod._autotune_cache_digest(paths[1], env)
    assert d0 and d0 == d1, "disjoint EP shape sets with matching launches must digest alike"
    ak.sidecar(paths[1]).write_text("other-launch\n")
    assert mod._autotune_cache_digest(paths[1], env) == "", "a changed launch must drop the cache"
    os.environ["DSV41_VERIFY_CAP"] = "conf:0.1"          # A/B-neutral switch: same fingerprint
    assert ak.launch_fingerprint() == ak.launch_fingerprint()
    fp_a = ak.launch_fingerprint()
    os.environ["DSV41_VERIFY_CAP"] = "3"
    assert ak.launch_fingerprint() == fp_a
    os.environ["SGLANG_RUN_ID"] = "sglang-run-1.0-1"          # per-boot id: must not enter the fingerprint
    fp_run = ak.launch_fingerprint()
    os.environ["SGLANG_RUN_ID"] = "sglang-run-2.0-2"
    assert ak.launch_fingerprint() == fp_run
    os.environ["SGLANG_SOMETHING_SHAPED"] = "1"
    assert ak.launch_fingerprint() != fp_a
    print("test_autotune_keep: ok")


if __name__ == "__main__":
    main()
