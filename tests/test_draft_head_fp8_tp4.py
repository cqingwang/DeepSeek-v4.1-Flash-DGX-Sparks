"""CPU check for adapter/draft_head_fp8_tp4.py and its selection in sitecustomize.

- the TP4 variant builds the same fp8 copy as adapter/draft_head_fp8.py (bitwise) and refuses a
  drifted draft class;
- sitecustomize installs draft_head_fp8_tp4 only when start-tp4.sh set DSV41_LAUNCHER=tp4
  and DSV41_DRAFT_HEAD_FP8_IMPL=tp4. ./start.sh always installs draft_head_fp8.py.
  Nothing is installed when DSV41_DRAFT_HEAD_FP8 is off. The Triton kernels are checked on the Spark.
"""
import importlib.util
import os
import sys
import types
from pathlib import Path
from unittest.mock import patch

os.environ["DSV41_DRAFT_HEAD_FP8"] = "1"
import torch  # noqa: E402

import draft_head_fp8 as default_impl  # noqa: E402
import draft_head_fp8_tp4 as tp4_impl  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def check_twin():
    w = (torch.randn(1000, 256) * 0.02).to(torch.bfloat16)
    q4, e4 = tp4_impl.make_twin(w)
    q, e = default_impl.make_twin(w)
    assert torch.equal(q4.view(torch.uint8), q.view(torch.uint8)) and torch.equal(e4, e), "fp8 copies differ"
    try:
        tp4_impl.install(types.SimpleNamespace(DeepseekV4ForCausalLMDSpark=type("X", (), {})))
    except RuntimeError:
        return
    raise AssertionError("drifted class accepted")


def check_selection():
    spec = importlib.util.spec_from_file_location("dsv41_sc_test", ROOT / "adapter/sitecustomize.py")
    sc = importlib.util.module_from_spec(spec)
    with patch.dict(os.environ, {"DSV41_SOURCE": ""}):
        spec.loader.exec_module(sc)

    class Loader:
        def create_module(self, spec):
            return None

        def exec_module(self, module):
            pass

    for env, expected in (({"DSV41_DRAFT_HEAD_FP8": "1"}, "default"),
                          ({"DSV41_DRAFT_HEAD_FP8": "1", "DSV41_DRAFT_HEAD_FP8_IMPL": "tp4"}, "default"),
                          ({"DSV41_LAUNCHER": "tp4", "DSV41_DRAFT_HEAD_FP8": "1", "DSV41_DRAFT_HEAD_FP8_IMPL": "tp4"}, "tp4"),
                          ({"DSV41_DRAFT_HEAD_FP8": "1", "DSV41_DRAFT_HEAD_FP8_IMPL": ""}, "default"),
                          ({"DSV41_DRAFT_HEAD_FP8": "0", "DSV41_DRAFT_HEAD_FP8_IMPL": "tp4"}, None)):
        calls = []
        fakes = {name: types.SimpleNamespace(install=lambda m, tag=tag: calls.append(tag))
                 for name, tag in (("draft_head_fp8", "default"), ("draft_head_fp8_tp4", "tp4"))}
        clean = {k: v for k, v in os.environ.items() if not k.startswith("DSV41_")}
        with patch.dict(os.environ, {**clean, **env}, clear=True), patch.dict(sys.modules, fakes):
            sc.EngramLoader(Loader()).exec_module(
                types.SimpleNamespace(__name__="sglang.srt.models.deepseek_v4_dspark"))
        assert calls == ([expected] if expected else []), (env, calls)


if __name__ == "__main__":
    check_twin()
    check_selection()
    print("test_draft_head_fp8_tp4: ok")
