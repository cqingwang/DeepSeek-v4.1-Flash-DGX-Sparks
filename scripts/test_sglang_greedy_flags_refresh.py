#!/usr/bin/env python3
"""Execute the staged SamplingBatchInfo filter/merge methods for SGLang #44923.

The production methods are extracted from the staged tree and run against CPU
stand-ins for the tensors, so no Torch or SGLang import is needed. The cases
follow upstream's TestGreedyFlags plus the DSpark one that matters here: a
sampled request joins an all-greedy decode batch and leaves again, after which
`is_all_greedy` must be true so the folded verify path is eligible again.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

TOP_K_ALL = 1 << 30


class _T:
    """List-backed tensor stand-in: index by another _T, concatenate via cat."""

    def __init__(self, values):
        self.values = list(values)

    def __getitem__(self, index):
        return _T(self.values[i] for i in index.values)

    def __len__(self):
        return len(self.values)


class _Torch:
    Tensor = _T

    @staticmethod
    def cat(tensors):
        return _T(v for t in tensors for v in t.values)


class _Orchestrator:
    def filter(self, _keep_indices_device):
        pass

    def merge(self, _other):
        pass


def _merge_bias_tensor(lhs, rhs, bs1, bs2, device, default):
    assert lhs is None and rhs is None
    return None


METHODS = (
    "__len__",
    "filter_batch",
    "merge_batch",
    "adjusted_filter_batch",
    "adjusted_merge_batch",
)


def load_class(python_root: Path):
    target = python_root / "sglang/srt/sampling/sampling_batch_info.py"
    tree = ast.parse(target.read_text(), filename=str(target))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SamplingBatchInfo"
    )
    methods = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name in METHODS
    ]
    missing = set(METHODS) - {node.name for node in methods}
    if missing:
        raise SystemExit(f"staged SamplingBatchInfo lacks {sorted(missing)}")
    stand_in = ast.ClassDef(
        name="SamplingBatchInfo",
        bases=[],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    if "type_params" in ast.ClassDef._fields:  # Python 3.12+
        stand_in.type_params = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            stand_in,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {"torch": _Torch, "merge_bias_tensor": _merge_bias_tensor}
    exec(compile(module, str(target), "exec"), namespace)
    return namespace["SamplingBatchInfo"]


def make(cls, greedy, track=True):
    info = cls()
    n = len(greedy)
    info.temperatures = _T(1.0 if g else 0.7 for g in greedy)
    info.top_ps = _T([1.0] * n)
    info.top_ks = _T(1 if g else TOP_K_ALL for g in greedy)
    info.min_ps = _T([0.0] * n)
    info.sampling_seed = None
    info.is_all_greedy = all(greedy)
    info.is_any_greedy = any(greedy)
    info.need_top_p_sampling = False
    info.need_top_k_sampling = not all(greedy)
    info.need_min_p_sampling = False
    info.npu_top_k_top_p_eligible = True
    info.greedy_rows = list(greedy) if track else None
    info.penalizer_orchestrator = _Orchestrator()
    info.has_custom_logit_processor = False
    info.logit_bias = None
    info.return_sampling_masks = None
    info.sampling_mask_batch_indices = None
    info.device = "cpu"
    return info


def keep(info, indices):
    info.filter_batch(list(indices), _T(indices))


def rows_match_top_ks(info):
    return info.greedy_rows == [k <= 1 for k in info.top_ks.values]


def test(python_root: Path) -> None:
    cls = load_class(python_root)

    # DSpark: an all-greedy decode batch, a sampled request joins and leaves.
    info = make(cls, [True, True])
    info.merge_batch(make(cls, [False]))
    assert info.greedy_rows == [True, True, False]
    assert not info.is_all_greedy and info.is_any_greedy
    keep(info, [0, 1])
    assert info.greedy_rows == [True, True] and rows_match_top_ks(info)
    assert info.is_all_greedy and info.is_any_greedy

    # The last greedy row leaves: is_any_greedy clears.
    info = make(cls, [False, False])
    info.merge_batch(make(cls, [True]))
    keep(info, [0, 1])
    assert info.greedy_rows == [False, False]
    assert not info.is_all_greedy and not info.is_any_greedy

    # Non-contiguous, reordered keep indices stay aligned with the tensors.
    info = make(cls, [True, False, True, False, True])
    keep(info, [3, 0, 4])
    assert info.greedy_rows == [False, True, True] and rows_match_top_ks(info)
    assert not info.is_all_greedy and info.is_any_greedy
    keep(info, [2, 1])
    assert info.greedy_rows == [True, True] and rows_match_top_ks(info)
    assert info.is_all_greedy
    info.merge_batch(make(cls, [False, True]))
    keep(info, [2, 0])
    assert info.greedy_rows == [False, True] and rows_match_top_ks(info)
    assert not info.is_all_greedy and info.is_any_greedy

    # A side built without per-row flags falls back to the old behaviour:
    # flags combined on merge and left unchanged on filter.
    for left, right in ((True, False), (False, True), (False, False)):
        info = make(cls, [True, True], track=left)
        info.merge_batch(make(cls, [False], track=right))
        assert info.greedy_rows is None
        assert not info.is_all_greedy and info.is_any_greedy
        keep(info, [0, 1])
        assert info.greedy_rows is None and len(info) == 2
        assert not info.is_all_greedy and info.is_any_greedy

    # copy_for_forward() shares the list object: merge and filter must not
    # mutate it in place.
    info = make(cls, [True, True])
    shared = info.greedy_rows
    info.merge_batch(make(cls, [False]))
    assert shared == [True, True]
    shared = info.greedy_rows
    keep(info, [0])
    assert shared == [True, True, False]


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} SGLANG_PYTHON_ROOT")
    test(Path(sys.argv[1]))
    print("SGLang #44923 greedy-flag refresh behavior: PASS")


if __name__ == "__main__":
    main()
