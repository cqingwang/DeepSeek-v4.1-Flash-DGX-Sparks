"""Keep the local ring image free of unused or ring-incompatible runtime branches."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class LocalRingRuntimeScopeTests(unittest.TestCase):
    def test_switch_fabric_two_shot_copy_is_not_shipped(self):
        self.assertFalse((ROOT / "adapter/roce_twoshot.py").exists())
        self.assertFalse((ROOT / "runtime/roce_twoshot_adapt.py").exists())
        self.assertFalse((ROOT / "runtime/b12x/b12x/comm/roce_twoshot").exists())
        self.assertFalse((ROOT / "runtime/b12x/twoshot-provenance.json").exists())
        self.assertFalse((ROOT / "tests/test_roce_twoshot.py").exists())

    def test_unused_prefill_only_hooks_are_not_shipped(self):
        self.assertFalse((ROOT / "adapter/moe_launch_prepared.py").exists())
        self.assertFalse((ROOT / "tests/test_moe_launch_prepared.py").exists())
        self.assertFalse((ROOT / "tests/test_indexer_nosync.py").exists())
        self.assertFalse((ROOT / "tests/gpu_test_indexer_nosync.py").exists())

        indexer = (ROOT / "adapter/indexer_chunked_v3.py").read_text()
        moe_adapter = (ROOT / "adapter/moe_b12x_next.py").read_text()
        ab_registry = (ROOT / "adapter/ab_variant.py").read_text()
        dockerfile = (ROOT / "Dockerfile.canary-roce").read_text()
        for source in (indexer, moe_adapter, ab_registry):
            self.assertNotIn("DSV41_INDEXER_NOSYNC", source)
            self.assertNotIn("DSV41_MOE_LAUNCH_PREPARED", source)
            self.assertNotIn("DSV41_ROCE_TWOSHOT", source)
        self.assertNotIn("test_indexer_nosync.py", dockerfile)
        self.assertNotIn("test_moe_launch_prepared.py", dockerfile)
        self.assertNotIn("roce_twoshot_adapt.py", dockerfile)
        self.assertIn("test_local_ring_runtime_scope.py", dockerfile)


if __name__ == "__main__":
    unittest.main()
