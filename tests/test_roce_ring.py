#!/usr/bin/env python3
"""CPU contracts for RoCEnante and the shared Spark ring configuration."""

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT.parent / "config.yaml"


def _load(name, *candidates):
    for candidate in candidates:
        if candidate.is_file():
            spec = importlib.util.spec_from_file_location(name, candidate)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError(candidates)


paths = _load(
    "roce_ring_paths",
    ROOT / "runtime/b12x/b12x/comm/roce_ring/_path_config.py",
    Path("/opt/b12x/b12x/comm/roce_ring/_path_config.py"),
)
plan = _load("ring_mesh_plan", ROOT / "scripts/ring_mesh/plan.py")

# CPU fixture captured from sparkring's rocenante_native_path_arguments for the configured ring.
NATIVE = {
    "0": [
        "1,0,rocep1s0f0,3,1", "1,1,roceP2p1s0f0,3,1",
        "2,0,rocep1s0f0,3,2", "2,1,roceP2p1s0f1,3,2",
        "3,0,rocep1s0f1,3,1", "3,1,roceP2p1s0f1,3,1",
    ],
    "1": [
        "0,0,rocep1s0f0,3,1", "0,1,roceP2p1s0f0,3,1",
        "2,0,rocep1s0f1,3,1", "2,1,roceP2p1s0f1,3,1",
        "3,0,rocep1s0f1,3,2", "3,1,roceP2p1s0f0,3,2",
    ],
    "2": [
        "0,0,rocep1s0f1,3,2", "0,1,roceP2p1s0f0,3,2",
        "1,0,rocep1s0f1,3,1", "1,1,roceP2p1s0f1,3,1",
        "3,0,rocep1s0f0,3,1", "3,1,roceP2p1s0f0,3,1",
    ],
    "3": [
        "0,0,rocep1s0f1,3,1", "0,1,roceP2p1s0f1,3,1",
        "1,0,rocep1s0f0,3,2", "1,1,roceP2p1s0f1,3,2",
        "2,0,rocep1s0f0,3,1", "2,1,roceP2p1s0f0,3,1",
    ],
}
MAPS = "1=0/2,2=0/3,3=1/3;0=0/2,2=1/3,3=1/2;0=1/2,1=1/3,3=0/2;0=1/3,1=0/3,2=0/2"


def synthetic_inventory(node, ring):
    ports = {}
    for prefix, rail in (("fabric", "ports"), ("fabric2", "ports2")):
        for index in (0, 1):
            netdev = node[f"{prefix}_if{index}"]
            hca = node[f"{prefix}_hca{index}"]
            address = node[f"{prefix}_ip{index}"]
            ports[netdev] = {
                "rdma": hca,
                "ipv4": address,
                "mac": "02:00:00:00:00:01",
                "mtu": int(ring["mtu"]),
                "roce_mtu": str(ring["roce_mtu"]),
                "state": "up",
                "gid": "0000:0000:0000:0000:0000:0000:" + plan._gid_tail(address),
                "gid_netdev": netdev,
                "tc_offload": str(ring["hw_tc_offload"]),
                "hairpin_num_queues": str(ring["hairpin_num_queues"]),
                "hairpin_queue_size": str(ring["hairpin_queue_size"]),
                "flow_steering_mode": str(ring["flow_steering_mode"]),
            }
    return {"host": node["node"], "ports": ports}


class PeerMapEnv(unittest.TestCase):
    def setUp(self):
        self.saved = {key: os.environ.pop(key, None) for key in (paths.PEER_HCA_MAP_ENV, paths.PEER_HCA_MAPS_ENV)}

    def tearDown(self):
        for key, value in self.saved.items():
            os.environ.pop(key, None)
            if value is not None:
                os.environ[key] = value

    def test_per_rank_selection(self):
        os.environ[paths.PEER_HCA_MAPS_ENV] = MAPS
        self.assertEqual(paths.peer_hca_map(4, 0, 4), ((-1, -1), (0, 2), (0, 3), (1, 3)))
        self.assertEqual(paths.peer_hca_map(4, 3, 4), ((1, 3), (0, 3), (0, 2), (-1, -1)))

    def test_single_map_wins(self):
        os.environ[paths.PEER_HCA_MAPS_ENV] = MAPS
        os.environ[paths.PEER_HCA_MAP_ENV] = "1=0/2,2=0/3,3=1/3"
        self.assertEqual(paths.peer_hca_map(4, 0, 4)[1], (0, 2))

    def test_wrong_count_refuses(self):
        os.environ[paths.PEER_HCA_MAPS_ENV] = MAPS.rsplit(";", 1)[0]
        with self.assertRaises(ValueError):
            paths.peer_hca_map(4, 0, 4)

    def test_two_hca_default_unchanged(self):
        self.assertEqual(paths.peer_hca_map(4, 1, 2), ((0, 1), (-1, -1), (0, 1), (0, 1)))


class Plan(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.snapshot = plan.load_ring_snapshot(CONFIG)
        cls.nodes = cls.snapshot["nodes"]
        cls.ring = cls.snapshot["ring"]
        cls.order = plan.ring_order(cls.nodes)
        cls.hcas = cls.nodes[cls.order[0]]["hca_all"].split(",")

    def test_physical_order_and_oriented_ports_come_from_config(self):
        self.assertEqual(self.order, ["spark-a", "spark-b", "spark-c", "spark-d"])
        oriented = plan.ring_ports(self.ring["nodes"], self.order)
        self.assertEqual(oriented[("spark-a", "clockwise", 0)], "enp1s0f0np0")
        self.assertEqual(oriented[("spark-a", "counter_clockwise", 0)], "enp1s0f1np1")
        self.assertEqual(oriented[("spark-b", "clockwise", 1)], "enP2p1s0f1np1")

    def test_live_inventory_is_compared_with_config(self):
        node = self.nodes["spark-a"]
        inv = synthetic_inventory(node, self.ring)
        self.assertEqual(plan.check_host("spark-a", inv, node, self.ring), [])
        inv["ports"][node["fabric_if0"]]["ipv4"] = "192.0.2.1/30"
        problems = plan.check_host("spark-a", inv, node, self.ring)
        self.assertTrue(any("IPv4" in problem and "spark-a" in problem for problem in problems))

    def test_peer_maps_match_sparkring_fixture_and_physical_rank_order(self):
        physical_maps = plan.peer_maps(NATIVE, self.order, self.order, self.hcas)
        self.assertEqual(physical_maps, MAPS)
        target = plan.query_target(CONFIG, "deepseek_v4.1_flash_switchless")
        runtime_order = plan.physical_runtime_order(target, self.order)
        self.assertEqual(runtime_order, ["spark-a", "spark-b", "spark-c", "spark-d"])
        runtime_maps = plan.peer_maps(NATIVE, self.order, self.order, self.hcas)
        self.assertEqual(runtime_maps, physical_maps)
        ring_env = plan.ring_environment(target, runtime_maps, ",".join(self.hcas), str(self.ring["roce_gid_index"]))
        self.assertIn(f"B12X_ROCE_PEER_HCA_MAPS={physical_maps}", ring_env)
        self.assertIn(f"B12X_NEXT_ROCE_GID_INDEX={self.ring['roce_gid_index']}", ring_env)

    def test_mesh_planner_rejects_nonphysical_target_rank_order(self):
        with self.assertRaises(SystemExit):
            plan.physical_runtime_order(
                {"runtime_rank_order": "spark-a spark-d spark-c spark-b"},
                self.order,
            )

    def test_topology_uses_shared_management_interface_and_ring_parameters(self):
        inventories = {node: synthetic_inventory(record, self.ring) for node, record in self.nodes.items()}
        document = plan.topology(inventories, self.order, self.snapshot)
        self.assertEqual(document["ranks"][0]["management_netdev"], "wlP9s9")
        self.assertEqual(document["expected_ethernet_mtu"], self.ring["mtu"])
        self.assertEqual(document["roce_gid_index"], self.ring["roce_gid_index"])
        self.assertEqual(document["expected_roce_mtu"], self.ring["roce_mtu"])

    def test_plan_writer_supports_runtime_pathlib(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "env.txt"
            plan.write_text(output, "value=1\n")
            self.assertEqual(output.read_bytes(), b"value=1\n")


class ImageOverlay(unittest.TestCase):
    PYNCCL = Path("/sgl-workspace/sglang/python/sglang/srt/distributed/device_communicators/pynccl.py")

    @unittest.skipUnless(PYNCCL.is_file(), "image only")
    def test_overlay_selects_package(self):
        src = self.PYNCCL.read_text()
        self.assertIn('"b12x.comm.roce_ring" if os.environ.get("DSV41_ROCE_RING", "0") == "1" else "b12x.comm.roce"', src)
        self.assertNotIn("from b12x.comm import roce\n", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
