#!/usr/bin/env python3
"""CPU checks for RoCEnante on a switchless ring (b12x.comm.roce_ring + spark/ring/mesh/plan.py).

- B12X_ROCE_PEER_HCA_MAPS: one ';'-separated map per rank is picked by rank, B12X_ROCE_PEER_HCA_MAP
  still wins, a wrong count refuses.
- ring_mesh planner topology fixtures are synthetic: rank order and reciprocal two-plane
  endpoints come from a parent-shaped config; live inventory can only validate those values.
  Duplicate ranks, non-neighbours, non-reciprocal IPs and inventory drift are rejected.
- every generated opposite-node path preserves PCIe-function identity; these tests do not encode
  a specific site's host, interface, HCA, GID or management address.
- in the image: the RoCEnante overlay selects the package from DSV41_ROCE_RING.

No GPU, torch or network. usage: python3 tests/test_roce_ring.py
"""
import ast
import hashlib
import importlib.util
import ipaddress
import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT.parent / "config.yaml"
ROCE_RING_CANDIDATES = (
    ROOT / "runtime/b12x/b12x/comm/roce_ring",
    ROOT.parent / "ring/b12x/comm/roce_ring",
    Path("/opt/b12x/b12x/comm/roce_ring"),
)
ROCE_RING = next(
    (path for path in ROCE_RING_CANDIDATES if (path / "_path_config.py").is_file()),
    ROCE_RING_CANDIDATES[0],
)
PROVENANCE_CANDIDATES = (
    ROOT.parent / "ring/b12x/roce_ring-provenance.json",
    ROOT / "runtime/b12x/roce_ring-provenance.json",
)
PROVENANCE_PATH = next((path for path in PROVENANCE_CANDIDATES if path.is_file()), PROVENANCE_CANDIDATES[0])
UINT32_MASK = 0xFFFFFFFF
CTRL = 0x1000
EPOCH = 0x2000


def _load(name, *candidates):
    for c in candidates:
        if c.is_file():
            spec = importlib.util.spec_from_file_location(name, c)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError(candidates)


paths = _load(
    "roce_ring_paths",
    *(path / "_path_config.py" for path in ROCE_RING_CANDIDATES),
)
plan = _load("ring_mesh_plan", ROOT.parent / "ring/mesh/plan.py")


class Uint32(int):
    def __new__(cls, value):
        return super().__new__(cls, int(value) & UINT32_MASK)

    def __add__(self, other):
        return Uint32(int(self) + int(other))


def simulate_timeout(filename, paths, sequence, timeout):
    """Run one kernel's timeout control flow against CPU-backed words."""
    source = ROCE_RING / filename
    tree = ast.parse(source.read_text())
    kernels = [
        node
        for cls in tree.body
        if isinstance(cls, ast.ClassDef)
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "kernel"
    ]
    if len(kernels) != 1:
        raise AssertionError(f"expected one kernel in {source}, found {len(kernels)}")
    kernel = kernels[0]
    kernel.decorator_list = []
    for argument in kernel.args.args:
        argument.annotation = None
    kernel.returns = None
    memory = {EPOCH: Uint32(sequence - 1)}
    waits = []

    def load(address):
        return Uint32(memory.get(address, 0))

    def store(address, value):
        memory[address] = Uint32(value)

    def atomic_add(address, value):
        previous = load(address)
        store(address, previous + value)
        return previous

    def spin(address, expected, limit):
        waits.append((address, int(expected), int(limit)))
        return Uint32(timeout)

    namespace = {
        "Uint32": Uint32,
        "Int32": int,
        "Int64": int,
        "PACK_BYTES": 16,
        "PATH_COUNT": 2,
        "cute": SimpleNamespace(
            arch=SimpleNamespace(
                thread_idx=lambda: (0, 0, 0),
                block_idx=lambda: (0, 0, 0),
                grid_dim=lambda: (1, 1, 1),
                sync_threads=lambda: None,
            )
        ),
        "cutlass": SimpleNamespace(const_expr=bool, range_constexpr=range),
        "ld_relaxed_gpu_u32": load,
        "ld_relaxed_sys_u32": load,
        "st_relaxed_sys_u32": store,
        "st_release_sys_u32": store,
        "st_release_gpu_u32": store,
        "atomic_add_relaxed_gpu_u32": atomic_add,
        "fence_sc_sys": lambda: None,
        "fence_sc_gpu": lambda: None,
        "spin_until_eq_acquire_sys": spin,
    }
    exec(compile(ast.Module(body=[kernel], type_ignores=[]), str(source), "exec"), namespace)
    runtime = SimpleNamespace(
        _threads=32,
        _world_size=4,
        _rank=1,
        _slots=2,
        _opposite_paths=paths,
        _flag_stride=128,
    )
    pointer = SimpleNamespace(toint=lambda: 0x3000)
    arguments = dict(
        self=runtime,
        input_ptr=pointer,
        output_ptr=pointer,
        nbytes=0,
        recv_base=0x4000,
        flag_base=0x8000,
        send_base=0xC000,
        ctrl_base=CTRL,
        slot_bytes=16,
        epoch_ptr=EPOCH,
        spin_limit=Uint32(1),
    )
    if filename == "_allgather_cute.py":
        arguments.update(shard_packs=0, row_packs=1)
    else:
        arguments.update(size_packs=0)
    namespace["kernel"](**arguments)
    if not waits:
        raise AssertionError("the simulated thread did not execute a peer wait")
    if any(expected != sequence for _, expected, _ in waits):
        raise AssertionError(f"kernel waited for a sequence other than {sequence}")
    return memory

MAPS = "1=1/3,2=0/3,3=0/2;0=0/2,2=1/3,3=1/2;0=1/2,1=0/2,3=1/3;0=1/3,1=0/3,2=0/2"


def synthetic_topology():
    names = [f"fabric-rank-{rank}" for rank in range(4)]
    ring_nodes = {}
    host_nodes = {}
    for rank, name in enumerate(names):
        endpoints_by_plane = {"ports": [], "ports2": []}
        neighbor_ranks = [(rank + 1) % 4, (rank - 1) % 4]
        if rank % 2:
            neighbor_ranks.reverse()
        for plane_index, plane in enumerate(("ports", "ports2")):
            for endpoint_index, neighbor_rank in enumerate(neighbor_ranks):
                low, high = sorted((rank, neighbor_rank))
                subnet = ipaddress.ip_network(
                    f"10.{210 + plane_index}.{low * 4 + high + 1}.0/30"
                )
                local_ip = subnet.network_address + (1 if rank == low else 2)
                peer_ip = subnet.network_address + (2 if rank == low else 1)
                endpoints_by_plane[plane].append({
                    "interface": f"fabric{plane_index}-rank{rank}-port{endpoint_index}",
                    "hca": f"rdma{plane_index}-rank{rank}-port{endpoint_index}",
                    "cidr": f"{local_ip}/30",
                    "neighbor": names[neighbor_rank],
                    "neighbor_ip": str(peer_ip),
                })
        ring_nodes[name] = {"rank": rank, **endpoints_by_plane}
        host_nodes[name] = {
            "rank": str(rank),
            "ssh": f"test-user@fabric-rank-{rank}.invalid",
            "management_if": f"management-rank-{rank}",
        }
    return {
        "nodes": host_nodes,
        "ring": {
            "nodes": ring_nodes,
            "mtu": 9000,
            "roce_mtu": 4096,
            "roce_gid_index": 7,
            "hw_tc_offload": "on",
            "flow_steering_mode": "hmfs",
            "hairpin_num_queues": 4,
            "hairpin_queue_size": 8192,
            "mesh": {
                "runtime_root": "/opt/spark-ring-mesh",
                "state_root": "/run/spark-ring-mesh",
                "sparkring_commit": "f16b5f43208c188c018c43820ccf52955006d1a5",
                "probe_source_sha256": "8684a6961b8e86aa474fa2310ff71e4cdf219a63a72ceb5593b2f95e54812792",
            },
        },
    }


def synthetic_inventory(document, node_name):
    ring = document["ring"]
    ports = {}
    for endpoint in plan.configured_endpoints(ring["nodes"][node_name]):
        ports[endpoint["interface"]] = {
            "rdma": endpoint["hca"],
            "ipv4": endpoint["cidr"],
            "mac": f"02:00:00:00:{ring['nodes'][node_name]['rank']:02x}:01",
            "mtu": ring["mtu"],
            "roce_mtu": str(ring["roce_mtu"]),
            "state": "up",
            "gid": f"0000:0000:0000:0000:0000:ffff:{plan._gid_tail(endpoint['cidr'])}",
            "gid_netdev": endpoint["interface"],
            "tc_offload": ring["hw_tc_offload"],
            "hairpin_num_queues": str(ring["hairpin_num_queues"]),
            "hairpin_queue_size": str(ring["hairpin_queue_size"]),
            "flow_steering_mode": ring["flow_steering_mode"],
        }
    return {"host": node_name, "ports": ports}


class PeerMapEnv(unittest.TestCase):
    def setUp(self):
        self.saved = {k: os.environ.pop(k, None) for k in (paths.PEER_HCA_MAP_ENV, paths.PEER_HCA_MAPS_ENV)}

    def tearDown(self):
        for k, v in self.saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v

    def test_per_rank_selection(self):
        os.environ[paths.PEER_HCA_MAPS_ENV] = MAPS
        self.assertEqual(paths.peer_hca_map(4, 0, 4), ((-1, -1), (1, 3), (0, 3), (0, 2)))
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
    def setUp(self):
        self.document = synthetic_topology()
        self.order = plan.validate_topology(self.document)
        self.invs = {
            name: synthetic_inventory(self.document, name)
            for name in self.order
        }

    def test_rank_order_and_peer_endpoints_come_from_parent_topology(self):
        self.assertEqual(self.order, [f"fabric-rank-{rank}" for rank in range(4)])
        for name in self.order:
            self.assertEqual(
                plan.check_host(
                    self.document["nodes"][name]["ssh"],
                    self.invs[name],
                    self.document["ring"]["nodes"][name],
                    self.document["ring"],
                ),
                [],
            )
        fabric = plan.topology(
            self.invs,
            self.document,
            self.order,
            self.document["ring"]["mesh"]["state_root"],
        )
        self.assertEqual(fabric["roce_gid_index"], 7)
        self.assertEqual(fabric["expected_ethernet_mtu"], 9000)
        for rank, name in enumerate(self.order):
            generated = fabric["ranks"][rank]
            self.assertEqual(generated["ssh_alias"], name)
            self.assertEqual(generated["management_netdev"], f"management-rank-{rank}")
            self.assertEqual(generated["ports"]["clockwise"][0]["peer_rank"], (rank + 1) % 4)
            self.assertEqual(generated["ports"]["counter_clockwise"][0]["peer_rank"], (rank - 1) % 4)
            for direction in ("clockwise", "counter_clockwise"):
                self.assertEqual(
                    {endpoint["function"] for endpoint in generated["ports"][direction]},
                    {0, 1},
                )

    def test_parent_topology_rejects_nonreciprocal_address_and_rank(self):
        broken_address = synthetic_topology()
        broken_address["ring"]["nodes"]["fabric-rank-0"]["ports"][0]["neighbor_ip"] = "203.0.113.99"
        with self.assertRaisesRegex(SystemExit, "not reciprocal"):
            plan.validate_topology(broken_address)

        duplicate_rank = synthetic_topology()
        duplicate_rank["ring"]["nodes"]["fabric-rank-3"]["rank"] = 2
        with self.assertRaisesRegex(SystemExit, "unique and exactly 0..3"):
            plan.validate_topology(duplicate_rank)

        non_neighbor = synthetic_topology()
        non_neighbor["ring"]["nodes"]["fabric-rank-0"]["ports"][0]["neighbor"] = "fabric-rank-2"
        with self.assertRaisesRegex(SystemExit, "neighbors do not match"):
            plan.validate_topology(non_neighbor)

    def test_live_inventory_only_validates_and_never_redefines_topology(self):
        name = self.order[0]
        live = self.invs[name]
        endpoint = self.document["ring"]["nodes"][name]["ports"][0]
        live["ports"][endpoint["interface"]]["ipv4"] = "198.51.100.1/30"
        live["ports"][endpoint["interface"]]["gid_netdev"] = "unexpected-interface"
        errors = plan.check_host(
            self.document["nodes"][name]["ssh"],
            live,
            self.document["ring"]["nodes"][name],
            self.document["ring"],
        )
        self.assertTrue(any("IPv4" in error for error in errors), errors)
        self.assertTrue(any("GID index" in error for error in errors), errors)


class TimeoutContract(unittest.TestCase):
    def test_provenance_hashes_cover_the_current_overlay(self):
        provenance = json.loads(PROVENANCE_PATH.read_text())
        self.assertEqual(set(provenance["source_files_sha256"]), {
            path.name for path in ROCE_RING.iterdir() if path.is_file()
        })
        package_root = ROCE_RING.parents[1]
        for relative_path, expected in provenance["deployed_files_sha256"].items():
            actual = hashlib.sha256((package_root / relative_path).read_bytes()).hexdigest()
            self.assertEqual(actual, expected, relative_path)

    def test_timeout_remains_visible_across_uint32_wrap(self):
        for filename in ("_allgather_cute.py", "_oneshot_cute.py"):
            for paths in (2, 4):
                for sequence in (1, 0x80000000, 0xFFFFFFFF, 0):
                    with self.subTest(filename=filename, paths=paths, sequence=sequence):
                        memory = simulate_timeout(filename, paths, sequence, timeout=True)
                        self.assertEqual(memory.get(CTRL + 24, 0), 1)
                        self.assertEqual(memory.get(EPOCH + 12, 0), 1)
                        self.assertEqual(memory[CTRL + 8], sequence)
                        self.assertEqual(memory[EPOCH], (sequence - 1) & UINT32_MASK)

    def test_successful_wrap_advances_epoch(self):
        for filename in ("_allgather_cute.py", "_oneshot_cute.py"):
            for paths in (2, 4):
                with self.subTest(filename=filename, paths=paths):
                    memory = simulate_timeout(filename, paths, 0, timeout=False)
                    self.assertEqual(memory[EPOCH], 0)
                    self.assertEqual(memory.get(CTRL + 24, 0), 0)
                    self.assertEqual(memory.get(EPOCH + 12, 0), 0)

    def test_proxy_diagnostic_counters_are_atomic(self):
        source = (ROCE_RING / "_roce_proxy.c").read_text()
        self.assertIn("atomic_uint_least32_t last_seq;", source)
        self.assertIn("atomic_uint_fast64_t ops_posted;", source)
        self.assertIn("atomic_uint_fast64_t writes_completed;", source)
        self.assertNotIn("\n    uint32_t last_seq;", source)
        self.assertNotIn("\n    uint64_t ops_posted;", source)
        self.assertNotIn("\n    uint64_t writes_completed;", source)

    def test_timeout_flag_has_a_distinct_control_word(self):
        runtime = (ROCE_RING / "roce_oneshot.py").read_text()
        self.assertIn("self._layout.ctrl_off + 28", runtime)
        self.assertIn("int(self._ctrl_np[6]) != 0", runtime)
        self.assertIn('"timed_out": bool(self._ctrl_words[6].item())', runtime)

    def test_host_health_uses_flag_and_unsigned_sequence(self):
        source = ROCE_RING / "roce_oneshot.py"
        tree = ast.parse(source.read_text())
        owner = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "RoceOneshotAllReduce"
        )
        selected = []
        for node in owner.body:
            if isinstance(node, ast.FunctionDef) and node.name in {"check_health", "poisoned"}:
                node.decorator_list = []
                selected.append(node)
        namespace = {}
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"), namespace)
        for sequence in (0, 1, 0x80000000, 0xFFFFFFFF):
            signed = sequence if sequence < 0x80000000 else sequence - (1 << 32)
            runtime = SimpleNamespace(
                _proxy=None,
                _ctrl_np=[0, 0, signed, 2, 0, 0, 1],
                rank=1,
            )
            with self.subTest(sequence=sequence):
                self.assertTrue(namespace["poisoned"](runtime))
                with self.assertRaisesRegex(RuntimeError, f"at sequence {sequence};"):
                    namespace["check_health"](runtime)
        runtime._ctrl_np = [0] * 7
        self.assertFalse(namespace["poisoned"](runtime))
        namespace["check_health"](runtime)


class ImageOverlay(unittest.TestCase):
    PYNCCL = Path("/sgl-workspace/sglang/python/sglang/srt/distributed/device_communicators/pynccl.py")

    @unittest.skipUnless(PYNCCL.is_file(), "image only")
    def test_overlay_selects_package(self):
        src = self.PYNCCL.read_text()
        self.assertIn('"b12x.comm.roce_ring" if os.environ.get("DSV41_ROCE_RING", "0") == "1" else "b12x.comm.roce"', src)
        self.assertNotIn("from b12x.comm import roce\n", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
