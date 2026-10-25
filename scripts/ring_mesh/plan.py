#!/usr/bin/env python3
"""Plan RoCEnante opposite-node paths from the parent Spark ring configuration.

The physical topology, management interface, HCAs, MTU and GID settings are read from
``spark/config.yaml`` through ``spark/program.py ring-topology --json``. Live inventory is
compared with that source before sparkring builds routes, hardware rules and source markers.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import shlex
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPARK_ROOT = HERE.parents[2]
DEFAULT_CONFIG = SPARK_ROOT / "config.yaml"
DEFAULT_TARGET = "deepseek_v4.1_flash_switchless"
RING_ENV_KEYS = (
    "DSV41_ROCE_RING",
    "SGLANG_ROCE_ALLREDUCE",
    "SGLANG_ROCE_MAX_SIZE",
    "DSV41_ROCE_GATHER",
    "B12X_ROCE_HCA",
    "B12X_ROCE_GID_INDEX",
    "B12X_NEXT_ROCE_GID_INDEX",
    "B12X_ROCE_PEER_HCA_MAPS",
    "B12X_ROCE_TWO_WAVE_THRESHOLD_BYTES",
)


def load_ring_snapshot(config_path: Path = DEFAULT_CONFIG) -> dict:
    """Use the parent parser as the only source and validator for ring configuration."""
    result = subprocess.run(
        [
            sys.executable,
            str(SPARK_ROOT / "program.py"),
            "ring-topology",
            "--json",
            "--config",
            str(config_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(result.stderr.strip() or "无法读取 Spark 环网配置")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"spark/program.py 返回无效环网 JSON: {exc}") from exc


def query_target(config_path: Path, target: str) -> dict[str, str]:
    """读取 deploy.sh 同一 query-model 输出，避免 planner 复制 target 环境值。"""
    result = subprocess.run(
        [
            sys.executable,
            str(SPARK_ROOT / "program.py"),
            "query-model",
            "--config",
            str(config_path),
            "--target",
            target,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(result.stderr.strip() or f"无法查询 target {target}")
    return dict(line.split("\t", 1) for line in result.stdout.splitlines())


def physical_runtime_order(target: dict[str, str], physical_order: list[str]) -> list[str]:
    """Fail closed if a target query ever diverges from the physical rank SSOT."""
    runtime_order = str(target.get("runtime_rank_order", "")).split()
    if runtime_order != physical_order:
        raise SystemExit(
            "target runtime rank 顺序必须直接继承 common.ring 物理 rank 顺序: "
            + " ".join(physical_order)
        )
    return runtime_order


def inventory(host: str, node: dict, ring: dict) -> dict:
    """Read hardware state for configured ports; interface/HCA pairs come from the source."""
    ports = [
        (node[f"fabric_if{index}"], node[f"fabric_hca{index}"])
        for index in (0, 1)
    ] + [
        (node[f"fabric2_if{index}"], node[f"fabric2_hca{index}"])
        for index in (0, 1)
    ]
    remote_command = "bash -s -- " + shlex.join(
        [str(ring["roce_gid_index"]), *(f"{netdev}:{rdma}" for netdev, rdma in ports)]
    )
    output = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", host, remote_command],
        input=(HERE / "inventory.sh").read_bytes(),
        capture_output=True,
        timeout=60,
        check=True,
    ).stdout
    return json.loads(output.decode())


def _gid_tail(cidr: str) -> str:
    if not cidr:
        return "-"
    octets = str(ipaddress.ip_interface(cidr).ip).split(".")
    return f"{int(octets[0]):02x}{int(octets[1]):02x}:{int(octets[2]):02x}{int(octets[3]):02x}"


def check_host(host: str, inv: dict, node: dict, ring: dict) -> list[str]:
    """Compare live devices with common.ring; return actionable mismatches."""
    problems: list[str] = []
    expected_ports = [
        (node[f"fabric_if{index}"], node[f"fabric_hca{index}"], node[f"fabric_ip{index}"])
        for index in (0, 1)
    ] + [
        (node[f"fabric2_if{index}"], node[f"fabric2_hca{index}"], node[f"fabric2_ip{index}"])
        for index in (0, 1)
    ]
    for netdev, rdma, expected_ip in expected_ports:
        port = inv["ports"].get(netdev)
        if port is None:
            problems.append(f"{host} {netdev}: configured interface is missing")
            continue
        if port["rdma"] != rdma:
            problems.append(f"{host} {netdev}: RDMA device {port['rdma']} (expected {rdma})")
        try:
            actual_ip = ipaddress.ip_interface(port["ipv4"])
            configured_ip = ipaddress.ip_interface(expected_ip)
            if actual_ip != configured_ip:
                problems.append(f"{host} {netdev}: IPv4 {actual_ip} (expected {configured_ip})")
        except ValueError:
            problems.append(f"{host} {netdev}: invalid IPv4 address {port['ipv4']!r}")
        if port["mtu"] != int(ring["mtu"]):
            problems.append(f"{host} {netdev}: MTU {port['mtu']} (expected {ring['mtu']})")
        if port["roce_mtu"] != str(ring["roce_mtu"]):
            problems.append(f"{host} {netdev}: RoCE MTU {port['roce_mtu']} (expected {ring['roce_mtu']})")
        if port["gid_netdev"] != netdev or not port["gid"].endswith(_gid_tail(expected_ip)):
            problems.append(f"{host} {netdev}: RoCEv2 GID index {ring['roce_gid_index']} does not map this port's IPv4")
        if port["tc_offload"] != str(ring["hw_tc_offload"]):
            problems.append(f"{host} {netdev}: hw-tc-offload {port['tc_offload']} (expected {ring['hw_tc_offload']})")
        if port["hairpin_num_queues"] != str(ring["hairpin_num_queues"]):
            problems.append(f"{host} {netdev}: hairpin_num_queues {port['hairpin_num_queues']} (expected {ring['hairpin_num_queues']})")
        if port["hairpin_queue_size"] != str(ring["hairpin_queue_size"]):
            problems.append(f"{host} {netdev}: hairpin_queue_size {port['hairpin_queue_size']} (expected {ring['hairpin_queue_size']})")
        if port["flow_steering_mode"] != str(ring["flow_steering_mode"]):
            problems.append(f"{host} {netdev}: flow_steering_mode {port['flow_steering_mode']} (expected {ring['flow_steering_mode']})")
        if port["state"] != "up":
            problems.append(f"{host} {netdev}: link state {port['state']}")
    return problems


def ring_order(nodes: dict[str, dict]) -> list[str]:
    """Return physical rank order from the validated common.ring snapshot."""
    try:
        order = sorted(nodes, key=lambda name: int(nodes[name]["rank"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"环节点必须包含有效 rank: {exc}") from exc
    if len(order) != 4 or {int(nodes[name]["rank"]) for name in order} != set(range(4)):
        raise SystemExit("环网计划要求 common.ring rank 0..3")
    return order


def ring_ports(nodes: dict[str, dict], order: list[str]) -> dict[tuple[str, str, int], str]:
    """Orient configured ports by rank and require both PCIe planes to share each neighbor."""
    if len(order) != 4 or set(order) != set(nodes):
        raise SystemExit("ring_ports 的节点顺序必须唯一覆盖四个配置节点")
    oriented: dict[tuple[str, str, int], str] = {}
    for rank, host in enumerate(order):
        for direction, peer_rank in (("clockwise", (rank + 1) % 4), ("counter_clockwise", (rank - 1) % 4)):
            neighbor = order[peer_rank]
            for plane_index, field in enumerate(("ports", "ports2")):
                matches = [port["interface"] for port in nodes[host][field] if port["neighbor"] == neighbor]
                if len(matches) != 1:
                    raise SystemExit(f"{host}: {direction} neighbor {neighbor} has {len(matches)} ports in {field}")
                oriented[(host, direction, plane_index)] = matches[0]
    return oriented


def topology(invs: dict, order: list[str], snapshot: dict, state_root: str = "/run/dsv41-mesh") -> dict:
    """Build the sparkring document using configured edges and verified live port details."""
    nodes = snapshot["nodes"]
    ring = snapshot["ring"]
    oriented = ring_ports(ring["nodes"], order)
    ranks = []
    for rank, host in enumerate(order):
        ports = {}
        for direction in ("clockwise", "counter_clockwise"):
            ports[direction] = []
            peer_rank = (rank + (1 if direction == "clockwise" else -1)) % 4
            for function in (0, 1):
                netdev = oriented[(host, direction, function)]
                port = invs[host]["ports"][netdev]
                ports[direction].append({
                    "function": function,
                    "netdev": netdev,
                    "rdma_device": port["rdma"],
                    "ipv4_cidr": str(ipaddress.ip_interface(port["ipv4"]).ip) + "/32",
                    "mac": port["mac"],
                    "peer_rank": peer_rank,
                    "peer_direction": "counter_clockwise" if direction == "clockwise" else "clockwise",
                    "peer_function": function,
                })
        ranks.append({
            "rank": rank,
            "ssh_alias": host,
            "management_netdev": nodes[host]["management_if"],
            "ports": ports,
        })
    return {
        "schema": "sparkring-cx7-hardware-diagonal-fabric/v1",
        "status": "research-only",
        "group_id": 77,
        "socket_direct_functions": 2,
        "flow_label_base": 16383,
        "shared_diagonal_flow_label": True,
        "endpoint_route_strategy": "adjacent_gateway",
        "standard_ether_type": 2048,
        "marked_ether_type": 34997,
        "standard_udp_destination_port": 4791,
        "roce_gid_index": int(ring["roce_gid_index"]),
        "bounded_runtime_seconds": 7200,
        "expected_ethernet_mtu": int(ring["mtu"]),
        "expected_roce_mtu": int(ring["roce_mtu"]),
        "orchestration": {
            "host_helper_path": "/UNSUPPORTED/external-fabric-orchestrator",
            "remote_state_root": state_root,
        },
        "ranks": ranks,
    }


def peer_maps(native_args: dict, order: list[str], hosts: list[str], hcas: list[str]) -> str:
    """Translate sparkring path devices to B12X indices for the target runtime rank order."""
    if len(set(order)) != 4 or len(set(hosts)) != 4 or len(hcas) != 4:
        raise SystemExit("B12X peer-map translation requires four unique nodes and four configured HCAs")
    tp_of = {host: hosts.index(host) for host in hosts}
    maps: list[str | None] = [None] * 4
    for ring_rank, specs in native_args.items():
        paths: dict[int, dict[int, int]] = {}
        for spec in specs:
            dest, function, rdma, _gid, _hops = spec.split(",")
            try:
                hca_index = hcas.index(rdma)
            except ValueError as exc:
                raise SystemExit(f"sparkring returned HCA {rdma} absent from common.ring endpoint HCA mappings") from exc
            paths.setdefault(tp_of[order[int(dest)]], {})[int(function)] = hca_index
        maps[tp_of[order[int(ring_rank)]]] = ",".join(
            f"{peer}={pair[0]}/{pair[1]}" for peer, pair in sorted(paths.items())
        )
    if any(row is None for row in maps):
        raise SystemExit("sparkring planner omitted a physical rank in B12X peer maps")
    return ";".join(row for row in maps if row is not None)


def ring_environment(target_query: dict[str, str], generated_maps: str, configured_hcas: str, gid_index: str) -> str:
    """Reuse query-model ring values and verify external planner paths against the same source."""
    values = dict(
        token.split("=", 1)
        for token in target_query["engine_extra_container_env"].split()
        if "=" in token
    )
    expected = {
        "B12X_ROCE_HCA": configured_hcas,
        "B12X_ROCE_GID_INDEX": gid_index,
        "B12X_NEXT_ROCE_GID_INDEX": gid_index,
        "B12X_ROCE_PEER_HCA_MAPS": generated_maps,
    }
    mismatches = {key: (values.get(key), value) for key, value in expected.items() if values.get(key) != value}
    if mismatches:
        raise SystemExit(f"sparkring 输出与 query-model/common.ring 不一致: {mismatches}")
    return " ".join(f"{key}={values[key]}" for key in RING_ENV_KEYS if key in values)


def write_text(path: Path, content: str) -> None:
    """Write UTF-8 text with an explicit trailing newline from its caller."""
    path.write_text(content, encoding="utf-8")


def host_scripts(plan, fabric, host_rank: int) -> tuple[str, str]:
    """Render idempotent per-node mesh-up/down scripts from sparkring's plan."""
    up = [
        "#!/usr/bin/env bash",
        "# Hardware-forwarded opposite-node paths for RoCEnante (generated by scripts/ring_mesh/plan.py). Idempotent.",
        "set -euo pipefail",
    ]
    down = ["#!/usr/bin/env bash", "# Removes exactly what mesh-up.sh creates.", "set -uo pipefail",
            "systemctl stop 'dsv41-mesh-marker@*' 2>/dev/null || true"]
    rules = [rule for rule in plan.tc_rules if rule.intermediate_rank == host_rank]
    qdiscs = sorted({rule.ingress_netdev for rule in rules})
    for netdev in qdiscs:
        up.append(f"tc qdisc show dev {netdev} | grep -q clsact || tc qdisc add dev {netdev} clsact")
    for rule in rules:
        add = fabric.tc_rule_command(rule, add=True)
        up.append(f"tc filter show dev {rule.ingress_netdev} ingress pref {rule.preference} | grep -q flower || {shlex.join(add)}")
        up.append(f"tc -s filter show dev {rule.ingress_netdev} ingress pref {rule.preference} | grep -q in_hw "
                  f"|| {{ echo 'rule pref {rule.preference} on {rule.ingress_netdev} is not in hardware'; exit 3; }}")
        down.append(f"tc filter del dev {rule.ingress_netdev} ingress pref {rule.preference} protocol 0x88b5 "
                    f"handle {rule.handle} flower 2>/dev/null || true")
    for netdev in qdiscs:
        down.append(f'[ -z "$(tc filter show dev {netdev} ingress; tc filter show dev {netdev} egress)" ] '
                    f"&& tc qdisc del dev {netdev} clsact 2>/dev/null || true")
    for route in (route for route in plan.routes if route.source_rank == host_rank):
        up.append(f"ip route show exact {route.destination_ipv4}/32 | grep -q . || {shlex.join(fabric.route_command(route, add=True))}")
        down.append(f"ip route del {route.destination_ipv4}/32 2>/dev/null || true")
    devices = sorted({marker.rdma_device for marker in plan.markers if marker.source_rank == host_rank})
    ports = sorted({marker.udp_source_port for marker in plan.markers})
    if ports != [65535]:
        raise SystemExit(f"unexpected marker UDP source ports {ports}; the marker unit assumes 65535")
    up.append(f"for device in {' '.join(devices)}; do systemctl start dsv41-mesh-marker@$device; done")
    up.append(f"sleep 2; for device in {' '.join(devices)}; do systemctl is-active --quiet dsv41-mesh-marker@$device "
              f"|| {{ echo marker $device not active; exit 4; }}; done")
    up.append("echo mesh-up OK")
    down.append("echo mesh-down done")
    return "\n".join(up) + "\n", "\n".join(down) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("hosts", nargs="*", help="optional SSH overrides in common.ring physical rank order")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="parent Spark config.yaml")
    parser.add_argument("--target", default=DEFAULT_TARGET, help="target ID whose runtime order/env are planned")
    parser.add_argument("--sparkring", required=True, help="FujitsuPolycom/sparkring checkout")
    parser.add_argument("--out", default="ring-mesh")
    arguments = parser.parse_args()

    config_path = Path(arguments.config)
    snapshot = load_ring_snapshot(config_path)
    target = query_target(config_path, arguments.target)
    nodes, ring = snapshot["nodes"], snapshot["ring"]
    physical_order = ring_order(nodes)
    runtime_order = physical_runtime_order(target, physical_order)
    if arguments.hosts and len(arguments.hosts) != 4:
        raise SystemExit("SSH override 必须按 common.ring 物理 rank 顺序提供四个地址")
    hosts_by_node = {
        node: endpoint
        for node, endpoint in zip(physical_order, arguments.hosts or [nodes[node]["ssh"] for node in physical_order])
    }
    invs = {
        node: inventory(hosts_by_node[node], nodes[node], ring)
        for node in physical_order
    }
    problems = [
        problem
        for node in physical_order
        for problem in check_host(hosts_by_node[node], invs[node], nodes[node], ring)
    ]
    if problems:
        raise SystemExit("live inventory does not match common.ring:\n  " + "\n  ".join(problems))

    sys.path.insert(0, str(Path(arguments.sparkring) / "spark_transport/fabric/cx7_hairpin_diagonal"))
    import fabric  # sparkring's tested route/tc planner

    output_dir = Path(arguments.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    fabric_document = topology(invs, physical_order, snapshot)
    write_text(output_dir / "fabric.json", json.dumps(fabric_document, indent=2) + "\n")
    plan = fabric.build_rocenante_plan(fabric.build_plan(fabric.load_topology(output_dir / "fabric.json")))
    for rank, node in enumerate(physical_order):
        up, down = host_scripts(plan, fabric, rank)
        write_text(output_dir / f"mesh-up-{node}.sh", up)
        write_text(output_dir / f"mesh-down-{node}.sh", down)

    hcas = nodes[physical_order[0]]["hca_all"].split(",")
    maps = peer_maps(fabric.rocenante_native_path_arguments(plan), physical_order, physical_order, hcas)
    env = ring_environment(target, maps, ",".join(hcas), str(ring["roce_gid_index"]))
    write_text(output_dir / "env.txt", env + "\n")
    print(f"physical ring ranks: {' -> '.join(physical_order)} -> {physical_order[0]}")
    print(f"target runtime ranks: {' -> '.join(runtime_order)}")
    print(f"verified ring hardware: MTU={ring['mtu']} RoCE-MTU={ring['roce_mtu']} GID={ring['roce_gid_index']} "
          f"hairpin={ring['hairpin_queue_size']}")
    print(f"wrote {output_dir}/fabric.json, mesh-up-*.sh, mesh-down-*.sh, env.txt\nEXTRA_CONTAINER_ENV additions:\n  {env}")


if __name__ == "__main__":
    main()
