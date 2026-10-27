#!/usr/bin/env python3
"""Plan hardware-forwarded opposite-node paths from the parent Spark topology SSOT.

Example:
    python3 scripts/ring_mesh/plan.py --sparkring ~/sparkring --out ring-mesh

Physical rank order, interface/HCA names, addresses, GID and management SSH targets are read from
the parent ``spark/config.yaml`` through ``program.py ring-topology --json``. Live inventory is
read-only and only checks that each configured endpoint matches the host. FujitsuPolycom/sparkring's
planner then builds the per-rank /32 routes, hardware-only tc redirect rules and source markers.

The generated ``fabric.json`` and ``mesh-up-<node>.sh`` / ``mesh-down-<node>.sh`` files are
site-specific artifacts. The script does not emit runtime environment variables: the parent
``program.py query-model`` and ``spark/deploy.sh`` own B12X/NCCL maps and serving settings.
Nothing on the hosts is changed. See docs/switchless-ring.md for the installation boundary.
"""
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
DEFAULT_PROGRAM = SPARK_ROOT / "program.py"
PLAN_RANKS = 4
PLAN_RAILS = (("ports", "fabric"), ("ports2", "fabric2"))


def load_parent_topology(program=DEFAULT_PROGRAM, config=DEFAULT_CONFIG):
    """Read physical endpoints and management SSH targets from the parent Spark SSOT."""
    command = [
        sys.executable,
        str(program),
        "ring-topology",
        "--config",
        str(config),
        "--json",
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=60)
    if result.returncode:
        raise SystemExit(f"parent ring-topology failed ({result.returncode}): {result.stderr.strip()}")
    try:
        document = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"parent ring-topology returned invalid JSON: {exc}") from exc
    validate_topology(document)
    return document


def _endpoint_for_neighbor(node_name, node, plane, neighbor):
    endpoints = node.get(plane)
    if not isinstance(endpoints, list) or len(endpoints) != 2:
        raise SystemExit(f"{node_name}: common.ring.{plane} must contain two endpoints")
    matches = [endpoint for endpoint in endpoints if endpoint.get("neighbor") == neighbor]
    if len(matches) != 1:
        raise SystemExit(
            f"{node_name}: common.ring.{plane} must have exactly one endpoint to {neighbor}"
        )
    endpoint = matches[0]
    required = ("interface", "hca", "cidr", "neighbor_ip")
    missing = [field for field in required if not endpoint.get(field)]
    if missing:
        raise SystemExit(f"{node_name}: common.ring.{plane} endpoint to {neighbor} lacks {missing}")
    return endpoint


def validate_topology(document):
    """Require a reciprocal, two-plane four-rank ring before using live inventory."""
    if not isinstance(document, dict):
        raise SystemExit("parent ring-topology must be a JSON object")
    ring = document.get("ring")
    ring_nodes = ring.get("nodes") if isinstance(ring, dict) else None
    node_records = document.get("nodes")
    if not isinstance(ring_nodes, dict) or not isinstance(node_records, dict):
        raise SystemExit("parent ring-topology must include ring.nodes and nodes")
    if len(ring_nodes) != PLAN_RANKS or set(ring_nodes) != set(node_records):
        raise SystemExit(f"parent common.ring must cover exactly {PLAN_RANKS} nodes")
    try:
        ranked = sorted((int(record["rank"]), name) for name, record in ring_nodes.items())
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit("parent common.ring nodes must each declare an integer rank") from exc
    if [rank for rank, _name in ranked] != list(range(PLAN_RANKS)):
        raise SystemExit("parent common.ring ranks must be unique and exactly 0..3")

    order = [name for _rank, name in ranked]
    rank_by_node = {name: rank for rank, name in ranked}
    for name in order:
        node = ring_nodes[name]
        management = node_records[name]
        if not management.get("ssh") or not management.get("management_if"):
            raise SystemExit(f"{name}: parent topology lacks management SSH/interface data")
        for plane, _label in PLAN_RAILS:
            endpoints = node.get(plane)
            if not isinstance(endpoints, list) or len(endpoints) != 2:
                raise SystemExit(f"{name}: common.ring.{plane} must contain two endpoints")
            neighbors = [endpoint.get("neighbor") for endpoint in endpoints]
            expected_neighbors = {
                order[(rank_by_node[name] - 1) % PLAN_RANKS],
                order[(rank_by_node[name] + 1) % PLAN_RANKS],
            }
            if len(set(neighbors)) != 2 or set(neighbors) != expected_neighbors:
                raise SystemExit(
                    f"{name}: common.ring.{plane} neighbors do not match adjacent physical ranks"
                )
            for endpoint in endpoints:
                neighbor = endpoint.get("neighbor")
                peer = ring_nodes.get(neighbor)
                if peer is None:
                    raise SystemExit(f"{name}: {plane} references unknown neighbor {neighbor}")
                reverse = _endpoint_for_neighbor(neighbor, peer, plane, name)
                try:
                    local_interface = ipaddress.ip_interface(endpoint["cidr"])
                    peer_interface = ipaddress.ip_interface(reverse["cidr"])
                    expected_peer_ip = ipaddress.ip_address(endpoint["neighbor_ip"])
                    reverse_peer_ip = ipaddress.ip_address(reverse["neighbor_ip"])
                except (KeyError, ValueError, TypeError) as exc:
                    raise SystemExit(f"{name}: invalid {plane} address for neighbor {neighbor}") from exc
                if (
                    expected_peer_ip != peer_interface.ip
                    or reverse_peer_ip != local_interface.ip
                    or local_interface.network != peer_interface.network
                ):
                    raise SystemExit(
                        f"{name}: {plane} endpoint to {neighbor} is not reciprocal in common.ring"
                    )
    return order


def configured_endpoints(node):
    return [
        endpoint
        for plane, _label in PLAN_RAILS
        for endpoint in node[plane]
    ]


def inventory(host, gid_index, endpoints):
    """Collect only the interfaces/HCA pairs named by the parent physical topology."""
    arguments = [str(gid_index)] + [
        f"{endpoint['interface']}:{endpoint['hca']}" for endpoint in endpoints
    ]
    remote_command = shlex.join(["bash", "-s", "--", *arguments])
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", host, remote_command],
        input=(HERE / "inventory.sh").read_bytes(),
        capture_output=True,
        timeout=60,
        check=False,
    )
    if result.returncode:
        message = result.stderr.decode(errors="replace").strip()
        raise SystemExit(f"inventory failed on {host} ({result.returncode}): {message}")
    try:
        return json.loads(result.stdout.decode())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"inventory on {host} returned invalid JSON: {exc}") from exc


def _gid_tail(cidr):
    address = ipaddress.ip_interface(cidr).ip
    octets = [int(part) for part in str(address).split(".")]
    return f"{octets[0]:02x}{octets[1]:02x}:{octets[2]:02x}{octets[3]:02x}"


def check_host(host, inv, node, ring):
    """Compare live host state with, but never derive, the parent topology."""
    bad = []
    expected_endpoints = configured_endpoints(node)
    live_ports = inv.get("ports", {})
    expected_names = {endpoint["interface"] for endpoint in expected_endpoints}
    if set(live_ports) != expected_names:
        bad.append(f"{host}: inventory endpoints differ from common.ring")
    for endpoint in expected_endpoints:
        netdev = endpoint["interface"]
        port = live_ports.get(netdev)
        if port is None:
            continue
        if port.get("rdma") != endpoint["hca"]:
            bad.append(f"{host} {netdev}: HCA {port.get('rdma')} (common.ring says {endpoint['hca']})")
        if port.get("state") != "up":
            bad.append(f"{host} {netdev}: link {port.get('state')} (up expected)")
        try:
            actual_ip = ipaddress.ip_interface(port.get("ipv4", ""))
            expected_ip = ipaddress.ip_interface(endpoint["cidr"])
        except (ValueError, TypeError):
            bad.append(f"{host} {netdev}: missing/invalid IPv4 address")
        else:
            if actual_ip != expected_ip:
                bad.append(f"{host} {netdev}: IPv4 {actual_ip} (common.ring says {expected_ip})")
        if str(port.get("mtu")) != str(ring["mtu"]):
            bad.append(f"{host} {netdev}: MTU {port.get('mtu')} (common.ring says {ring['mtu']})")
        if str(port.get("roce_mtu")) != str(ring["roce_mtu"]):
            bad.append(
                f"{host} {netdev}: RoCE MTU {port.get('roce_mtu')} "
                f"(common.ring says {ring['roce_mtu']})"
            )
        if port.get("gid_netdev") != netdev or not port.get("gid", "").endswith(_gid_tail(endpoint["cidr"])):
            bad.append(f"{host} {netdev}: configured RoCE GID index does not map this IPv4 endpoint")
        if port.get("tc_offload") != str(ring["hw_tc_offload"]):
            bad.append(f"{host} {netdev}: hw-tc-offload {port.get('tc_offload')}")
        if port.get("flow_steering_mode") != str(ring["flow_steering_mode"]):
            bad.append(f"{host} {netdev}: flow_steering_mode {port.get('flow_steering_mode')}")
        for field in ("hairpin_num_queues", "hairpin_queue_size"):
            if str(port.get(field)) != str(ring[field]):
                bad.append(f"{host} {netdev}: {field} {port.get(field)} (common.ring says {ring[field]})")
    return bad


def topology(invs, document, order, state_root="/run/dsv41-mesh"):
    ring = document["ring"]
    node_records = document["nodes"]
    rank_by_node = {name: rank for rank, name in enumerate(order)}
    ranks = []
    for rank, name in enumerate(order):
        node = ring["nodes"][name]
        management = node_records[name]
        ports = {}
        for direction, offset in (("clockwise", 1), ("counter_clockwise", -1)):
            peer_rank = (rank + offset) % PLAN_RANKS
            peer_name = order[peer_rank]
            peer_direction = "counter_clockwise" if offset == 1 else "clockwise"
            ports[direction] = []
            for function, (plane, _label) in enumerate(PLAN_RAILS):
                endpoint = _endpoint_for_neighbor(name, node, plane, peer_name)
                live = invs[name]["ports"][endpoint["interface"]]
                local_ip = ipaddress.ip_interface(endpoint["cidr"]).ip
                ports[direction].append({
                    "function": function,
                    "netdev": endpoint["interface"],
                    "rdma_device": endpoint["hca"],
                    "ipv4_cidr": f"{local_ip}/32",
                    "mac": live["mac"],
                    "peer_rank": peer_rank,
                    "peer_direction": peer_direction,
                    "peer_function": function,
                })
        ranks.append({
            "rank": rank,
            "ssh_alias": management["ssh"],
            "management_netdev": management["management_if"],
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


def host_scripts(plan, fabric, host_rank):
    """(mesh-up, mesh-down) for one host from sparkring's RoCEnante plan."""
    up = ["#!/usr/bin/env bash",
          "# Hardware-forwarded opposite-node paths for RoCEnante (generated by scripts/ring_mesh/plan.py). Idempotent.",
          "set -euo pipefail"]
    down = ["#!/usr/bin/env bash", "# Removes exactly what mesh-up.sh creates.", "set -uo pipefail",
            "systemctl stop 'dsv41-mesh-marker@*' 2>/dev/null || true"]
    rules = [rule for rule in plan.tc_rules if rule.intermediate_rank == host_rank]
    qdiscs = sorted({rule.ingress_netdev for rule in rules})
    for netdev in qdiscs:
        up.append(f"tc qdisc show dev {netdev} | grep clsact >/dev/null || tc qdisc add dev {netdev} clsact")
    for rule in rules:
        add = fabric.tc_rule_command(rule, add=True)
        up.append(f"tc filter show dev {rule.ingress_netdev} ingress pref {rule.preference} | grep flower >/dev/null || {shlex.join(add)}")
        up.append(f"tc -s filter show dev {rule.ingress_netdev} ingress pref {rule.preference} | grep in_hw >/dev/null "
                  f"|| {{ echo 'rule pref {rule.preference} on {rule.ingress_netdev} is not in hardware'; exit 3; }}")
        down.append(f"tc filter del dev {rule.ingress_netdev} ingress pref {rule.preference} protocol 0x88b5 "
                    f"handle {rule.handle} flower 2>/dev/null || true")
    for netdev in qdiscs:
        down.append(f'[ -z "$(tc filter show dev {netdev} ingress; tc filter show dev {netdev} egress)" ] '
                    f"&& tc qdisc del dev {netdev} clsact 2>/dev/null || true")
    for route in (route for route in plan.routes if route.source_rank == host_rank):
        up.append(f"ip route show exact {route.destination_ipv4}/32 | grep . >/dev/null || {shlex.join(fabric.route_command(route, add=True))}")
        down.append(f"ip route del {route.destination_ipv4}/32 2>/dev/null || true")
    devices = sorted({marker.rdma_device for marker in plan.markers if marker.source_rank == host_rank})
    ports = sorted({marker.udp_source_port for marker in plan.markers})
    if ports != [65535]:
        raise SystemExit(f"unexpected marker UDP source ports {ports}; the marker unit assumes 65535")
    up.append(f"for d in {' '.join(devices)}; do systemctl start dsv41-mesh-marker@$d; done")
    up.append(f"sleep 2; for d in {' '.join(devices)}; do systemctl is-active --quiet dsv41-mesh-marker@$d "
              f"|| {{ echo marker $d not active; exit 4; }}; done")
    up.append("echo mesh-up OK")
    down.append("echo mesh-down done")
    return "\n".join(up) + "\n", "\n".join(down) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "hosts",
        nargs="*",
        help="deprecated audit list; if supplied it must exactly match the parent rank order",
    )
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="parent spark/config.yaml")
    ap.add_argument("--program", type=Path, default=DEFAULT_PROGRAM, help="parent spark/program.py")
    ap.add_argument("--sparkring", required=True, help="FujitsuPolycom/sparkring checkout")
    ap.add_argument("--out", default="ring-mesh", help="directory for generated host scripts")
    a = ap.parse_args()
    document = load_parent_topology(a.program, a.config)
    order = validate_topology(document)
    ssh_targets = [document["nodes"][name]["ssh"] for name in order]
    if a.hosts and a.hosts not in (order, ssh_targets):
        ap.error("positional hosts cannot override parent common.ring rank/SSH order")
    sys.path.insert(0, str(Path(a.sparkring) / "spark_transport/fabric/cx7_hairpin_diagonal"))
    import fabric  # sparkring's planner

    ring = document["ring"]
    invs = {
        name: inventory(
            document["nodes"][name]["ssh"],
            ring["roce_gid_index"],
            configured_endpoints(ring["nodes"][name]),
        )
        for name in order
    }
    problems = [
        problem
        for name in order
        for problem in check_host(
            document["nodes"][name]["ssh"],
            invs[name],
            ring["nodes"][name],
            ring,
        )
    ]
    if problems:
        raise SystemExit("hosts not ready for the mesh:\n  " + "\n  ".join(problems))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "fabric.json").write_text(
        json.dumps(topology(invs, document, order), indent=2) + "\n",
        newline="\n",
    )
    plan = fabric.build_rocenante_plan(fabric.build_plan(fabric.load_topology(out / "fabric.json")))
    for rank, h in enumerate(order):
        up, down = host_scripts(plan, fabric, rank)
        (out / f"mesh-up-{h}.sh").write_text(up, newline="\n")
        (out / f"mesh-down-{h}.sh").write_text(down, newline="\n")
    print(f"physical rank order (parent common.ring): {' -> '.join(order)} -> {order[0]}")
    print(
        f"live inventory matches configured fabric: MTU={ring['mtu']}, "
        f"RoCE MTU={ring['roce_mtu']}, GID index={ring['roce_gid_index']}"
    )
    print(
        f"wrote {out}/fabric.json and mesh-up-<node>.sh / mesh-down-<node>.sh; "
        "runtime env remains parent-managed"
    )


if __name__ == "__main__":
    main()
