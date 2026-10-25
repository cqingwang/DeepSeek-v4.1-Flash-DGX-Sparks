"""Pure configuration contract for RoCEnante peer paths."""

from __future__ import annotations

import os
from collections.abc import Sequence


PATH_COUNT = 2
MAX_PATH_COUNT = 4
PEER_HCA_MAP_ENV = "B12X_ROCE_PEER_HCA_MAP"
PEER_HCA_MAPS_ENV = "B12X_ROCE_PEER_HCA_MAPS"
OPPOSITE_PATHS_ENV = "B12X_ROCE_OPPOSITE_PATHS"
def opposite_path_count(world_size: int, hca_count: int) -> int:
    """Resolve the research-only opposite-rank path count from the environment."""

    raw = os.getenv(OPPOSITE_PATHS_ENV, "").strip()
    if not raw:
        return PATH_COUNT
    try:
        count = int(raw)
    except ValueError as exc:
        raise ValueError(f"{OPPOSITE_PATHS_ENV} must be 2 or 4") from exc
    if count not in (PATH_COUNT, MAX_PATH_COUNT):
        raise ValueError(f"{OPPOSITE_PATHS_ENV} must be 2 or 4")
    if count == MAX_PATH_COUNT and (world_size != 4 or hca_count != 4):
        raise RuntimeError(
            f"{OPPOSITE_PATHS_ENV}=4 requires four ranks and four active RDMA devices"
        )
    return count


def peer_path_count(
    world_size: int, rank: int, peer: int, opposite_paths: int
) -> int:
    """Return two paths for a neighbor and the configured count for the TP4 opposite."""

    if peer == rank:
        return 0
    if world_size == 4 and (peer - rank) % world_size == 2:
        return opposite_paths
    return PATH_COUNT


def peer_hca_map(
    world_size: int,
    rank: int,
    hca_count: int,
    opposite_paths: int = PATH_COUNT,
) -> tuple[tuple[int, ...], ...]:
    """Resolve local HCA indices per peer from ``B12X_ROCE_PEER_HCA_MAP``.

    Entries use ``peer=hca0/hca1`` for neighbors. With four opposite paths,
    the opposite-rank entry has four distinct HCA indices. A two-HCA runtime
    defaults every non-local peer to ``0/1``; runtimes with more HCAs require
    an explicit mapping because local and reciprocal indices can differ.
    ``B12X_ROCE_PEER_HCA_MAPS`` carries one such map per rank, ``;``-separated in rank order, so a
    single launcher environment serves every rank; ``B12X_ROCE_PEER_HCA_MAP`` wins when both are set.
    """

    if hca_count < PATH_COUNT:
        raise RuntimeError(
            f"RoCE two-path transport needs at least {PATH_COUNT} active RDMA devices, "
            f"got {hca_count}"
        )
    raw = os.getenv(PEER_HCA_MAP_ENV, "").strip()
    per_rank = os.getenv(PEER_HCA_MAPS_ENV, "").strip()
    if not raw and per_rank:
        maps = [item.strip() for item in per_rank.split(";")]
        if len(maps) != world_size:
            raise ValueError(
                f"{PEER_HCA_MAPS_ENV} needs {world_size} ';'-separated maps, got {len(maps)}"
            )
        raw = maps[rank]
    if not raw:
        if hca_count == PATH_COUNT:
            return tuple(
                (-1, -1) if peer == rank else (0, 1)
                for peer in range(world_size)
            )
        raise RuntimeError(
            f"{PEER_HCA_MAP_ENV} is required when more than two RDMA devices are selected; "
            "use peer=path0/path1 entries such as 1=0/2,2=0/3,3=1/3"
        )
    parsed: dict[int, tuple[int, ...]] = {}
    for entry in raw.split(","):
        item = entry.strip()
        try:
            peer_text, paths_text = item.split("=", 1)
            path_text = paths_text.split("/")
            peer = int(peer_text)
            expected_paths = peer_path_count(
                world_size, rank, peer, opposite_paths
            )
            if len(path_text) != expected_paths:
                raise ValueError
            paths = tuple(int(value) for value in path_text)
        except ValueError as exc:
            raise ValueError(
                f"invalid {PEER_HCA_MAP_ENV} entry {item!r}; expected the configured "
                "number of slash-separated HCA indices"
            ) from exc
        if peer < 0 or peer >= world_size or peer == rank:
            raise ValueError(
                f"{PEER_HCA_MAP_ENV} peer {peer} is not a non-local rank in "
                f"[0,{world_size})"
            )
        if peer in parsed:
            raise ValueError(f"{PEER_HCA_MAP_ENV} repeats peer {peer}")
        if len(set(paths)) != len(paths) or any(
            hca < 0 or hca >= hca_count for hca in paths
        ):
            raise ValueError(
                f"{PEER_HCA_MAP_ENV} peer {peer} needs distinct HCA indices in "
                f"[0,{hca_count}) for every configured path"
            )
        parsed[peer] = paths
    expected = set(range(world_size)) - {rank}
    if set(parsed) != expected:
        missing = sorted(expected - set(parsed))
        unexpected = sorted(set(parsed) - expected)
        raise ValueError(
            f"{PEER_HCA_MAP_ENV} must map every non-local rank exactly once; "
            f"missing={missing} unexpected={unexpected}"
        )
    return tuple(
        (-1, -1) if peer == rank else parsed[peer] for peer in range(world_size)
    )


def validate_four_path_tp4_mapping(
    hca_names: Sequence[str],
    world_size: int,
    rank: int,
    mapping: tuple[tuple[int, ...], ...],
    opposite_paths: int,
) -> None:
    """Validate the configured four-path structure without pinning site HCA names."""

    if opposite_paths != MAX_PATH_COUNT:
        return
    if world_size != 4 or len(hca_names) != MAX_PATH_COUNT or len(set(hca_names)) != MAX_PATH_COUNT:
        raise RuntimeError(
            f"{OPPOSITE_PATHS_ENV}=4 requires four ranks and four distinct configured RDMA devices"
        )
    if rank < 0 or rank >= world_size or len(mapping) != world_size:
        raise RuntimeError(f"{OPPOSITE_PATHS_ENV}=4 requires a complete configured rank map")
    direct_hcas: set[int] = set()
    for peer, paths in enumerate(mapping):
        distance = (peer - rank) % world_size
        expected_path_count = 0 if peer == rank else (
            MAX_PATH_COUNT if distance == 2 else PATH_COUNT
        )
        if peer == rank:
            valid = paths == (-1, -1)
        else:
            valid = (
                len(paths) == expected_path_count
                and len(set(paths)) == expected_path_count
                and all(0 <= hca < MAX_PATH_COUNT for hca in paths)
            )
            if distance == 2:
                valid = valid and set(paths) == set(range(MAX_PATH_COUNT))
            else:
                overlap = direct_hcas.intersection(paths)
                valid = valid and not overlap
                direct_hcas.update(paths)
        if not valid:
            raise RuntimeError(
                f"{OPPOSITE_PATHS_ENV}=4 has an invalid configured peer map for rank {rank}, peer {peer}"
            )
    if direct_hcas != set(range(MAX_PATH_COUNT)):
        raise RuntimeError(
            f"{OPPOSITE_PATHS_ENV}=4 requires the configured direct-neighbor paths to cover four distinct RDMA devices"
        )


__all__ = [
    "MAX_PATH_COUNT",
    "OPPOSITE_PATHS_ENV",
    "PATH_COUNT",
    "PEER_HCA_MAP_ENV",
    "PEER_HCA_MAPS_ENV",
    "opposite_path_count",
    "peer_hca_map",
    "peer_path_count",
    "validate_four_path_tp4_mapping",
]
