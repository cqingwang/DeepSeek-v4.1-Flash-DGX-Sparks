"""验证 TP4/ring 启动链路不会被官方同步覆盖。"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _env(path: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in (ROOT / path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value.strip().strip('"').strip("'")
    return values


def test_tp4_templates_keep_production_defaults() -> None:
    """模板即使脱离父控制器启动，也必须落在 TP4/ring 的安全配方。"""
    for name in (".env.example", ".env.tp4.example"):
        values = _env(name)
        assert values["NNODES"] == "4"
        assert values["TP_SIZE"] == "4"
        assert values["EP_SIZE"] == "2"
        assert values["DIST_PORT"] == "25000"
        assert values["WEIGHTS_MODE"] == "local"
        assert values["NFS_SHARE"] == "0"
        assert values["NCCL_HOST_DIR"] == "/opt/nccl-ringonly"
        assert values["DSV41_ADAPTIVE_CHUNK"] == "1"
        assert values["DSV41_INDEXER_BUDGET_MIB"] == "256"


def test_launcher_retains_ring_and_local_weight_contract() -> None:
    source = (ROOT / "start.sh").read_text(encoding="utf-8")
    for marker in (
        "/opt/nccl-ringonly/libnccl.so.2",
        "NCCL_IB_PEER_HCA",
        "EXTRA_DOCKER_ENV",
        "WEIGHTS_MODE",
        "WORKER_MODEL_DIR",
        "scripts/nccl_selfcheck.sh",
        "scripts/gate.sh",
    ):
        assert marker in source, f"启动器缺少必要能力: {marker}"


def test_build_and_runtime_overlay_contract() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "patch_encoding_dsv41.py" in dockerfile
    assert (ROOT / "runtime/patch_encoding_dsv41.py").is_file()
    assert (ROOT / "scripts/nccl_selfcheck.sh").is_file()
    assert (ROOT / "scripts/gate.sh").is_file()


def test_official_adapter_additions_are_wired() -> None:
    source = (ROOT / "adapter/sitecustomize.py").read_text(encoding="utf-8")
    assert "deepseek_v4_memory_pool" in source
    assert "deepseek_v4_backend" in source
    assert (ROOT / "adapter/indexer_budget.py").is_file()
    metrics = (ROOT / "adapter/kv_pool_metrics.py").read_text(encoding="utf-8")
    assert "DeepSeekV4TokenToKVPool" in metrics
