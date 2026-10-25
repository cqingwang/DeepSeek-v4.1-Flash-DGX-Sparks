"""验证 TP4/ring 启动链路不会被官方同步覆盖。"""

from pathlib import Path
import os
import subprocess
import tempfile


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
    """模板保留 serving defaults；父仓负责写入活动 ring/backend 环境。"""
    for name in (".env.example", ".env.tp4.example"):
        values = _env(name)
        assert values["NNODES"] == "4"
        assert values["TP_SIZE"] == "4"
        assert values["EP_SIZE"] == "1"
        assert values["DIST_PORT"] == "25000"
        assert values["WEIGHTS_MODE"] == "local"
        assert values["NFS_SHARE"] == "0"
        assert values["DSV41_ADAPTIVE_CHUNK"] == "1"
        assert values["DSV41_INDEXER_BUDGET_MIB"] == "256"


def test_launcher_retains_ring_and_local_weight_contract() -> None:
    source = (ROOT / "start.sh").read_text(encoding="utf-8")
    for marker in (
        '"$NCCL_HOST_DIR:$NCCL_CONTAINER_DIR:ro"',
        '"LD_PRELOAD=$NCCL_PIN_CONTAINER $NCCL_CONTAINER_DIR/$NCCL_SO_NAME"',
        "nccl_validate_managed_tp4_transport_policy",
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


def test_nccl_selfcheck_requires_target_owned_library_directory() -> None:
    script = ROOT / "scripts/nccl_selfcheck.sh"
    with tempfile.TemporaryDirectory() as temp_dir:
        debug_dir = Path(temp_dir) / "debug"
        debug_dir.mkdir()
        (debug_dir / "nccl.log").write_text(
            "RING-ONLY vfixture\n4 coll channels\n"
            "NET/IB : Using [0]roce0 [1]roce1 [2]roce2 [3]roce3\n",
            encoding="utf-8",
        )
        env = os.environ.copy()
        env.pop("NCCL_HOST_DIR", None)
        result = subprocess.run(
            ["bash", str(script), str(debug_dir)],
            cwd=ROOT,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "NCCL_HOST_DIR" in result.stderr
        assert "受管 target 配置" in result.stderr


def test_nccl_selfcheck_accepts_managed_switchless_signature_and_derives_hca_count() -> None:
    script = ROOT / "scripts/nccl_selfcheck.sh"
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        debug_dir = root / "debug"
        library_dir = root / "library"
        debug_dir.mkdir()
        library_dir.mkdir()
        (library_dir / "libnccl.so.2.30.7").write_text(
            "Tree transport setup disabled by NCCL_SWITCHLESS_RING_ONLY\n"
            "PAT transport setup disabled by NCCL_SWITCHLESS_RING_ONLY\n",
            encoding="utf-8",
        )
        matrix_rows = "\n".join(
            f"    {function:12} | 1 0 1 | 0 1 0 0 0 0 0"
            for function in ("Broadcast", "Reduce", "AllGather", "ReduceScatter", "AllReduce")
        )
        (debug_dir / "nccl.log").write_text(
            "Tree transport setup disabled by NCCL_SWITCHLESS_RING_ONLY\n"
            "PAT transport setup disabled by NCCL_SWITCHLESS_RING_ONLY\n"
            "Enabled NCCL Func/Proto/Algo Matrix:\n"
            "     Function | LL LL128 Simple | Tree Ring CollNetDirect CollNetChain NVLS NVLSTree PAT\n"
            f"{matrix_rows}\n"
            "4 coll channels\n"
            "NET/IB : Using [0]roce0:1/RoCE [1]roce1:1/RoCE [RO]; OOB mgmt0\n",
            encoding="utf-8",
        )
        env = os.environ.copy()
        env.update(
            NCCL_HOST_DIR=str(library_dir),
            NCCL_SWITCHLESS_RING_ONLY="1",
            IB_HCA="roce0,roce1",
        )
        result = subprocess.run(
            ["bash", str(script), str(debug_dir)],
            cwd=ROOT,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "Tree=0 / Ring=1" in result.stdout
        assert "NCCL RoCE HCA 顺序与 common.ring 完全一致: roce0,roce1（2 个）" in result.stdout
        assert "managed switchless ring policy" in result.stdout


def test_official_adapter_additions_are_wired() -> None:
    source = (ROOT / "adapter/sitecustomize.py").read_text(encoding="utf-8")
    assert "deepseek_v4_memory_pool" in source
    assert "deepseek_v4_backend" in source
    assert (ROOT / "adapter/indexer_budget.py").is_file()
    metrics = (ROOT / "adapter/kv_pool_metrics.py").read_text(encoding="utf-8")
    assert "DeepSeekV4TokenToKVPool" in metrics


def test_sitecustomize_conflict_merge_keeps_local_and_upstream_hooks() -> None:
    """合并两边 adapter 时保留 UMA 防护和 Engram L2 prefetch hook。"""
    import py_compile

    source = ROOT / "adapter/sitecustomize.py"
    py_compile.compile(str(source), doraise=True)
    text = source.read_text(encoding="utf-8")
    assert "DSV41_SERIAL_WEIGHT_LOAD" in text
    assert "DSV41_L2_PREFETCH_ENGRAM" in text
    assert "install_engram_prefetch(module)" in text


def test_tp4_template_matches_the_managed_hardware_ring_contract() -> None:
    values = _env(".env.tp4.example")
    container_env = values["EXTRA_CONTAINER_ENV"].split()
    for managed_ring_key in (
        "DSV41_ROCE_RING",
        "SGLANG_ROCE_ALLREDUCE",
        "SGLANG_ROCE_MAX_SIZE",
        "DSV41_ROCE_GATHER",
        "B12X_ROCE_HCA",
        "B12X_ROCE_GID_INDEX",
        "B12X_ROCE_PEER_HCA_MAPS",
        "B12X_ROCE_TWO_WAVE_THRESHOLD_BYTES",
    ):
        assert not any(token.startswith(f"{managed_ring_key}=") for token in container_env)

    for managed_ring_key in (
        "NCCL_NET",
        "NCCL_IB_DISABLE",
        "NCCL_P2P_DISABLE",
        "NCCL_SHM_DISABLE",
        "NCCL_HOST_DIR",
        "NCCL_CONTAINER_DIR",
        "NCCL_IB_GID_INDEX_FORCE",
        "NCCL_IB_SUBNET_PREFIX_LEN",
        "IB_HCA",
    ):
        assert managed_ring_key not in values
