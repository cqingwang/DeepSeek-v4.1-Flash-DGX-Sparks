#!/usr/bin/env python3
"""把已构建镜像压平成一个可由 docker load 恢复的 Docker archive。

远端 Docker 的经典 save/exporter 在本机大镜像上不可用时，仍保留运行时
配置并生成确定性的单层制品，供物理环网逐跳传输。该路径只处理镜像文件，
不会读取模型权重挂载，也不会修改任何运行中的容器。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tarfile
import tempfile
from pathlib import Path


def run(command: list[str], *, stdout=None) -> subprocess.CompletedProcess:
    return subprocess.run(command, check=True, stdout=stdout)


def image_inspect(image: str) -> dict:
    result = subprocess.run(
        ["docker", "image", "inspect", image],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)[0]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def flat_config(inspect: dict, rootfs_digest: str) -> dict:
    config = dict(inspect.get("Config") or {})
    result = {
        "architecture": inspect.get("Architecture", "arm64"),
        "os": inspect.get("Os", "linux"),
        "created": inspect.get("Created"),
        "config": config,
        "rootfs": {"type": "layers", "diff_ids": [f"sha256:{rootfs_digest}"]},
        "history": [{
            "created": inspect.get("Created"),
            "created_by": "flat rootfs export for ring transfer",
        }],
    }
    if inspect.get("Variant"):
        result["variant"] = inspect["Variant"]
    return result


def split_tag(image: str) -> tuple[str, str]:
    last = image.rsplit("/", 1)[-1]
    if ":" not in last:
        return image, "latest"
    repository, tag = image.rsplit(":", 1)
    return repository, tag


def build_archive(image: str, output: Path) -> tuple[str, str, str]:
    inspect = image_inspect(image)
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="dsv41-flat-", dir=output.parent) as temp:
        staging = Path(temp)
        rootfs = staging / "rootfs.tar"
        container_name = f"dsv41-flat-export-{output.stem}"
        container_id = subprocess.run(
            ["docker", "create", "--name", container_name, image],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        try:
            with rootfs.open("wb") as target:
                run(["docker", "export", container_id], stdout=target)
        finally:
            subprocess.run(["docker", "rm", "-f", container_id], check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        rootfs_digest = sha256_file(rootfs)
        blobs = staging / "blobs" / "sha256"
        blobs.mkdir(parents=True)
        layer = blobs / rootfs_digest
        rootfs.rename(layer)

        config = flat_config(inspect, rootfs_digest)
        config_path = blobs / "config.pending"
        config_digest = write_json(config_path, config)
        config_path.rename(blobs / config_digest)

        manifest = [{
            "Config": f"blobs/sha256/{config_digest}",
            "RepoTags": [image],
            "Layers": [f"blobs/sha256/{rootfs_digest}"],
        }]
        write_json(staging / "manifest.json", manifest)
        repository, tag = split_tag(image)
        write_json(staging / "repositories", {repository: {tag: rootfs_digest}})
        partial = output.with_suffix(output.suffix + ".partial")
        if partial.exists():
            partial.unlink()
        with tarfile.open(partial, "w") as archive:
            for name in ("manifest.json", "repositories"):
                archive.add(staging / name, arcname=name, recursive=False)
            archive.add(blobs / config_digest,
                        arcname=f"blobs/sha256/{config_digest}", recursive=False)
            archive.add(layer, arcname=f"blobs/sha256/{rootfs_digest}", recursive=False)
        partial.replace(output)

    return rootfs_digest, config_digest, sha256_file(output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rootfs_digest, config_digest, archive_digest = build_archive(args.image, args.output)
    print(f"rootfs_sha256={rootfs_digest}")
    print(f"config_sha256={config_digest}")
    print(f"archive_sha256={archive_digest}")
    print(f"archive={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
