# 工程记忆

## 2026-10-27：knapcio/main 部署后的 prose c1/c4 专项诊断

- **范围与现场：** managed `deepseek_v4.1_flash_switchless`，固定模型
  `deepseek-ai/DeepSeek-V4.1-Flash-1017`，sparkDash 1.8.8 参考协议为 256 new tokens、
  temperature 0、thinking off。本轮不重启、不重建、不改环网；四个受管管理 IP 的容器
  `deepseek_v4.1_flash_switchless-r0..r3` 均为运行态，模型 bind mount 为只读，rank/world
  为 0..3/4。
- **专项结果：** 仅纳入 prose c1/c4 各三轮算术平均：c1 `84.84/85.85/84.76`，平均
  `85.15 tok/s`，低于 README 参考 `91.3`；c4 `173.41/175.12/175.25`，平均
  `174.59 tok/s`，达到参考 `174.0`。本地 SparkDash 结果存于
  `docs/results/decodebench-prose-c1-c4-sparkdash-1.8.8-2026-10-27.json`。其他 decode
  类型不属于本次专项验收。
- **单请求诊断（不计入均值）：** 对固定 SparkDash hash-map prose prompt 发送一个 c1 请求，
  256/256 completion tokens，`spec_verify_ct=82`、
  `spec_accept_length=3.1219512195`、`spec_accept_rate=0.4243902439`、
  174/410 correct/proposed drafts。非流式端到端用时 3.37 s 包含 prefill，不能当作 decode
  step time。请求前 `num_running_reqs=0`、`num_queue_reqs=0`，请求后服务回到空闲。
- **运行 env 核验：** 四机实际 `NCCL_BUFFSIZE=8388608`、`NCCL_MIN_NCHANNELS=4`、
  `NCCL_MAX_NCHANNELS=4`、`NCCL_NET=IB`、`NCCL_ALGO=RING`、`SGLANG_ROCE_MAX_SIZE=262144`；
  `DSV41_CERT_HEAD`、`DSV41_SPEC_SYNC_FREE`、`DSV41_EAGER_GLUE`、`DSV41_L2_PREFETCH`、
  `DSV41_DRAFT_TAU`、`DSV41_VERIFY_CAP` 等当前 profile 标志已实际注入。`.env.tp4` 顶层
  留存的 1 MiB/8-channel 值被 managed TP4 launcher 的 target-owned
  `EXTRA_DOCKER_ENV` 覆盖路径排除，实际容器值与父仓 `spark/config.yaml` 相符；没有基于这组
  静态差异改环境或重启。
- **限制与后续：** 本轮没有证明 c1 缺口由 speculative acceptance、单步延迟或 physical-ring
  collective 的哪一项单独导致；没有修改 `/30`、rank/HCA/GID、ring-only、256 KiB cap、
  serving config 或 GPU 时钟。c4 已过、c1 未过，整体性能目标保持未完成；下一步须先基于
  有效环境和现场计数选一个 ring-safe 单变量 A/B，结果仍按 prose c1/c4 各三轮平均记录。

## 2026-10-27：镜像归档校验拒绝 Docker/OCI config 路径

- **范围与环境：** `scripts/ship_image_ring.sh::archive_image_id` 对 Spark 上
  `docker save` 归档的身份校验；本机 macOS arm64/Python 3.13，Spark 为 Linux/aarch64。
- **症状与 RED 证据：** Spark 归档 `manifest.json` 中 `Config` 为
  `blobs/sha256/1c1088290a0a9ab0c65235c3097de942e204dea27251dda8268f3f5cff29f4cf`，旧函数报
  `unsafe Docker image config path`。补入 OCI 格式成功样本后，修复前
  `python3 tests/test_offline_build.py` → 26 tests、1 failure，失败明确指向该 OCI config path。
- **根因：** 路径检查只允许单段 `<digest>.json` 的 legacy Docker archive 格式，误将标准
  OCI-style `blobs/sha256/<digest>` 当作不安全路径。
- **修复：** 只允许严格匹配 `[0-9a-f]{64}.json` 或
  `blobs/sha256/[0-9a-f]{64}`；保持绝对路径、`..` 和未知路径拒绝。测试覆盖两种合法路径、
  traversal、未知目录以及 RepoTag 不匹配。
- **红绿验证：** RED 为上述 OCI 子测失败；GREEN：
  `python3 tests/test_offline_build.py` → 26 passed，
  `bash -n scripts/ship_image_ring.sh` 与针对脚本/测试的 `git diff --check` 通过。
- **现场状态与适用边界：** spark-a 已构建镜像仍在且 ID 为
  `sha256:5c300d07b3b571f11afe93c4ac2685412d683c34a63e48047a552e85ceafac27`；远端旧 archive
  的 config ID 是 `sha256:1c1088290a0a9ab0c65235c3097de942e204dea27251dda8268f3f5cff29f4cf`，
  不可用于当前镜像。后续应从现有镜像重新保存归档，不需重新 build；分发和部署尚未完成。

## 2026-10-27：TP4 ring-helper 单测错误要求重复合成 GID 扩展变量

- **范围与环境：** `files/nccl.sh` 的 managed TP4 helper 与
  `tests/test_switchless_ring.py`；spark-a Linux/aarch64、Python 3.12.3、GNU Bash 5.2.21。
- **症状与 RED 证据：** 同步后运行 `python3 tests/test_switchless_ring.py`，88 项中仅
  `DualPciDomainTests.test_target_owned_extension_policy_is_not_synthesized_twice` 失败，称
  helper 输出缺少 `NCCL_IB_EXTENDED_IPV4_GIDS=1`。同轮
  `SwitchOnTests.test_managed_transport_env_is_injected_once_per_rank` 通过。
- **根因：** TP4 环网 helper 按设计不重复合成 parent-owned GID policy；生产
  `docker_common_args()` 与 `worker_env_lines()` 分别从 managed target 字段注入该变量，
  而既有集成合同验证 head/worker 每 rank 恰好一次。单测只观察 helper，却要求它也注入，
  与 helper 的去重边界冲突。
- **修复：** 将 helper 单测改为断言 managed TP4 下不重复输出该变量；保留集成合同对 head、
  worker 实际参数中唯一一次注入的断言。同步修正文档注释，区分 target 字段与
  `extra_docker_env` 中的 companion PCI 参数；不改变 launcher 参数、不改变拓扑或 NCCL policy。
- **红绿回归：** RED 为 spark-a 原始源码上的上述 1 项失败。GREEN：修正并重同步后，
  `python3 tests/test_switchless_ring.py` → 88 passed，
  `python3 tests/test_upstream_launcher_ports.py` → 13 passed；同轮
  `python3 -m unittest` 按 `.github/workflows/offline-tests.yml` 清单运行 → 157 passed、2 skipped，
  两个直接脚本合同通过。远端缺少 pytest，故 TP4 pytest 合同未在目标机重复；此前本地
  `tests/test_ring_tp4_contract.py` 为 8 passed。四节点关键文件 SHA-256 一致，
  `bash -n start.sh start-tp4.sh stop.sh files/nccl.sh` 通过。
- **限制：** 这些结果只证明同步源码和 CPU 合同；尚未构建新镜像、部署新容器、验证实际 NCCL
  数据面或跑速度矩阵。
- **适用边界：** managed TP4 ring 中，parent-owned GID 参数只由 launcher 集成层注入；
  helper 局部测试须与真实参数组装边界一致，不能把“helper 未输出”误判为生产参数缺失。

## 2026-10-27：受管 TP4 忽略 GID force=-1 并错误回退到 index 3

- **范围与环境：** 父仓 `spark/config.yaml → program.py → deploy.sh` 到子模块
  `files/nccl.sh` / `start.sh` 的 managed TP4 switchless RoCE GID 配置链；macOS 控制端、
  Python 3.13，Spark 目标为 Bash 5。
- **症状与 RED 证据：** 父仓 query 为该 target 提供
  `NCCL_IB_GID_INDEX_FORCE=-1`，代表自动选择而非固定 GID。新增测试先失败：
  `python3 -m unittest -v
  tests.test_switchless_ring.ManagedB12xPolicyTests.test_managed_gid_force_minus_one_does_not_fall_back_to_index_three`
  得到 `auto= index= resolved=3`，但预期为 `auto=1 index= resolved=auto`。
- **根因：** managed `start.sh` 的 `resolved_gid()` 只读取 `NCCL_IB_GID_INDEX`、
  address-detected GID 与 standalone fallback；它没有消费父级 `NCCL_IB_GID_INDEX_FORCE`。
  Spark 的 `HEAD_IP` / worker rendezvous 地址属于管理网，通常不会匹配 RoCE GID，因而
  force=-1 会落入旧的 index 3 fallback，意外把 managed NCCL 固定到一个 GID index。
- **修复：** `nccl_validate_managed_tp4_transport_policy()` 现在要求 force 为 `-1` 或非负
  整数，并拒绝与独立 `NCCL_IB_GID_INDEX` 冲突；`nccl_apply_managed_tp4_gid_policy()` 将
  `-1` 映射到 `NCCL_IB_GID_AUTO=1`、不设置 index，将非负值映射为经校验的显式 index。
  `resolved_gid()` 在 managed TP4 下遵守同一语义。`nccl_preflight` 仍逐所选 HCA 验证有效
  IPv4 RoCEv2 GID；当返回 `auto` 时，容器 argv 不注入 `NCCL_IB_GID_INDEX`。未更改
  `/30`、HCA/rank 顺序、物理接线或 B12X parent-owned peer map。
- **GREEN 与范围：** `python3 -m unittest -v
  tests.test_switchless_ring.ManagedB12xPolicyTests` → 6 passed（覆盖 auto、显式 index、
  非法及冲突值）；`python3 -m pytest -q tests/test_ring_tp4_contract.py` → 8 passed；
  离线 host-side unittest → 157 passed、2 skipped；layer20/prefill-interval/bounded-replay
  Dockerfile 包装测试 → 21 passed；`python3 tests/test_offline_build.py` → 26 passed；
  父仓 ring SSOT/build/sync safety → 49 passed。
- **适用边界/待办：** 这些是源码与离线合同结果，不代表 Spark 已同步、构建、部署或 GID
  运行态已验证。受 managed GID force 影响的 `tests/test_switchless_ring.py` 完整 Bash 测试
  需在 Spark Bash 5 上执行；部署后还要检查实际容器未被注入 `NCCL_IB_GID_INDEX`，并通过
  NCCL route/QP 日志和请求时 HCA 计数证明环网数据面。

## 2026-10-27：上游端口兼容测试未模拟有效运行配置

- **范围与环境：** `tests/test_launcher_build.py` 的 Bash fixture 与
  `tests/test_upstream_launcher_ports.py`；macOS arm64、Python 3.13、自带 Bash 3.2。
- **症状/证据：** 定向组合出现 3 项失败、85 passed、2 skipped、49 subtests：
  NFS `cmd_pack` 用例报 `MODEL_DIR is missing/incomplete`，因为测试虽传入 volume/NFS
  设置，fixture 随后仍强制覆盖为 TP4 `WEIGHTS_MODE=local`、`NFS_SHARE=0`；空 GID
  worker-env 用例以 `NCCL_PIN_CONTAINER` / `NCCL_NET: unbound variable` 失败，因为 fixture
  未提供完整 NCCL base settings。修复上述输入后，旧断言仍期望探测 2 个 worker，但当前
  TP4 fixture 实际有 3 个 worker，单项结果为 31 passed、1 failed。
- **根因：** 兼容测试沿用 TP3/NFS 假设，却复用了强制设置 TP4 本地权重的测试切片；
  worker-env 用例则把不完整 shell 环境传给启用 `set -u` 的真实函数。
- **修复：** fixture 仅在未显式提供时才默认 `WEIGHTS_MODE=local` 与 `NFS_SHARE=0`；
  NFS 测试显式设 `WEIGHTS_MODE=volume` 并分别覆盖 NFS 开/关；TP4 NFS 共享检查期望三个
  worker；GID 测试复用完整 `BASE_SETTINGS` 并提供显式 shim。仅修改测试夹具/断言，不改生产
  launcher、受管权重或环网配置。
- **红绿回归：** RED：上述定向组合 3 项失败；第一次修 fixture 后
  `python3 -m pytest -q tests/test_launcher_build.py tests/test_upstream_launcher_ports.py`
  为 31 passed、1 failed（worker 数断言）。GREEN：修正 worker 数后同一命令为
  32 passed、3 subtests；扩展组合
  `python3 -m pytest -q tests/test_ring_tp4_contract.py tests/test_roce_ring.py tests/test_launcher_build.py tests/test_offline_build.py tests/test_upstream_launcher_ports.py tests/test_adapter_engine_pins.py tests/test_b12x_memory_requirements.py tests/test_ring_mesh_pipefail.py tests/test_row_store_keepalive.py tests/test_sps_engram_guard.py`
  为 88 passed、2 skipped、49 subtests。
- **限制：** `tests/test_prefill_sp.py` 在本机收集阶段因缺少 `torch` 报
  `ModuleNotFoundError`，未安装依赖；该用例未计入 GREEN 组合。

## 2026-10-26：RoCEnante 源归属合同被 Python 字节码缓存误报

- **范围与环境：** `tests/test_launcher_build.py::CanaryDockerfileTests.test_roce_image_embeds_runtime_provenance_files`；macOS arm64、Python 3.13。
- **症状/证据：** 定向合同在完整 ring/build 测试集中失败，称 `runtime/b12x/b12x/comm/roce_ring` 不应存在。现场该目录仅含忽略的 `__pycache__/_path_config.cpython-313.pyc`，没有 Python/C 源；Dockerfile 已从 BuildKit `spark-ring` context 复制权威实现。
- **根因：** 合同把目录存在等同于 vendored 源存在；Python 导入留下的字节码缓存可创建空源目录，导致测试结果依赖开发机历史状态。
- **修复：** 合同仍要求最终 RoCE Dockerfile 从 `spark-ring` 拷贝 ring 实现，并检查 target 路径下实际文件时排除 `__pycache__`；未删除缓存、未改变 source owner 或镜像构建上下文。
- **红绿证据：** RED：`python3 -m pytest -q tests/test_ring_tp4_contract.py tests/test_switchless_ring.py tests/test_roce_ring.py tests/test_launcher_build.py tests/test_offline_build.py tests/test_upstream_launcher_ports.py` 中 `test_roce_image_embeds_runtime_provenance_files` 因仅缓存目录存在而失败。GREEN：修复后 `python3 -m pytest -q tests/test_launcher_build.py` → 13 passed、3 subtests passed。
- **限制：** 同一批次的 shell 测试另受 macOS Bash 3.2 不支持 `${var^^}` 与空数组 `set -u` 影响；不能据此判定 Spark Linux ring 代码失败，须在 Bash 5 环境重跑。

## 2026-10-26：镜像构建误执行依赖父仓配置的 RoCE 拓扑合同

- **范围与环境：** `tests/test_roce_ring.py` 在 Spark 主机与 container build 中的运行边界；物理拓扑唯一来源仍是父仓 `spark/config.yaml` / `spark/program.py`。
- **症状/证据：** spark-a retry5 镜像包装测试在 `tests/test_roce_ring.py` 执行时退出，日志为 `/opt/sglang/bin/python3: can't open file '/opt/program.py': [Errno 2] No such file or directory`。使用现存镜像并只读挂载新测试源码，可复现同一失败；镜像内没有父仓 `/opt/program.py` 与站点 `config.yaml`。
- **根因：** `Plan.setUpClass()` 调父仓 `program.py` 读取本地物理环，但 Dockerfile 在纯运行镜像内调用整份 host topology 测试。把父仓 config/program 复制入镜像会将站点 HCA/IP/拓扑烘焙进通用制品，违反 SSOT 边界。
- **修复：** 对依赖父仓 `config.yaml` 的 `Plan` 测试类添加 `skipUnless(CONFIG.is_file())`；Spark 主机保留完整 topology 合同，镜像缺少父仓配置时只执行自包含的 peer-map 与 image-overlay 测试。未向镜像加入站点配置。
- **红绿验证：** RED：修复前 `docker run --rm --entrypoint python3 -v /opt/spark/deepseek_v4.1_flash_switchless:/opt/dsv41:ro dsv41-4x-spark:canary-roce /opt/dsv41/tests/test_roce_ring.py` 返回 1，报 `/opt/program.py` 缺失。GREEN：spark-a host `python3 tests/test_roce_ring.py` → 11 tests、1 个预期 image-only skip；相同 container 命令 → 11 tests、6 个预期 parent-config skips，其余均通过。
- **限制/下一步：** macOS 系统 Python 未安装 PyYAML，不能用本机解释器运行该父仓解析用例；已由 spark-a host Python 验证。Docker rebuild、image distribute/deploy、RoCE 请求流量与 speed matrix 仍未完成。

## 2026-10-26：Canary verifier 合并 RUN 后的包装合同误报

- **范围与环境：** `Dockerfile.canary`、`Dockerfile.canary-roce` 的 pinned SGLang verifier；spark-a 为 Linux/arm64、Docker `overlay2`，base 镜像 77 层、overlay2 总层数上限 128。
- **症状/证据：** 镜像构建已通过 SGLang pinned patch verifier 与 CPU falsifier，但 `test_sglang_layer20_tail_patch.py` 的两个 recipe 子用例失败：期望独立 `RUN python3 ...verify_sglang_layer20_tail.py...`，实际 verifier 位于 `RUN set -eu; \` 的续行命令。spark-a 日志 `/var/tmp/dsv41-canary-roce-knapcio-main-20261026-retry2.build.log` 报 6 tests、2 failures。修正该合同后，代理构建又在 `test_sglang_prefill_interval_min_bs_patch.py` 的两个 recipe 子用例发现同形状过期断言，见 `/var/tmp/dsv41-canary-roce-knapcio-main-20261026-retry4.build.log`；静态扫描还发现 bounded-replay 包装合同有相同断言。
- **根因：** 为避免最终镜像超过 overlay2 的 128 层上限，多个 verifier/falsifier 被有意合并到单个 fail-fast `RUN set -eu`；layer20、prefill-interval 和 bounded-replay 包装合同仍把 verifier 与独立 Dockerfile `RUN` 绑定，未覆盖多命令构建层。
- **修复：** 三个包装合同均按反斜杠续行解析 Dockerfile `RUN` 指令，断言各自 verifier 位于 `RUN set -eu;` 块中，同时保留两种 recipe 都含相应 CPU 包装测试的断言。未拆回独立层，未修改 Docker daemon/storage driver。
- **红绿回归：** RED：layer20 6 tests 中 2 个子测失败；prefill-interval 9 tests 中 2 个子测失败；bounded-replay 6 tests 中 2 个子测失败，均报告独立 `RUN python3 ...` 字符串缺失。GREEN：相同三条命令分别 6/6、9/9、6/6 通过；`python3 tests/test_offline_build.py` 为 23/23。spark-a retry4 已通过 layer20 6 项后在 prefill 两子测失败，因此尚无构建产物。
- **构建网络与边界：** spark-a 首轮未设置代理时，固定 B12X archive 的直连 GitHub `curl -m 900` 等待近两分钟；取消未产出镜像的构建后，验证 `192.168.2.114:7890` 对固定 URL 返回 HTTP 200，并用 HTTP(S)_PROXY 参数重跑。最新包装合同修复仍需单向同步后在 spark-a 成功构建并实测 overlay2 层数，再通过 `scripts/ship_image_ring.sh` 沿 `spark/config.yaml.common.ring` 物理相邻 hop 单份导出、校验及逐节点 `docker load`。不在 worker 重建，不改物理环地址/HCA/GID/拓扑；部署、RoCE 数据面和性能矩阵仍须各自取证。
