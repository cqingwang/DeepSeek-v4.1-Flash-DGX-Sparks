#!/bin/bash
# nccl_selfcheck.sh — 校验 NCCL 初始化是否符合无交换机环网预期
#
# 检查项（任一失败即非零退出）：
#   1. 目标配置选择的 ring-only NCCL 路径在日志中有对应补丁证据
#   2. 算法矩阵中 Tree 全部为 0、Ring 全部为 1（Tree 未参与，Ring 强制）
#   3. 通道数 > 0
#   4. common.ring 派生的 HCA 全部被 NCCL 采用
# 仅记录（不判失败）：GDR 状态、协议集。
#
# 用法：nccl_selfcheck.sh [debug_dir] [expected_hca_count]
#   默认 debug_dir=$HOME/nccl-debug；HCA 数量从部署器写入的 IB_HCA 派生。

set -uo pipefail

DBG_DIR="${1:-$HOME/nccl-debug}"
fail=0

if [[ -z "${NCCL_HOST_DIR:-}" ]]; then
  echo "[x] NCCL_HOST_DIR 必须由受管 target 配置提供" >&2
  exit 1
fi

managed_hcas="${IB_HCA:-${NCCL_IB_HCA:-}}"
managed_hcas="${managed_hcas#=}"
if [[ -z "$managed_hcas" ]]; then
  echo "[x] IB_HCA 必须由 common.ring 派生的受管 target 配置提供" >&2
  exit 1
fi
IFS=',' read -r -a managed_hca_list <<< "$managed_hcas"
EXPECT_HCA="${#managed_hca_list[@]}"
for hca in "${managed_hca_list[@]}"; do
  if [[ -z "$hca" || "$hca" == *[[:space:]]* ]]; then
    echo "[x] IB_HCA 不是有效的逗号分隔 HCA 清单: $managed_hcas" >&2
    exit 1
  fi
done
if [[ -n "${2:-}" && "$2" != "$EXPECT_HCA" ]]; then
  echo "[x] expected_hca_count=$2 与 common.ring 派生的 IB_HCA 数量 $EXPECT_HCA 不一致" >&2
  exit 1
fi

log=$(ls -t "$DBG_DIR"/*.log 2>/dev/null | head -1)
if [[ -z "${log:-}" ]]; then
  echo "[x] 未找到 NCCL 调试日志（$DBG_DIR/*.log）——确认 NCCL_DEBUG_FILE 已设且服务已启动"
  exit 1
fi
echo "[i] 检查日志: $log"

# 1. ring-only 补丁签名
if grep -q 'RING-ONLY v' "$log"; then
  ver=$(grep -oE 'RING-ONLY v[0-9a-zA-Z._-]+' "$log" | head -1)
  echo "[+] $ver 补丁生效"
elif [[ "${NCCL_SWITCHLESS_RING_ONLY:-0}" == 1 ]] \
  && grep -q 'Tree transport setup disabled by NCCL_SWITCHLESS_RING_ONLY' "$log" \
  && grep -q 'PAT transport setup disabled by NCCL_SWITCHLESS_RING_ONLY' "$log"; then
  echo "[+] managed switchless ring policy 生效（Tree/PAT transport 已禁用）"
else
  echo "[x] 未发现与 target NCCL ring policy 相符的补丁签名"
  fail=1
fi

# 2. 算法矩阵：Tree 全 0 / Ring 全 1
#    表行形如： "    AllReduce |      1         1         1   |      0       1  ..."
matrix_rows=$(grep -cE '^\s+(Broadcast|Reduce|AllGather|ReduceScatter|AllReduce)\s+\|' "$log" || true)
if [[ "${matrix_rows:-0}" -ge 5 ]]; then
  bad=$(grep -E '^\s+(Broadcast|Reduce|AllGather|ReduceScatter|AllReduce)\s+\|' "$log" \
        | awk -F'|' '{n=split($3,a," "); tree=a[1]; ring=a[2];
                      if (tree+0 != 0 || ring+0 != 1) print "  BAD: " $0}')
  if [[ -z "$bad" ]]; then
    echo "[+] 算法矩阵 OK：Tree=0 / Ring=1（$matrix_rows 个集合函数全部强制 Ring）"
  else
    echo "[x] 算法矩阵异常：Tree 未禁用或 Ring 未启用"
    echo "$bad"
    fail=1
  fi
else
  echo "[!] 日志中未找到算法矩阵（NCCL_DEBUG 可能不是 INFO）——跳过"
fi

# 3. 通道数
chans=$(grep -oE '[0-9]+ coll channels' "$log" | head -1 | grep -oE '^[0-9]+' || echo 0)
if [[ "${chans:-0}" -gt 0 ]]; then
  echo "[+] 集合通道数: $chans"
else
  echo "[x] 未发现集合通道"
  fail=1
fi

# 4. RoCE 设备清单必须与 common.ring 派生的 HCA 顺序完全一致
net_line=$(grep 'NET/IB : Using' "$log" | head -1 || true)
actual_hcas=$(sed -E 's/^.*NET\/IB : Using //; s/ \[RO\].*$//; s/\[[0-9]+\]([^:]+):[^ ]+/\1/g' <<< "$net_line")
actual_hcas="${actual_hcas// /,}"
if [[ -n "$net_line" && "$actual_hcas" == "$managed_hcas" ]]; then
  echo "[+] NCCL RoCE HCA 顺序与 common.ring 完全一致: ${actual_hcas}（$EXPECT_HCA 个）"
else
  echo "[x] NCCL RoCE HCA 与 common.ring 不一致: actual=${actual_hcas:-缺失} expected=$managed_hcas"
  fail=1
fi

# 5. 仅记录：GDR 状态（GB10 上预期为 0，全走 host-staging）
gdr=$(grep -oE 'cuMemGdrSupport [0-9]+' "$log" | head -1 || true)
echo "[i] ${gdr:-cuMemGdrSupport 未知}（GB10 预期 0 = 无 GDR，CPU-path）"

# 6. 补丁谱系识别（仅记录）
lib=""
for candidate in "$NCCL_HOST_DIR"/libnccl.so*; do
  [[ -f "$candidate" ]] || continue
  lib="$candidate"
  break
done
if [[ -n "$lib" ]]; then
  # grep -c prints "0" and exits 1 on no match, so do not append another 0.
  sk=$(strings "$lib" 2>/dev/null | grep -c SKIP_TREE_CONNECT)
  dis=$(grep -icE 'transport setup disabled' "$log")
  sk=${sk:-0}; dis=${dis:-0}
  if grep -q 'RING-ONLY v' "$log"; then
    echo "[i] NCCL 补丁谱系: SparkRing/PR#3 路线（运行日志含 RING-ONLY 适配记录）"
  elif [[ "$dis" -ge 2 && "${NCCL_SWITCHLESS_RING_ONLY:-0}" == 1 ]]; then
    echo "[i] NCCL 补丁谱系: managed switchless 路线（Tree/PAT transport 禁用，skip-tree=${sk}）"
  elif [[ "$dis" -eq 0 ]]; then
    echo "[i] NCCL 补丁谱系: LuZ 路线（无 SWITCHLESS_RING_ONLY；靠算法矩阵 Tree=0 达成 ring-only）"
  else
    echo "[i] NCCL 补丁谱系: 未知/混合"
  fi
else
  echo "[i] 未找到 $lib，跳过谱系识别"
fi

echo
if [[ "$fail" -eq 0 ]]; then
  echo "[+] NCCL 自检通过"
else
  echo "[x] NCCL 自检发现问题（见上）"
fi
exit "$fail"
