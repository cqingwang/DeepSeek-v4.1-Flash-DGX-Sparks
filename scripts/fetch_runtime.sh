#!/usr/bin/env bash
# Fetch third-party target sources at their pinned commits and stage shared Spark ring
# sources from the parent spark/ring/b12x module.
#
#   scripts/fetch_runtime.sh OUT_DIR [sglang-canary] [b12x] [b12x-next]   (default: all three)
#
#   OUT_DIR/sglang-canary/python   SGLang dsv4.1 branch python/ tree (unpatched; the Dockerfile
#                                  applies runtime/sglang-rocenante.patch + runtime/roce_tp4_adapt.py)
#   OUT_DIR/b12x/{b12x,LICENSE}    SG17 b12x plus the shared b12x.comm.roce_ring package
#   OUT_DIR/b12x_next/{b12x_next,LICENSE}
#                                  b12x main renamed to the package b12x_next, patched
#
# The Dockerfiles run this in their `fetch` stage. On a host it needs bash, curl, tar, perl,
# patch and sha256sum (or shasum). Nothing is trusted without its sha256.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="${MANIFEST:-$ROOT/runtime/sources.manifest}"
SPARK_RING_DIR="${SPARK_RING_DIR:-$ROOT/../ring/b12x}"
OUT="${1:?usage: scripts/fetch_runtime.sh OUT_DIR [sglang-canary] [b12x] [b12x-next]}"
shift
COMPONENTS=("$@")
[[ ${#COMPONENTS[@]} -gt 0 ]] || COMPONENTS=(sglang-canary b12x b12x-next)

die() { echo "fetch_runtime: $*" >&2; exit 1; }
if command -v sha256sum >/dev/null; then sha256() { sha256sum "$1" | cut -d' ' -f1; }
else sha256() { shasum -a 256 "$1" | cut -d' ' -f1; }; fi

# manifest row for a component id: "commit sha256 url"
row() {
  local line
  line=$(awk -v id="$1" '$1 == id { print $2, $3, $4; n++ } END { exit n != 1 }' "$MANIFEST") \
    || die "manifest must have exactly one row for $1"
  printf '%s\n' "$line"
}

# fetch ID DEST: download the manifest url for ID to DEST and check its sha256
fetch() {
  local id="$1" dest="$2" commit sum url got
  read -r commit sum url <<<"$(row "$id")"
  [[ "$commit" =~ ^[0-9a-f]{40}$ && "$sum" =~ ^[0-9a-f]{64}$ ]] || die "$id: malformed manifest row"
  [[ "$url" == https://*"$commit"* ]] || die "$id: url does not name commit $commit"
  curl -fsSL --retry 3 --retry-delay 5 -m 900 -o "$dest" "$url" || die "$id: download failed ($url)"
  got=$(sha256 "$dest")
  [[ "$got" == "$sum" ]] || die "$id: sha256 $got, manifest says $sum"
  echo "  $id @ ${commit:0:12}  sha256 ok"
}

# local files used as patches/overrides must match the manifest too
check_local() {
  local path="$1" want got
  want=$(awk -v p="$path" '$1 == "local" && $2 == p { print $3 }' "$MANIFEST")
  [[ -n "$want" ]] || die "manifest has no local row for $path"
  got=$(sha256 "$ROOT/$path")
  [[ "$got" == "$want" ]] || die "$path: sha256 $got, manifest says $want"
}

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"

fetch_sglang_canary() {
  echo "sglang-canary"
  fetch sglang-canary "$tmp/sglang.tgz"
  rm -rf "$OUT/sglang-canary"
  mkdir -p "$OUT/sglang-canary" "$tmp/sglang"
  tar -xzf "$tmp/sglang.tgz" -C "$tmp/sglang" --strip-components=1
  [[ -d "$tmp/sglang/python/sglang" ]] || die "sglang-canary: unexpected archive layout"
  mv "$tmp/sglang/python" "$OUT/sglang-canary/python"
  read -r commit _ <<<"$(row sglang-canary)"
  echo "$commit" > "$OUT/sglang-canary/REF"
}

fetch_b12x() {
  echo "b12x (SG17 + roce_ring)"
  fetch b12x-sg17 "$tmp/b12x-sg17.tgz"
  rm -rf "$OUT/b12x"
  mkdir -p "$OUT/b12x" "$tmp/sg17"
  tar -xzf "$tmp/b12x-sg17.tgz" -C "$tmp/sg17"
  [[ -d "$tmp/sg17/b12x/comm/roce" && -f "$tmp/sg17/LICENSE" ]] || die "b12x-sg17: unexpected archive layout"
  mv "$tmp/sg17/b12x" "$tmp/sg17/LICENSE" "$OUT/b12x/"
  check_local scripts/b12x-required-gid.patch
  (cd "$OUT/b12x" && patch -p1 --forward --quiet --no-backup-if-mismatch < "$ROOT/scripts/b12x-required-gid.patch") \
    || die "b12x required-GID patch did not apply"
}

fetch_b12x_next() {
  echo "b12x-next (b12x main as b12x_next)"
  fetch b12x-main "$tmp/b12x-main.tgz"
  local commit dst="$OUT/b12x_next" patch_file="$ROOT/scripts/b12x_next-compact-n64-m64.patch"
  read -r commit _ <<<"$(row b12x-main)"
  rm -rf "$dst"
  mkdir -p "$dst" "$tmp/main"
  tar -xzf "$tmp/b12x-main.tgz" -C "$tmp/main" --strip-components=1
  [[ -d "$tmp/main/b12x" && -f "$tmp/main/LICENSE" ]] || die "b12x-main: unexpected archive layout"
  cp -R "$tmp/main/b12x" "$dst/b12x_next"
  cp "$tmp/main/LICENSE" "$dst/LICENSE"
  find "$dst/b12x_next" -name __pycache__ -type d -prune -exec rm -rf {} +
  # Both b12x revisions live in one image (RoCEnante needs the SG17 one), so main is renamed:
  # every whole-word `b12x` identifier/string in .py/.c/.cpp/.h -> `b12x_next` (imports,
  # CompileJob "module:function" strings, torch.library namespaces, cache paths), and every
  # B12X_* knob -> B12X_NEXT_*, so compile/tuning caches and knobs stay separate.
  find "$dst/b12x_next" -type f \( -name '*.py' -o -name '*.c' -o -name '*.cpp' -o -name '*.h' \) -print0 \
    | xargs -0 perl -pi -e 's/\bb12x\b(?!-)/b12x_next/g; s/\bB12X_(?!NEXT_)/B12X_NEXT_/g'
  echo "$commit" > "$dst/b12x_next/SOURCE_COMMIT"
  # admit the M64 tile for compact-N64 (N=576, EP1) prefill capacities (b12x pins them to M16)
  check_local scripts/b12x_next-compact-n64-m64.patch
  (cd "$dst" && patch -p0 --forward --quiet < "$patch_file") || die "compact-n64-m64 patch did not apply"
  grep -q "_compact_n64_tiles" "$dst/b12x_next/moe/fused_moe/_tuning.py" || die "compact-n64-m64 patch did not apply"
  sha256 "$patch_file" | cut -c1-16 > "$dst/b12x_next/SOURCE_PATCH"
  # The measured images never carried sequence/engram (an op package unused by the MoE path;
  # a repository-wide `engram/` ignore rule kept it out). Left out so the build matches them.
  rm -rf "$dst/b12x_next/sequence/engram"
  local left
  left=$(find "$dst/b12x_next" -name '*.py' -print0 | xargs -0 perl -ne 'print "$ARGV\n" if /\bb12x\b(?!-)/ || /\bB12X_(?!NEXT_)/' | sort -u)
  [[ -z "$left" ]] || die "unrenamed b12x references in: $left"
}

copy_shared_ring_runtime() {
  local source="$SPARK_RING_DIR/comm"
  [[ -d "$OUT/b12x/b12x/comm" || -d "$OUT/b12x_next/b12x_next/comm" ]] || return 0
  [[ -f "$source/roce_env.py" && -f "$source/roce_ring/roce_oneshot.py" ]] \
    || die "shared Spark ring source is missing under $SPARK_RING_DIR"
  if [[ -d "$OUT/b12x/b12x/comm" ]]; then
    cp "$source/roce_env.py" "$OUT/b12x/b12x/comm/roce_env.py"
    rm -rf "$OUT/b12x/b12x/comm/roce_ring"
    cp -R "$source/roce_ring" "$OUT/b12x/b12x/comm/roce_ring"
  fi
  if [[ -d "$OUT/b12x_next/b12x_next/comm" ]]; then
    cp "$source/roce_env.py" "$OUT/b12x_next/b12x_next/comm/roce_env.py"
  fi
}

for c in "${COMPONENTS[@]}"; do
  case "$c" in
    sglang-canary) fetch_sglang_canary ;;
    b12x) fetch_b12x ;;
    b12x-next) fetch_b12x_next ;;
    *) die "unknown component $c (sglang-canary, b12x, b12x-next)" ;;
  esac
done
copy_shared_ring_runtime
check_local runtime/sglang-rocenante.patch
check_local runtime/flash_mla_sm120.canary.py
echo "fetched into $OUT: ${COMPONENTS[*]}"
