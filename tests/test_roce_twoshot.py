#!/usr/bin/env python3
"""CPU checks for the two-shot RoCE all-reduce (b12x.comm.roce_twoshot, flashinfer#6915 port).

- Default off is production: b12x.comm.roce is byte-identical to the 1059dfe copy, the copy's
  one-shot and all-gather kernels differ from it only by the kind word and their compile ids, and
  DSV41_ROCE_TWOSHOT unset selects b12x.comm.roce in the patched pynccl.py.
- The kernel, the proxy and the host agree on the control-record words, the ABI and the chunk cap;
  the proxy releases pipe_send only after every write completed and refuses two-shot ops after a
  32-bit sequence wrap (the only case in which a stale ready / flag / pipe_done word can equal seq).
- The two-shot owner reduces with the one-shot's own code (sources in rank order, FP32, one
  rounding), and a byte-level model of all four ranks (kernel phases + proxy posts, using the
  runtime's geometry) reproduces the one-shot result bit for bit on every rank, writes every pack
  exactly once and never reads a receive word no peer wrote; uneven shards, empty shards and the
  64-chunk cap included.
- Routing is a pure function of the shape and replicated configuration (rank-invariant), 0 = off.
- The adapter reads DSV41_ROCE_TWOSHOT_ROWS per call through ab_variant.env; the A/B registry
  tells 0, 48 and 64 apart; sitecustomize and l2_prefetch know the new module.
- With gcc and libibverbs headers (Linux, the image): the proxy builds with -Wall -Wextra -Werror
  and roce_layout matches the Python model of the region.

No GPU, torch or network. usage: PYTHONPATH=adapter python3 tests/test_roce_twoshot.py
"""
import ast
import ctypes
import hashlib
import importlib.util
import math
import os
import random
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
B12X = ROOT / "runtime/b12x/b12x/comm"
if not (B12X / "roce_twoshot").is_dir():  # an overlay image: the package is installed in /opt/b12x only
    B12X = Path("/opt/b12x/b12x/comm")
PROD = B12X / "roce"
TWO = B12X / "roce_twoshot"
ADAPTER = ROOT / "adapter"
RUNTIME = ROOT / "runtime"

# b12x.comm.roce as shipped in main 1059dfe (production since 2026-10-23). The two-shot work must
# never touch it: DSV41_ROCE_TWOSHOT unset keeps this exact transport.
PRODUCTION_ROCE_SHA256 = {
    "__init__.py": "17070372720121037728d2a3fdd9f53739929c751f1e17f16a124ce8991c589e",
    "_allgather_cute.py": "4ec1bb11cdba1f0a23c30b9cebc366381417a1cf18eef7f7dbbb2567190bdb33",
    "_cute_intrinsics.py": "11737796a80a0966c04ee347d5ef83a662bde16dc188022134d2cbaa1567410b",
    "_oneshot_cute.py": "e796a354c363be8153b9112f5b46b6249c7762e0293c2e3d8541ba036bcf3802",
    "_proxy.py": "db0aaca54df8589f65500ee3a07a39a1d005262d9821b3f9e01ebced7523ced0",
    "_roce_proxy.c": "a35f54cf75d6abf2427c5d144300647685816060348883d2cbf51d153efc5c70",
    "api.py": "aaa3dac7a1e06cd8ee54ac1d722dfa1a58d1c1a02118f12b54ed9b763d6fed68",
    "roce_oneshot.py": "df4bcacb3acb6d9a43955857238975285dd185eacf3db6493da481b563aeb213",
}


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


plan = _load("twoshot_plan", TWO / "_twoshot_plan.py")


def c_defines(text):
    return {m.group(1): int(m.group(2)) for m in re.finditer(r"^#define (\w+) (\d+)\b", text, re.M)}


# -- bf16 / fp32 arithmetic of the kernels (cvt.rn, FP32 adds) ------------------------------------

def f32(x):
    try:
        return struct.unpack("<f", struct.pack("<f", x))[0]
    except OverflowError:  # rounds past FLT_MAX: the FP32 add gives +-inf
        return math.copysign(math.inf, x)


def bf16_to_f32(bits):
    return struct.unpack("<f", struct.pack("<I", (bits & 0xFFFF) << 16))[0]


def f32_to_bf16(x):
    bits = struct.unpack("<I", struct.pack("<f", x))[0]
    if (bits & 0x7F800000) == 0x7F800000 and bits & 0x007FFFFF:
        return 0x7FFF
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF


def reduce_pack(sources):
    """The one-shot kernel's pack reduction: accumulator = source 0, then += sources 1.. in rank
    order, all in FP32 (a double sum of two floats rounded to float is the correctly rounded FP32
    sum), then one RNE conversion per lane (_store_accumulator)."""
    out = []
    for lane in range(len(sources[0])):
        acc = bf16_to_f32(sources[0][lane])
        for src in sources[1:]:
            acc = f32(acc + bf16_to_f32(src[lane]))
        out.append(f32_to_bf16(acc))
    return tuple(out)


def random_bf16(rng):
    pick = rng.random()
    if pick < 0.02:
        return rng.choice((0x0000, 0x8000, 0x0001, 0x8001, 0x7F7F, 0xFF7F))  # +-0, subnormal, max
    return f32_to_bf16(rng.uniform(-4, 4) * 2.0 ** rng.randint(-20, 20))


# -- a byte-level model of the protocol (kernel phases + proxy posts) ------------------------------

def simulate_twoshot(inputs, world, chunk_packs_min, max_chunks=plan.MAX_CHUNKS, pipe_packs=None):
    """Run the two-shot op on ``world`` ranks; ``inputs[r]`` is a list of packs (tuples of 8 bf16).

    Returns (outputs per rank, stats). Raises AssertionError on a protocol violation."""
    total = len(inputs[0])
    nbytes = total * plan.PACK_BYTES
    chunk_packs, chunks = plan.pipe_geometry(nbytes, world, chunk_packs_min, max_chunks)
    assert 1 <= chunks <= max_chunks
    pipe_packs = total if pipe_packs is None else pipe_packs
    shard_cap = -(-pipe_packs // world)  # in packs (the C rounds the bytes up to 4 KiB as well)
    widest = max(plan.shard_start(j + 1, total, world) - plan.shard_start(j, total, world)
                 for j in range(world))
    assert widest <= shard_cap and chunks * chunk_packs >= widest
    send = [[None] * total for _ in range(world)]                   # pipe_send per rank
    recv = [[None] * (world * 2 * shard_cap) for _ in range(world)]  # pipe_recv[src][phase]
    out = [[None] * total for _ in range(world)]
    writes = [[0] * total for _ in range(world)]
    flags = set()                                                   # (dst, phase, src, chunk)

    def chunk_range(shard, chunk):
        first = plan.shard_start(shard, total, world)
        end = plan.shard_start(shard + 1, total, world)
        lo = min(first + chunk * chunk_packs, end)
        return first, lo, min(end, lo + chunk_packs)

    def post(rank, phase, chunk):                                   # post_pipe_chunk
        for p in range(world):
            if p == rank:
                continue
            first, lo, hi = chunk_range(p if phase == 0 else rank, chunk)
            base = (rank * 2 + phase) * shard_cap + (lo - first)
            for i in range(lo, hi):
                assert send[rank][i] is not None, "proxy posted an unstaged pack"
                assert (lo - first) + (i - lo) < shard_cap
                recv[p][base + (i - lo)] = send[rank][i]
            flags.add((p, phase, rank, chunk))

    for c in range(chunks):                                          # 2. stage, ready_rs[c]
        for r in range(world):
            for shard in range(world):
                if shard != r:
                    _, lo, hi = chunk_range(shard, c)
                    for i in range(lo, hi):
                        send[r][i] = inputs[r][i]
        for r in range(world):
            post(r, 0, c)
    for c in range(chunks):                                          # 3. reduce own shard, ready_ag[c]
        for r in range(world):
            assert all((r, 0, q, c) in flags for q in range(world) if q != r)
            own_first, lo, hi = chunk_range(r, c)
            for i in range(lo, hi):
                sources = []
                for src in range(world):
                    if src == r:
                        sources.append(inputs[r][i])
                    else:
                        v = recv[r][src * 2 * shard_cap + (i - own_first)]
                        assert v is not None, "reduced a receive word no peer wrote"
                        sources.append(v)
                value = reduce_pack(sources)
                out[r][i] = value
                send[r][i] = value
                writes[r][i] += 1
        for r in range(world):
            post(r, 1, c)
    for c in range(chunks):                                          # 4. gather
        for r in range(world):
            assert all((r, 1, q, c) in flags for q in range(world) if q != r)
            for shard in range(world):
                if shard != r:
                    first, lo, hi = chunk_range(shard, c)
                    for i in range(lo, hi):
                        v = recv[r][(shard * 2 + 1) * shard_cap + (i - first)]
                        assert v is not None, "gathered a receive word no peer wrote"
                        out[r][i] = v
                        writes[r][i] += 1
    return out, {"writes": writes, "chunks": chunks, "chunk_packs": chunk_packs}


def simulate_oneshot(inputs, world):
    total = len(inputs[0])
    reduced = [reduce_pack([inputs[s][i] for s in range(world)]) for i in range(total)]
    return [list(reduced) for _ in range(world)]


class ProductionUntouched(unittest.TestCase):
    def test_production_package_is_byte_identical(self):
        got = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(PROD.iterdir()) if p.is_file() and p.suffix in (".py", ".c")}
        self.assertEqual(got, PRODUCTION_ROCE_SHA256)

    def _normalized(self, path, drop_doc_head):
        text = path.read_text()
        text = re.sub(r"\n                    # kind\[slot\] = 0: a one-shot payload.*?\n"
                      r"                    \)\n", "\n", text, flags=re.S)
        text = re.sub(r"        # ds41 sequence-wrap guard:.*?            cute\.arch\.sync_threads\(\)\n", "",
                      text, flags=re.S)
        if drop_doc_head:
            text = re.sub(r'\n\nb12x\.comm\.roce_twoshot copy:.*?the compile ids are ``comm\.roce_twoshot\.\*``\.\n',
                          "\n", text, count=1, flags=re.S)
        return text.replace("comm.roce_twoshot.", "comm.roce.")

    def test_copy_kernels_differ_only_by_the_kind_word_and_wrap_guard(self):
        self.assertEqual((TWO / "_cute_intrinsics.py").read_bytes(), (PROD / "_cute_intrinsics.py").read_bytes())
        for name, kinds in (("_oneshot_cute.py", 1), ("_allgather_cute.py", 2)):
            with self.subTest(name=name):
                text = (TWO / name).read_text()
                self.assertEqual(text.count("ctrl_base + Int64(32) + slot * Int64(4), Uint32(0)"), kinds)
                # kind is stored before the fence that precedes the doorbell
                for m in re.finditer(r"Int64\(32\) \+ slot \* Int64\(4\), Uint32\(0\)\s*\)\s*"
                                     r"fence_sc_sys\(\)\s*st_relaxed_sys_u32\(ctrl_base, seq\)", text):
                    kinds -= 1
                self.assertEqual(kinds, 0, "kind=0 must precede fence_sc_sys + the seq doorbell")
                self.assertEqual(self._normalized(TWO / name, True), (PROD / name).read_text())


class ControlRecord(unittest.TestCase):
    C = c_defines((TWO / "_roce_proxy.c").read_text())

    def test_kernel_and_proxy_agree(self):
        src = (TWO / "_twoshot_cute.py").read_text()
        kernel = {m.group(1): int(m.group(2)) for m in re.finditer(r"^_CTRL_(\w+) = (\d+)", src, re.M)}
        self.assertEqual(kernel, {k: self.C["CTRL_" + k] for k in kernel})
        self.assertEqual(set(kernel), {"SLOT_KIND", "SLOT_CHUNK_PACKS", "SLOT_CHUNKS", "PIPE_DONE",
                                       "READY_RS", "READY_AG"})
        self.assertEqual(self.C["CTRL_SLOT_KIND"] * 4, 32)        # the one-shot kind store
        self.assertEqual(self.C["CTRL_SLOT_NBYTES"] * 4, 16)      # nbytes[slot] of every kernel
        self.assertEqual(self.C["CTRL_ERROR"] * 4, 8)
        self.assertLessEqual((self.C["CTRL_READY_AG"] + self.C["ROCE_MAX_CHUNKS"]) * 4,
                             self.C["ROCE_CTRL_BYTES"])
        self.assertLessEqual(self.C["CTRL_READY_RS"] + self.C["ROCE_MAX_CHUNKS"], self.C["CTRL_READY_AG"])
        self.assertLess(self.C["CTRL_SLOT_CHUNKS"] + 1, self.C["CTRL_PIPE_DONE"])
        self.assertEqual(self.C["ROCE_MAX_CHUNKS"], plan.MAX_CHUNKS)
        runtime = (TWO / "roce_oneshot.py").read_text()
        self.assertIn(f"ctrl_off + {self.C['CTRL_PIPE_DONE'] * 4} :", runtime)

    def test_abi_is_distinct_from_production(self):
        self.assertEqual(self.C["ROCE_ABI_VERSION"], 103)
        self.assertEqual(c_defines((PROD / "_roce_proxy.c").read_text())["ROCE_ABI_VERSION"], 3)
        proxy = (TWO / "_proxy.py").read_text()
        self.assertIn("ABI_VERSION = 103", proxy)
        self.assertIn('f"roce_twoshot_proxy-{digest}.so"', proxy)

    def test_proxy_release_order(self):
        c = (TWO / "_roce_proxy.c").read_text()
        body = c[c.index("static int run_pipe_op("):c.index("static void *proxy_main(")]
        drained = body.index("outstanding_writes(c) == 0")
        done = body.index("__atomic_store_n(&ctrl[CTRL_PIPE_DONE], seq, __ATOMIC_RELEASE);")
        self.assertLess(drained, done)  # pipe_send is released only after every write completed
        # equality on seq is exact until the 32-bit sequence wraps: a two-shot op after a wrap fails closed
        self.assertLess(body.index("if (c->seq_wrapped) {"), body.index("while (rs < chunks || ag < chunks) {"))
        main = c[c.index("static void *proxy_main("):]
        self.assertLess(main.index("c->seq_wrapped = 1;"), main.index("kind == KIND_TWOSHOT"))
        self.assertIn("kind == KIND_ONESHOT", main)
        self.assertIn("kind == KIND_TWOSHOT", main)
        self.assertIn('"unknown op kind %u at seq %u"', main)

    def test_oneshot_posting_is_unchanged(self):
        """post_striped carries the stock post_op's work requests: same flag address per HCA, same
        stripe split, same quarter-queue bound."""
        prod = (PROD / "_roce_proxy.c").read_text()
        two = (TWO / "_roce_proxy.c").read_text()
        for line in ("uint32_t stripe_packs = total_packs / (uint32_t)c->n_hca;",
                     "while (hca->outstanding[p] >= ROCE_SEND_DEPTH / 4) {",
                     "flag_wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;"):
            self.assertIn(line, prod)
        self.assertIn("uint32_t stripe_packs = packs / (uint32_t)c->n_hca;", two)
        self.assertIn("while (hca->outstanding[p] >= ROCE_SEND_DEPTH / 4) {", two)
        self.assertIn("flag_wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;", two)
        # stock flag address: flag_off + ((rank * SLOTS + slot) * n_hca + h) * STRIDE
        self.assertIn("((uint64_t)c->rank * ROCE_SLOTS + slot) * (uint64_t)c->n_hca + (uint64_t)h) * ROCE_FLAG_STRIDE",
                      re.sub(r"\s+", " ", prod))
        self.assertIn("c->flag_off + ((uint64_t)c->rank * ROCE_SLOTS + slot) * (uint64_t)c->n_hca * ROCE_FLAG_STRIDE",
                      re.sub(r"\s+", " ", two))
        self.assertIn("remote + flag_off + (uint64_t)h * ROCE_FLAG_STRIDE", two)


class SameArithmetic(unittest.TestCase):
    """The owner's reduction is the one-shot's code: same helpers, sources in rank order."""

    def _reduce_calls(self, path, func):
        tree = ast.parse(path.read_text())
        calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == func:
                for sub in ast.walk(node):
                    if isinstance(sub, ast.For) and isinstance(sub.iter, ast.Call) and \
                            ast.unparse(sub.iter) == "cutlass.range_constexpr(self._world_size)":
                        calls.append(ast.unparse(sub))
        return calls

    def test_owner_reduce_matches_oneshot(self):
        one = self._reduce_calls(TWO / "_oneshot_cute.py", "kernel")
        two = self._reduce_calls(TWO / "_twoshot_cute.py", "kernel")
        self.assertEqual(len(one), 1)
        reduce_loops = [t for t in two if "_accumulate_words" in t]
        self.assertEqual(len(reduce_loops), 1)
        for text in (one[0], reduce_loops[0]):
            self.assertIn("self._accumulate_words(accumulator, words, source == 0)", text)
            self.assertIn("if cutlass.const_expr(source == self._rank):", text)
            self.assertIn("words = ld_global_v4_u32(input_base + offset)", text)
        src = (TWO / "_twoshot_cute.py").read_text()
        self.assertIn("class _RoceTwoshotLaunch(_RoceOneshotLaunch):", src)
        self.assertNotIn("def _accumulate_words", src)   # inherited, not redefined
        self.assertNotIn("def _store_accumulator", src)
        self.assertEqual(src.count("self._store_accumulator(output_base + offset, accumulator)"), 1)
        self.assertEqual(src.count("self._store_accumulator(send_base + offset, accumulator)"), 1)

    def test_kernel_split_matches_plan(self):
        src = (TWO / "_twoshot_cute.py").read_text()
        self.assertIn("return Int32((Int64(shard) * Int64(size_packs)) // Int64(self._world_size))", src)
        c = (TWO / "_roce_proxy.c").read_text()
        self.assertIn("return (uint32_t)(((uint64_t)j * total) / (uint64_t)c->world);", c)


class ProtocolModel(unittest.TestCase):
    def _check(self, world, packs, chunk_packs_min, seed, pipe_packs=None):
        rng = random.Random(seed)
        inputs = [[tuple(random_bf16(rng) for _ in range(8)) for _ in range(packs)] for _ in range(world)]
        got, stats = simulate_twoshot(inputs, world, chunk_packs_min, pipe_packs=pipe_packs)
        want = simulate_oneshot(inputs, world)
        for r in range(world):
            self.assertEqual(stats["writes"][r], [1] * packs, f"rank {r} pack coverage")
            self.assertEqual(got[r], want[r], f"rank {r} differs from the one-shot")
        return stats

    def test_bit_identical_to_oneshot(self):
        cases = [(4, 61440 // 16, 8192), (4, 245760 // 16, 8192), (4, 491520 // 16, 8192),
                 (4, 983040 // 16, 8192), (4, 1966080 // 16, 8192),      # c1 c4 c8 c16 rows; fp32 verify AR
                 (4, 1, 8192), (4, 3, 8192), (4, 5, 1), (4, 7, 2),       # empty and uneven shards
                 (2, 9, 1), (3, 10, 2), (8, 33, 1), (4, 997, 3)]
        for world, packs, chunk, in cases:
            with self.subTest(world=world, packs=packs, chunk=chunk):
                self._check(world, packs, chunk, seed=world * 100003 + packs)

    def test_chunk_cap(self):
        stats = self._check(4, 4 * 64 * 3, 1, seed=7)       # shard 192 packs: 64 chunks of 3
        self.assertEqual((stats["chunks"], stats["chunk_packs"]), (plan.MAX_CHUNKS, 3))
        stats = self._check(4, 4 * 64 * 3 + 5, 1, seed=8)   # shard 194: chunks grow to 4 packs
        self.assertEqual((stats["chunks"], stats["chunk_packs"]), (49, 4))
        for packs in range(1, 3000, 7):
            chunk_packs, chunks = plan.pipe_geometry(packs * 16, 4, 1)
            self.assertLessEqual(chunks, plan.MAX_CHUNKS)
            self.assertGreaterEqual(chunks * chunk_packs, -(-packs // 4))

    def test_smaller_message_in_a_wider_pipe(self):
        self._check(4, 37, 2, seed=11, pipe_packs=131072)

    def test_inplace_safe_order(self):
        """In place (out is inp): a peer shard's pack is overwritten (gather) only after its chunk
        was staged by every block, because the gather waits for flags that need ready_rs."""
        src = (TWO / "_twoshot_cute.py").read_text()
        stage = src.index("# 2. stage every peer's shard")
        reduce_ = src.index("# 3. reduce our shard")
        gather = src.index("# 4. gather every peer's reduced shard")
        self.assertLess(stage, reduce_)
        self.assertLess(reduce_, gather)
        self.assertIn("self._wait_chunk(\n                        flag_base, 1, chunk", src)



UINT32 = 0xFFFFFFFF
CTRL, EPOCH, POISON, COUNTERS = 0x1000, 0x2000, 0x2100, 0x2200


class U32(int):
    def __new__(cls, value):
        return super().__new__(cls, int(value) & UINT32)

    def __add__(self, other):
        return U32(int(self) + int(other))


def run_kernel(filename, cls_name, sequence, args):
    """Execute one copy kernel's control flow for thread 0 of block 0 (grid 1) against CPU words;
    the payload / flag accessors record every touch (fault injection)."""
    tree = ast.parse((TWO / filename).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls_name)
    kernel = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "kernel")
    kernel.decorator_list, kernel.returns = [], None
    for a in kernel.args.args:
        a.annotation = None
    mem = {EPOCH: U32(sequence - 1)}
    log = {"spin": [], "payload": 0}

    def load(addr):
        return U32(mem.get(addr, 0))

    def store(addr, value):
        mem[addr] = U32(value)

    def add(addr, value):
        prev = load(addr)
        store(addr, prev + value)
        return prev

    def spin(addr, expected, limit):
        log["spin"].append((addr, int(expected)))
        return U32(0)  # every flag "ready": the worst case for a guard that must not look at them

    def payload(*a):
        log["payload"] += 1
        return (U32(0),) * 4

    ns = {"Uint32": U32, "Int32": int, "Int64": int, "PACK_BYTES": 16,
          "cute": SimpleNamespace(arch=SimpleNamespace(thread_idx=lambda: (0, 0, 0), block_idx=lambda: (0, 0, 0),
                                                       grid_dim=lambda: (1, 1, 1), sync_threads=lambda: None),
                                  make_rmem_tensor=lambda *a: [0.0] * 8),
          "cutlass": SimpleNamespace(const_expr=bool, range_constexpr=range, range=lambda n, unroll=1: range(int(n)),
                                     min=min, Float32=float, Uint32=U32),
          "ld_relaxed_gpu_u32": load, "ld_relaxed_sys_u32": load, "st_relaxed_sys_u32": store,
          "st_release_gpu_u32": store, "atomic_add_relaxed_gpu_u32": add, "fence_sc_sys": lambda: None,
          "fence_sc_gpu": lambda: None, "spin_until_eq_acquire_sys": spin,
          "ld_global_v4_u32": payload, "ld_relaxed_sys_v4_u32": payload, "st_global_v4_u32": payload,
          "_CTRL_SLOT_KIND": 8, "_CTRL_SLOT_CHUNK_PACKS": 10, "_CTRL_SLOT_CHUNKS": 12, "_CTRL_PIPE_DONE": 48,
          "_CTRL_READY_RS": 64, "_CTRL_READY_AG": 128}
    exec(compile(ast.Module(body=[kernel], type_ignores=[]), filename, "exec"), ns)
    rt = SimpleNamespace(_threads=32, _world_size=4, _rank=1, _slots=2, _flag_stride=128, _hca_count=2,
                         _max_chunks=64, _col_packs=(1, 1, 1, 1), _pack_elems=8,
                         _shard_start=lambda shard, size: int(shard) * int(size) // 4)
    rt._wait_chunk = lambda *a: [spin(0, sequence, 0) for _ in range(1)]
    rt._arrive = lambda *a: None
    ptr = SimpleNamespace(toint=lambda: 0x3000)
    base = dict(self=rt, input_ptr=ptr, output_ptr=ptr, ctrl_base=CTRL, epoch_ptr=EPOCH, poison_ptr=POISON,
                spin_limit=U32(1))
    base.update(args)
    params = [a.arg for a in kernel.args.args]
    ns["kernel"](**{k: v for k, v in base.items() if k in params})
    return mem, log


KERNELS = [
    ("_oneshot_cute.py", "_RoceOneshotLaunch",
     dict(size_packs=0, nbytes=0, recv_base=0x4000, flag_base=0x8000, send_base=0xC000, slot_bytes=16,
          stage_counter_ptr=COUNTERS, tail_counter_ptr=COUNTERS + 4)),
    ("_allgather_cute.py", "_RoceAllGatherLaunch",
     dict(shard_packs=0, nbytes=0, row_packs=1, recv_base=0x4000, flag_base=0x8000, send_base=0xC000,
          slot_bytes=16, stage_counter_ptr=COUNTERS, tail_counter_ptr=COUNTERS + 4)),
    ("_allgather_cute.py", "_RoceAllGatherColumnsLaunch",
     dict(rows=0, in_row_packs=1, in_col_off=0, nbytes=0, recv_base=0x4000, flag_base=0x8000,
          send_base=0xC000, slot_bytes=16, stage_counter_ptr=COUNTERS, tail_counter_ptr=COUNTERS + 4)),
    ("_twoshot_cute.py", "_RoceTwoshotLaunch",
     dict(size_packs=0, nbytes=0, chunk_packs=1, chunks=0, send_base=0xC000, recv_base=0x4000,
          flag_base=0x8000, shard_cap=4096, counter_base=COUNTERS)),
]


class SequenceWrap(unittest.TestCase):
    """At sequence 0 (after a 32-bit wrap) no copy kernel may touch a flag or payload,
    accept anything, or advance the epoch; it must leave a nonzero, seq-independent error and poison."""

    def test_sequence_zero_fails_closed_in_every_kernel(self):
        for filename, cls, args in KERNELS:
            with self.subTest(kernel=cls):
                mem, log = run_kernel(filename, cls, 0, args)
                self.assertEqual(log["spin"], [], "waited on a flag at sequence 0")
                self.assertEqual(log["payload"], 0, "touched payload at sequence 0")
                self.assertEqual(mem.get(CTRL + 8), UINT32)
                self.assertEqual(mem.get(CTRL + 12), UINT32)
                self.assertEqual(mem.get(POISON), UINT32)
                self.assertEqual(mem[EPOCH], UINT32, "the epoch moved")
                self.assertNotIn(CTRL, mem, "rang the doorbell at sequence 0")

    def test_poisoned_runtime_stays_dead(self):
        for filename, cls, args in KERNELS:
            with self.subTest(kernel=cls):
                mem, log = run_kernel(filename, cls, 0, args)
                # the next launch: the epoch did not move, so it is sequence 0 again and poisoned
                self.assertEqual(mem[EPOCH] + 1 & UINT32, 0)

    def test_normal_sequence_is_unaffected(self):
        for filename, cls, args in KERNELS:
            for seq in (1, 5, UINT32):
                with self.subTest(kernel=cls, seq=seq):
                    mem, log = run_kernel(filename, cls, seq, args)
                    self.assertTrue(log["spin"], "the guard fired at a valid sequence")
                    self.assertEqual(mem.get(CTRL + 8, 0), 0)
                    self.assertEqual(mem.get(POISON, 0), 0)
                    self.assertEqual(mem[EPOCH], seq)

    def _health(self):
        tree = ast.parse((TWO / "roce_oneshot.py").read_text())
        consts = {}
        for n in tree.body:
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and \
                    n.targets[0].id in ("SEQUENCE_GUARD", "WRAP_SENTINEL"):
                consts[n.targets[0].id] = eval(compile(ast.Expression(n.value), "c", "eval"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RoceOneshotAllReduce")
        fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "check_health")
        ns = dict(consts)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "h", "exec"), ns)
        return ns["check_health"], consts

    def test_host_guard(self):
        check, c = self._health()
        self.assertGreater(UINT32 - c["SEQUENCE_GUARD"], 1000)  # >> collectives between two checks
        rt = SimpleNamespace(_proxy=None, _seq_exhausted=False, rank=0, _ctrl_np=[0] * 8)
        rt._ctrl_np[0] = c["SEQUENCE_GUARD"] - 1
        check(rt)  # healthy below the guard
        for word in (c["SEQUENCE_GUARD"], c["SEQUENCE_GUARD"] - (1 << 32)):  # int32 view of the same bits
            rt = SimpleNamespace(_proxy=None, _seq_exhausted=False, rank=0, _ctrl_np=[word, 0, 0, 0, 0, 0, 0, 0])
            with self.assertRaises(RuntimeError):
                check(rt)
            rt._ctrl_np[0] = 7  # sticky: a later healthy-looking word does not revive it
            with self.assertRaises(RuntimeError):
                check(rt)
        rt = SimpleNamespace(_proxy=None, _seq_exhausted=False, rank=2, _ctrl_np=[0, 0, -1, -1, 0, 0, -1, 0])
        with self.assertRaisesRegex(RuntimeError, "sequence 0"):
            check(rt)  # the kernel's wrap sentinel (int32 view) is reported as such

    def test_route_receipt_names_the_variant(self):
        src = (TWO / "roce_oneshot.py").read_text()
        self.assertIn('"ROCE_TWOSHOT_ROUTE rank=%d min_rows=%d rows=%d bytes=%d dtype=%s"', src)
        self.assertIn("if route not in self._twoshot_routes_logged:", src)

class Routing(unittest.TestCase):
    def test_default_off(self):
        self.assertEqual(plan.DEFAULT_MIN_ROWS, 0)
        for rows in (1, 6, 96, 4096):
            self.assertFalse(plan.uses_pipe(rows, rows * 10240, pipe_bytes=2 << 20, min_rows=0))

    def test_threshold(self):
        row = 10240  # one bf16 row of the 5120-wide residual
        for rows, want in ((6, False), (24, False), (47, False), (48, True), (96, True),
                           (204, True), (205, False)):  # 205 * 10240 > 2 MiB
            with self.subTest(rows=rows):
                self.assertEqual(plan.uses_pipe(rows, rows * row, pipe_bytes=2 << 20, min_rows=48), want)
        self.assertFalse(plan.uses_pipe(64, 64 * 10 + 8, pipe_bytes=2 << 20, min_rows=48))  # not 16 B
        self.assertFalse(plan.uses_pipe(64, 64 * row, pipe_bytes=0, min_rows=48))

    def test_rows_of_is_shape_only(self):
        self.assertEqual(plan.rows_of((96, 5120)), 96)
        self.assertEqual(plan.rows_of((96, 2, 2560)), 96)
        self.assertEqual(plan.rows_of((491520,)), 0)  # 1-D tensors never take the two-shot path
        self.assertEqual(plan.rows_of(()), 0)

    def test_runtime_routes_with_the_plan(self):
        tree = ast.parse((TWO / "roce_oneshot.py").read_text())
        fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "uses_twoshot")
        text = ast.unparse(fn)
        self.assertIn("rows_of(tuple(inp.shape))", text)
        self.assertIn("inp.numel() * inp.element_size()", text)
        self.assertNotIn("data_ptr", text)  # never a pointer
        runtime = (TWO / "roce_oneshot.py").read_text()
        self.assertIn('"pipe_min_rows": self.pipe_min_rows,', runtime)  # checked equal on all ranks
        self.assertIn("may not exceed max_size", runtime)

    def test_geometry(self):
        self.assertEqual(plan.pipe_geometry(983040, 4, 8192), (8192, 2))
        self.assertEqual(plan.pipe_geometry(61440, 4, 8192), (8192, 1))
        self.assertEqual(plan.pipe_geometry(16, 4, 8192), (8192, 1))
        with self.assertRaises(ValueError):
            plan.pipe_geometry(24, 4, 8192)


class Adapter(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, str(ADAPTER))
        self.saved = {k: os.environ.get(k) for k in ("DSV41_ROCE_TWOSHOT_ROWS",)}

    def tearDown(self):
        sys.path.remove(str(ADAPTER))
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        sys.modules.pop("roce_twoshot", None)

    def _fake_module(self):
        cls = type("RoceOneshotAllReduce", (), {"pipe_min_rows_fn": None})
        return types.SimpleNamespace(RoceOneshotAllReduce=cls), cls

    def test_install_reads_per_call(self):
        import roce_twoshot
        module, cls = self._fake_module()
        os.environ["DSV41_ROCE_TWOSHOT_ROWS"] = "48"
        roce_twoshot.install(module)
        self.assertEqual(cls.pipe_min_rows_fn(), 48)
        os.environ["DSV41_ROCE_TWOSHOT_ROWS"] = "0"
        self.assertEqual(cls.pipe_min_rows_fn(), 0)
        os.environ.pop("DSV41_ROCE_TWOSHOT_ROWS")
        self.assertEqual(cls.pipe_min_rows_fn(), 0)
        roce_twoshot.install(module)  # idempotent

    def test_negative_refused_at_install(self):
        import roce_twoshot
        module, _ = self._fake_module()
        os.environ["DSV41_ROCE_TWOSHOT_ROWS"] = "-1"
        with self.assertRaises(ValueError):
            roce_twoshot.install(module)

    def test_ab_registry(self):
        import ab_variant
        self.assertIn("DSV41_ROCE_TWOSHOT_ROWS", ab_variant.RUNTIME)
        self.assertIn("roce_twoshot", ab_variant.HASHED_SOURCES)
        norm = [ab_variant._norm({"DSV41_ROCE_TWOSHOT_ROWS": v})["DSV41_ROCE_TWOSHOT_ROWS"]
                for v in (None, "", "0", "48", "64")]
        self.assertEqual(norm, [0, 0, 0, 48, 64])

    def test_sitecustomize_and_l2_prefetch_know_the_module(self):
        site = (ADAPTER / "sitecustomize.py").read_text()
        self.assertEqual(site.count("'b12x.comm.roce_twoshot.roce_oneshot'"), 3)
        self.assertIn("os.environ.get('DSV41_ROCE_TWOSHOT', '0').strip() == '1'", site)
        l2 = (ADAPTER / "l2_prefetch.py").read_text()
        self.assertIn('"b12x.comm.roce_twoshot.roce_oneshot")', l2)


class OverlaySelection(unittest.TestCase):
    FIXTURE = (
        '        if roce_enabled and (os.environ.get("SGLANG_ROCE_ALLREDUCE", "0") == "1"\n'
        '                             or os.environ.get("SGLANG8_ROCE_ALLREDUCE", "0") == "1"):\n'
        '            import importlib\n'
        '            roce = importlib.import_module(\n'
        '                "b12x.comm.roce_ring" if os.environ.get("DSV41_ROCE_RING", "0") == "1" else "b12x.comm.roce")\n'
        '\n'
        '            self.roce = None\n'
    )
    PYNCCL = Path("/sgl-workspace/sglang/python/sglang/srt/distributed/device_communicators/pynccl.py")

    def _select(self, src, environ):
        """Evaluate the patched selection the way pynccl.py runs it."""
        lines = [line for line in src.splitlines() if line.strip() != "import importlib"]
        body = "def f(roce_enabled, os, importlib):\n" + "\n".join(
            "    " + line[8:] if line.strip() else "" for line in lines)
        body += "\n        return roce\n"
        ns = {}
        exec(compile(body, "<pynccl>", "exec"), ns)
        fake_os = types.SimpleNamespace(environ=environ)
        fake_importlib = types.SimpleNamespace(import_module=lambda name: name)
        return ns["f"](True, fake_os, fake_importlib)

    def test_selection(self):
        adapt = _load("roce_twoshot_adapt", RUNTIME / "roce_twoshot_adapt.py")
        out = adapt.adapt("class X:\n    def __init__(self):\n" + self.FIXTURE)
        snippet = out.split("    def __init__(self):\n", 1)[1].split("\n\n")[0] + "\n"
        on = {"SGLANG_ROCE_ALLREDUCE": "1"}
        self.assertEqual(self._select(snippet, dict(on)), "b12x.comm.roce")
        self.assertEqual(self._select(snippet, dict(on, DSV41_ROCE_TWOSHOT="0")), "b12x.comm.roce")
        self.assertEqual(self._select(snippet, dict(on, DSV41_ROCE_TWOSHOT="1")), "b12x.comm.roce_twoshot")
        self.assertEqual(self._select(snippet, dict(on, DSV41_ROCE_RING="1")), "b12x.comm.roce_ring")
        with self.assertRaises(RuntimeError):
            self._select(snippet, dict(on, DSV41_ROCE_RING="1", DSV41_ROCE_TWOSHOT="1"))
        self.assertEqual(adapt.adapt(out), out)  # idempotent
        with self.assertRaises(SystemExit):
            adapt.adapt("nothing to patch\n")

    def test_dockerfile_runs_it(self):
        docker = (ROOT / "Dockerfile.canary-roce").read_text() if (ROOT / "Dockerfile.canary-roce").is_file() \
            else Path("/opt/dsv41/Dockerfile.canary-roce").read_text()
        self.assertIn("python3 /tmp/roce_tp4_adapt.py && python3 /tmp/roce_twoshot_adapt.py", docker)
        self.assertIn("tests/test_roce_twoshot.py", docker)

    @unittest.skipUnless(PYNCCL.is_file(), "image only")
    def test_image_pynccl(self):
        src = self.PYNCCL.read_text()
        self.assertIn('else ("b12x.comm.roce_twoshot" if os.environ.get("DSV41_ROCE_TWOSHOT", "0") == "1" '
                      'else "b12x.comm.roce"))', src)


def _verbs_available():
    cc = shutil.which("gcc") or shutil.which("cc")
    return cc is not None and any(Path(p, "infiniband/verbs.h").is_file()
                                  for p in ("/usr/include", "/usr/local/include"))


@unittest.skipUnless(_verbs_available(), "needs gcc and libibverbs headers (Linux)")
class NativeProxy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.lib_path = os.path.join(cls.tmp, "proxy.so")
        cmd = [shutil.which("gcc") or "cc", "-O2", "-std=gnu11", "-Wall", "-Wextra", "-Werror", "-shared",
               "-fPIC", "-o", cls.lib_path, str(TWO / "_roce_proxy.c"), "-libverbs", "-lpthread"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise AssertionError(proc.stderr)
        cls.lib = ctypes.CDLL(cls.lib_path)
        cls.lib.roce_layout.restype = ctypes.c_int
        cls.lib.roce_layout.argtypes = [ctypes.c_int, ctypes.c_uint64, ctypes.c_uint64,
                                        ctypes.POINTER(ctypes.c_uint64)]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def layout(self, world, slot, pipe):
        out = (ctypes.c_uint64 * 12)()
        rc = self.lib.roce_layout(world, slot, pipe, out)
        return rc, [int(v) for v in out]

    def test_abi(self):
        self.lib.roce_abi_version.restype = ctypes.c_int
        self.assertEqual(self.lib.roce_abi_version(), 103)

    def test_layout_model(self):
        C = ControlRecord.C
        for world in (2, 4, 8):
            for slot, pipe in ((2 << 20, 0), (2 << 20, 2 << 20), (4096, 4096), (2 << 20, 1 << 20)):
                with self.subTest(world=world, slot=slot, pipe=pipe):
                    rc, o = self.layout(world, slot, pipe)
                    self.assertEqual(rc, 0)
                    recv, flag, send, ctrl, total, stride, slots, ps, pr, pf, cap, mc = o
                    self.assertEqual((recv, stride, slots, mc), (0, 128, 2, 64))
                    self.assertEqual(flag, world * 2 * slot)
                    self.assertEqual(send, flag + world * 2 * C["ROCE_MAX_HCAS"] * 128)
                    self.assertEqual(ctrl, send + 2 * slot)
                    self.assertEqual(ps, ctrl + C["ROCE_CTRL_BYTES"])
                    self.assertEqual(pr, ps + pipe)
                    self.assertEqual(cap % 4096, 0)
                    self.assertGreaterEqual(cap, -(-pipe // 16 // world) * 16)
                    self.assertEqual(pf, pr + world * 2 * cap)
                    self.assertEqual(total, pf + (0 if pipe == 0 else 2 * world * 64 * C["ROCE_MAX_HCAS"] * 128))
        self.assertNotEqual(self.layout(4, 2 << 20, 100)[0], 0)  # pipe not 4 KiB aligned
        self.assertNotEqual(self.layout(1, 2 << 20, 0)[0], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
