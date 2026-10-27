"""Packed-shard parity and the NVMe keepalive: reads only while lookups arrive, none when off.

Run on Linux after building adapter/librow_store.so. The shard must live on a filesystem that
accepts O_DIRECT (not tmpfs): set ROW_STORE_TEST_DIR, default the current directory. Without
O_DIRECT the test reports skipped.
"""
import ctypes as C
import json
import os
from pathlib import Path
import random
import struct
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
ROWS, LO, HI, LAYER = 4099, 1000, 3000, 1
MAGIC = 0x31344e4531565344


def child(directory):
    C.CDLL(None).prctl(29, 1, 0, 0, 0)                  # PR_SET_TIMERSLACK: 1 ns, not inherited 50 us+
    lib = C.CDLL(str(ROOT/'adapter/librow_store.so'))
    lib.row_store_keepalive_reads.restype = C.c_uint64
    reads = lib.row_store_keepalive_reads
    U, P = C.c_uint64, C.c_void_p
    lib.row_store_open.argtypes = [C.c_char_p, U, U, U, U]
    lib.row_store_open.restype = P
    lib.row_store_close.argtypes = [P]
    lib.row_store_range.argtypes = [P, U, U]
    lib.row_store_attach_packed.argtypes = [P, C.c_char_p, U]
    lib.row_store_attach_packed.restype = C.c_int
    lib.row_store_stats.argtypes = [P, C.POINTER(U)]

    class Work(C.Structure):
        _fields_ = [('store', P), ('ids', P), ('weights', P), ('scales', P), ('count', U)]

    lib.row_store_lookup.argtypes = [C.POINTER(Work)]
    rng = random.Random(1023)
    weights, scales = rng.randbytes(ROWS * 256), rng.randbytes(ROWS * 8)
    with tempfile.TemporaryDirectory(dir=directory) as tmp:
        table = Path(tmp, 'table.bin')
        table.write_bytes(weights + scales)
        packed = Path(tmp, 'packed.bin')
        header = struct.pack('<6Q', MAGIC, LAYER, LO, HI, ROWS, 264).ljust(4096, b'\0')
        body = b''.join(weights[i*256:(i+1)*256] + scales[i*8:(i+1)*8] for i in range(LO, HI))
        packed.write_bytes(header + body + bytes(-len(body) % 4096))
        store = lib.row_store_open(str(table).encode(), ROWS, 0, len(weights), 272 * 64)
        assert store
        lib.row_store_range(store, LO, HI)
        if lib.row_store_attach_packed(store, str(packed).encode(), LAYER) != 1:
            print(json.dumps(dict(skipped='no O_DIRECT on ' + directory)))
            return

        def lookup(seed):
            r = random.Random(seed)
            ids = [r.randrange(ROWS) for _ in range(500)]
            idx = (C.c_int64 * len(ids))(*ids)
            w, s = C.create_string_buffer(len(ids) * 256), C.create_string_buffer(len(ids) * 8)
            lib.row_store_lookup(C.byref(Work(store, C.addressof(idx), C.addressof(w),
                                              C.addressof(s), len(ids))))
            for k, i in enumerate(ids):
                own = LO <= i < HI
                assert w.raw[k*256:(k+1)*256] == (weights[i*256:(i+1)*256] if own else bytes(256))
                assert s.raw[k*8:(k+1)*8] == (scales[i*8:(i+1)*8] if own else bytes(8))

        lookup(0)
        stats = (U * 9)()
        lib.row_store_stats(store, stats)
        assert stats[8] == 1 and stats[1] > 0          # packed path, real misses
        base = reads()
        deadline = time.monotonic() + 1.5               # within the 2 s window after a lookup
        while reads() - base < 5 and time.monotonic() < deadline:
            time.sleep(0.01)
        active = reads() - base
        time.sleep(2.5)                                 # past the 2 s window
        base = reads()
        time.sleep(0.5)
        idle = reads() - base
        lookup(1)                                       # a lookup re-arms it, rows still exact
        lib.row_store_close(store)
        print(json.dumps(dict(active_reads=active, idle_reads=idle, misses=stats[1])))


def run(env_value):
    env = dict(os.environ)
    if env_value is None:
        env.pop('DSV41_ENGRAM_KEEPALIVE_MS', None)
    else:
        env['DSV41_ENGRAM_KEEPALIVE_MS'] = env_value
    out = subprocess.run([sys.executable, __file__, '--child',
                          os.environ.get('ROW_STORE_TEST_DIR', os.getcwd())],
                         env=env, check=True, capture_output=True, text=True, timeout=60).stdout
    return json.loads(out.strip().splitlines()[-1])


if __name__ == '__main__':
    if sys.argv[1:2] == ['--child']:
        child(sys.argv[2])
        sys.exit(0)
    default = run(None)
    if 'skipped' in default:
        print(json.dumps(default))
        sys.exit(0)
    off, bad = run('0'), run('-1')
    assert default['active_reads'] >= 5, default
    assert default['idle_reads'] == 0, default
    assert off['active_reads'] == 0 and off['idle_reads'] == 0, off
    assert bad['active_reads'] == 0 and bad['idle_reads'] == 0, bad   # out of range: off
    print(json.dumps(dict(default=default, off=off, out_of_range=bad, passed=True)))
