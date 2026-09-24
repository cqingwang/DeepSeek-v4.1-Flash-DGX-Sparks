"""Column-split of chosen ReplicatedLinear layers over the TP group, bit-identical or not at all.

DSV41_REPLICATED_SPLIT=wqkv_a (comma list of prefix suffixes). A ReplicatedLinear makes every TP
rank stream the whole weight for the same output. Here each rank runs the SAME quantized linear on
its slice of 128-row weight tiles (tiles spread as evenly as possible, e.g. 4/4/3/3 of 14) and the
columns are all-gathered (padded to the widest slice; RoCE at small M, a byte copy).

Exactness is checked, not assumed: at the first eager call with M <= DSV41_REPLICATED_SPLIT_MAX_M,
each rank runs the stock layer and its slice on the same random inputs and compares the columns bit
for bit; the ranks agree over the TP group (MIN) and either all use the split for that layer or
all keep the stock path. Graph capture only ever sees the decided path.
"""
import copy
import os

import torch

SPEC = [s.strip() for s in os.environ.get("DSV41_REPLICATED_SPLIT", "").split(",") if s.strip()]
MAX_M = int(os.environ.get("DSV41_REPLICATED_SPLIT_MAX_M", "96"))
TILE = 128
_LOG = {"built": 0, "on": 0}


def _scale_params(layer):
    n, k = layer.weight.shape
    out = []
    for name, p in layer.named_parameters(recurse=False):
        if name == "weight" or p is None or p.dim() < 1:
            continue
        if p.dim() >= 2 and p.shape[0] == n:
            out.append((name, "rows"))
        elif p.dim() == 1 and p.numel() == n * (k // 32):
            out.append((name, "flat"))           # 128x4-swizzled: 128-row tiles are contiguous
    return out


def _ranges(n, world):
    tiles = n // TILE
    base, extra = divmod(tiles, world)
    counts = [base + (1 if r < extra else 0) for r in range(world)]
    starts = [sum(counts[:r]) * TILE for r in range(world)]
    return [(s, s + c * TILE) for s, c in zip(starts, counts)], max(counts) * TILE


def _proxy(layer, n0, n1):
    proxy = copy.copy(layer)
    proxy._parameters = dict(layer._parameters)
    proxy._buffers = dict(layer._buffers)
    proxy._modules = dict(layer._modules)
    k = layer.weight.shape[1]
    proxy._parameters["weight"] = torch.nn.Parameter(layer.weight.data[n0:n1].contiguous(), requires_grad=False)
    for name, kind in _scale_params(layer):
        p = layer._parameters[name]
        sl = p.data[n0:n1] if kind == "rows" else p.data[n0 * (k // 32):n1 * (k // 32)]
        proxy._parameters[name] = torch.nn.Parameter(sl.contiguous(), requires_grad=False)
    for attr in ("output_size", "output_size_per_partition"):
        if hasattr(layer, attr):
            setattr(proxy, attr, n1 - n0)
    proxy._dsv41_split = None
    return proxy


def _rows(x):
    """M of a plain activation or of a pre-quantized Mxfp8SwizzledInput (data, scales)."""
    if isinstance(x, torch.Tensor):
        return x.shape[0] if x.dim() == 2 else -1
    data = getattr(x, "data", None)
    return data.shape[0] if isinstance(data, torch.Tensor) and data.dim() == 2 else -1


def _build(layer, orig_forward, group, x_real):
    world, rank = group.world_size, group.rank_in_group
    n, k = layer.weight.shape
    ok = n % TILE == 0 and n // TILE >= world and layer.weight.dtype == torch.float8_e4m3fn
    state = None
    if ok:
        ranges, width = _ranges(n, world)
        n0, n1 = ranges[rank]
        try:
            proxy = _proxy(layer, n0, n1)
            dev = layer.weight.device
            g = torch.Generator(device=dev).manual_seed(4321)
            inputs = [x_real] + [torch.randn((rows, k), generator=g, device=dev, dtype=torch.bfloat16)
                                 for rows in (1, 6, 16)]
            for x in inputs:
                full = orig_forward(layer, x)[0]
                part = orig_forward(proxy, x)[0]
                ok = ok and bool(torch.equal(full[:, n0:n1], part))
            state = (proxy, ranges, width)
        except Exception as exc:
            print(f"[replicated_split] {layer._dsv41_prefix}: slice failed on rank {rank}: {exc!r}", flush=True)
            ok = False
    flag = torch.tensor([1 if ok else 0], dtype=torch.int32, device=layer.weight.device)
    torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN, group=group.device_group)
    agreed = bool(flag.item())
    _LOG["built"] += 1
    _LOG["on"] += int(agreed)
    if rank == 0 and (_LOG["built"] <= 2 or not agreed):
        print(f"[replicated_split] {layer._dsv41_prefix} {n}x{k}: split {'ON' if agreed else 'OFF'} "
              f"(rank 0 bit-exact {ok}; {_LOG['on']}/{_LOG['built']} layers on so far)", flush=True)
    return state if agreed else False


def install(linear_module):
    """sglang.srt.layers.linear: ReplicatedLinear, per-instance by prefix."""
    if not SPEC:
        return
    cls = linear_module.ReplicatedLinear
    orig_init, orig_forward = cls.__init__, cls.forward

    def __init__(self, *a, **kw):
        orig_init(self, *a, **kw)
        prefix = kw.get("prefix", "") or ""
        self._dsv41_prefix = prefix
        self._dsv41_split = None if any(prefix.endswith(s) for s in SPEC) else False

    def forward(self, x, *a, **kw):
        st = getattr(self, "_dsv41_split", False)
        m = _rows(x) if st is not False else -1
        if st is False or a or kw or m <= 0 or m > MAX_M:
            return orig_forward(self, x, *a, **kw)
        from sglang.srt.distributed import get_tp_group
        group = get_tp_group()
        if group.world_size == 1:
            return orig_forward(self, x)
        if st is None:
            if torch.cuda.is_current_stream_capturing():
                return orig_forward(self, x)
            st = self._dsv41_split = _build(self, orig_forward, group, x)
            if st is False:
                return orig_forward(self, x)
        proxy, ranges, width = st
        local = orig_forward(proxy, x)[0]
        if local.shape[1] < width:
            local = torch.nn.functional.pad(local, (0, width - local.shape[1]))
        gathered = group.all_gather(local.contiguous(), dim=-1)           # [M, world * width]
        if all(b - a == width for a, b in ranges):
            return gathered, None
        parts = [gathered[:, r * width:r * width + (b - a)] for r, (a, b) in enumerate(ranges)]
        return torch.cat(parts, dim=-1), None

    cls.__init__ = __init__
    cls.forward = forward
    print(f"[replicated_split] armed for {SPEC} (rows <= {MAX_M})", flush=True)
