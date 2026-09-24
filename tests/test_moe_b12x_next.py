"""GPU test for adapter/moe_b12x_next.py on the real engine method (dsv4.1 f80c91a4b image).

Drives the engine's own Mxfp4FlashinferCutlassMoEMethod with the adapter installed:
  create_weights -> FusedMoE._weight_loader_impl (the real TP narrowing, real checkpoint slices)
  -> process_weights_after_loading -> create_moe_runner -> apply -> FusedOpPool fused func,
on a FusedMoE shell that carries only the attributes those methods read (no distributed init).

Cases (each loads CKPT_EXPERTS experts of a layer, not all 384, to stay inside ~2 GB):
  EP1  N=576  target layers (topk 6), moe_tp_rank 3 and 0      vs fp32 reference from the checkpoint
  EP1  N=576  draft layer mtp.0 (128 experts, topk 3)          vs fp32 reference
  EP2  N=1152 target layer, ep_rank 1, moe_tp_rank 1            vs stock FlashInfer CUTLASS + fp32 ref
for M in {6, 48, 96, 1024} (+ 5000 > top capacity for the chunk path), with CUDA-graph padding
rows (topk id -1), CUDA-graph capture/replay of the decode path, and a dense M sweep of the
capacity ladder.

usage (inside `docker run --rm --gpus all` of the image, see RESULTS / report):
  PYTHONPATH=/ds41/adapter:/b12x_next python3 /ds41/tests/test_moe_b12x_next.py [--quick]
env: MODEL_DIR (default /model), TEST_MAX_GB (2.0), CKPT_EXPERTS (32), DSV41_MOE_B12X_NEXT_TUNE.
"""
import json
import os
import sys
import time
from types import SimpleNamespace as NS

os.environ.setdefault("DSV41_MOE_B12X_NEXT", "1")
os.environ.setdefault("SGLANG_FLASHINFER_MOE_FUSED_FINALIZE", "0")
os.environ.setdefault("DSV41_MOE_B12X_NEXT_MEMSTATS", "1")

import torch  # noqa: E402

QUICK = "--quick" in sys.argv
MODEL = os.environ.get("MODEL_DIR", "/model")
NE = int(os.environ.get("CKPT_EXPERTS", "32"))
K_HID, N_FULL, LIMIT = 5120, 2304, 10.0
torch.cuda.set_per_process_memory_fraction(
    float(os.environ.get("TEST_MAX_GB", "2.0")) * 2**30 / torch.cuda.get_device_properties(0).total_memory)
DEV = torch.device("cuda", torch.cuda.current_device())
RESULTS = {"cases": [], "sweep": {}, "graph": [], "memory": {}}


class _Backend:
    """get_moe_runner_backend()/get_moe_a2a_backend() stand-in: every is_*() False except `true`."""

    def __init__(self, *true):
        self.true = set(true)

    def __getattr__(self, name):
        if name.startswith("is_"):
            return lambda: name in self.true
        raise AttributeError(name)


# ---------------------------------------------------------------------------- engine + stubs
import sglang.srt.layers.quantization.fp8 as fp8  # noqa: E402
import sglang.srt.layers.quantization.mxfp4_flashinfer_cutlass_moe as mq  # noqa: E402
import sglang.srt.layers.moe.moe_runner.flashinfer_cutlass as fic  # noqa: E402
import sglang.srt.layers.moe.moe_runner.runner as runner_mod  # noqa: E402
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE  # noqa: E402
from sglang.srt.layers.moe.moe_runner.base import FusedOpPool, MoeRunnerConfig  # noqa: E402
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput  # noqa: E402
from sglang.srt.layers.moe.topk import StandardTopKOutput  # noqa: E402

PLATFORM = NS(is_sm120=True, is_sm90=False, is_sm100=False)
_parallel = NS(tp_size=4)
fp8.get_parallel = lambda: _parallel
fp8.get_moe_runner_backend = lambda: _Backend("is_flashinfer_mxfp4")
fp8.get_moe_a2a_backend = lambda: _Backend()
fp8.get_platform = lambda: PLATFORM
mq.get_platform = lambda: PLATFORM
mq.get_exec = lambda: NS(moe=NS(flashinfer_mxfp4_moe_precision="default"))
mq.log_info_on_rank0 = lambda *a, **k: None
fic.get_tp_group = lambda: NS(world_size=1)
fic.is_allocation_symmetric = lambda: False


class _Runner:
    """MoeRunner stand-in: the real FusedOpPool entry, as MoeRunner.__init__ resolves it."""

    def __init__(self, backend, config):
        self.config = config
        self.fused = FusedOpPool.get_fused_func("none", backend.value)

    def run(self, dispatch_output, quant_info, lora_info=None):
        return self.fused(dispatch_output, quant_info, self.config)


runner_mod.MoeRunner = _Runner

import moe_b12x_next as ad  # noqa: E402

ad.install_method(mq)
ad.install_runner(fic)
ad._B["is_allocation_symmetric"] = lambda: False
assert FusedOpPool.get_fused_func("none", "flashinfer_mxfp4") is ad._fused_b12x_next
QUANT_CFG = NS(use_mxfp8=False, weight_block_size=[32, 32], is_fp4_experts=True, dequant_fp4_to_fp8=False,
               is_checkpoint_fp8_serialized=True, activation_scheme="dynamic", get_name=lambda: "fp8")

# ---------------------------------------------------------------------------- checkpoint
from safetensors import safe_open  # noqa: E402

_index = json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))["weight_map"]


def ckpt(name):
    with safe_open(os.path.join(MODEL, _index[name]), "pt", device="cpu") as f:
        return f.get_tensor(name)


def expert_tensors(prefix, e):
    out = {}
    for w in ("w1", "w2", "w3"):
        out[w] = ckpt(f"{prefix}.ffn.experts.{e}.{w}.weight")
        out[w + "s"] = ckpt(f"{prefix}.ffn.experts.{e}.{w}.scale")
    return out


# ---------------------------------------------------------------------------- layer shell
def make_layer(prefix, *, n_global, topk, ep_size, ep_rank, moe_tp_size, moe_tp_rank, process="adapter"):
    """FusedMoE shell + the engine's method; loads this rank's experts through the real loader."""
    e_local = n_global // ep_size
    n = N_FULL // moe_tp_size
    layer = FusedMoE.__new__(FusedMoE)
    torch.nn.Module.__init__(layer)
    cfg = MoeRunnerConfig(num_experts=n_global, num_local_experts=e_local, hidden_size=K_HID,
                          intermediate_size_per_partition=n, layer_id=0, top_k=topk, num_fused_shared_experts=0,
                          params_dtype=torch.bfloat16, activation="silu", is_gated=True,
                          routed_scaling_factor=None, swiglu_limit=LIMIT)
    for k, v in dict(layer_id=0, top_k=topk, hidden_size=K_HID, num_experts=n_global, num_fused_shared_experts=0,
                     moe_ep_size=ep_size, moe_ep_rank=ep_rank, moe_tp_size=moe_tp_size, moe_tp_rank=moe_tp_rank,
                     _num_global_routed=n_global, _expert_storage_rank=ep_rank, _num_local_routed=e_local,
                     num_local_experts=e_local, _has_fused_shared=False, intermediate_size_per_partition=n,
                     use_triton_kernels=False, use_presharded_weights=False, use_flashinfer_trtllm_moe=False,
                     moe_runner_config=cfg, quant_config=QUANT_CFG, scheme=None, with_bias=False,
                     reduce_results=False).items():
        setattr(layer, k, v)
    method = mq.Mxfp4FlashinferCutlassMoEMethod(fp8.Fp8MoEMethod(QUANT_CFG), prefix=prefix)
    layer.quant_method = method
    with torch.device(DEV):
        method.create_weights(layer, e_local, K_HID, n, torch.bfloat16,
                              weight_loader=FusedMoE.weight_loader.__get__(layer))
    method.create_moe_runner(layer, cfg)
    src = {}
    for le in range(e_local):
        g = ep_rank * e_local + le
        t = expert_tensors(prefix, g)
        src[le] = t
        for w, shard in (("w1", "w1"), ("w3", "w3"), ("w2", "w2")):
            FusedMoE._weight_loader_impl(layer, param=getattr(layer, "w13_weight" if w != "w2" else "w2_weight"),
                                         loaded_weight=t[w], weight_name=f"experts.{w}.weight", shard_id=shard,
                                         expert_id=le)
            FusedMoE._weight_loader_impl(layer, param=getattr(layer, "w13_weight_scale_inv" if w != "w2"
                                                              else "w2_weight_scale_inv"),
                                         loaded_weight=t[w + "s"], weight_name=f"experts.{w}.weight_scale_inv",
                                         shard_id=shard, expert_id=le)
    torch.cuda.synchronize()
    if process == "adapter":
        method.process_weights_after_loading(layer)
    else:
        type(method)._dsv41_b12x_next_orig_process(method, layer)
    torch.cuda.synchronize()
    return layer, method, src, None


# ---------------------------------------------------------------------------- reference
LUT = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def dq(packed, scale):
    p = packed.view(torch.uint8).to(DEV)
    lut = LUT.to(DEV)
    v = torch.stack([lut[(p & 15).long()], lut[(p >> 4).long()]], -1).reshape(p.shape[0], -1)
    s = torch.exp2(scale.view(torch.uint8).to(DEV).float() - 127)
    return (v.view(v.shape[0], -1, 32) * s[..., None]).view(v.shape[0], -1)


@torch.no_grad()
def reference(src, x, ids, w, *, e_local, ep_rank, moe_tp_size, moe_tp_rank):
    """fp32 MoE from the checkpoint slices, narrowed here independently of the engine loader."""
    n = N_FULL // moe_tp_size
    r0, r1 = moe_tp_rank * n, (moe_tp_rank + 1) * n
    out = torch.zeros(x.shape[0], K_HID, device=DEV)
    xf = x.float()
    for g in torch.unique(ids).tolist():
        le = g - ep_rank * e_local
        if g < 0 or not (0 <= le < e_local):
            continue
        rows, slots = (ids == g).nonzero(as_tuple=True)
        t = src[le]
        gate = xf[rows] @ dq(t["w1"][r0:r1], t["w1s"][r0:r1]).t()
        up = xf[rows] @ dq(t["w3"][r0:r1], t["w3s"][r0:r1]).t()
        act = torch.nn.functional.silu(gate.clamp(max=LIMIT)) * up.clamp(-LIMIT, LIMIT)
        w2 = dq(t["w2"][:, r0 // 2:r1 // 2].contiguous(), t["w2s"][:, r0 // 32:r1 // 32].contiguous())
        out.index_add_(0, rows, (act @ w2.t()) * w[rows, slots][:, None])
    return out


def routing(m, n_global, topk, seed, pad_rows=1):
    g = torch.Generator(device=DEV)
    g.manual_seed(seed)
    ids = torch.rand(m, n_global, device=DEV, generator=g).argsort(dim=1)[:, :topk].to(torch.int32)
    w = torch.softmax(torch.randn(m, topk, device=DEV, generator=g), dim=1).float()
    if pad_rows and m > pad_rows:
        ids[-pad_rows:] = -1                     # CUDA-graph padding rows
    x = torch.randn(m, K_HID, device=DEV, generator=g).to(torch.bfloat16)
    return x, ids.contiguous(), w


def apply(layer, method, x, ids, w):
    d = StandardDispatchOutput(hidden_states=x, hidden_states_scale=None,
                               topk_output=StandardTopKOutput(topk_weights=w, topk_ids=ids, router_logits=None))
    return method.apply(layer, d).hidden_states


def err(a, b):
    a, b = a.float(), b.float()
    d = a - b
    return {"rel_l2": round(float(d.norm() / b.norm().clamp_min(1e-20)), 5), "max_abs": round(float(d.abs().max()), 4),
            "cos": round(float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0)), 6),
            "ref_max": round(float(b.abs().max()), 3)}


def check_case(name, layer, method, src, geo, ms, *, fi=None):
    for m in ms:
        x, ids, w = routing(m, geo["n_global"], geo["topk"], 100 + m)
        t0 = time.time()
        out = apply(layer, method, x, ids, w)
        torch.cuda.synchronize()
        rec = {"case": name, "M": m, "eager_ms": round((time.time() - t0) * 1e3, 2)}
        ref = None
        if m <= 1024 or fi is None:
            ref = reference(src, x, ids, w, e_local=geo["n_global"] // geo["ep_size"], ep_rank=geo["ep_rank"],
                            moe_tp_size=geo["moe_tp_size"], moe_tp_rank=geo["moe_tp_rank"])
            rec["vs_fp32_ref"] = err(out, ref)
            assert rec["vs_fp32_ref"]["rel_l2"] < 0.08, rec
        if fi is not None and m <= 1024:          # FlashInfer's own workspace at M>1024 exceeds the 2 GB cap
            fo = apply(fi[0], fi[1], x, ids, w)
            rec["vs_flashinfer"] = err(out, fo)
            rec["flashinfer_vs_fp32_ref"] = err(fo, ref)
            assert rec["vs_flashinfer"]["rel_l2"] < 0.05, rec
        pad = out[-1].float().abs().max().item() if m > 1 else 0.0
        rec["pad_row_max_abs"] = pad
        assert pad == 0.0, rec
        assert torch.isfinite(out).all()
        RESULTS["cases"].append(rec)
        print(json.dumps(rec), flush=True)


def graph_case(name, layer, method, geo, m):
    x, ids, w = routing(m, geo["n_global"], geo["topk"], 7 + m)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        apply(layer, method, x, ids, w)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = apply(layer, method, x, ids, w)
    x2, ids2, w2 = routing(m, geo["n_global"], geo["topk"], 900 + m)
    x.copy_(x2), ids.copy_(ids2), w.copy_(w2)
    g.replay()
    torch.cuda.synchronize()
    eager = apply(layer, method, x2, ids2, w2)
    eager2 = apply(layer, method, x2, ids2, w2)
    e = err(out, eager)
    e_run = err(eager2, eager)          # run-to-run spread of eager itself (atomic reduction order)
    reps = 200
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        g.replay()
    b.record()
    b.synchronize()
    rec = {"case": name, "M": m, "graph_vs_eager": e, "eager_vs_eager": e_run,
           "bit_identical": bool(torch.equal(out, eager)), "eager_deterministic": bool(torch.equal(eager2, eager)),
           "replay_us": round(a.elapsed_time(b) * 1000 / reps, 1)}
    # graph replay must match eager to within eager's own run-to-run spread (bf16 ulps)
    assert e["rel_l2"] < max(1e-3, 3 * e_run["rel_l2"]) and e["rel_l2"] < 0.02, rec
    RESULTS["graph"].append(rec)
    print(json.dumps(rec), flush=True)
    del g


def sweep(name, layer, method, geo, ms):
    bad = []
    t0 = time.time()
    for m in ms:
        x, ids, w = routing(m, geo["n_global"], geo["topk"], m, pad_rows=0)
        try:
            out = apply(layer, method, x, ids, w)
            if not bool(torch.isfinite(out).all()):
                bad.append((m, "nonfinite"))
        except Exception as exc:  # noqa: BLE001
            bad.append((m, repr(exc)[:200]))
    torch.cuda.synchronize()
    RESULTS["sweep"][name] = {"count": len(ms), "failures": bad[:20], "n_fail": len(bad),
                              "seconds": round(time.time() - t0, 1)}
    print(json.dumps({"sweep": name, **RESULTS["sweep"][name]}), flush=True)
    assert not bad, bad[:5]


def sizing():
    """Scratch bytes per capacity for the production geometries (heuristic plans; no weights needed)."""
    B = ad._b12x()
    fm = B["fm"]
    from b12x_next.moe.fused_moe import _impl
    out = {}
    for name, (e, n, k) in {"ep1_target": (384, 576, 6), "ep1_draft": (128, 576, 3),
                            "ep2_target": (192, 1152, 6), "ep2_draft": (64, 1152, 3)}.items():
        wp = fm.plan_weights(
            source=fm.PackedSource(format=fm.PackedSourceFormat.MXFP4_E8M0_K32, w13_layout=fm.W13Layout.W13),
            activation=fm.ActivationSpec(mode=fm.ActivationMode.A8, nonlinearity="silu", io_dtype=torch.bfloat16,
                                         swiglu_limit=LIMIT),
            geometry=fm.MoEGeometry(num_experts=e, hidden_size=K_HID, intermediate_size=n))
        from b12x_next.moe.fused_moe import _preparation as fp, _tuning as ft
        from b12x_next.preparation import FrozenMapping
        from b12x_next.preparation.device import detect_device
        ident = detect_device(DEV).identity
        fake = NS(plan=wp, num_experts=e, hidden_size=K_HID, intermediate_size=n,
                  _impl=NS(can_share_input=lambda **kw: False))
        row = {}
        for cap in (6, 96, 1024, 4096):
            q = fp._query(fake, fm.ExecutionCapacity(max_tokens=cap, top_k=k), cap, fm.RoutingSpec(),
                          fp._control_snapshot(), FrozenMapping())
            caps = fp._lower_caps(q, ft._default_config(q, ident), wp._impl, DEV)
            row[cap] = round(_impl.tp_moe_required_nbytes(caps) / 2**20, 1)
        out[name] = row
    RESULTS["scratch_mb_by_capacity"] = out
    print(json.dumps({"scratch_mb_by_capacity": out}), flush=True)


def main():
    try:
        sizing()
    except Exception as exc:  # noqa: BLE001 - informational
        print(f"sizing failed: {exc!r}", flush=True)
    ms = [6, 48, 96, 1024] if QUICK else [6, 48, 96, 1024, 5000]
    torch.cuda.reset_peak_memory_stats()
    # ---- EP1 target (N=576), two layers sharing one geometry, different TP ranks
    geo1 = dict(n_global=NE, topk=6, ep_size=1, ep_rank=0, moe_tp_size=4, moe_tp_rank=3)
    l3, m3, s3, _ = make_layer("layers.3", **{k: geo1[k] for k in ("topk", "ep_size", "ep_rank", "moe_tp_size", "moe_tp_rank")},
                               n_global=NE)
    check_case("ep1_target_layer3_tp3", l3, m3, s3, geo1, ms)
    geo1b = dict(geo1, moe_tp_rank=0)
    l20, m20, s20, _ = make_layer("layers.20", **{k: geo1b[k] for k in ("topk", "ep_size", "ep_rank", "moe_tp_size", "moe_tp_rank")},
                                  n_global=NE)
    check_case("ep1_target_layer20_tp0", l20, m20, s20, geo1b, [6, 96])
    check_case("ep1_target_layer3_tp3_after_layer20", l3, m3, s3, geo1, [6])   # shared plan, own experts
    graph_case("ep1_target_layer3", l3, m3, geo1, 6)
    graph_case("ep1_target_layer3", l3, m3, geo1, 96)
    # ---- EP1 draft (mtp.0: 128 experts, top-3)
    geod = dict(n_global=NE, topk=3, ep_size=1, ep_rank=0, moe_tp_size=4, moe_tp_rank=2)
    ld, md, sd, _ = make_layer("mtp.0", **{k: geod[k] for k in ("topk", "ep_size", "ep_rank", "moe_tp_size", "moe_tp_rank")},
                               n_global=NE)
    check_case("ep1_draft_mtp0_tp2", ld, md, sd, geod, [5, 40, 80, 1024])
    graph_case("ep1_draft_mtp0", ld, md, geod, 5)
    RESULTS["memory"]["after_ep1_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    del l20, m20, s20, ld, md, sd
    torch.cuda.empty_cache()
    # ---- EP2 target (N=1152), ep_rank 1, vs stock FlashInfer on a second copy of the same slice
    geo2 = dict(n_global=2 * NE, topk=6, ep_size=2, ep_rank=1, moe_tp_size=2, moe_tp_rank=1)
    kw2 = {k: geo2[k] for k in ("topk", "ep_size", "ep_rank", "moe_tp_size", "moe_tp_rank")}
    lb, mb, sb, _ = make_layer("layers.3", n_global=2 * NE, **kw2)
    lf, mf, _, _ = make_layer("layers.3", n_global=2 * NE, process="flashinfer", **kw2)
    check_case("ep2_target_layer3_rank1", lb, mb, sb, geo2, ms, fi=(lf, mf))
    graph_case("ep2_target_layer3", lb, mb, geo2, 6)
    graph_case("ep2_target_layer3", lb, mb, geo2, 96)
    del lf, mf
    torch.cuda.empty_cache()
    # ---- capacity ladder sweep (every M up to 1100, then strided to past the top capacity)
    sweep_ms = list(range(1, 1101)) + list(range(1101, 4400, 13)) if not QUICK else list(range(1, 200))
    sweep("ep2_target", lb, mb, geo2, sweep_ms)
    sweep("ep1_target", l3, m3, geo1, sweep_ms)
    RESULTS["memory"]["peak_total_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    RESULTS["geometries"] = ad.geometry_reports()
    print("RESULTS " + json.dumps(RESULTS, default=str), flush=True)
    print("ALL OK", flush=True)


if __name__ == "__main__":
    main()
