"""CPU check for adapter/replicated_split.py and adapter/draft_main_proj.py: tile ranges, the weight
and scale slices a proxy layer gets, the Mxfp8SwizzledInput row count, and the draft main_proj fp8
shard being an exact re-encoding of on-grid weights. The bit-identity gate itself runs at boot."""
import collections
import os

os.environ["DSV41_REPLICATED_SPLIT"] = "wqkv_a,engram.wkv"
os.environ["DSV41_DRAFT_MAIN_PROJ_SPLIT"] = "1"
import torch  # noqa: E402

import draft_main_proj as dmp  # noqa: E402
import replicated_split as rs  # noqa: E402


class Toy(torch.nn.Module):
    def __init__(self, n, k):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(n * k).view(n, k).float().to(torch.float8_e4m3fn),
                                         requires_grad=False)
        self.weight_scale_inv_swizzled = torch.nn.Parameter(torch.arange(n * k // 32, dtype=torch.int32),
                                                            requires_grad=False)
        self.output_size = n


def main():
    assert rs._ranges(1792, 4) == ([(0, 512), (512, 1024), (1024, 1408), (1408, 1792)], 512)
    assert rs._ranges(25600, 4) == ([(0, 6400), (6400, 12800), (12800, 19200), (19200, 25600)], 6400)
    toy = Toy(512, 64)
    assert rs._scale_params(toy) == [("weight_scale_inv_swizzled", "flat")]
    p = rs._proxy(toy, 128, 384)
    assert torch.equal(p.weight.float(), toy.weight.float()[128:384])
    assert torch.equal(p.weight_scale_inv_swizzled, toy.weight_scale_inv_swizzled[128 * 2:384 * 2])
    assert p.output_size == 256 and toy.output_size == 512 and toy.weight.shape[0] == 512
    Mx = collections.namedtuple("Mxfp8SwizzledInput", "data scales")
    assert rs._rows(Mx(torch.zeros(6, 64), torch.zeros(1))) == 6 and rs._rows(torch.zeros(3, 8)) == 3
    assert rs._rows(torch.zeros(2, 3, 4)) == -1

    # draft_main_proj: fp8 values times a power of two per (row, 32 columns) re-encode exactly
    q = (torch.randn(64, 256) * 40).to(torch.float8_e4m3fn).float()
    e = torch.randint(-12, -2, (64, 8)).float()
    w = (q.view(64, 8, 32) * torch.exp2(e)[..., None]).view(64, 256).to(torch.bfloat16)
    w8, s = dmp.quant_rowscale(w)
    back = (w8.float().view(64, 8, 32) * torch.exp2(s.float() - 127)[..., None]).view(64, 256)
    assert torch.equal(back.to(torch.bfloat16), w)
    print("test_replicated_split: ok")


if __name__ == "__main__":
    main()
