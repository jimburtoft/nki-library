"""Simulator check for the nkilib CTE ROW row-quant shared-scratch change.

Runs nkilib mlp (CTE, ROW FP8) from whatever nkilib is first on PYTHONPATH and saves the output.
Compare a stock run against a patched run (``--compare a.pt b.pt``).
  NKI_SIMULATOR=1 NEURON_PLATFORM_TARGET_OVERRIDE=trn2 VLLM_NEURON_CPU_MODE=1 python sim_cte_row_shared_abs.py --T 1024 --out stock.pt
  (swap in this branch's mlp_cte_basic_quantization.py)  ... --T 1024 --out patched.pt
"""
import argparse
import os

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--T", type=int, default=512)
ap.add_argument("--H", type=int, default=1024)
ap.add_argument("--I", type=int, default=5120)
ap.add_argument("--out", default="")
ap.add_argument("--compare", nargs=2, default=None)
args = ap.parse_args()

if args.compare:
    a, b = (torch.load(p) for p in args.compare)
    print("torch.equal:", torch.equal(a["out"], b["out"]),
          "max|d|:", float((a["out"].float() - b["out"].float()).abs().max()),
          "cos vs ref a/b:", a["cos"], b["cos"])
    raise SystemExit(0)

import nkilib  # noqa: E402
import vllm_neuron.functional as NF  # noqa: E402
from nkilib.core.utils.common_types import QuantizationType  # noqa: E402

print("nkilib from", os.path.dirname(nkilib.__file__), flush=True)
torch.manual_seed(0)
H, I, T = args.H, args.I, args.T
FP8MAX = 240.0


def q(w):
    s = (w.float().abs().amax(dim=0, keepdim=True) / FP8MAX).clamp(min=1e-12)
    return (w.float() / s).clamp(-FP8MAX, FP8MAX).to(torch.float8_e4m3fn), s


Wg, Wu, Wd = torch.randn(H, I) / H**0.5, torch.randn(H, I) / H**0.5, torch.randn(I, H) / I**0.5
qg, sg = q(Wg)
qu, su = q(Wu)
qd, sd = q(Wd)
x = (torch.randn(T, H) * 2.0).to(torch.bfloat16)
ones = torch.ones(1, H, dtype=torch.bfloat16)
packed = NF.rmsnorm_quant(x, ones, None, eps=1e-5, quantization_type=QuantizationType.ROW)
out = NF.mlp(packed, qg, qu, qd, quantization_type=QuantizationType.ROW,
             gate_w_scale=sg.expand(128, I).contiguous(), up_w_scale=su.expand(128, I).contiguous(),
             down_w_scale=sd.expand(128, H).contiguous(), output_dtype="bfloat16")
pk = packed.view(torch.uint8)
xd = pk[:, :H].view(torch.float8_e4m3fn).float() * pk[:, H:].contiguous().view(torch.float32)
ref = (torch.nn.functional.silu(xd @ (qg.float() * sg)) * (xd @ (qu.float() * su))) @ (qd.float() * sd)
cos = torch.nn.functional.cosine_similarity(out.float().flatten(), ref.flatten(), dim=0).item()
print(f"T={T} H={H} I={I}: cos vs dequant reference = {cos:.6f}", flush=True)
if args.out:
    torch.save({"out": out.cpu(), "cos": cos}, args.out)
print("SIM_DONE")
