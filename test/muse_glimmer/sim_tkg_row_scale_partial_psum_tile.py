"""Simulator check of the TKG LHS/RHS-swap down projection row-scale slice on a partial last PSUM tile
(a per-channel FP8 ROW scale). Runs NF.mlp ROW at TKG sizes on the simulator with the plugin default (column tiling)
and with column_tiling=False (forces down_projection_lhs_rhs_swap), against a dequant reference.
H=6656 -> H1=52, H1_shard=26 at LNC2; with the swap path the issue shows whenever 26 % (512 // T) != 0.
  NKI_SIMULATOR=1 NEURON_PLATFORM_TARGET_OVERRIDE=trn2 VLLM_NEURON_CPU_MODE=1 python sim_tkg_row_scale_partial_psum_tile.py
"""
import os

import torch

import vllm_neuron.functional as NF
from nkilib.core.utils.common_types import QuantizationType

torch.manual_seed(0)
H, I = int(os.environ.get("H", 6656)), int(os.environ.get("I", 1280))
FP8MAX = 240.0


def q(w):
    s = (w.float().abs().amax(dim=0, keepdim=True) / FP8MAX).clamp(min=1e-12)
    return (w.float() / s).clamp(-FP8MAX, FP8MAX).to(torch.float8_e4m3fn), s


Wg, Wu, Wd = torch.randn(H, I) / H**0.5, torch.randn(H, I) / H**0.5, torch.randn(I, H) / I**0.5
qg, sg = q(Wg)
qu, su = q(Wu)
qd, sd = q(Wd)
kw = dict(quantization_type=QuantizationType.ROW, gate_w_scale=sg.expand(128, I).contiguous(),
          up_w_scale=su.expand(128, I).contiguous(), down_w_scale=sd.expand(128, H).contiguous(),
          output_dtype="bfloat16")
for T in [int(t) for t in os.environ.get("TS", "13,32,48,64").split(",")]:
    x = (torch.randn(T, H) * 0.5).to(torch.bfloat16)
    ref = (torch.nn.functional.silu(x.float() @ (qg.float() * sg)) * (x.float() @ (qu.float() * su))) @ (qd.float() * sd)
    row = [f"T={T:3d} (swap-affected={26 % max(512 // T, 1) != 0 and 26 > 512 // T})"]
    for label, ct in (("default", None), ("swap", False)):
        try:
            out = NF.mlp(x, qg, qu, qd, column_tiling=ct, **kw)
            rel = ((out.float() - ref).norm() / ref.norm()).item()
            row.append(f"{label} rel {rel * 100:6.2f} %")
        except Exception as e:  # noqa: BLE001
            row.append(f"{label} FAILED {type(e).__name__}: {str(e)[:80]}")
    print(" | ".join(row), flush=True)
print("CHECK_DONE")
