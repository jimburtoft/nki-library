"""Simulator check: nkilib mlp CTE with projection biases at I >= 4096 (ViT-style fc1 -> GELU -> fc2).

Before the fix, the I >= 4096 sharding rule forced I-sharding even with biases, and I-sharding does not
apply biases, so the output was silently missing the bias. After the fix, I >= 4096 with biases shards
on BxS and the biases are applied.

  NKI_SIMULATOR=1 NEURON_PLATFORM_TARGET_OVERRIDE=trn2 VLLM_NEURON_CPU_MODE=1 I=4096 python sim_mlp_cte_bias_large_i.py
"""
import os

import torch
import torch.nn.functional as F

import vllm_neuron.functional as NF
from nkilib.core.utils.common_types import ActFnType

torch.manual_seed(0)
H, I, T = int(os.environ.get("H", 1024)), int(os.environ.get("I", 4096)), int(os.environ.get("T", 256))
x = torch.randn(T, H).to(torch.bfloat16)
w1 = (torch.randn(H, I) / H**0.5).to(torch.bfloat16)
b1 = (torch.randn(I) * 0.5).to(torch.bfloat16)
w2 = (torch.randn(I, H) / I**0.5).to(torch.bfloat16)
b2 = (torch.randn(H) * 0.3).to(torch.bfloat16)
erf = lambda z: 0.5 * z * (1 + torch.erf(z / 2**0.5))  # noqa: E731


def cos(a, b):
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


ref = erf(x.float() @ w1.float() + b1.float()) @ w2.float() + b2.float()
nob = erf(x.float() @ w1.float()) @ w2.float()
out = NF.mlp(x, w1, w1, w2, skip_gate=True, act_fn=ActFnType.GELU, up_b=b1.unsqueeze(0), down_b=b2.unsqueeze(0))
c = cos(out, ref)
print(f"H={H} I={I} T={T}: cos vs reference {c:.6f} | cos vs bias-dropped reference {cos(out, nob):.6f}", flush=True)
print("PASS" if c > 0.9995 else "FAIL")
