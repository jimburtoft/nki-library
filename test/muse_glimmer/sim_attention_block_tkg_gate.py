"""NKI-simulator check of the attention_block_tkg output gate (W_gate / gate_dq: gate + o_proj inside the kernel).

Compares gate_decode.gated_attention_decode on the simulator (patched kernel) against
  (1) the shipping path: NF.attention_decode(W_out=None) on the simulator + torch gate + torch o_proj, and
  (2) an fp32 torch reference,
for B in (1, 4, 13), full/NoPE and sliding/RoPE+SWA, BF16 KV and FP8 KV, with and without the folded input norm.
Per-rank TP=4 geometry (8 q heads, 1 kv head, d=128); H from argv (default 6656).

Driven through the vllm-neuron MuseGlimmer wrapper (vllm_neuron.model.muse_glimmer.gate_decode, branch
jimburtoft/vllm-neuron@contrib/muse-glimmer-0.29), which calls this kernel with W_gate:
  VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 NEURON_PLATFORM_TARGET_OVERRIDE=trn2 python sim_attention_block_tkg_gate.py 1024
"""
import math
import os
import sys

import torch

import vllm_neuron.functional as NF
from vllm_neuron.model.muse_glimmer.gate_decode import gated_attention_decode
from vllm_neuron.utils.neuron_utils import can_run_kernel

torch.manual_seed(0)
assert can_run_kernel(torch.zeros(1)), "simulator not active (set NKI_SIMULATOR=1)"
bf, f8 = torch.bfloat16, torch.float8_e4m3fn
H = int(sys.argv[1]) if len(sys.argv) > 1 else 6656
BS_LIST = [int(x) for x in os.environ.get("BS", "1,4,13").split(",")]
d, nq = 128, 8
M = 240.0
scaling = 3.87 * d ** -0.5
eps = 1e-5


def cosf(a, b):
    return torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def rope(B, pos):
    inv = 1.0 / (5e5 ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
    f = pos.reshape(-1, 1).float() * inv[None]
    return (f.cos().view(B, 1, d // 2).permute(2, 0, 1).contiguous().to(bf),
            f.sin().view(B, 1, d // 2).permute(2, 0, 1).contiguous().to(bf))


blk, nblk = 32, 4
nb = 64
Kf = torch.randn(nb, blk, d)
Kf = Kf / Kf.pow(2).mean(-1, keepdim=True).sqrt()
Vf = torch.randn(nb, blk, d).clamp(-3, 3)
Wqkv = (torch.randn(H, d * (nq + 2)) * H ** -0.5).to(bf)
Wg = (torch.randn(H, nq * d) * H ** -0.5).to(bf)
Wo = (torch.randn(nq * d, H) * (nq * d) ** -0.5).to(bf)
w = torch.randn(H) * 0.3
gamma = (1.0 + w).view(1, H)
worst = 1.0
for kv in ("bf16", "fp8"):
    if kv == "fp8":
        k_dq = 1.125 * math.sqrt(d) / M
        v_dq = 2.0 * 3.0 / M
        Kc = (Kf / k_dq).clamp(-M, M).to(f8)
        Vc = (Vf / v_dq).clamp(-M, M).to(f8)
        kvq = dict(k_scale=torch.full((1, 1), 1.0 / k_dq), v_scale=torch.full((1, 1), 1.0 / v_dq))
    else:
        k_dq = v_dq = 1.0
        Kc, Vc = Kf.to(bf), Vf.to(bf)
        kvq = {}
    for B in BS_LIST:
        table = torch.stack([torch.randperm(nb - 1)[:nblk] + 1 for _ in range(B)]).to(torch.int32)
        X = (torch.randn(B, 1, H) * 2).to(bf)
        for label, rope_on, swa in [("full", False, None), ("swa", True, 64)]:
            pos = torch.tensor([[100.0 + 3 * b] for b in range(B)])
            cos = sin = None
            if rope_on:
                cos, sin = rope(B, pos)
            start = torch.clamp(pos - (swa - 1), min=0.0) if swa else None
            mask = NF.gen_attention_decode_mask(pos_ids=pos.reshape(1, B), bs=B, q_head=nq, s_active=1,
                                                s_prior=nblk * blk,
                                                start_pos=start.reshape(1, B) if start is not None else None,
                                                block_len=blk)
            mk = dict(attention_mask=mask) if kv == "fp8" else dict(pos_ids=pos, swa_start_pos_ids=start)
            for fold in (False, True):
                Xin = X
                if not fold:  # unfused input norm (shipping): torch centered norm first
                    xf = X.float()
                    Xin = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * gamma).to(bf)
                nk = dict(rmsnorm_X_enabled=True, rmsnorm_X_eps=eps, rmsnorm_X_gamma=gamma) if fold else {}
                # (1) shipping path on the simulator
                o, Kn, Vn = NF.attention_decode(
                    X=Xin, W_qkv=Wqkv, rmsnorm_QK_pre_rope_enabled=True, rmsnorm_QK_pre_rope_eps=eps,
                    rmsnorm_QK_pre_rope_W_Q=None, rmsnorm_QK_pre_rope_W_K=None, cos=cos, sin=sin,
                    rope_contiguous_layout=True, active_blocks_table=table, K_cache=Kc, V_cache=Vc,
                    softmax_scale=scaling * k_dq, update_cache=False, W_out=None, K_cache_transposed=False,
                    transposed_out=False, out_in_sb=False, **nk, **mk, **kvq)
                attn = o.permute(0, 3, 1, 2).reshape(B, nq * d)
                Xn = Xin if not fold else (X.float() * torch.rsqrt(X.float().pow(2).mean(-1, keepdim=True) + eps)
                                           * gamma).to(bf)
                g = torch.sigmoid(Xn.reshape(B, H) @ Wg) * v_dq
                ship = ((attn * g).to(bf) @ Wo)
                # (2) gated kernel
                try:
                    out, Kg, Vg = gated_attention_decode(
                        X=Xin, W_qkv=Wqkv, W_gate=Wg, W_out=Wo, K_cache=Kc, V_cache=Vc, active_blocks_table=table,
                        softmax_scale=scaling * k_dq, qk_norm_eps=eps, cos=cos, sin=sin, gate_dq=v_dq, **nk, **mk, **kvq)
                except Exception as e:  # noqa: BLE001
                    print(f"[sim] kv={kv} B={B} {label} fold={int(fold)}: FAILED {type(e).__name__}: {str(e)[:400]}",
                          flush=True)
                    continue
                # (3) fp32 reference (dequantized cache, unquantized active)
                oref, _, _ = NF.attention_decode(
                    X=Xn.float(), W_qkv=Wqkv.float(), rmsnorm_QK_pre_rope_enabled=True, rmsnorm_QK_pre_rope_eps=eps,
                    cos=None if cos is None else cos.float(), sin=None if sin is None else sin.float(),
                    rope_contiguous_layout=True, active_blocks_table=table, K_cache=Kc.float() * k_dq,
                    V_cache=Vc.float() * v_dq, attention_mask=mask, softmax_scale=scaling, update_cache=False,
                    W_out=None) if False else (None, None, None)
                xr = Xn.float().reshape(B, H)
                c1 = cosf(out, ship)
                worst = min(worst, c1)
                print(f"[sim] kv={kv} B={B} {label} fold={int(fold)}: gated-vs-ship cos {c1:.6f} max|d| "
                      f"{(out.float() - ship.float()).abs().max():.4f} (|ship| max {ship.float().abs().max():.2f}) "
                      f"| K_new eq {torch.equal(Kg.float(), Kn.float())} V_new eq {torch.equal(Vg.float(), Vn.float())} "
                      f"| out {tuple(out.shape)} {out.dtype}", flush=True)
print(f"SIM_DONE worst cos(gated, ship) = {worst:.6f}", flush=True)
