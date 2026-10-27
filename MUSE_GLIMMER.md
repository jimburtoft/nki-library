# nkilib for Muse-Glimmer-30B on Trainium2 (SDK 2.33)

This branch is the public **`2.33`** tag of `aws-neuron/nki-library` (`726c501`, the nkilib bundled in the Neuron SDK 2.33 `nki` 0.7.0 wheel) plus the kernel changes found while porting [`meta-models/Muse-Glimmer-30B`](https://huggingface.co/meta-models/Muse-Glimmer-30B) to vLLM-Neuron.

The vLLM-Neuron side is [`jimburtoft/vllm-neuron@contrib/muse-glimmer-0.29`](https://github.com/jimburtoft/vllm-neuron/tree/contrib/muse-glimmer-0.29). That branch already works with stock SDK 2.33 nkilib: it carries a local copy of the gated decode kernel and works around the other issues. This branch is the upstream form of those changes.

| Commit | Kernel | Change | Evidence |
|---|---|---|---|
| `mlp_cte ROW: share the discarded row-quant \|x\| scratch` | `core/mlp/mlp_cte/basic/mlp_cte_basic_quantization.py` | FP8 ROW prefill MLP allocated one `[P, I]` fp32 scratch for the discarded `\|x\|` output per 128-row BxS subtile. With 1024-row tiles (T > 512) at a large per-core I, this overflows the stack and the kernel fails validation (`NCC_INKI016 Stack out of memory`). One shared buffer is exact. | **Simulator:** output `torch.equal` to stock at T = 512/1024/2048. **trn2, H=6656, I=5120 at LNC2:** T = 1024/2048/4096 now validate; output equal to 512-row chunked prefill; MLP time -15 to -17 % vs chunking. Adds 2 ROW test cases at the failing shape. |
| `attention_block_tkg: optional sigmoid output gate` | `experimental/transformer/attention_block_tkg.py` | New keyword-only `W_gate`, `W_gate_scale`, `gate_dq`. The kernel computes `out = (attn * sigmoid(X_norm @ W_gate) * gate_dq) @ W_out`, so models with an output gate between attention and o_proj keep o_proj inside the decode kernel. The gate projection is issued before attention, so its weight DMA overlaps KV traffic. No change when `W_gate` is None. | **Simulator (B 1/4, NoPE and RoPE+SWA, BF16 and FP8 KV, ± fused rmsnorm_X):** cos >= 0.999995 vs stock kernel + torch gate/o_proj; K/V identical. **trn2 (30B decoder, TP=4):** decode step -4 / -9 / -12 / -16 % at batch 13/32/48/58; about +3 % at batch 1. |
| `mlp_cte: keep the projection-bias guard when I >= 4096` | `core/mlp/mlp_cte/mlp_cte_sharding.py` | The `intermediate_size >= 4096` rule forced I-sharding even with projection biases. I-sharding does not apply biases, so a ViT-style fc1/fc2 MLP silently lost its biases. | **Simulator, skip_gate + GELU + up/down bias, I=4096:** stock cos 0.888 vs reference, patched 0.999988; I=2048 unchanged. Adds a test case. |
| `attention_segmented_cte: document unaligned prior_tokens` | `core/attention/attention_segmented_cte.py` | `prior_tokens` must be a multiple of `block_size`. It is a runtime tensor, so this can't be asserted at trace time, and an unaligned value silently attends to the wrong KV. Docstring only; the caller-side fix is in vllm-neuron (block-aligned prefill chunks under speculative decoding). | Simulator: max error 3.9 unaligned vs 0.03 aligned |
| `tkg down projection: row-scale slice on the partial last PSUM tile` | `core/mlp/mlp_tkg/` and `core/moe/moe_tkg/` `mlp_tkg_down_projection_lhs_rhs_swap.py` (+ 3 experimental/model copies) | The LHS/RHS-swap down projection sliced the per-channel (FP8 ROW) dequant scale with `index * size`; on the last, partial PSUM tile `size` is smaller than the stride, so the wrong scale columns were read whenever `H1_shard % (512 // T) != 0`. Use the tile's `start_offset` / `end_offset`. Same fix as on `gptoss-trn2-perf-2.33` (`fc5fec9`). Not on MuseGlimmer's default path (its ROW decode MLP uses column tiling), but it affects any ROW caller of the swap path, e.g. FP8 MoE decode. | Simulator, mlp_tkg ROW forced onto the swap path, H=6656 (H1_shard 26 at LNC2): T=32 / 48 / 64 stock 8.4 / 6.4 / - % rel error -> 0.23-0.24 % (same as the column-tiling path); T=13 (one tile) unchanged |

**Also observed, not changed here:**
- **FP8 ROW CTE MLP at per-core intermediate sizes that are not a multiple of 128.** Seen during the port with nkilib 0.6.0 (SDK 2.32): cos 0.986 vs 0.999 at I = 4992/2496. It does **not** reproduce with this 2.33 base in the simulator (cos 0.9988 at I = 2432/2496/4992, same as 2560/5120), so no change is proposed.
- **`attention_cte` with sequence-packed bounds still computes fully masked KV tiles.** A ViT with windowed attention pays for the full block. A tile-skip based on the bounds would remove that work; see the sequence-packing tile-pruning change tracked separately.
- **`cascaded_max` (argmax) asserts when one stage would exceed the `nc_find_index8` limit of 16,384 elements.** This happens at 256 verify rows over a 50k vocab shard. vLLM-Neuron falls back to `torch.max` there.

## Use

nkilib ships inside the `nki` wheel as the top-level package `nkilib`. To use this branch, put `src/nkilib_src` first on `PYTHONPATH`. That shadows the bundled copy, and nothing needs to be reinstalled:

```bash
git clone -b contrib/muse-glimmer-2.33 https://github.com/jimburtoft/nki-library.git
export PYTHONPATH=$PWD/nki-library/src/nkilib_src:$PYTHONPATH
python -c "import nkilib, os; print(os.path.dirname(nkilib.__file__))"   # must point into the clone
```

The simulator checks are in `test/muse_glimmer/`. They need the vLLM-Neuron branch above on `PYTHONPATH` (for `vllm_neuron.functional`), plus `NKI_SIMULATOR=1 NEURON_PLATFORM_TARGET_OVERRIDE=trn2 VLLM_NEURON_CPU_MODE=1`.
