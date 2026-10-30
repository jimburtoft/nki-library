# Gemma3 (head_dim 256) decode kernel patches for Trn2 (based on NKI Library 2.33)

This branch is the public `2.33` NKI Library release plus changes to two decode attention kernels for
`d_head > 128` (Gemma3 uses `head_dim` 256). A caller with `d_head <= 128` gets the same code paths as stock 2.33.

Use it with the Gemma3 model on the `gemma3-trn2-perf-0.29` branch of
<https://github.com/jimburtoft/vllm-neuron>. That model code checks for the
`SUPPORTS_D256_OUT_PROJ` and `SUPPORTS_D256_BLOCK_KV_UPDATE` markers below. With the stock NKI Library it keeps
the output projection and the KV cache write outside the kernel, which is correct but slower.

## Changes

| Kernel | Change | Marker / switch |
|---|---|---|
| `experimental/transformer/attention_block_tkg.py` | In-kernel output projection for `d_head = n * 128`: the d-tiled attention output is consumed as `n` 128-wide virtual heads, so `out = sum_n attn[n] @ W[n]` runs in `output_projection_tkg` unchanged. Stock asserts `d_head <= 128`. | `SUPPORTS_D256_OUT_PROJ` |
| `experimental/transformer/attention_block_tkg.py` | In-kernel block-KV cache update for `d_head = n * 128` (bf16, non-transposed K cache): the K scatter transposes per 128-wide d-tile. Stock asserts `d_head <= 128`. Also B > 128 Q/K tiling for the d-tiled path. | `SUPPORTS_D256_BLOCK_KV_UPDATE` |
| `experimental/transformer/attention_block_tkg.py` | Multi-KV-head V extraction at `S_tkg == 1`: one DMA per batch tile instead of one per (batch, head). | `NKILIB_ABT_V_EXTRACT_PER_HEAD=1` restores the per-head copy |
| `core/attention/attention_tkg.py` | `d_head > 128` block-KV decode (PE-transpose K path): stitch the active-token K/V once per group of batches instead of once per batch. | `D_TILED_STITCH`; `NKILIB_ATTN_TKG_D_TILED_STITCH_BYTES=0` restores the per-batch stitch |

Effect inside Gemma3-12B on one Trn2 device (TP=4, batch 83, 2,048-token prompts): decode step -10.6 %
(sliding-window layers -16 %, global layers -11 %), offline output throughput +9.5 % vs the stock kernels in the same session.
Outputs pass the same fp32 first-divergence accuracy gate as the stock kernels.

## Install into the SDK 2.33 vLLM environment

```bash
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_29_0_1_2_0/bin/activate
pip install --no-deps --force-reinstall "git+https://github.com/jimburtoft/nki-library.git@gemma3-trn2-perf-2.33"
rm -rf ~/.cache/neuron_libtorch     # the compile cache does not track kernel source
python -c "import importlib; m = importlib.import_module('nkilib.experimental.transformer.attention_block_tkg'); print(m.SUPPORTS_D256_OUT_PROJ, m.SUPPORTS_D256_BLOCK_KV_UPDATE)"   # True True
```
