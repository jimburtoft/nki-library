# GPT-OSS decode/prefill kernel patches for Trn2 (based on NKI Library 2.33)

This branch is the public `2.33` NKI Library release plus small changes to six kernels (several changes in `bwmm_shard_on_I`).
The changes make GPT-OSS (and other MoE models with a small per-rank intermediate size)
faster on Trn2. Each change is opt-in by shape or input layout. A caller that passes the
stock shapes and layouts gets the same results as stock 2.33.

Use it with the GPT-OSS model changes on the `gptoss-trn2-perf-0.29` branch of
<https://github.com/jimburtoft/vllm-neuron>. That model code checks for the
`SUPPORTS_GATE_UP_MAJOR` and `SWA_SINK_STATIC` markers below. With stock nkilib it keeps the
stock layout and the stock prefill MoE kernel.

## Changes

| Kernel | Change | Effect (GPT-OSS 20B, trn2.3xlarge, TP=4) |
|---|---|---|
| `core/moe/moe_tkg` gate/up projection | Pack all I tiles of a projection into one PSUM bank (gate in bank 0, up in bank 1, down from bank 2). Stock gives each 128-row I tile its own bank, so gate, up, and down compete for 8 banks and run one expert at a time. | Decode |
| `core/moe/moe_tkg` down projection | Down-weight DMA on hardware DGE, on the Scalar queue for T >= 8 | Decode |
| `core/moe/moe_tkg/mlp_tkg_constants.py` | In auto-alloc mode, the weight ring was fixed at 2 slots. Down now gets all its tiles and gate/up gets 2x the required tiles. | Decode |
| `core/moe/moe_tkg` selective/all-expert | Accept gate/up weights gate/up-major as `[E, 2, H, I]` (detected by `shape[1] == 2`). This gives 2.5x larger DMA packets than `[E, H, 2, I]`. Marker: `SUPPORTS_GATE_UP_MAJOR`. | Decode, BS >= 8 |
| `core/moe/moe_cte/bwmm_shard_on_I.py` | Per-expert weight DMAs on hardware DGE instead of software DGE on GpSimd, which is the busiest prefill engine | TTFT -5 to -8% |
| `core/moe/moe_cte/bwmm_shard_on_I.py` per-channel FP8 | FP8 weight tiles go straight into `nc_matmul` (FP8 x BF16, FP32 accumulation; scale applied at PSUM eviction) instead of being copied to BF16 first. The down-projection scale row is broadcast to all 128 partitions by one DMA (partition stride 0) instead of 4 Vector stream shuffles per tile. The per-channel scales of a block are loaded with one DMA (gate/up) and one DMA per 1024-wide H tile (down) instead of one per 128- or 512-wide tile. `NKILIB_FP8_DIRECT_MATMUL=0` / `NKILIB_DOWN_SCALE_DMA_BCAST=0` / `NKILIB_FP8_SCALE_COALESCE=0` restore the old paths. Markers: `FP8_DIRECT_MATMUL`, `DOWN_SCALE_DMA_BCAST`, `FP8_SCALE_COALESCE`. | FP8 prefill without the BF16 expert copy (`GPTOSS_HYBRID_PREFILL=0`): TTFT +32 % -> +5 % vs the BF16 copy |
| `core/moe/moe_cte/bwmm_shard_on_I.py` gate/up layout | Accept the gate/up weight gate/up-major as `[E, 2, H, I]` (detected by `shape[1] == 2`), the same layout `moe_tkg` reads. One FP8 tensor then serves prefill and decode. Marker: `bwmm_shard_on_I.SUPPORTS_GATE_UP_MAJOR`. | FP8 without the BF16 copy: decode as fast as with the copy |
| `core/moe/moe_cte/bwmm_shard_on_I.py` down bias | The [1, H] down-projection bias row is broadcast to all 128 partitions by one DMA (partition stride 0) instead of Vector stream shuffles, once per block. Applies to BF16 and FP8. `NKILIB_DOWN_BIAS_DMA_BCAST=0` restores the shuffles. Marker: `DOWN_BIAS_DMA_BCAST`. | Prefill MoE kernel -9 % (BF16 and FP8); with it, FP8 without the BF16 copy matches the BF16 copy |
| `core/attention/attention_segmented_cte.py` | Remove a dynamic loop in the sliding-window sink path that only re-copied `sink` into `sink_masked`. With a dynamic-loop MoE kernel in the same graph (for example `shard_on_i_hybrid`), neuronx-cc 2.27 failed with `NCC_IGCA024` for 6+ layers. Marker: `SWA_SINK_STATIC`. | Enables hybrid prefill MoE |
| `experimental/transformer/attention_block_tkg.py` | GQA V extraction: one strided DMA per KV head instead of one 128-byte DMA per (batch, head) | Decode, BS > 1 |

Measured together with the vLLM 0.29 model branch (trn2.3xlarge, TP=4, FP8 experts), removing only
these kernel changes (and the hybrid prefill MoE kernel they enable) makes TTFT 1.5-1.6x slower and
decode 1.18-1.25x slower (1 to 32 concurrent requests). Greedy outputs are identical with and without them.

All changes are bit-identical to stock on the NKI CPU simulator for the shapes tested, and
token-identical on Trn2 hardware. The per-channel FP8 change is exact against a BF16-dequantized reference
in the simulator (the scales are powers of two); the down-bias change is bit-identical to the shuffles.

FP8 prefill without the BF16 copy, trn2 (4 cores, TP=4, same session), time to first token at 2,048 / 4,096 tokens:
FP8 + BF16 copy 127.1 / 249.6 ms; **FP8 only 126.9 / 249.2 ms** (equal); FP8 only before the down-bias change
135.3 / 264.2 ms; FP8 with the stock upcast 167.2 / 325.8 ms. Kernel level (`bwmm_shard_on_I`, H 3072, I 768,
E 8, top-4, 1024 tokens, cycles): stock BF16 1.462-1.465 G; this branch BF16 1.296-1.300 G (-11 %), FP8
1.359-1.364 G (-7 % vs stock BF16); the stock FP8 upcast is 2.10 G. Dropping the copy doubles the KV cache (1.01M vs 0.54M tokens at
`gpu_memory_utilization` 0.80). GSM8K-CoT (250 questions): 0.864 vs 0.868 with the copy.

## Install into the SDK 2.33 vLLM environment

```bash
# Activate the vLLM 0.29 environment first (DLAMI venv or the vLLM 0.29 container).
pip install "git+https://github.com/jimburtoft/nki-library.git@gptoss-trn2-perf-2.33"
python -c "from nkilib.core.moe.moe_tkg import selective_expert_impl as m; print(m.SUPPORTS_GATE_UP_MAJOR)"   # prints True
python -c "from nkilib.core.moe.moe_cte import bwmm_shard_on_I as c; print(c.FP8_DIRECT_MATMUL, c.DOWN_SCALE_DMA_BCAST, c.FP8_SCALE_COALESCE, c.DOWN_BIAS_DMA_BCAST)"   # prints True True True True
# The vLLM-Neuron compile cache does not key on kernel source. Clear it after any kernel change:
rm -rf ~/.cache/neuron_libtorch
```

To go back to the bundled kernels, run `pip uninstall nki-library` (or set
`NKILIB_FORCE_BUNDLED_LIBRARY=true`) and clear the compile cache again.
