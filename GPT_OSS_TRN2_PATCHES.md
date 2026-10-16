# GPT-OSS decode/prefill kernel patches for Trn2 (based on NKI Library 2.32)

This branch is the public `2.32` NKI Library release plus small changes to five kernels.
The changes make GPT-OSS (and other MoE models with a small per-rank intermediate size)
faster on Trn2. Each change is opt-in by shape or input layout. A caller that passes the
stock shapes and layouts gets the same results as stock 2.32.

Use it with the GPT-OSS model changes on the `gptoss-trn2-perf-0.24` branch of
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
| `core/attention/attention_segmented_cte.py` | Remove a dynamic loop in the sliding-window sink path that only re-copied `sink` into `sink_masked`. With a dynamic-loop MoE kernel in the same graph (for example `shard_on_i_hybrid`), neuronx-cc 2.27 failed with `NCC_IGCA024` for 6+ layers. Marker: `SWA_SINK_STATIC`. | Enables hybrid prefill MoE |
| `experimental/transformer/attention_block_tkg.py` | GQA V extraction: one strided DMA per KV head instead of one 128-byte DMA per (batch, head) | Decode, BS > 1 |

All changes are bit-identical to stock on the NKI CPU simulator for the shapes tested, and
token-identical on Trn2 hardware.

## Install into the SDK 2.32 vLLM venv

```bash
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0/bin/activate
pip install "git+https://github.com/jimburtoft/nki-library.git@gptoss-trn2-perf-2.32"
python -c "import nkilib.core.moe.moe_tkg.selective_expert_impl as m; print(m.SUPPORTS_GATE_UP_MAJOR)"
# The vLLM-Neuron compile cache does not key on kernel source. Clear it after any kernel change:
rm -rf ~/.cache/neuron_libtorch
```

To go back to the bundled kernels, run `pip uninstall nki-library` (or set
`NKILIB_FORCE_BUNDLED_LIBRARY=true`) and clear the compile cache again.
