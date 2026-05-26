# NKI Library (GLM-5 Fork)

> **This is a fork of [aws-neuron/nki-library](https://github.com/aws-neuron/nki-library) on branch `feature/selection-bias-routing`.**
> **Based on SDK 2.29 nkilib (commit `d2ad3a5`, "NKI Lib 2026-04-13").**

## Fork Purpose

This fork adds **post-activation selection bias** and **routed scaling factor** support to the `router_topk` and `moe_block_tkg` kernels for GLM-5 (and DeepSeek-V3 style) MoE routing.

### Modified Files (4 files, 1 commit: `dc74d3d`)

| File | Change |
|------|--------|
| `src/nkilib_src/nkilib/core/router_topk/router_topk.py` | Added `selection_bias` param (post-sigmoid, pre-TopK) and `routed_scaling_factor` (post-L1-norm scaling) |
| `src/nkilib_src/nkilib/core/router_topk/router_topk_torch.py` | PyTorch reference implementation matching NKI kernel |
| `src/nkilib_src/nkilib/core/moe_block/moe_block_tkg.py` | Pass `selection_bias` and `routed_scaling_factor` through to `router_topk` |
| `src/nkilib_src/nkilib/core/subkernels/rmsnorm_tkg.py` | NKI 0.3.0 `tensor_reduce` axis fix (later reverted -- NKI 0.3.0 supports `axis=2` natively) |

### Why This Fork Is Needed

Upstream nkilib's `router_bias` (`w_bias`) is a **pre-activation** bias added to raw logits before sigmoid. GLM-5 requires a **post-activation** selection bias:

```python
# Upstream: sigmoid(logits + w_bias) -- bias changes BOTH selection AND weights
# GLM-5:   selection_scores = sigmoid(logits) + selection_bias  -- bias only for TopK selection
#           weights = sigmoid(logits)[topk_indices]             -- UN-BIASED values for weighting
```

These are mathematically different (`sigmoid(x+b) != sigmoid(x) + b`). The upstream `router_bias` cannot replicate GLM-5's routing behavior where the bias influences which experts are selected but does NOT affect the affinity weights used for combining expert outputs.

### SDK 2.29 vs 2.30 Analysis (2026-05-26)

This fork is based on SDK 2.29 kernels. The 2.30 upstream (`origin/main`) has significant changes, but **none provide performance improvements for GLM-5's code path**:

| Category | SDK 2.30 Changes | Impact for GLM-5 |
|----------|-----------------|-------------------|
| **MoE TKG (non-MX path)** | Refactored into helper functions (`load_all_expert_affinities`, `broadcast_all_expert_affinity`, etc.), `safe_tensor_view` wrappers, `transposed_out` support | **No perf change** -- same algorithm, same DMA patterns |
| **MoE TKG (MX path)** | Matmul loop reorder (H->I->4_I), pre-allocated PSUM, STATIC_MX/ROW_MX modes, `.view()` bitcasts | **Not applicable** -- GLM-5 uses standard FP8, not MX-packed weights |
| **Router TopK** | +31/-19 lines: `@nki.jit` decorator, `skip_store_router_logits` allowed | **No perf change** |
| **MoE Block TKG** | +178/-29 lines: `is_all_expert_dynamic`, `block_size`, `inp_layout`/`outp_layout` enums | **Not applicable** at BS=1 (T=1 doesn't meet dynamic mode requirements) |
| **RMSNorm TKG** | New `rmsnorm_tkg_th` (T-on-partition layout), `_rmsnorm_tkg_dloc` (dynamic mode) | **Not applicable** -- GLM-5 uses existing layout |
| **gen4 assertion** | MX weights restricted to Trn3+ (`nisa.get_nc_version() >= gen4`) | **Not applicable** -- GLM-5 doesn't use MX weights |

**Decision: Remain on SDK 2.29 kernels.** Rebasing onto 2.30 would require resolving conflicts across hundreds of lines of refactoring for zero performance benefit on GLM-5's code path (non-MX `_all_expert_moe_tkg` with standard FP8 per-tensor-symmetric quantization).

### When to Rebase onto 2.30+

Consider rebasing if:
1. Upstream adds native `selection_bias` support (post-activation, pre-TopK)
2. GLM-5 moves to MXFP4/MXFP8 quantization (would benefit from MX path optimizations)
3. GLM-5 moves to Trn3 (hardware MX acceleration)
4. A bug fix in 2.30 affects GLM-5's code path

---

The NKI Library provides pre-built reference kernels you can use directly in your model development with the AWS Neuron SDK and NKI.
These kernel APIs provide the default classes, functions, and parameters you can use to integrate the NKL kernels into your models.
More details can be found in the [NKI Library Documentation](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/library/api/index.html)

## Kernel Reference

| Kernel API                                                                                                                                                   | Description                                                                                                                                                                          |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Attention CTE Kernel](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/core/attention/attention_cte.py)                         | The kernel implements attention with support for multiple variants and optimizations.                        |
| [Attention TKG Kernel](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/core/attention/attention_tkg.py)                         | The kernel implements attention specifically optimized for token generation use cases.                       |
| [MLP Kernel](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/core/mlp/mlp.py)                                                   | The kernel implements a Multi-Layer Perceptron with optional normalization fusion and various optimizations.           |
| [MoE CTE Kernel](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/core/moe/moe_cte/)                       | The kernel implements Mixture of Experts optimized for Context Encoding use cases.              |
| [MoE TKG Kernel](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/core/moe/moe_tkg/moe_tkg.py)                                   | The kernel implements Mixture of Experts optimized for Token Generation use cases.                           |
| [Output Projection CTE Kernel](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/core/output_projection/output_projection_cte.py) | The kernel computes the output projection operation optimized for Context Encoding use cases.           |
| [Output Projection TKG Kernel](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/core/output_projection/output_projection_tkg.py) | The kernel computes the output projection operation optimized for Token Generation use cases.           |
| [QKV Kernel](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/core/qkv/qkv.py)                                                   | The kernel performs Query-Key-Value projection with optional normalization fusion.                                     |
| [RMSNorm-Quant Kernel](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/core/rmsnorm/rmsnorm_quant.py)                           | The kernel performs optional RMS normalization followed by quantization to `fp8`.                            |
| [RoPE Kernel](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/core/embeddings/rope.py)                                          | The kernel applies Rotary Position Embedding to input embeddings with optional LNC sharding.                 |
| [Router Top-K Kernel](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/core/router_topk/router_topk.py)                          | The kernel computes router logits and top-K selection for Mixture of Experts models.                         |
| [Cumsum Kernel](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/core/cumsum/cumsum.py)                                          | The kernel computes cumulative sum along the last dimension.                                                 |

### Experimental Kernels

| Kernel API                                                                                                                                                   | Description                                                                                                                                                                          |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Attention Block TKG Kernel](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/transformer/attention_block_tkg.py)  | The kernel implements fused attention block for TKG with RMSNorm, QKV, RoPE, and output projection.          |
| [Cross Entropy Kernel](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/loss/cross_entropy.py)                      | The kernel implements memory-efficient cross entropy loss forward and backward passes for large vocabularies. |
| [Depthwise Conv1D Kernel](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/conv/depthwise_conv1d.py)  | The kernel implements depthwise 1D convolution using implicit GEMM.          |
| [Blockwise MM Backward Kernel](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/moe/bwd/blockwise_mm_backward.py) | The kernel implements blockwise matrix multiplication backward pass for dropless Mixture of Experts. |
| [Conv1D Kernel](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/conv/conv1d.py)  | The kernel implements 1D convolution using a filter replication strategy. |
| [Dynamic Shape Kernels](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/dynamic_shapes/) | The kernels dynamic input shapes with dynamic loop tiling on dynamic dimension. |
| [Fine-Grained AllGather Kernel](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/collectives/fg_allgather.py) | The kernel implements fine-grained ring-based all-gather. |
| [FGCC Kernel](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/collectives/fgcc.py) | The kernel implements fused all-gather and matrix multiplication (Fine-Grained Gather Collective Compute). |
| [Transformer TKG Kernel](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/transformer/transformer_tkg.py) | The kernel implements a transformer forward pass megakernel optimized for token generation (TKG). |

## Integration with the Neuron Compiler

The Neuron compiler includes a bundled version of this package within `neuronx-cc`, accessible under the `nkilib` Python namespace (for example, `import nkilib`). This bundled version is referred to as "bundled nkilib" throughout this guide. Bundled nkilib has been validated to work with that particular compiler version and can be used out of the box.

If you want to contribute a kernel change or use the latest kernels, you can integrate with this package directly.

> **Note:** Unlike bundled nkilib, **kernels from this package are not guaranteed to be compatible with the latest release of the Neuron compiler**. To start from a known good commit compatible with your compiler version, find the branch corresponding to your compiler version in this repository.

### Installation
1. Install `neuronx-cc` as usual (most likely already done). For more information, see the [Neuron Quick Start
Guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/quick-start/index.html).
2. Install this package into the same virtual environment as the rest of your project:
   ```bash
   pip install nki-library
   ```
3. Import and use kernels as usual. This package automatically replaces bundled nkilib kernels with the content of this package. No code changes are required.

### Uninstalling
To uninstall, run the following command:
```bash
pip uninstall nki-library
```

After uninstalling, the compiler falls back to the bundled nkilib.

### Controlling which package gets loaded
To _temporarily_ revert to the bundled version of nkilib, set the `NKILIB_FORCE_BUNDLED_LIBRARY` environment variable to a truthy value:
```bash
export NKILIB_FORCE_BUNDLED_LIBRARY=true
```

On the next execution of neuronx-cc, it will use the bundled version of nkilib. To go back to the kernels from this package, unset `NKILIB_FORCE_BUNDLED_LIBRARY`

```bash
unset NKILIB_FORCE_BUNDLED_LIBRARY
```
