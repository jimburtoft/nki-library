# nkilib champion for Qwen3-ASR (SDK 2.33)

This branch is the public **`2.33`** tag of `aws-neuron/nki-library` (`726c501`) plus three decode kernel improvements. It is meant to run with the Qwen3-ASR vllm-neuron champion branch (`jimburtoft/vllm-neuron`, branch `qwen3-asr/0.29-champion`).

| commit | kernel | change | measured on trn2 (SDK 2.33 HW) |
|---|---|---|---|
| `attention_block_tkg` / `qkv_tkg` | At B·S == 1 the QKV weight is stationary and the output is d-major. That removes 8 serial Vector PSUM adds and 6 head transposes, and fuses Q\|K RMSNorm and a single RoPE. | kernel **−10.0 % (LNC2) / −12.3 % (LNC1)** at a 2048-hidden decoder shape |
| `attention_tkg` | Batched block-KV decode (trn2, bf16, d_head 128, s_active 1, ≥ 32 batches/core): **stitch** writes the active token's K/V once per DMA-batch group, not per sequence. **dt4** keeps the fast K `dma_transpose` gather at the 512 decode bucket (resized block_len 4) instead of falling back to the PE path. Default `NKILIB_CR17_VARIANT=dt4+stitch`; `stock` turns both off. | kernel **−19 to −24 %** at bucket 1024, **−54 to −57 %** at bucket 512; output identical |
| `attention_tkg` (packed FP8 KV) | **fp8st**: the same group stitch for the packed FP8 KV layout. The active token goes into the packed pair's lane, once per group. On by default; it only acts when the KV cache is packed FP8. | **−24 to −39 %** vs stock packed FP8 (b32–64, buckets 512/1024); output bit-identical to it |
| `mlp_tkg` | At T == 1 with gate/up column tiling off, the down projection runs in column-tiling mode: 24 matmuls instead of 96 strided ones, plus one contiguous output store. | kernel **−6.1 to −7.5 % (LNC2)**, neutral at LNC1, output bit-identical |

- **Scope:** every other shape and caller runs the existing code. The gates are in the commit messages.
- **Tests:** stock vs patched, the nkilib suites on SDK 2.33 hardware gave identical pass/fail. The new tests are `test_attn_blk_fast` (2 configs), `test_mlp_tkg_single_token_down_column_tiling` (5 cases) and `test/unit/nkilib/core/mlp/test_mlp_tkg_single_token_gate.py`.
- **Qwen3-ASR single stream** (TP4, LNC2, 30 s audio): E2E p50 360 → 342 ms with the attention change. The mlp change adds a further −9 ms.
- **Qwen3-ASR throughput** (TP1, 30 s, with `decode_context_length_buckets [512]` in vllm-neuron): **1,712–1,722 RTFx** at 8 replicas (LNC1) against 1,517–1,526 for stock nkilib, and **1,508–1,516** at 4 replicas (LNC2) against 1,366–1,383. Decode ITL at 51 running goes 44.0 → 34.6 ms. WER 6.01 %.
  - stitch alone at bucket 1024: +2.6 % / +3.4 %.
  - dt4 with the 512 bucket: a further +5.7 % / +8.4 %.
- **Qwen3-ASR with FP8 KV** (vllm-neuron branch `qwen3-asr/fp8-kv`, `--kv-cache-dtype fp8_e4m3` + `fp8_packed_kv`): **1,784–1,792 RTFx** at TP1 × DP8 LNC1 MNS48, against 1,717–1,721 for bf16 KV. WER 6.07 %. KV capacity doubles (105k tokens per replica), but MNS 64/96 is not faster.
- **Variants evaluated and rejected** for the batched-decode change. These stay behind `NKILIB_CR17_ALLOW_EXPERIMENTAL=1` for reproduction only:
  - `evict` (one PSUM eviction per 32 batches) gives **wrong results on hardware**, though it is bit-identical in the simulator.
  - `kcap64`/`kcap128` (coarser K gather) is 31–38 % slower.
  - `ktpe` (K via dma_copy + PE transpose) is 5–10 % slower.
  - `vbuf` and `vskip` (persistent V buffers, and oob skip on the V padding) are neutral to −7 %, so they are not enabled by default.
- **`kskip`/`kvskip`** (oob skip on the K gather) **hang the trn2 device** and are refused.

## Use
nkilib ships inside the `nki` wheel as the top-level package `nkilib`. To use this branch, put `src/nkilib_src` first on `PYTHONPATH`. The bundled copy is then shadowed, and nothing needs to be reinstalled:

```bash
git clone -b qwen3-asr/2.33-champion https://github.com/jimburtoft/nki-library.git
export PYTHONPATH=$PWD/nki-library/src/nkilib_src:$PYTHONPATH
python -c "import nkilib; print(nkilib.__file__)"   # must point into nki-library/
```

**Use private NKI caches.** The NKI trace-cache key does not hash nkilib source, so a shared cache silently serves the stock kernels. Set a fresh `NKI_TRACE_CACHE_URL` and `NKI_COMPILE_CACHE_URL` (and `VLLM_CACHE_ROOT`) for each nkilib revision. The vllm-neuron champion's `qwen3_asr/setup_champion.sh` does all of this.

Requires the SDK 2.33 toolchain: nki 0.7.x, neuronx-cc 2.28.x.
