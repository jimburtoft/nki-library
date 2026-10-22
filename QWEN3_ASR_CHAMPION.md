# nkilib champion for Qwen3-ASR (SDK 2.33)

This branch is the public **`2.33`** tag of `aws-neuron/nki-library` (`726c501`) plus two single-token decode kernel improvements. It is meant to run with the Qwen3-ASR vllm-neuron champion branch (`jimburtoft/vllm-neuron`, branch `qwen3-asr/0.29-champion`).

| commit | kernel | change | measured on trn2 (SDK 2.33 HW) |
|---|---|---|---|
| `attention_block_tkg` / `qkv_tkg` | At B·S == 1 the QKV weight is stationary and the output is d-major. That removes 8 serial Vector PSUM adds and 6 head transposes, and fuses Q\|K RMSNorm and a single RoPE. | kernel **−10.0 % (LNC2) / −12.3 % (LNC1)** at a 2048-hidden decoder shape |
| `mlp_tkg` | At T == 1 with gate/up column tiling off, the down projection runs in column-tiling mode: 24 matmuls instead of 96 strided ones, plus one contiguous output store. | kernel **−6.1 to −7.5 % (LNC2)**, neutral at LNC1, output bit-identical |

- **Scope:** every other shape and caller runs the existing code. The gates are in the commit messages.
- **Tests:** stock vs patched, the nkilib suites on SDK 2.33 hardware gave identical pass/fail. The new tests are `test_attn_blk_fast` (2 configs), `test_mlp_tkg_single_token_down_column_tiling` (5 cases) and `test/unit/nkilib/core/mlp/test_mlp_tkg_single_token_gate.py`.
- **Qwen3-ASR single stream** (TP4, LNC2, 30 s audio): E2E p50 360 → 342 ms with the attention change. The mlp change adds a further −9 ms.

## Use
nkilib ships inside the `nki` wheel as the top-level package `nkilib`. To use this branch, put `src/nkilib_src` first on `PYTHONPATH`. The bundled copy is then shadowed, and nothing needs to be reinstalled:

```bash
git clone -b qwen3-asr/2.33-champion https://github.com/jimburtoft/nki-library.git
export PYTHONPATH=$PWD/nki-library/src/nkilib_src:$PYTHONPATH
python -c "import nkilib; print(nkilib.__file__)"   # must point into nki-library/
```

**Use private NKI caches.** The NKI trace-cache key does not hash nkilib source, so a shared cache silently serves the stock kernels. Set a fresh `NKI_TRACE_CACHE_URL` and `NKI_COMPILE_CACHE_URL` (and `VLLM_CACHE_ROOT`) for each nkilib revision. The vllm-neuron champion's `qwen3_asr/setup_champion.sh` does all of this.

Requires the SDK 2.33 toolchain: nki 0.7.x, neuronx-cc 2.28.x.
