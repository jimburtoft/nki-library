# Sequence-packing tile pruning -- status

**Correct and hardware-validated for seqlen <= 8192. Multi-section (seqlen > 8192) is a known gap.**

Supersedes the earlier `HW_VALIDATION_WARNING.md`: the numerically-broken per-tile
implementation has been replaced.

## What works

`attention_cte(..., segment_cu_seqlens=tuple(cu_seqlens))` prunes MM1 and the exp pass for K
tiles whose every (q, k) pair is outside the packed-sequence bounds.

Hardware-verified on trn2.3xlarge (DLAMI 20260818 / SDK 2.32, neuronx-cc 2.27.5334, NKI 0.6.0):

| Config | MAC ratio | wall-clock | realized | bit-identical |
|---|---|---|---|---|
| 8192 = 8 x 1024 non-causal | 1.77x | **1.33x** | 75% | yes |
| 8192 = 8 x 1024 causal | 1.71x | **1.35x** | 79% | yes |
| 8192 = 16 x 512 non-causal | 1.87x | **1.34x** | 72% | yes |
| 4096 = 4 x 1024 non-causal | 1.59x | **1.27x** | 80% | yes |
| 4096 unaligned [1500,1300,796,500] | 1.41x | **1.19x** | 85% | yes |
| 4096 single segment (control) | 1.00x | 1.00x | - | yes |

`test_attention_cte_seqpack_prune.py`: **35 passed, 1 xfailed**, including 20 device
output-neutrality cases across LNC=1 and LNC=2.

## Why MM2 is deliberately NOT pruned

The kernel's SBUF/PSUM buffers are ring/modulo-allocated for the 2-deep software pipeline:

| Buffer | block_dim | num_free_tiles | effect |
|---|---|---|---|
| `mm1_masked` | `[num_grps, n_lt]` | `[2, n_lt]` | modulo 2 over Q groups |
| `mm1_partial_max` | `[num_grps]` | `[2]` | modulo 2 over Q groups |
| `exp_sb` | `[num_grps, n_lt]` | `[1, n_lt]` | modulo 1 over Q groups |
| `mm2_psum` | bank `(4 + large_tile_idx % (banks-4))` | shared | `nc_matmul` accumulates |

The recycled axis is the **Q-group** axis, while pruning varies *within* a Q group. So simply
skipping a tile leaves a **previous Q group's finite scores** at that offset -- not zeros or
`-inf` -- and those leak into the softmax max, the softmax denominator, and the PV accumulation.

The working implementation therefore prunes MM1 and exp *and writes the algebraic identity into
every consumer buffer the skipped tile owns*: `_SEQPACK_SKIP_FILL` (-30000.0, which exponentiates
to 0 without overflowing when the running-max bias is added) into `mm1_masked`, and 0.0 into
`exp_sb` and `exp_tp_sb`. `mm1_partial_max` and `exp_partial_sum` are already memset per group.

**MM2 is left intact on purpose**: with exp always writing `exp_tp_sb`, MM2's input is fully
defined and PSUM accumulation retains a well-defined first write. Pruning MM2 as well is where the
remaining upside lies (the broken version reached 3.8-14x MAC reduction), but it requires an
explicit PSUM-bank initialization or forced-first-write story. Do not attempt it without one.

Note: `_FLOAT32_MIN` is **not** a usable fill value here -- it overflowed to `2.5e38` in the
output once the running-max bias was applied.

## Known gap: multi-section (seqlen > 8192)

Tracked by `test_seqpack_prune_multisection_output_neutral` as a **strict xfail**, so it will fail
loudly when fixed. At 16384 = 16 x 1024 the prune produces `nan`; the cross-section running-max /
running-sum accumulation is not yet handled. This is the highest-value remaining work -- measured
2.85x wall-clock / 3.53x MACs at that shape.

## Validation lesson

The original implementation passed predicate soundness (211,655 tiles), NKI simulator
bit-identity (7/7), compiler `mac_count`, lowered-IR matmul counts, and an inertness regression --
and was still wrong on hardware (cos 0.158 vs 0.999987). `nki.simulate` models logical buffers
independently and cannot reproduce modulo SBUF aliasing or PSUM accumulation.

**For any optimization that changes which instructions execute inside this kernel, run the
hardware output-neutrality test first, on one small config, before investing in breadth.** The
local ladder took hours; the hardware run that falsified it took 90 seconds.

## Reproduce

```bash
export PYTHONPATH=<repo>/src/nkilib_src:$PYTHONPATH
python3 -m pytest test/integration/nkilib/core/attention/test_attention_cte_seqpack_prune.py -q
python3 prototype/bench_hardware.py --iters 20   # wall-clock + device correctness
```
