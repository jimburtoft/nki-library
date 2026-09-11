# Sequence-packing tile pruning -- status

**Correct and hardware-validated, including multi-section (seqlen > 8192).**

Supersedes the earlier `HW_VALIDATION_WARNING.md`. The multi-section gap noted in the previous
revision of this file is now **fixed**.

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

**Multi-section (this is where it matters -- the customer targets 128K via context parallelism,
so per-rank lengths are the relevant shapes):**

| Config | sections | MAC ratio | **wall-clock** | bit-identical |
|---|---|---|---|---|
| 8192 = 8 x 1024 | 1 | 1.77x | **1.34x** | yes (0/64) |
| 16384 = 16 x 1024 (CP=8 rank of 128K) | 2 | 3.53x | **2.93x** | yes (0/128) |
| 32768 = 32 x 1024 (CP=4 rank of 128K) | 4 | 7.07x | **5.95x** | yes (0/256) |
| 65536 = 64 x 1024 (CP=2 rank of 128K) | 8 | **14.14x** | **12.15x** | yes (0/512) |

Realized efficiency (wall-clock / MAC ratio) *improves* with length: 76%, 83%, 84%, 86%.

Savings grow with sequence length, as expected: the dense grid is O(seqlen^2) while useful work
is only O(seqlen x segment). **The optimization is most valuable exactly where the customer needs
it.**

`test_attention_cte_seqpack_prune.py`: **36 passed, 0 xfailed**, including device
output-neutrality across LNC=1 and LNC=2 and five multi-section configs.

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

## Multi-section support (how it was fixed)

Two distinct defects, both specific to sequence packing:

**1. Accumulators were initialized on section 0 instead of the group's first LIVE section.**

The flash accumulators (`mm1_running_max`, `exp_running_sum`, `flash_attn_correction_factor`) are
initialized on `sp.section_idx == 0` and updated thereafter. That is correct for causal and dense
masking because their liveness is **monotone** in section index -- if a section is live for a
group, every earlier section is too, so the first live section is always section 0.

**Sequence-packing liveness is not monotone.** A Q group belongs to one segment `[a, b)` and is
live only in sections overlapping `[a, b)` -- a contiguous *middle band*, not a prefix. A group
whose segment begins past the first section is skipped in section 0, so its initializer never
ran and the update path read uninitialized accumulators. Symptom: `nan`.

Fixed by `_is_first_live_section()`, used in place of `sp.section_idx == 0` at all five
accumulator-init / write-vs-accumulate branches. It falls back to `sp.section_idx == 0` whenever
no compile-time layout is available, so existing behavior is untouched.

**2. The empty-span guard returned the wrong answer for "is there a NEXT section".**

`_has_any_compute_bounds_section` originally returned `True` for a section past the end of active
K. That looks like the safe, conservative choice for "should I skip this tile?" -- and it is. But
callers also ask the same question about the *next* section to compute
`is_last_section_with_compute`, where `True` means *"more work is coming"*. For a group whose only
live section was the last one, that suppressed the final `1/sum` normalization and silently
emitted **unnormalized** output.

Symptom: finite values, no `nan`, cos 0.718, and *exactly* the 64 of 128 Q groups whose only live
section was section 1. Fixed by returning `False` for an empty span.

**Diagnostic worth reusing**: a per-Q-group error map identified both defects immediately. The
first gave `nan`; the second gave a clean partition where every wrong group shared the same
liveness pattern (`livesec=[1]`). A scalar cosine would have shown "0.718, still broken" and
nothing more.

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
