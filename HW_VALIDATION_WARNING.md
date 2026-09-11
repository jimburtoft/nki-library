# WARNING: `feature/seqpack-tile-prune` is NOT numerically correct on hardware

**Status as of 2026-09-11: DO NOT USE. DO NOT SHIP. DO NOT SEND TO A CUSTOMER.**

Commit `dbddfa3` on this branch passed every compile-time and simulator check but **fails
numerical validation on real trn2 hardware**. It was validated on the NKI simulator, which
does not model the SBUF/PSUM reuse this optimization interacts with.

## What was measured (trn2.3xlarge, neuronx-cc 2.27.5334, NKI 0.6.0, LNC=2)

Config: seqlen 4096 packed as 4 x 1024, non-causal, d=128, bs=1, bf16.

| Variant | MACs | latency | cos vs CPU ref | bad q-groups |
|---|---|---|---|---|
| baseline (no descriptor) | 4,362,076,160 | 0.161 ms | **0.999987** | 0 / 32 |
| MM1 prune only | 2,751,463,424 | 0.142 ms | 0.666 | — |
| exp prune only | 4,362,076,160 | 0.155 ms | **nan** | — |
| MM2 prune only | 2,751,463,424 | 0.148 ms | 0.618 | 13 / 32 |
| MM1+exp (MM2 intact) | 2,751,463,424 | 0.092 ms | **nan** | 32 / 32 |
| MM1+MM2 | 1,140,850,688 | 0.136 ms | **nan** | — |
| **all three (as committed)** | 1,140,850,688 | **0.084 ms** | **0.158** | 14 / 32 |

**Every** pruning variant is wrong. Also confirmed wrong at LNC=1 (cos 0.564), so this is
not an LNC-sharding artifact.

The performance result is real and worth noting: **1.92x wall-clock** (0.161 -> 0.084 ms),
which is 50% of the 3.82x MAC reduction. The idea works; this implementation does not.

## Root cause

The kernel's SBUF and PSUM buffers are **ring/modulo-allocated for a 2-deep software
pipeline**, not per-Q-group:

- `mm1_masked`      `block_dim=[num_grps, n_lt]`, `num_free_tiles=[2, n_lt]`  -> modulo 2
- `mm1_partial_max` `block_dim=[num_grps]`,       `num_free_tiles=[2]`        -> modulo 2
- `exp_sb`          `block_dim=[num_grps, n_lt]`, `num_free_tiles=[1, n_lt]`  -> modulo 1
- `mm2_psum` PSUM bank address = `(4 + large_tile_idx % (banks-4)) * PSUM_BANK_SIZE`, shared
  across Q groups; `nc_matmul` into PSUM **accumulates**, so the first write of an
  accumulation group establishes the bank contents.

Skipping a tile therefore does **not** leave zeros or `-inf`. It leaves whatever Q group
`g-1`/`g-2` wrote at that offset: real, finite scores from a *different* query group and a
*different* packed segment. Those leak into the softmax max, the softmax denominator, and
the PV accumulation as if in-bounds.

The error map is structured, which is the fingerprint: q groups 0-15 exact, 16-29 wrong,
30-31 exact. The first two segments alias onto themselves harmlessly; the leak becomes
visible once the pipeline crosses a segment boundary.

Attempted point fixes that did **not** work (14/32 bad in each case):
1. memset `exp_tp_sb` on a pruned tile
2. memset `mm1_masked` to `_FLOAT32_MIN` on a pruned tile (leaked `2.5e38` into the output)
3. both together

## Why the simulator missed it

`nki.simulate` models each logical buffer independently. It does not reproduce modulo SBUF
aliasing or PSUM bank accumulation semantics, so the stale-data path is invisible to it.
This is exactly the failure mode the workspace steering doc warns about: *simulator green
is not ship-ready*.

## What a correct implementation needs

Tile pruning cannot be bolted onto the selection predicates alone. It must be co-designed
with the buffer lifecycle. Any real fix has to establish, per pruned tile, that **every**
consumer either (a) is pruned under a provably identical predicate, or (b) reads a location
that has been explicitly initialized to the algebraic identity for that consumer
(`-inf` pre-exp, `0` post-exp, `0` for PSUM accumulation) -- and that PSUM accumulation
groups still have a well-defined first write.

The likely tractable direction is to prune at **whole-large-tile or whole-section
granularity** rather than per 512/128 tile, since that aligns with the buffer recycling
boundaries instead of cutting across them.

## Reproduce

```bash
# on a trn2 instance, with this branch on PYTHONPATH
python3 prototype/bench_hardware.py --only 3      # correctness + wall clock
python3 iso.py                                    # per-site isolation
python3 iso4.py                                   # per-q-group error map
```

Raw logs: `working/zyphra_troubleshooting/task010/results/hw/`.
