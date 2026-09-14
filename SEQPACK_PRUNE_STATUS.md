# Sequence-packing tile pruning -- status

**Correct and hardware-validated, including multi-section (seqlen > 8192) and striped CP.**

> ## ⚠ APPLICABILITY LIMIT: ONE NEFF PER PACKING LAYOUT
>
> `segment_cu_seqlens` is a **trace-time** parameter (the NKI tracer treats tuples as shape
> parameters), so **every distinct packing layout produces a distinct graph and therefore a
> distinct NEFF.** Measured, compile-only, at seqlen 4096 with 4 segments:
>
> | | distinct graphs across 12 random layouts |
> |---|---|
> | without descriptor | **1** (one NEFF serves every layout) |
> | with descriptor | **12** (a NEFF per layout) |
>
> MAC counts differ per layout too (2.75G / 2.82G / 2.95G) -- the graph is genuinely specialized
> to the boundary positions, which is both why it is fast and why it cannot be layout-agnostic.
>
> **Consequence:**
>
> | Workload | Verdict |
> |---|---|
> | **Fixed**-resolution packing (same layout every step, e.g. always 8 x 1024) | **1 NEFF. Full benefit. Use it.** |
> | **Variable**-resolution packing (layout changes per batch) | **Not viable as built** -- recompile per batch. For 4 resolutions filling 8192 there are ~47M possible ordered layouts. |
>
> **Tested mitigation that does NOT work**: quantizing descriptor boundaries to the 512-key tile
> grid. 12 layouts still produced 12 graphs -- the cut positions retain ~7 distinct slots each, so
> the combinatorics survive. Coarser buckets (2048) would collapse the count but gut the pruning.
>
> **Mitigations that do work:**
> 1. A **uniform `segment_len: int`** parameter instead of full `cu_seqlens` -- one integer, so one
>    NEFF per segment length. Covers fixed-resolution packing exactly. **Not yet implemented; this
>    is the highest-value next step.**
> 2. **Caller-side bucketing**: canonicalize layouts to a small fixed set (pad each image to a
>    power-of-2 bucket, or sort segments so only the multiset matters -- 165 multisets vs 47M
>    ordered layouts). This is a caller change, not a kernel change.
>
> Probes: `task010/prototype/recompile_probe.py`, `quant_probe.py`,
> `tile_rounding_analysis.py`, `bench_tile_rounding_device.py`.
>
> ### Tile-grid rounding: measured on device, and it does NOT help
>
> The natural idea is to round document boundaries out to the 512-key tile grid, so layouts
> differing only within a tile share a NEFF. An analytical tile-count model predicted this was
> free-to-positive (~1.07x throughput, 3x fewer NEFFs). **The device disagrees.**
>
> seqlen 16384, 12 random variable-document batches, LNC=2, figure of merit is **real (unpadded)
> tokens/s** -- padding is not useful work:
>
> | strategy | NEFFs | real tok | latency | real Mtok/s | net |
> |---|---|---|---|---|---|
> | no descriptor (no pruning) | 1 | 15735 | 2.181 ms | 7.21 | 0.39x |
> | **exact boundaries** | 12 | 15608 | 0.848 ms | 18.40 | **1.00x** |
> | **sort only, no rounding** | 12 | 15608 | 0.834 ms | 18.71 | **1.02x** |
> | round 256 + sort | 12 | 14354 | 0.821 ms | 17.48 | 0.95x |
> | round 512 + sort | 11 | 12319 | 0.805 ms | 15.31 | **0.83x** |
> | round 1024 + sort | 7 | 10191 | 0.800 ms | 12.74 | 0.69x |
>
> **Why the model was wrong**: it treated live-tile count as proportional to time. On device
> latency is nearly flat across grids (0.848 -> 0.800 ms, only -6%) while padding destroys real
> tokens fast (-21% at grid 512). The kernel is **not tile-bound at this shape**, so the tile
> saving is not cashable while the padding cost is immediate. Rounding trades 17-31% of real
> throughput for a modest NEFF reduction.
>
> **What the same run establishes**: pruning itself is worth **2.58x latency / 2.55x real-token
> throughput** (2.181 -> 0.848 ms) on **variable** document layouts. That is the headline, and it
> is now device-measured rather than inferred.
>
> **Sorting alone is free** (+2%, no padding) and is the part that can bound the NEFF count, since
> only the multiset of document sizes matters. In this 12-batch sample every multiset was distinct
> so no collapse was visible; a larger sweep is needed to quantify it, but it cannot hurt.
>
> **Recommendation: exact boundaries plus caller-side sorting. Do not round.**

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

## Striped context parallelism (WORKING, device-verified)

| Config (global / segment / cp_degree) | causal | MAC ratio | bit-identical |
|---|---|---|---|
| 32768 / 1024 / 4 | yes | 1.80x | **yes** (0/64) |
| 32768 / 1024 / 4 | no | 1.87x | **yes** (0/64) |
| 32768 / 2048 / 8 | yes | 1.65x | **yes** (0/32) |
| 65536 / 1024 / 4 | no | **3.74x** | **yes** (0/128) |
| 65536 / 1024 / 8 | yes | 1.80x | **yes** (0/64) |
| 16384 / 1024 / 2 | no | 1.87x | **yes** (0/64) |

Full suite: **47 passed, 0 failed** (40 non-CP + 6 striped output-neutrality + 1 guard).

### The coordinate contract (this is what took three attempts)

Read off the library's own bound builder,
`test/integration/nkilib/utils/sequence_packing_helpers.py::cu_seqlens_to_striped_bounds`:

```python
local_start = cu_seqlens[i]     // cp_degree
local_end   = cu_seqlens[i + 1] // cp_degree
bound_min[local_start:local_end] = local_start
bound_max[local_start:local_end] = local_end
```

So bounds are in **LOCAL** coordinates -- global boundaries **divided** by `cp_degree` -- and are
**identical on every rank**, provided every boundary is a multiple of `cp_degree`. In local
coordinates the packed layout is still plain block-diagonal.

This is corroborated by `nisa.range_select`'s documented semantics: it compares
`range_start + lane` against the bounds, and the kernel passes `range_start = k_start_pos`, a
**local** tile offset. The bounds must therefore be local, which is exactly what the caller builds.

`_segment_spans_local()` implements precisely this: divide the spans by `cp_degree`, then run the
ordinary contiguous overlap test. `_striped_prune_is_safe()` enforces the divisibility precondition
and declines otherwise (falling back to computing every tile), because without it the local layout
differs per rank and no rank-independent compile-time decision exists.

### Two earlier attempts failed, and both failed the same way

1. A **stride-walk model**: local `i` on rank `r` maps to global `i * cp_degree + r`, tested against
   **global** spans. Wrong coordinate space *and* needlessly complex. Device result: cos 0.51.
2. A **CPU reference** written to judge whether the resulting delta was benign. It scored cos 0.126
   against the **unmodified** kernel, so it could not arbitrate anything -- yet it printed a
   reassuring verdict.

Worse, an isolation test reported "0 unsound prunes, mapping agrees with the mask" -- because it was
fed the same stride assumption as ground truth. **It confirmed the model against itself.**

**The lesson worth keeping: when a kernel's convention is unclear, read it off a real caller or the
instruction's documented semantics. Do not infer it from a layout's name, and never let a
self-authored oracle adjudicate a self-authored model.** The trustworthy check throughout was
bit-identity against the same kernel with the feature disabled, which requires no semantic model at
all.

## Striped context parallelism -- design notes (for whoever resumes this)## Striped context parallelism -- design notes (for whoever resumes this)

**Why this matters more than the raw numbers above**: `attention_cte` requires
`cp_striped_input=True` when sequence packing is combined with CP (`attention_cte.py:677` --
contiguous CP is explicitly unsupported with packing). Until this change, all three predicates
bailed out to "no pruning" under striped input, so **for any packed workload under context
parallelism the optimization was completely inert.** Since CP is how long sequences are reached,
that covered the case that actually matters.

Striped CP distributes the sequence round-robin: local position `i` on rank `r` is global position
`i * D + r`, `D = global_cp_deg`. A contiguous local range therefore maps to an arithmetic
progression of stride `D`, not a contiguous global span.

**Prunability survives striping** -- a strided set of 128 local positions still lands in only a
few segments:

| Global seqlen | segment | D | local len | live tiles | ceiling |
|---|---|---|---|---|---|
| 131072 | 1024 | 4 | 32768 | 1.6% | 64x |
| 131072 | 1024 | 8 | 16384 | 3.1% | 32x |
| 131072 | 1024 | 16 | 8192 | 6.2% | 16x |
| 131072 | 4096 | 8 | 16384 | 3.1% | 32x |
| 131072 | 1536 (unaligned) | 8 | 16384 | 4.2% | 24x |
| 65536 | 1000 (unaligned) | 8 | 8192 | 9.2% | 10.9x |

**The SPMD constraint, and why this is expressible at all**: one NEFF runs on every rank, and
`cp_offset` is a *runtime* tensor -- so a compile-time prune decision must not depend on the rank.
It does not: changing `r` shifts every global position by the same constant `< D`, which cannot
move a position across a segment boundary **as long as every segment is longer than `D`**. That
precondition is enforced by `_striped_prune_is_safe()`, which disables pruning (returns "keep
everything") when any segment is shorter than the CP degree, rather than silently mis-pruning.

Verified at trace time by `verify_striped_cp.py`: **0 unsound prunes and 0 over-conservative tiles
(the predicate is exact) across 12 configurations**, decision confirmed **rank- and
ring-step-independent**, and the short-segment guard degrades correctly.

**Two real defects were found and fixed along the way** (both still worth keeping):

1. **Striped with unknown degree would crash.** `ring_attention_fwd.py:1133-1134` gates
   `cp_offset`/`global_cp_deg` on `use_causal_mask` while forwarding `cp_striped_input`
   unconditionally, so a *non-causal* ring call arrives with `striped=True, global_cp_deg=None`.
   The mapping would have dereferenced `None`. Now refuses to prune, since the data really is
   interleaved and treating it as contiguous would be silently wrong.
2. **The descriptor's coordinate system was under-specified.** `segment_cu_seqlens` is now
   documented and asserted to be in **GLOBAL** coordinates -- the same system as
   `bound_min`/`bound_max`. Without CP that equals `seqlen_q`; under striped CP it is
   `seqlen_q * global_cp_deg`. The original assert compared against the local length and rejected
   every legitimate CP call.

**Non-causal CP is now ENABLED in this kernel (verified on device).** Two asserts previously
rejected it -- `"CP currently only supports causal attn"` and `"Striped CP requires
causal_mask=True"`. Both are lifted **for the sequence-packed case only**: packing supplies its own
per-query mask via `bound_min`/`bound_max`, so the causal term is simply *absent* rather than
missing, and there is nothing for CP to reconcile against. Also fixed: the bound construction no
longer adds `cp_offset` to what is a constant sentinel upper bound on the non-causal packed path
(`range_sel_ubs` is `seqlen_k_active`, not a per-row causal position, so adding the offset inflated
it meaninglessly).

Verified on trn2 (LNC=2, global 8192, `cp_deg=4`, ranks 0 and 2): compiles, runs, and is
**bit-identical** with and without the descriptor. Full suite: **40 passed**.

This matters because it removes a restriction I had previously reported as structural. It is a
parameter-level gate, not an architectural limit -- non-packed non-causal CP remains rejected, as it
should be, since it genuinely has no mask to fall back on.

**Still blocked upstream for the customer's exact workload**: the ring wrappers assert
`bound_min/bound_max require use_causal_mask=True` (`ring_attention_fwd.py:995`,
`ring_attention_bwd.py:133`). A non-causal packed ViT under ring CP therefore cannot pass bounds
through the wrapper today, independent of anything in this file.

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

## Coordination with the 128K max-seqlen work (Task 009)

Another agent (`opus-seqlen-128k`) is working on the same file for a different goal: reaching
seqlen 131072. There is **no git conflict today** -- that work has no branch on this fork yet -- but
there is one **design interaction that must not be lost**.

### The collision: rolling the loop nest would delete this optimization

Task 009's most promising mitigation is converting `attention_cte`'s **unrolled Python loop nest**
into a **rolled `nl.fori_loop`**, because the 128K ceiling is a trace-time/`colz` blob-size problem
(their gates A-E all pass on GA, colz flat at +0.2% for 8x trips).

**But everything in this file depends on the loops being unrolled.** Compile-time tile pruning *is*
trace-time specialization:

- `_has_any_compute_bounds` / `_has_any_compute_bounds_section` / `_is_first_live_section` --
  ~20 call sites, all plain-Python `if`/`continue` on loop indices
- `_segment_spans_local()` iteration at two sites
- and the pre-existing `_has_any_compute_causal` / `_has_any_compute_swa` skips this work is modelled on

A rolled loop must convert every one of those to **runtime predication**, at which point the tiles
are still *issued* and the saving disappears. Task 009's own notes say as much: rolling *"gives up
causal tile pruning (~half the tiles)"*. The same applies, in full, to bound-based pruning.

### Why the two are not simply in opposition

The tension is real but narrower than it looks:

- **Rolling helps the causal long-context case**, where the ceiling binds and pruning is worth ~2x.
- **Pruning helps the non-causal packed case**, where the ceiling binds *less* (Task 009 notes
  *"non-causal ViT pays little"*) and pruning is worth **3.5x-14x**.

For a **non-causal packed ViT** -- the workload driving this -- pruning is worth substantially more
than rolling, and pruning already reaches 65536 today. So the sequencing that preserves both is:

1. Prefer **query chunking** (Task 009's validated caller-side mitigation) to reach 128K, since it
   leaves the loop nest unrolled and keeps pruning intact.
2. Treat **rolling as an upstream ask with a PoC**, as Task 009 already frames it -- and if it is
   ever adopted, it needs a `segment_cu_seqlens`-aware story, or a packed workload silently loses
   3.5x-14x.

**Do not land a rolled loop nest on this branch without re-running
`test_attention_cte_seqpack_prune.py`.** The MAC-floor assertions in that suite are exactly the
tripwire: they fail if pruning becomes inert, which is the failure mode rolling would cause.

### Non-overlapping parts (safe to develop in parallel)

Task 009's other three mitigations do not touch anything here:

| 009 item | Where | Overlaps this work? |
|---|---|---|
| `colz` uint32 overflow | closed-source `_nki` C++ | no (AWS ticket) |
| `_MAX_SEQLEN` warn -> assert | this file, but a different region | **no line overlap** (verified) |
| lowering-stampede pre-lock | framework `nki_kernel.py` | no |
| query chunking | caller / trainer | no |

## Reproduce

```bash
export PYTHONPATH=<repo>/src/nkilib_src:$PYTHONPATH
python3 -m pytest test/integration/nkilib/core/attention/test_attention_cte_seqpack_prune.py -q
python3 prototype/bench_hardware.py --iters 20   # wall-clock + device correctness
```
