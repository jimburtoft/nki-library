"""Hardware correctness tests for compile-time sequence-packing tile pruning.

WHY THIS FILE EXISTS
--------------------
The `segment_cu_seqlens` optimization prunes K tiles that are entirely outside the
packed-sequence bounds. Its first implementation passed *every* static and simulator
check and was still numerically wrong on hardware:

    predicate soundness (211,655 tiles, 0 unsound)   PASS
    NKI simulator bit-identity (7/7 configs)         PASS
    compiler mac_count reduction + 1.00x control     PASS
    lowered-IR matmul counts                         PASS
    regression: inert without the descriptor         PASS
    ---> hardware numerical validation                FAIL  (cos 0.158 vs 0.999987)

Root cause: the kernel's SBUF/PSUM buffers are ring/modulo-allocated for a 2-deep
software pipeline (`mm1_masked` num_free_tiles=[2,..], `mm1_partial_max` [2], `exp_sb`
[1,..], and `mm2_psum` banks addressed by `large_tile_idx % (banks-4)` and accumulated
into by `nc_matmul`). Skipping a tile leaves a *previous Q group's* finite scores rather
than zeros or -inf, and those leak into the softmax max, the softmax denominator, and the
PV accumulation.

`nki.simulate` models each logical buffer independently and does not reproduce modulo SBUF
aliasing or PSUM accumulation, so it is structurally incapable of catching this. Therefore
**these tests must run on hardware**, and they are the gate for this feature.

DESIGN NOTES -- what makes these tests able to catch the bug
------------------------------------------------------------
1. They compare against the *unpruned kernel* (same bounds tensors, descriptor withheld),
   not only against a CPU reference. The prune is specified to be output-neutral, so any
   deviation is a defect by definition and needs no tolerance argument.
2. They use enough Q groups to exercise the pipeline. The bug is invisible with <= 2
   segments and few Q groups, because the modulo-2 aliasing lands on the same segment.
   Configurations therefore span >= 4 segments and >= 16 Q groups.
3. They assert per-Q-group, so a failure localizes to a group index. The original failure
   signature was "groups 0-15 exact, 16-29 wrong, 30-31 exact", which is what identified
   pipeline aliasing as the cause. A scalar cosine would have hidden that structure.
4. They include a single-segment control that must be bit-identical for a trivial reason
   (nothing is prunable), which distinguishes "prune is broken" from "descriptor plumbing
   is broken".
5. They cover both LNC=1 and LNC=2, because a plausible-but-wrong hypothesis was that this
   was an LNC-sharding artifact. It is not, and the tests should keep proving that.
"""

import numpy as np
import pytest

import nki.language as nl

from nkilib.core.attention.attention_cte import attention_cte


_Q_GRP_SZ = 128


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _cu_from_spec(seqlen, spec):
    """spec is either a uniform segment length (int) or an explicit list of lengths."""
    if isinstance(spec, int):
        assert seqlen % spec == 0, f"{seqlen} not divisible by {spec}"
        return tuple(range(0, seqlen + 1, spec))
    cu = [0]
    for x in spec:
        cu.append(cu[-1] + x)
    assert cu[-1] == seqlen, f"segment lengths sum to {cu[-1]}, expected {seqlen}"
    return tuple(cu)


def _bounds_from_cu(cu, seqlen, bs):
    """The (bound_min, bound_max) device tensors the caller would pass, per query row."""
    lo = np.zeros(seqlen, dtype=np.float32)
    hi = np.zeros(seqlen, dtype=np.float32)
    for i in range(len(cu) - 1):
        lo[cu[i] : cu[i + 1]] = cu[i]
        hi[cu[i] : cu[i + 1]] = cu[i + 1]
    return (
        np.tile(lo.reshape(1, seqlen, 1), (bs, 1, 1)).astype(np.float32),
        np.tile(hi.reshape(1, seqlen, 1), (bs, 1, 1)).astype(np.float32),
    )


def _cpu_reference(q, k, v, lo, hi, causal):
    """Dense-then-mask reference. q:[b,sq,d] k:[b,d,sk] v:[b,sk,d] -> [b,sq,d].

    Chunked over queries so long sequences do not materialize an [sq, sk] score matrix.
    Rows whose mask is entirely True (possible only for degenerate bounds) are defined to
    produce 0, matching the kernel's all-masked behavior rather than propagating NaN.
    """
    b, sq, d = q.shape
    sk = k.shape[2]
    out = np.zeros((b, sq, d), dtype=np.float32)
    chunk = 512
    for bi in range(b):
        kk = k[bi].astype(np.float32).T
        vv = v[bi].astype(np.float32)
        kidx = np.arange(sk)[None, :]
        for s in range(0, sq, chunk):
            e = min(s + chunk, sq)
            scores = q[bi, s:e].astype(np.float32) @ kk.T
            mask = (kidx < lo[bi, s:e].reshape(-1, 1)) | (kidx >= hi[bi, s:e].reshape(-1, 1))
            if causal:
                mask = mask | (kidx > np.arange(s, e)[:, None])
            scores = np.where(mask, -np.inf, scores)
            mx = scores.max(axis=-1, keepdims=True)
            mx = np.where(np.isfinite(mx), mx, 0.0)
            ex = np.exp(scores - mx)
            ex = np.where(np.isfinite(ex), ex, 0.0)
            out[bi, s:e] = (ex / np.maximum(ex.sum(axis=-1, keepdims=True), 1e-30)) @ vv
    return out


def _cos(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 and nb == 0.0:
        return 1.0
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(a @ b / (na * nb))


def _per_group_max_err(a, b, seqlen):
    """Max abs error per 128-row Q group. Returns a list of (group_index, error)."""
    a = a.reshape(-1, seqlen, a.shape[-1])
    b = b.reshape(-1, seqlen, b.shape[-1])
    err = np.abs(a - b).max(axis=(0, 2))  # per query row, worst over batch and d
    out = []
    for g in range(0, (seqlen + _Q_GRP_SZ - 1) // _Q_GRP_SZ):
        s = g * _Q_GRP_SZ
        e = min(s + _Q_GRP_SZ, seqlen)
        out.append((g, float(err[s:e].max())))
    return out


def _run_on_device(q, k, v, bmin, bmax, causal, cu, lnc, tag):
    """Compile and execute attention_cte on hardware. cu=None means no descriptor."""
    from nki.compiler.ncc_driver import CompileOptions, compile_bir_to_neff
    from nki.compiler.driver import compile_to_bir
    from nki.compiler.frontend import TracerFrontend

    inputs = dict(
        q=q, k=k, v=v, bound_min=bmin, bound_max=bmax,
        scale=1.0, causal_mask=causal, tp_q=True, tp_k=False, tp_out=False,
    )
    if cu is not None:
        inputs["segment_cu_seqlens"] = tuple(cu)

    opts = CompileOptions(
        target="trn2", lnc=lnc,
        output_path=f"/tmp/seqpack_{tag}.neff",
        artifacts_dir=f"/tmp/seqpack_{tag}_art",
    )
    bir = compile_to_bir(attention_cte, frontend=TracerFrontend(), inputs=inputs, compile_opts=opts)
    arg_names = [s.name for s in bir.descriptor.input_specs]
    out_names = [s.name for s in bir.descriptor.output_specs]
    arrays = [inputs[n] for n in arg_names if isinstance(inputs.get(n), np.ndarray)]
    ck = compile_bir_to_neff(opts, bir, arrays, arg_names, out_names)
    res = ck.run(**{n: inputs[n] for n in arg_names if isinstance(inputs.get(n), np.ndarray)})
    outs = res.outputs
    arr = outs[0] if isinstance(outs, (list, tuple)) else (
        list(outs.values())[0] if isinstance(outs, dict) else outs)
    return np.asarray(arr, dtype=np.float32), int(bir.mac_count)


def _make_inputs(bs, seqlen, d, seed=0xC0FFEE):
    rng = np.random.default_rng(seed)
    return (
        (rng.standard_normal((bs, seqlen, d)) * 0.5).astype(np.float32),
        (rng.standard_normal((bs, d, seqlen)) * 0.5).astype(np.float32),
        (rng.standard_normal((bs, seqlen, d)) * 0.5).astype(np.float32),
    )


# ---------------------------------------------------------------------------
# The configuration matrix.
#
# Deliberately includes cases that the original defect DID and DID NOT show up in, so a
# regression cannot be masked by picking friendly shapes:
#   * >= 4 segments and >= 16 Q groups -> exercises modulo-2 pipeline aliasing
#   * 2 segments / few Q groups        -> the "looks fine" cases
#   * single segment                   -> control, nothing prunable
#   * unaligned boundaries             -> partially-live tiles at 512 granularity
# ---------------------------------------------------------------------------
SEQPACK_PRUNE_CASES = [
    # (id, bs, seqlen, d, segment_spec, causal)
    ("s4096_4x1024_nocausal",      1, 4096, 128, 1024, False),   # the original failure
    ("s4096_4x1024_causal",        1, 4096, 128, 1024, True),
    ("s4096_8x512_nocausal",       1, 4096, 128, 512,  False),
    ("s8192_8x1024_nocausal",      1, 8192, 128, 1024, False),
    ("s8192_16x512_nocausal",      1, 8192, 128, 512,  False),
    ("s2048_2x1024_nocausal",      1, 2048, 128, 1024, False),   # few groups: passed before
    ("s4096_unaligned_nocausal",   1, 4096, 128, [1500, 1300, 796, 500], False),
    ("s4096_unaligned_causal",     1, 4096, 128, [1500, 1300, 796, 500], True),
    ("s4096_1seg_control",         1, 4096, 128, 4096, False),   # control: nothing prunable
    ("bs2_s2048_4x512_nocausal",   2, 2048, 128, 512,  False),
]

# MULTI-SECTION configurations (seqlen > 8192, so more than one flash section).
#
# These are the lengths that actually matter for the customer: their ViT targets 128K via
# context parallelism, so the relevant shapes are the PER-RANK lengths. They are also where the
# payoff is largest -- waste grows with sequence length, since the dense grid is O(seqlen^2)
# while useful work is only O(seqlen * segment).
#
# Multi-section requires the flash cross-section accumulators (mm1_running_max, exp_running_sum,
# flash_attn_correction_factor) to be initialized on a group's FIRST LIVE section rather than on
# section 0. Sequence-packing liveness is a contiguous middle band, not a prefix, so a group
# whose segment starts past the first section is skipped in section 0 -- see
# _is_first_live_section in attention_cte.py.
SEQPACK_PRUNE_MULTISECTION_CASES = [
    # (id, bs, seqlen, d, segment_spec, causal)
    ("s16384_16x1024_nocausal", 1, 16384, 128, 1024, False),   # CP=8 rank of 128K
    ("s16384_16x1024_causal",   1, 16384, 128, 1024, True),
    ("s16384_32x512_nocausal",  1, 16384, 128, 512,  False),
    ("s24576_24x1024_nocausal", 1, 24576, 128, 1024, False),   # 3 sections (odd count)
    ("s32768_32x1024_nocausal", 1, 32768, 128, 1024, False),   # CP=4 rank of 128K
]


@pytest.mark.parametrize("lnc", [1, 2], ids=["lnc1", "lnc2"])
@pytest.mark.parametrize(
    "bs, seqlen, d, seg_spec, causal",
    [c[1:] for c in SEQPACK_PRUNE_CASES],
    ids=[c[0] for c in SEQPACK_PRUNE_CASES],
)
def test_seqpack_prune_is_output_neutral_on_device(bs, seqlen, d, seg_spec, causal, lnc):
    """Passing segment_cu_seqlens must not change the output. Bit-identical is required.

    Both arms receive identical bound_min/bound_max device tensors; the only difference is
    whether the trace-time descriptor is supplied. The prune is specified as output-neutral,
    so exact equality is the correct assertion -- no tolerance is warranted, and using one
    would let real aliasing bugs through.
    """
    cu = _cu_from_spec(seqlen, seg_spec)
    q, k, v = _make_inputs(bs, seqlen, d)
    bmin, bmax = _bounds_from_cu(cu, seqlen, bs)

    base, base_macs = _run_on_device(q, k, v, bmin, bmax, causal, None, lnc, f"b{seqlen}_{lnc}")
    pruned, pruned_macs = _run_on_device(q, k, v, bmin, bmax, causal, cu, lnc, f"p{seqlen}_{lnc}")

    if np.array_equal(base, pruned):
        return

    # Report per-Q-group so the failure localizes. The original defect produced a
    # contiguous band of wrong groups starting mid-sequence, which is the signature of
    # ring-buffer aliasing rather than a masking error.
    groups = _per_group_max_err(base, pruned, seqlen)
    bad = [(g, e) for g, e in groups if e > 0.0]
    detail = ", ".join(f"grp{g}:{e:.3e}" for g, e in bad[:12])
    pytest.fail(
        f"segment_cu_seqlens changed the output (must be bit-identical).\n"
        f"  lnc={lnc} seqlen={seqlen} segments={len(cu)-1} causal={causal}\n"
        f"  MACs {base_macs:,} -> {pruned_macs:,}\n"
        f"  max|delta| = {float(np.max(np.abs(base - pruned))):.6e}\n"
        f"  cos(base, pruned) = {_cos(base, pruned):.8f}\n"
        f"  {len(bad)}/{len(groups)} Q groups differ: {detail}"
        + ("" if len(bad) <= 12 else f" ... (+{len(bad)-12} more)")
    )


@pytest.mark.parametrize(
    "bs, seqlen, d, seg_spec, causal",
    [c[1:] for c in SEQPACK_PRUNE_CASES if c[1] == 1 and c[2] <= 4096],
    ids=[c[0] for c in SEQPACK_PRUNE_CASES if c[1] == 1 and c[2] <= 4096],
)
def test_seqpack_prune_matches_cpu_reference(bs, seqlen, d, seg_spec, causal):
    """Both arms must match a dense-then-mask CPU reference.

    Guards against the case where the prune and the baseline are consistently wrong
    together -- which output-neutrality alone would not detect.
    """
    cu = _cu_from_spec(seqlen, seg_spec)
    q, k, v = _make_inputs(bs, seqlen, d)
    bmin, bmax = _bounds_from_cu(cu, seqlen, bs)
    ref = _cpu_reference(q, k, v, bmin[:, :, 0], bmax[:, :, 0], causal)

    base, _ = _run_on_device(q, k, v, bmin, bmax, causal, None, 2, f"rb{seqlen}")
    pruned, _ = _run_on_device(q, k, v, bmin, bmax, causal, cu, 2, f"rp{seqlen}")

    cos_base = _cos(base.reshape(ref.shape), ref)
    cos_pruned = _cos(pruned.reshape(ref.shape), ref)
    # bf16-class accumulation in a dense-vs-flash comparison; 0.999 is comfortably strict
    # relative to the observed 0.999987 for a correct kernel, and far above the 0.158 the
    # broken implementation produced.
    assert cos_base > 0.999, f"BASELINE kernel disagrees with reference: cos={cos_base:.6f}"
    assert cos_pruned > 0.999, (
        f"PRUNED kernel disagrees with reference: cos={cos_pruned:.6f} "
        f"(baseline cos={cos_base:.6f}) -- prune is numerically incorrect"
    )


@pytest.mark.parametrize(
    "seqlen, seg, expected_min_ratio",
    [
        # Floors for the CORRECT implementation (MM1 + exp pruned; MM2 intentionally left
        # intact so exp_tp_sb is always fully written and PSUM keeps a well-defined first
        # write). Measured values are given for reference.
        #
        # NOTE: earlier, higher floors (3.0/6.0/6.0/12.0) came from an implementation that
        # ALSO pruned MM2 and was numerically WRONG on hardware. Do not restore them without
        # a correctness story for PSUM accumulation.
        (4096, 1024, 1.4),   # measured 1.59x
        (4096, 512, 1.7),    # measured 1.87x
        (8192, 1024, 1.6),   # measured 1.77x
        (8192, 512, 1.7),    # measured 1.87x
    ],
    ids=["s4096_4seg", "s4096_8seg", "s8192_8seg", "s8192_16seg"],
)
def test_seqpack_prune_reduces_macs(seqlen, seg, expected_min_ratio):
    """The prune must actually remove arithmetic, per the compiler's own MAC accounting.

    Compile-only, so it is cheap and runs without a device. Catches a silently inert
    descriptor -- e.g. a predicate wired inside an `if ac.use_swa:` block, which was a real
    defect during development: it under-pruned to 1.59x instead of 3.82x while still
    "working".
    """
    from nki.compiler.ncc_driver import CompileOptions
    from nki.compiler.driver import compile_to_bir
    from nki.compiler.frontend import TracerFrontend

    cu = _cu_from_spec(seqlen, seg)
    bmin, bmax = _bounds_from_cu(cu, seqlen, 1)
    macs = {}
    for use in (False, True):
        inputs = dict(
            q=np.zeros((1, seqlen, 128), np.float32),
            k=np.zeros((1, 128, seqlen), np.float32),
            v=np.zeros((1, seqlen, 128), np.float32),
            bound_min=bmin, bound_max=bmax,
            scale=1.0, causal_mask=False, tp_q=True, tp_k=False, tp_out=False,
        )
        if use:
            inputs["segment_cu_seqlens"] = cu
        opts = CompileOptions(
            target="trn2", lnc=2,
            output_path=f"/tmp/mac_{seqlen}_{seg}_{int(use)}.neff",
            artifacts_dir=f"/tmp/mac_{seqlen}_{seg}_{int(use)}_art",
        )
        macs[use] = int(
            compile_to_bir(attention_cte, frontend=TracerFrontend(),
                           inputs=inputs, compile_opts=opts).mac_count
        )
    ratio = macs[False] / max(macs[True], 1)
    assert ratio >= expected_min_ratio, (
        f"MAC reduction {ratio:.2f}x is below the expected {expected_min_ratio}x "
        f"({macs[False]:,} -> {macs[True]:,}). The descriptor may be wired at the wrong "
        f"scope and only partially effective."
    )


def test_seqpack_descriptor_absent_is_inert():
    """Without the descriptor, the emitted graph must be unchanged.

    Protects existing callers: this feature must be strictly opt-in.
    """
    from nki.compiler.ncc_driver import CompileOptions
    from nki.compiler.driver import compile_to_bir
    from nki.compiler.frontend import TracerFrontend

    seqlen = 4096
    cu = _cu_from_spec(seqlen, 1024)
    bmin, bmax = _bounds_from_cu(cu, seqlen, 1)
    results = {}
    for causal in (False, True):
        inputs = dict(
            q=np.zeros((1, seqlen, 128), np.float32),
            k=np.zeros((1, 128, seqlen), np.float32),
            v=np.zeros((1, seqlen, 128), np.float32),
            bound_min=bmin, bound_max=bmax,
            scale=1.0, causal_mask=causal, tp_q=True, tp_k=False, tp_out=False,
        )
        opts = CompileOptions(
            target="trn2", lnc=2,
            output_path=f"/tmp/inert_{int(causal)}.neff",
            artifacts_dir=f"/tmp/inert_{int(causal)}_art",
        )
        results[causal] = int(
            compile_to_bir(attention_cte, frontend=TracerFrontend(),
                           inputs=inputs, compile_opts=opts).mac_count
        )
    # These are the values the unpatched 2.32 kernel emits; they must not drift.
    assert results[False] == 4_362_076_160, f"non-causal MACs drifted: {results[False]:,}"
    assert results[True] == 2_382_364_672, f"causal MACs drifted: {results[True]:,}"


@pytest.mark.parametrize(
    "seqlen, seg",
    [(4096, 1024), (8192, 1024), (4096, 512)],
    ids=["s4096_4seg", "s8192_8seg", "s4096_8seg"],
)
def test_seqpack_prune_predicate_is_sound(seqlen, seg):
    """The predicate must never prune a tile that contains an in-bounds (q, k) pair.

    Pure trace-time arithmetic, so this needs no device. It is necessary but NOT sufficient:
    the original defect passed this over 211,655 tiles and was still wrong on hardware,
    because the failure was in the *consequence* of skipping (stale ring-buffer data), not
    in the decision. Keep this test, but never treat it as the gate.
    """
    from nkilib.core.attention.attention_cte import _has_any_compute_bounds

    cu = _cu_from_spec(seqlen, seg)
    spans = [(cu[i], cu[i + 1]) for i in range(len(cu) - 1)]
    lo = np.zeros(seqlen, dtype=np.int64)
    hi = np.zeros(seqlen, dtype=np.int64)
    for s, e in spans:
        lo[s:e] = s
        hi[s:e] = e

    class _AC:
        segment_spans = spans
        kvp_group_size = 0
        cp_strided_q_slicing = False
        cp_striped_input = False
        seqlen_q = seqlen

    ac = _AC()
    unsound = []
    for tile_w in (512, 128):
        for g in range((seqlen + _Q_GRP_SZ - 1) // _Q_GRP_SZ):
            qs = g * _Q_GRP_SZ
            qe = min(qs + _Q_GRP_SZ, seqlen)
            for ks in range(0, seqlen, tile_w):
                ke = min(ks + tile_w, seqlen)
                live = bool(((lo[qs:qe] < ke) & (ks < hi[qs:qe])).any())
                if live and not _has_any_compute_bounds(g, ks, ke - ks, ac):
                    unsound.append((tile_w, g, ks))
    assert not unsound, (
        f"{len(unsound)} unsound prunes (tile has in-bounds pairs but predicate said skip); "
        f"first few: {unsound[:5]}"
    )


@pytest.mark.parametrize(
    "bs, seqlen, d, seg_spec, causal",
    [c[1:] for c in SEQPACK_PRUNE_MULTISECTION_CASES],
    ids=[c[0] for c in SEQPACK_PRUNE_MULTISECTION_CASES],
)
def test_seqpack_prune_multisection_output_neutral(bs, seqlen, d, seg_spec, causal):
    """Same contract as the single-section test, for seqlen > 8192.

    Kept separate from the single-section matrix because the failure modes are different: these
    exercise the cross-section flash accumulators, which single-section configs never touch.
    Measured MAC reduction and wall-clock at these lengths (LNC=2, d=128, non-causal):

        16384 (2 sections)  3.53x MACs -> 2.93x wall-clock
        32768 (4 sections)  7.07x MACs -> 5.95x wall-clock

    i.e. this is where the optimization actually pays, so a regression here matters more than
    one at 4096.
    """
    cu = _cu_from_spec(seqlen, seg_spec)
    q, k, v = _make_inputs(bs, seqlen, d)
    bmin, bmax = _bounds_from_cu(cu, seqlen, bs)
    base, _ = _run_on_device(q, k, v, bmin, bmax, causal, None, 2, f"msb{seqlen}")
    pruned, _ = _run_on_device(q, k, v, bmin, bmax, causal, cu, 2, f"msp{seqlen}")
    assert np.array_equal(base, pruned), (
        f"multi-section prune changed the output: "
        f"max|delta|={float(np.max(np.abs(base - pruned))):.3e}, cos={_cos(base, pruned):.6f}"
    )


# ---------------------------------------------------------------------------
# Striped context parallelism.
#
# This is the configuration that matters for reaching long global sequence lengths:
# attention_cte requires cp_striped_input=True whenever sequence packing is combined with CP.
#
# The coordinate contract, taken from the library's own bound builder
# (test/integration/nkilib/utils/sequence_packing_helpers.py::cu_seqlens_to_striped_bounds):
#
#     local_start = cu_seqlens[i]     // cp_degree
#     local_end   = cu_seqlens[i + 1] // cp_degree
#
# i.e. bounds are in LOCAL coordinates (global boundaries divided by cp_degree) and are
# IDENTICAL on every rank, provided every boundary is a multiple of cp_degree. That is what
# makes a single compile-time prune decision valid for one SPMD binary.
#
# These tests deliberately build bounds with that helper rather than by hand. Two earlier
# attempts at this failed because the test invented its own coordinate model and then agreed
# with itself -- see SEQPACK_PRUNE_STATUS.md.
# ---------------------------------------------------------------------------
SEQPACK_PRUNE_STRIPED_CASES = [
    # (id, global_seqlen, segment, cp_degree, causal)
    ("g32768_seg1024_D4_causal",   32768, 1024, 4, True),
    ("g32768_seg1024_D4_nocausal", 32768, 1024, 4, False),
    ("g32768_seg2048_D8_causal",   32768, 2048, 8, True),
    ("g65536_seg1024_D4_nocausal", 65536, 1024, 4, False),
    ("g65536_seg1024_D8_causal",   65536, 1024, 8, True),
    ("g16384_seg1024_D2_nocausal", 16384, 1024, 2, False),
]


def _run_striped(local_len, d, causal, cu, cp_deg, use_seg, lnc, tag):
    from nki.compiler.ncc_driver import CompileOptions, compile_bir_to_neff
    from nki.compiler.driver import compile_to_bir
    from nki.compiler.frontend import TracerFrontend
    # Relative import when run inside the test package; path fallback when this file is run
    # standalone (the full harness pulls in internal-only deps such as aws_embedded_metrics).
    try:
        from ....utils.sequence_packing_helpers import cu_seqlens_to_striped_bounds
    except ImportError:
        import os
        import sys

        _here = os.path.dirname(os.path.abspath(__file__))
        _utils = os.path.normpath(os.path.join(_here, "..", "..", "utils"))
        if _utils not in sys.path:
            sys.path.insert(0, _utils)
        from sequence_packing_helpers import cu_seqlens_to_striped_bounds

    bmin1, bmax1 = cu_seqlens_to_striped_bounds(np.asarray(cu), cu[-1], cp_deg)
    bmin = bmin1.reshape(1, local_len, 1).astype(np.float32)
    bmax = bmax1.reshape(1, local_len, 1).astype(np.float32)
    rng = np.random.default_rng(11)
    q = (rng.standard_normal((1, local_len, d)) * 0.5).astype(np.float32)
    k = (rng.standard_normal((1, d, local_len)) * 0.5).astype(np.float32)
    v = (rng.standard_normal((1, local_len, d)) * 0.5).astype(np.float32)
    inp = dict(q=q, k=k, v=v, bound_min=bmin, bound_max=bmax, scale=1.0, causal_mask=causal,
               tp_q=True, tp_k=False, tp_out=False,
               cp_offset=np.zeros((1, 1), dtype=np.int32),
               global_cp_deg=cp_deg, cp_striped_input=True)
    if use_seg:
        inp["segment_cu_seqlens"] = tuple(cu)
    opts = CompileOptions(target="trn2", lnc=lnc,
                          output_path=f"/tmp/striped_{tag}.neff",
                          artifacts_dir=f"/tmp/striped_{tag}_art")
    bir = compile_to_bir(attention_cte, frontend=TracerFrontend(), inputs=inp, compile_opts=opts)
    an = [x.name for x in bir.descriptor.input_specs]
    on = [x.name for x in bir.descriptor.output_specs]
    ck = compile_bir_to_neff(opts, bir,
                             [inp[n] for n in an if isinstance(inp.get(n), np.ndarray)], an, on)
    res = ck.run(**{n: inp[n] for n in an if isinstance(inp.get(n), np.ndarray)})
    outs = res.outputs
    arr = outs[0] if isinstance(outs, (list, tuple)) else (
        list(outs.values())[0] if isinstance(outs, dict) else outs)
    return np.asarray(arr, np.float32).reshape(local_len, d), int(bir.mac_count)


@pytest.mark.parametrize(
    "global_len, seg, cp_deg, causal",
    [c[1:] for c in SEQPACK_PRUNE_STRIPED_CASES],
    ids=[c[0] for c in SEQPACK_PRUNE_STRIPED_CASES],
)
def test_seqpack_prune_striped_cp_output_neutral(global_len, seg, cp_deg, causal):
    """Under striped CP, the descriptor must not change the output.

    Measured MAC reductions at these shapes: 1.65x-3.74x, all bit-identical.
    """
    local_len = global_len // cp_deg
    cu = tuple(range(0, global_len + 1, seg))
    base, mb = _run_striped(local_len, 128, causal, cu, cp_deg, False, 2, f"b{global_len}_{cp_deg}")
    pruned, mp = _run_striped(local_len, 128, causal, cu, cp_deg, True, 2, f"p{global_len}_{cp_deg}")

    assert mb > mp, (
        f"descriptor did not reduce MACs under striped CP ({mb:,} -> {mp:,}); "
        f"the prune may be silently inert"
    )
    if np.array_equal(base, pruned):
        return
    groups = _per_group_max_err(base, pruned, local_len)
    bad = [(g, e) for g, e in groups if e > 0.0]
    pytest.fail(
        f"striped-CP prune changed the output (must be bit-identical).\n"
        f"  global={global_len} seg={seg} cp_deg={cp_deg} causal={causal} local={local_len}\n"
        f"  MACs {mb:,} -> {mp:,}\n"
        f"  max|delta| = {float(np.max(np.abs(base - pruned))):.6e}\n"
        f"  cos = {_cos(base, pruned):.8f}\n"
        f"  {len(bad)}/{len(groups)} Q groups differ: "
        + ", ".join(f"grp{g}:{e:.2e}" for g, e in bad[:10])
    )


def test_seqpack_prune_striped_guard_requires_divisible_boundaries():
    """Boundaries not divisible by cp_degree must disable pruning, not mis-prune.

    That precondition is what makes the local layout identical on every rank; without it no
    rank-independent compile-time decision exists.
    """
    from nkilib.core.attention.attention_cte import _striped_prune_is_safe

    class _AC:
        cp_striped_input = True
        global_cp_deg = 8
        segment_spans = None

    ac = _AC()
    ac.segment_spans = [(0, 1024), (1024, 2048)]
    assert _striped_prune_is_safe(ac) is True, "divisible boundaries should permit pruning"
    # 1004 % 8 == 4, so the local layout would differ per rank and no rank-independent
    # compile-time decision exists. (Note 1000 % 8 == 0 -- picking a genuinely indivisible
    # boundary matters for this test to mean anything.)
    ac.segment_spans = [(0, 1004), (1004, 2048)]
    assert _striped_prune_is_safe(ac) is False, "non-divisible boundary must disable pruning"
    ac.global_cp_deg = None
    assert _striped_prune_is_safe(ac) is False, "unknown degree must disable pruning"
