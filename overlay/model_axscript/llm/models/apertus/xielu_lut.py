# Copyright Axelera AI, 2026 (model_axscript conventions this file follows)
# SPDX-License-Identifier: Apache-2.0
"""The xIELU lookup table: a chained piecewise-linear fit of ``g(m) = expm1(m) - m``.

Why only ``g`` goes through a table
-----------------------------------
xIELU (Apertus's MLP activation) is, per element,

    x > 0:  alpha_p * x**2 + beta * x
    x <= 0: alpha_n * (expm1(min(x, eps)) - x) + beta * x

With ``r = relu(x)`` and ``m = min(x, 0) = x - r`` this is, for every x,

    xielu(x) = alpha_p * r**2 + beta * x + alpha_n * g(m),    g(m) = expm1(m) - m

(for x > 0, m = 0 and g(0) = 0; for x <= 0, r = 0). The quadratic and linear terms are
computed exactly on the DPU, so they need no calibrated domain and the positive side can
never clip inside the activation. Only ``g`` -- bounded, convex, ``g(0) = 0``, approaching
the line ``-1 - m`` as ``m -> -inf`` (the gap is ``e**m``) -- goes through a table. The table
does not depend on calibration or on the layer: one table serves the whole model. The
``eps`` clamp (``-1e-6`` in the checkpoint) changes the result by ~``alpha_n * 1e-6`` near
zero, far below the int8 step that follows.

Hardware table model
--------------------
One table is one PWORD: 16 delimiters, 16 slopes, 16 offsets, 16 zeros. Delimiter ``i`` is
the EXCLUSIVE upper edge of bin ``i``: bin ``i`` holds ``delims[i-1] <= x < delims[i]``, so an
input equal to a delimiter belongs to the next bin. Bin 0's segment applies to every input
below ``delims[0]``, and the output is 0 at and above ``delims[15]``. Chained tables are
ADDITIVE. The SDK documents none of this (only comments in ``model_axscript/llm/luts.py``
hint at it); it was measured against the compiler's own evaluator
(``compute_lut_numpy_single_batch``), which this module's model reproduces bit for bit, and
the op that uses the table has been checked on a Metis device.

Layout of an ``n``-table chain over ``[lo, 0]``:

* table 0, bin 0: the asymptote ``-1 - m`` (error-centred), covering ``(-inf, lo]``;
* table k > 0, bin 0: identically zero, covering everything up to table k-1's last edge
  (where table k-1 hands over by outputting 0);
* every other bin: a secant of ``g`` with its offset centred, so the error is +-half the
  secant error;
* the last bin of the last table ends exactly at 0 and is NOT centred: its secant passes
  through ``(0, 0)``. The hardware already outputs 0 at ``m = 0`` (the last edge is
  exclusive); the line through the origin keeps ``g(0) = 0`` even if a device disagrees.

That is ``15 * n`` secant bins. Breakpoints are placed greedily from 0 leftwards for a
common error target and the target is bisected until the bins exactly reach ``lo``, where
the asymptote's own error ``e**lo / 2`` equals the target. ``lo`` is therefore an output of
the fit, not a calibrated input.

float18
-------
Table parameters are stored and evaluated in float18. When the compiler wheel is importable
its own rounding (``np_round_float32_to_float18``) and packer (``pack_lut_params_to_pword``)
are used. Otherwise this module falls back to a model with float32's exponent range and
``FLOAT18_MANTISSA_BITS`` mantissa bits -- an ASSUMPTION, since float18's layout is not
documented.
"""

from __future__ import annotations

from dataclasses import dataclass
import functools
import math

import numpy as np

PWORD = 64
#: Bins per hardware table (one PWORD holds 16 delimiters, 16 slopes, 16 offsets).
BINS = 16
#: float18 mantissa width for the fallback rounding. Measured against the compiler's
#: np_round_float32_to_float18: 1 sign, 7 exponent, 10 mantissa bits (largest value 2**64).
#: The fallback models the mantissa only, not the narrower exponent range.
FLOAT18_MANTISSA_BITS = 10
#: Tables in the chain the kernel declares (``xielu_lut`` is ``(XIELU_LUT_ROWS,)``).
XIELU_LUT_ROWS = 2
#: Name of the RangeObserver site on the up_proj output (diagnostics only: the table is
#: fixed). Kept out of ``luts.generate_lut_table``'s closed function set.
XIELU_RANGE_FUNCTION = "xielu"


def g(m):
    """``expm1(m) - m``: the only part of xIELU that needs a table (float64)."""
    m = np.asarray(m, dtype=np.float64)
    return np.expm1(m) - m


def _secant(a: float, b: float) -> tuple[float, float]:
    """Slope and offset of the chord of ``g`` over ``[a, b]``."""
    ga, gb = float(g(a)), float(g(b))
    slope = (gb - ga) / (b - a)
    return slope, ga - slope * a


def _secant_error(a: float, b: float) -> float:
    """Largest ``chord - g`` over ``[a, b]`` (g is convex, so the chord lies above it).

    The maximum is where ``g'(x) = e**x - 1`` equals the chord slope ``s``, i.e. at
    ``x = log1p(s)`` (``s > -1`` always, since ``g' > -1``).
    """
    s, c = _secant(a, b)
    x = math.log1p(s)
    x = min(max(x, a), b)
    return s * x + c - float(g(x))


def _widest_bin(right: float, max_err: float, floor: float) -> float:
    """The leftmost ``a >= floor`` whose chord over ``[a, right]`` errs by at most ``max_err``."""
    if _secant_error(floor, right) <= max_err:
        return floor
    lo, hi = floor, right
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _secant_error(mid, right) <= max_err:
            hi = mid
        else:
            lo = mid
    return hi


def _place_breakpoints(target: float, n_secants: int, floor: float = -40.0):
    """Edges ``[lo, ..., 0]`` for an error target, or None if ``n_secants`` cannot reach.

    ``lo`` is where the centred asymptote's error ``e**lo / 2`` meets the target. The last
    bin (ending at 0) is uncentred, so its full chord error must meet the target; every
    other bin is centred, so twice the target is allowed.
    """
    lo = max(math.log(2.0 * target), floor)
    edges = [0.0]
    right = 0.0
    for k in range(n_secants):
        allowed = target if k == 0 else 2.0 * target
        left = _widest_bin(right, allowed, lo)
        edges.append(left)
        right = left
        if left <= lo:
            return list(reversed(edges))
    return None


@dataclass(frozen=True)
class LutRow:
    """One hardware table: 16 upper bin edges, 16 slopes, 16 offsets (float32, f18-rounded)."""

    delims: np.ndarray
    slopes: np.ndarray
    offsets: np.ndarray


@dataclass(frozen=True)
class XieluLut:
    """The fitted chain plus the numbers a report or a test wants."""

    rows: tuple[LutRow, ...]
    lo: float  # left end of the secant domain; below it the asymptote bin applies
    target_error: float  # the per-bin error the placement was solved for (before rounding)

    def table(self) -> np.ndarray:
        """The packed ``(n_luts, 64)`` float32 table the kernel's ``xielu_lut`` buffer takes."""
        return np.stack([_pack_row(r) for r in self.rows]).astype(np.float32)


def round_float18(x) -> np.ndarray:
    """Round to float18: the compiler's own rounding when present, else the assumed model."""
    x = np.asarray(x, dtype=np.float32)
    try:
        from tvm.relay.backend.contrib.axelera.utils.float18 import (
            np_round_float32_to_float18,
        )
    except ImportError:
        return _round_float18_model(x).reshape(x.shape)
    return np.asarray(np_round_float32_to_float18(x), dtype=np.float32).reshape(x.shape)


def _round_float18_model(x: np.ndarray, mantissa_bits: int = FLOAT18_MANTISSA_BITS) -> np.ndarray:
    """Round-to-nearest-even onto float32's exponent range with ``mantissa_bits`` mantissa bits.

    Implemented on the float32 bit pattern: drop ``23 - mantissa_bits`` low mantissa bits
    with round-half-to-even. Infinities and NaNs pass through.
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    drop = 23 - mantissa_bits
    bits = x.view(np.uint32).astype(np.uint64)
    half = np.uint64(1 << (drop - 1))
    lsb = (bits >> np.uint64(drop)) & np.uint64(1)
    rounded = (bits + half - np.uint64(1) + lsb) & ~np.uint64((1 << drop) - 1)
    out = rounded.astype(np.uint32).view(np.float32)
    return np.where(np.isfinite(x), out, x).astype(np.float32)


def _pack_row(row: LutRow) -> np.ndarray:
    """One PWORD: the compiler's packer when present, else ``[delims, slopes, offsets, 0]``."""
    try:
        from axelera.compiler.alto.axol.utils.lut_chain import pack_lut_params_to_pword
        from tvm.relay.backend.contrib.axelera.utils.lut import LUTParams
    except ImportError:
        return np.concatenate(
            [row.delims, row.slopes, row.offsets, np.zeros(BINS, np.float32)]
        ).astype(np.float32)
    return np.asarray(
        pack_lut_params_to_pword(LUTParams(row.delims, row.slopes, row.offsets)),
        dtype=np.float32,
    )


def _make_rows(edges: list[float], n_luts: int) -> tuple[LutRow, ...]:
    """Lay ``edges`` ([lo, ..., 0], 15*n_luts + 1 of them) out as an additive chain."""
    n_sec = len(edges) - 1
    per_table = BINS - 1
    if n_sec != per_table * n_luts:
        raise ValueError(f"expected {per_table * n_luts} secant bins, got {n_sec}")
    # Round the edges first and fit each chord on the ROUNDED edges, so the stored segments
    # stay continuous at the breakpoints the hardware actually compares against.
    edges_r = round_float18(np.asarray(edges, dtype=np.float32)).astype(np.float64)
    edges_r[-1] = 0.0
    if np.any(np.diff(edges_r) <= 0):
        raise ValueError("float18 rounding collapsed two breakpoints; use fewer bins")
    lo = float(edges_r[0])

    rows = []
    for t in range(n_luts):
        delims = np.empty(BINS, np.float64)
        slopes = np.empty(BINS, np.float64)
        offsets = np.empty(BINS, np.float64)
        first = t * per_table  # index into edges_r of this table's first secant's left edge
        delims[0] = edges_r[first]
        if t == 0:
            # Centred asymptote: g(m) - (-1 - m) = e**m in (0, e**lo] below lo.
            slopes[0] = -1.0
            offsets[0] = -1.0 + 0.5 * math.exp(lo)
        else:
            slopes[0] = 0.0
            offsets[0] = 0.0
        for b in range(1, BINS):
            a, c = float(edges_r[first + b - 1]), float(edges_r[first + b])
            s, _ = _secant(a, c)
            # Rounding-aware: round the slope FIRST, then centre the offset on the residual
            # of the rounded line. At 9-10 mantissa bits the slope's own rounding error is
            # comparable to the chord error, so centring the unrounded chord leaves it biased.
            s = float(round_float18(np.float32(s)))
            last_overall = t == n_luts - 1 and b == BINS - 1
            if last_overall:
                o = 0.0  # pass through (0, 0): g(0) = 0 whichever bin m = 0 falls in
            else:
                grid = np.linspace(a, c, 257)
                resid = g(grid) - s * grid
                o = 0.5 * (float(resid.max()) + float(resid.min()))
            delims[b] = c
            slopes[b] = s
            offsets[b] = o
        rows.append(
            LutRow(
                delims=round_float18(delims),
                slopes=round_float18(slopes),
                offsets=round_float18(offsets),
            )
        )
    return tuple(rows)


@functools.lru_cache(maxsize=8)
def fit_xielu_lut(n_luts: int = XIELU_LUT_ROWS) -> XieluLut:
    """Fit the ``n_luts``-table chain for ``g`` (minimax over the whole real half-line)."""
    if n_luts < 1:
        raise ValueError("n_luts must be >= 1")
    n_sec = (BINS - 1) * n_luts
    lo_t, hi_t = 1e-9, 0.5
    for _ in range(100):
        mid = math.sqrt(lo_t * hi_t)
        if _place_breakpoints(mid, n_sec) is None:
            lo_t = mid
        else:
            hi_t = mid
    target = hi_t
    edges = _place_breakpoints(target, n_sec)
    # The greedy pass may finish with bins to spare; split the widest bins until every one
    # of the n_sec secants is used (finer bins only lower the error).
    while len(edges) - 1 < n_sec:
        widths = np.diff(edges)
        k = int(np.argmax(widths))
        edges.insert(k + 1, 0.5 * (edges[k] + edges[k + 1]))
    rows = _make_rows(edges, n_luts)
    # lo as stored (float18-rounded), which is where the hardware switches to the asymptote.
    return XieluLut(rows=rows, lo=float(rows[0].delims[0]), target_error=target)


def xielu_lut_table(n_luts: int = XIELU_LUT_ROWS) -> np.ndarray:
    """The packed ``(n_luts, 64)`` table for the kernel's ``xielu_lut`` buffer."""
    return fit_xielu_lut(n_luts).table()


def unpack_table(table: np.ndarray) -> list[LutRow]:
    """Inverse of the fallback packer: ``(n, 64)`` -> rows."""
    table = np.asarray(table, dtype=np.float32).reshape(-1, PWORD)
    return [LutRow(r[0:16], r[16:32], r[32:48]) for r in table]


def eval_row_model(x: np.ndarray, row: LutRow) -> np.ndarray:
    """This module's model of one hardware table (see the module docstring).

    Bin ``i`` holds ``delims[i-1] <= x < delims[i]``; bin 0 extends to -inf; 0 at and above
    the last edge. The input is float18-rounded, then ``slope * x`` is rounded to float18 and
    the sum with the offset rounded again -- the order that reproduces the compiler's
    ``compute_lut_numpy_single_batch`` bit for bit (tested).
    """
    xr = round_float18(np.asarray(x, dtype=np.float32))
    idx = np.searchsorted(row.delims, xr, side="right")
    inside = idx < BINS
    idc = np.minimum(idx, BINS - 1)
    y = round_float18(round_float18(row.slopes[idc] * xr) + row.offsets[idc])
    return np.where(inside, y, np.float32(0.0)).astype(np.float32)


def eval_chain_model(x: np.ndarray, table: np.ndarray) -> np.ndarray:
    """The chained table on ``x`` under this module's hardware model (tables summed in float32,
    as the SDK evaluator does; the DPU op rounds the sum, see ops/xielu_quantize.py)."""
    acc = None
    for row in unpack_table(table):
        y = eval_row_model(x, row)
        acc = y if acc is None else (acc + y).astype(np.float32)
    return acc


def eval_chain_sdk(x: np.ndarray, table: np.ndarray) -> np.ndarray:
    """The chained table on ``x`` under the SDK's own evaluator (needs the compiler wheel).

    Same primitive ``luts.apply_chained_lut`` uses, so this is the device model the rest of
    the toolkit trusts.
    """
    from tvm.relay.backend.contrib.axelera.utils.lut import compute_lut_numpy_single_batch

    xn = np.asarray(x, dtype=np.float32)
    out = np.zeros_like(xn, dtype=np.float32)
    for row in np.asarray(table, dtype=np.float32).reshape(-1, PWORD):
        out += compute_lut_numpy_single_batch(xn, row[0:16], row[16:32], row[32:48], True)
    return out


def apply_xielu_lut_torch(m, table: np.ndarray | None = None):
    """Torch-side model of the device's ``g`` table, for the fake-quant reference.

    ``luts.apply_chained_lut`` only knows the SDK's built-in functions, so this
    is its xIELU counterpart: the SDK evaluator when importable, else this module's model.
    """
    import torch

    tbl = xielu_lut_table() if table is None else table
    mn = m.detach().to(torch.float32).cpu().numpy()
    try:
        out = eval_chain_sdk(mn, tbl)
    except ImportError:
        out = eval_chain_model(mn, tbl)
    return torch.from_numpy(np.asarray(out, dtype=np.float32)).to(device=m.device, dtype=m.dtype)
