#
# Copyright Axelera AI, 2026 (structure follows ops/relu2_quantize.py and ops/pointwise_lut.py)
# SPDX-License-Identifier: Apache-2.0
#
"""Fused xIELU + per-channel activation quantize (llm.xielu_quantize).

    r = relu(x),  m = x - r = min(x, 0)
    y = alpha_p * r**2 + beta * x + alpha_n * lut(m)          # lut(m) ~ expm1(m) - m
    out[i, j] = round_clip_int8(y * scale[j] + zero_point[j])

Apertus's MLP middle section in ONE DPU datapath, as ``llm.relu2_quantize`` is for
Domyn: between ``up_proj`` and ``down_proj`` sit the activation and the down-projection's
input quantize, and the intermediate is the widest tensor in the layer (21504 channels).

Why the split form: the quadratic and linear terms are computed exactly (DPU multiplies),
so they need no calibrated domain and the positive side can never saturate a table. Only
the bounded convex term ``g(m) = expm1(m) - m`` goes through a chained additive LUT, fitted
once, independent of calibration and layer (``models/apertus/xielu_lut.py``). The table's
bin 0 holds the asymptote ``-1 - m``, which the hardware extends below the first delimiter,
and the last bin passes through 0 at ``m = 0``.

Layout matches llm.quantize / llm.relu2_quantize: ``x`` is ``(N, C_blocks)`` (a free view
of the up_proj output, fp16), ``qparams`` the PRE-INTERLEAVED ``(2*C_blocks,)`` [scale, zp,
...] buffer. x streams on IFD0, qparams on IFD1. Two small extra operands are staged in L1
and read as broadcast scalars inside the DPU block (the way pointwise_lut reads its table):

* ``coef`` ``(3,)``: one PWORD each holding ``alpha_p``, ``alpha_n``, ``beta`` broadcast
  (per-layer constants, folded through softplus at build time);
* ``lut`` ``(n_luts,)``: the chained table, one PWORD per table.

DPU cost: ~11 operations per element against relu2_quantize's ~5 (prelu, sub, n_luts LUTs
+ (n_luts-1) adds, 2 muls for alpha_p*r*r, mul+add for beta*x, mul+add for alpha_n*g, mul+add
for the quantize). Status: ``emulate()`` matches the float formula within one int8 step, and
``implementation()`` matches ``emulate()`` on a Metis device, up to the full 21504 width.
"""
from axelera.compiler.alto.axol import atomic
from axelera.compiler.alto.axol.common import ramp
from axelera.compiler.alto.compiler.config import HardwareGeneration
from axelera.compiler.alto.compiler.device import PWORD
from axelera.compiler.alto.compiler.dtypes import MetisFloats
from axelera.compiler.alto.compiler.interpreter import emulation as emu
from axelera.compiler.alto.compiler.op_registry import Arg, BufferInfo, ScriptOp, register_op
import axelera.compiler.alto.script as A
import axelera.compiler.backend.tir.ops as tir_ops
import numpy as np
from tvm import tir

# mypy: disable-error-code="attr-defined, index, arg-type"

#: Row order of the ``coef`` operand.
COEF_ALPHA_P, COEF_ALPHA_N, COEF_BETA = 0, 1, 2
COEF_ROWS = 3


class XieluQuantize(ScriptOp):
    """out = round_clip_int8(xielu(x) * scale + zero_point), per-channel. See module docstring."""

    x = Arg(MetisFloats[tir.Buffer, "..."], description="(N, C_blocks) activation, fp16/fp32.")
    qparams = Arg(
        MetisFloats[tir.Buffer, "..."],
        description="(2*C_blocks,) interleaved per-channel [scale[0], zp[0], scale[1], zp[1], ...].",
    )
    coef = Arg(
        MetisFloats[tir.Buffer, "..."],
        description="(3,) broadcast PWORDs [alpha_p, alpha_n, beta] (effective, post-softplus).",
    )
    lut = Arg(
        MetisFloats[tir.Buffer, "..."],
        description="(n_luts,) chained additive LUT for g(m) = expm1(m) - m over m <= 0.",
    )
    out = Arg(MetisFloats[tir.Buffer, "..."] | BufferInfo, default=None, is_output=True, kw=True)
    generation = Arg(HardwareGeneration | None, default=None, kw=True)

    def _n(self) -> int:
        return int(self.x.shape[0])

    def _c(self) -> int:
        return int(self.x.shape[1])

    def _n_luts(self) -> int:
        return int(self.lut.shape[0])

    def validate(self) -> None:
        c = self._c()
        n_qp = int(np.prod([int(s) for s in self.qparams.shape]))
        if n_qp != 2 * c:
            raise ValueError(
                f"qparams must have 2*{c} = {2 * c} PWORD blocks (interleaved scale,zp per "
                f"channel block of x), got shape {tuple(int(s) for s in self.qparams.shape)}."
            )
        n_coef = int(np.prod([int(s) for s in self.coef.shape]))
        if n_coef != COEF_ROWS:
            raise ValueError(f"coef must be {COEF_ROWS} PWORDs [alpha_p, alpha_n, beta]; got {n_coef}")
        if self._n_luts() < 1:
            raise ValueError("lut must hold at least one table")

    def infer_outputs(self) -> dict[str, BufferInfo]:
        return {"out": BufferInfo(shape=(self._n(), self._c()), dtype="int8x64", scope=A.DDR)}

    def implementation(self) -> None:
        n, c, n_luts = self._n(), self._c(), self._n_luts()
        f32, f18 = "float32x64", "float18x64"
        xdt, sdt = self.x.dtype, self.qparams.dtype
        c_pe = self.x.shape[1]

        # Tile-invariant operands, staged once: the interleaved qparams (streamed on IFD1,
        # as relu2_quantize), the three coefficients and the table rows (read as broadcast
        # scalars inside the DPU block, as pointwise_lut reads its table and post scalar).
        params_l1 = atomic.copy(self.qparams, scope=A.L1)
        coef = A.alloc_buffer((COEF_ROWS,), f32, scope=A.L1)
        lut = A.alloc_buffer((n_luts,), f32, scope=A.L1)
        atomic.copy(self.coef, output=coef)
        atomic.copy(self.lut, output=lut)
        zero_const = A.alloc_constant(
            np.zeros((1, PWORD), dtype=np.float32), dtype="float32x64", scope=A.DDR
        )

        xin = A.alloc_buffer((n, c), xdt, scope=A.L1)
        oout = A.alloc_buffer((n, c), "int8x64", scope=A.L1)

        with A.tiles(n) as t:
            xt = xin.tile(t, dim=0)
            ot = oout.tile(t, dim=0)
            atomic.copy(self.x[t, ramp(c_pe)], output=xt)
            rows = t.lanes
            with A.script():
                # One Datapath PER ROW, as llm.quantize / relu2_quantize: each invocation
                # streams only this row's c values + the 2c interleaved params, so the IFD
                # FIFO never accumulates a whole tile.
                for i in A.serial(rows):
                    with A.Datapath(core=None):
                        fx = A.decl_stream(xdt)
                        fp = A.decl_stream(sdt)
                        o = A.decl_stream("int8x64")
                        for j in A.serial(c):
                            with A.IFD(num=0):
                                fx.push(xt[i, j])
                        for k in A.serial(2 * c):
                            with A.IFD(num=1):
                                fp.push(params_l1[k])
                        with A.DPU():
                            xr = A.var(f18)
                            sr = A.var(f18)
                            zr = A.var(f18)
                            r = A.var(f18)
                            m = A.var(f18)
                            gm = A.var(f18)
                            la = A.var(f18)
                            y = A.var(f18)
                            u = A.var(f18)
                            for j in A.serial(c):
                                xr = A.cast(fx.pop(), f18)
                                sr = A.cast(fp.pop(), f18)
                                zr = A.cast(fp.pop(), f18)
                                r = A.dpu.ops.prelu(xr, A.cast(zero_const[0], f18))  # relu
                                m = xr - r  # min(x, 0)
                                gm = tir_ops.lut(m, A.cast(lut[0], f18))
                                for k in A.serial(1, n_luts):
                                    la = tir_ops.lut(m, A.cast(lut[k], f18))
                                    gm = gm + la
                                y = r * r
                                y = y * A.cast(coef[COEF_ALPHA_P], f18)
                                u = xr * A.cast(coef[COEF_BETA], f18)
                                y = y + u
                                u = gm * A.cast(coef[COEF_ALPHA_N], f18)
                                y = y + u
                                y = y * sr
                                y = y + zr
                                o.push(A.cast(y, "int8x64"))  # round + clip to [-128, 127]
                        for j in A.serial(c):
                            with A.ODR():
                                ot[i, j] = o.pop()
            atomic.copy(ot, output=self.out[t, ramp(c_pe)])

    def emulate(self) -> None:  # type: ignore[override]
        """Bit-level model of the datapath above: every DPU result rounded to float18."""
        n, c, n_luts = self._n(), self._c(), self._n_luts()
        x = emu.rf18(self.x.reshape(n, c, PWORD).astype(np.float32))
        qp = self.qparams.reshape(2 * c, PWORD).astype(np.float32)
        s = emu.rf18(qp[0::2])[None]
        z = emu.rf18(qp[1::2])[None]
        coef = self.coef.reshape(COEF_ROWS, PWORD).astype(np.float32)
        alpha_p = emu.rf18(coef[COEF_ALPHA_P])
        alpha_n = emu.rf18(coef[COEF_ALPHA_N])
        beta = emu.rf18(coef[COEF_BETA])
        lut = self.lut.reshape(n_luts, PWORD).astype(np.float32)

        r = np.maximum(x, np.float32(0.0))
        m = emu.rf18(x - r)
        flat = m.reshape(1, -1)
        gm = emu.apply_lut(flat, lut[0:1])
        for k in range(1, n_luts):
            gm = emu.rf18(gm + emu.apply_lut(flat, lut[k : k + 1]))
        gm = gm.reshape(n, c, PWORD)
        y = emu.rf18(emu.rf18(r * r) * alpha_p)
        y = emu.rf18(y + emu.rf18(x * beta))
        y = emu.rf18(y + emu.rf18(gm * alpha_n))
        q = emu.rf18(emu.rf18(y * s) + z)
        np.copyto(self.out, np.clip(np.round(q), -128, 127).astype(np.int8))


xielu_quantize = register_op(
    XieluQuantize,
    "llm.xielu_quantize",
    generation=[HardwareGeneration.OMEGA, HardwareGeneration.EUROPA],
    atomic=False,
)
