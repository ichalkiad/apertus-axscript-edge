# Copyright Axelera AI, 2026 (kernel body follows models/domyn/metis_impl.py, the QK-norm
#   path models/qwen3/multicore_impl.py, the backbone is models/multicore_layer.py)
# SPDX-License-Identifier: Apache-2.0
"""Apertus 1.5 Metis implementation, registered as ``apertus-metis``.

The device half of the port: the alto-script decoder-layer kernel, its per-layer constant
binding, and the host-side seams that differ from the shared multicore backbone
(``models.multicore_layer``). The host contract is the production one -- a single
precombined RoPE trig (sin||cos), an int8 ``{0, -127}`` 1-head broadcast causal mask, and
``seq_pos`` -- and ``cores > 1`` head-shards SDPA and N-shards the two MLP matmuls.

Where each part comes from:

* **Attention projections / SDPA / KV cache**: the Llama-3.1-8B shape (32 q / 8 kv heads,
  head_dim 128, separate q/k/v matmuls) with Domyn's op order (the K chain authored before
  the V projection so k_heads is freed before V stages its tiles).
* **QK-Norm**: Qwen3's two-arm ``_norm_rope_quant``: one fused ``rms_norm_dpu_reduce`` (norm
  weight + RoPE + quantize) at decode, norm -> ``axol.core.rope`` -> ``llm.quantize`` at
  prefill, whose trig would overflow L1 in the fused form.
* **Hidden RMSNorms**: ``hidden_norm_quant`` with Apertus's OWN folds (``attention_layernorm``
  -> ``q_proj_qparams``, ``feedforward_layernorm`` -> ``up_proj_qparams``): the shared
  ``POST_ATTN_NORM_FOLD`` names a SwiGLU gate. At hidden 4096 the default
  spec takes the single-pass dpu_reduce arm.
* **MLP**: Domyn's un-gated structure (up_proj fp16 out -> fused activation + quantize ->
  down_proj) with ``llm.xielu_quantize`` in place of ``llm.relu2_quantize``, run in TWO
  halves of the intermediate dim (``MLP_PARTS``): down_proj's K of 21504 exceeds what one
  matmul can hold in IMC (K <= 16384 at int8), so up_proj is split along N and
  down_proj along K, and the halves' outputs are added. Two extra per-layer operands:
  ``xielu_coef`` (alpha_p, alpha_n, beta) and ``xielu_lut``.
* **Host RoPE**: built from the exported ``rope_inv_freq`` (llama3, theta 4e6, factor 32),
  as Domyn's ``_trig`` does -- NOT Llama's ``_rope_tables``, which hard-code Llama 3.2's
  scaling.
* **Head**: the shared device head (``lm_head_kernel``), built at ``head_vocab_size`` rows
  (131072, the checkpoint's pruned head) rather than ``vocab_size`` (266752, the embedding),
  which the shared kernel assumes are equal. ``has_lm_head=False`` is what
  turns the device head ON. The host head stays available on the same
  artefact through ``force_host_head`` (``lm_head_weight`` is in host_params).

Body: Llama/Domyn's segment-per-op schedule, plus Qwen3's free decode-time reshape in place
of the head transpose. Not yet carried over (performance follow-ups): the K-split weight
shards (``<name>1``, streamed on Metis's two DMA channels), ``pipeline_weights`` at decode,
and a fused q||k||v projection.
"""

# flake8: noqa: F821  -- A.Buffer dtype-string annotations look like F821 names

import dataclasses
from functools import partial
from typing import TYPE_CHECKING

import numpy as np

from ...implementation import register_implementation
from ..multicore_layer import (
    PWORD,
    ModelSpec,
    MulticoreDecoderImplementation,
    NormFold,
    add_norm_consts,
    check_norm_folds,
    hidden_norm_quant,
    interleave_qparams,
    layer_dims,
    quantize_activation,
)

if TYPE_CHECKING:
    from .apertus import ApertusConfig

#: has_qk_norm=True (per-head q/k RMSNorm); has_lm_head=False, which -- despite the name --
#: means "no HOST lm_head kernel", i.e. this implementation provides the DEVICE head
#: (``provides_device_head = not has_lm_head``). Default norm arm: at hidden
#: 4096 (64 D-blocks) the single-pass rms_norm_dpu_reduce, as Llama-3.1-8B and Qwen3-8B.
APERTUS_SPEC = ModelSpec(has_qk_norm=True, has_lm_head=False)

#: Hidden-norm folds: each norm's affine weight is folded into the activation-quantize scale
#: of the projection it feeds. Declared here because the shared defaults name SwiGLU paths.
ATTN_NORM_FOLD = NormFold("q_proj_qparams", "attention_layernorm")
FFN_NORM_FOLD = NormFold("up_proj_qparams", "feedforward_layernorm")
APERTUS_NORM_FOLDS = (ATTN_NORM_FOLD, FFN_NORM_FOLD)

#: Rows of the per-layer xIELU operands (see ops/xielu_quantize.py).
XIELU_COEF_ROWS = 3
XIELU_LUT_ROWS = 2

#: The MLP runs in this many parts of the intermediate dim ("lo", "hi"): see mlp_part_dims.
MLP_PARTS = ("lo", "hi")
#: K-blocks of an int8 weight stripe that fit one core's IMC (1 MiB / 4 KiB per 64x64 tile).
#: quantized_matmul_activation allocates its IMC operand over the FULL K (both DDR shards
#: included), so a matmul whose K exceeds this cannot be built.
IMC_MAX_K_BLOCKS = 256


def mlp_part_dims(intermediate_size: int) -> tuple[int, int]:
    """(channels, PWORD blocks) of one MLP part, checked against the IMC limit."""
    blocks = intermediate_size // PWORD
    if intermediate_size % PWORD or blocks % len(MLP_PARTS):
        raise ValueError(
            f"intermediate_size {intermediate_size} must split into {len(MLP_PARTS)} whole "
            f"PWORD-block parts"
        )
    part_pw = blocks // len(MLP_PARTS)
    if part_pw > IMC_MAX_K_BLOCKS:
        raise ValueError(
            f"an MLP part of {part_pw} K-blocks exceeds the {IMC_MAX_K_BLOCKS} an int8 weight "
            f"stripe may hold in IMC; split the MLP into more parts"
        )
    return part_pw * PWORD, part_pw


def make_apertus_layer_kernel(
    config: "ApertusConfig",
    ctx_len: int,
    phase: str,
    chunk_len: int | None = None,
    cores: int = 1,
):
    """Build the ApertusLayerFunctional IRModule for one Apertus decoder layer.

    ``ctx_len`` sizes the KV cache / mask; ``phase`` is "decode" (seq_len 1) or "prefill"
    (seq_len = chunk_len or ctx_len). Chunked prefill builds the query side for chunk_len
    positions while the KV cache stays full ctx_len; the host threads it via ``seq_pos``.
    """
    from axelera.compiler.alto import axol
    from axelera.compiler.alto import script as A
    from axelera.compiler.alto.axol.common import BufferInfo
    from axelera.compiler.alto.axol.utils.lut_chain import ChainedLUTMethod

    from ...ops.quantized_matmul_activation import (
        quantized_matmul_activation,
        quantized_matmul_activation_multicore,
    )
    from ...ops.rms_norm_dpu_reduce import rms_norm_dpu_reduce
    from ...ops.xielu_quantize import xielu_quantize

    if config.head_dim % (2 * PWORD):
        raise ValueError(
            f"head_dim must be a multiple of {2 * PWORD} (RoPE pairs whole PWORDs); "
            f"got {config.head_dim}"
        )
    d = layer_dims(
        config,
        head_dim_pw=config.head_dim // PWORD,
        logical_head_dim=config.head_dim,
        ctx_len=ctx_len,
        phase=phase,
        chunk_len=chunk_len,
        spec=APERTUS_SPEC,
    )
    seq_len, hidden_dim, hidden_dim_pw = d.seq_len, d.hidden_dim, d.hidden_dim_pw
    head_dim, head_dim_pw = config.head_dim, d.head_dim_pw
    num_q_heads, num_kv_heads, context_len = d.num_q_heads, d.num_kv_heads, d.context_len
    q_out_dim, q_out_dim_pw, kv_out_dim_pw = d.q_out_dim, d.q_out_dim_pw, d.kv_out_dim_pw
    part_dim, part_pw = mlp_part_dims(config.intermediate_size)

    # Op + cores kwarg selected at trace time so the body has one unconditional call per op.
    _multicore = bool(cores and cores > 1)
    _sdpa_cores_kw = {"cores": cores} if _multicore else {}
    _mlp_matmul = (
        quantized_matmul_activation_multicore if _multicore else quantized_matmul_activation
    )
    _mlp_cores_kw = {"cores": cores} if _multicore else {}

    def _to_heads(x3, heads):
        """(1, seq, heads*D) -> (1, heads, seq, D). At decode (seq 1) a free reshape: swapping
        a length-1 axis moves no data, where a transpose still costs its own segment."""
        x4 = axol.atomic.reshape(x3, shape=(1, seq_len, heads, head_dim_pw))
        if seq_len == 1:
            return axol.atomic.reshape(x4, shape=(1, heads, seq_len, head_dim_pw))
        return axol.core.transpose(x4, dims=(0, 2, 1, 3))

    def _norm_rope_quant(x, consts, weight_buffer, quant_params, trig, heads):
        """Per-head RMSNorm -> RoPE -> int8 quantize (Qwen3's two arms, minus Europa glue)."""
        if seq_len == 1:
            return rms_norm_dpu_reduce(
                x,
                consts,
                normalized_shape=(head_dim,),
                weight_buffer=weight_buffer,
                rope_trig=trig,
                quant_params=quant_params,
                invsqrt_method="additive",
                output=BufferInfo(dtype="int8x64"),
            )
        normed = rms_norm_dpu_reduce(
            x,
            consts,
            normalized_shape=(head_dim,),
            weight_buffer=weight_buffer,
            invsqrt_method="additive",
        )
        return quantize_activation(
            axol.core.rope(normed, trig), quant_params, (heads, seq_len), head_dim_pw
        )

    @A.ir_module
    class ApertusLayerFunctional:
        @A.prim_func
        def main(
            seq_pos: A.Buffer((1,), "uint64x64", scope=A.DDR),
            hidden_states: A.Buffer((1, seq_len, hidden_dim_pw), "float32x64", scope=A.DDR),
            # [inv-sqrt LUT rows, 1/D, eps] in one buffer: see pack_norm_consts.
            rms1_consts: A.Buffer((4,), "float32x64", scope=A.DDR),
            q_proj_qparams: A.Buffer((2 * hidden_dim_pw,), "float32x64", scope=A.DDR),
            q_proj_w: A.Buffer((q_out_dim_pw, hidden_dim), "int8x64", scope=A.DDR),
            q_proj_dq: A.Buffer((1, 1, 2 * q_out_dim_pw), "float32x64", scope=A.DDR),
            k_proj_w: A.Buffer((kv_out_dim_pw, hidden_dim), "int8x64", scope=A.DDR),
            k_proj_dq: A.Buffer((1, 1, 2 * kv_out_dim_pw), "float32x64", scope=A.DDR),
            v_proj_w: A.Buffer((kv_out_dim_pw, hidden_dim), "int8x64", scope=A.DDR),
            v_proj_dq: A.Buffer((1, 1, 2 * kv_out_dim_pw), "float32x64", scope=A.DDR),
            # Per-head QK-norm weights, D-major for dpu_reduce, and their norm consts.
            q_norm_w: A.Buffer((head_dim_pw,), "float32x64", scope=A.DDR),
            q_norm_consts: A.Buffer((4,), "float32x64", scope=A.DDR),
            k_norm_w: A.Buffer((head_dim_pw,), "float32x64", scope=A.DDR),
            k_norm_consts: A.Buffer((4,), "float32x64", scope=A.DDR),
            rope_trig: A.Buffer((1, seq_len, head_dim_pw), "float32x64", scope=A.DDR),
            k_cache_qparams: A.Buffer((2 * head_dim_pw,), "float32x64", scope=A.DDR),
            v_qparams: A.Buffer((2 * head_dim_pw,), "float16x64", scope=A.DDR),
            k_cache: A.Buffer((1, num_kv_heads, context_len, head_dim_pw), "int8x64", scope=A.DDR),
            v_cache: A.Buffer((1, num_kv_heads, context_len, head_dim_pw), "int8x64", scope=A.DDR),
            q_attn_qparams: A.Buffer((2 * head_dim_pw,), "float32x64", scope=A.DDR),
            sdpa_exp_lut: A.Buffer((2,), "float32x64", scope=A.DDR),
            sdpa_inv_lut: A.Buffer((2,), "float32x64", scope=A.DDR),
            sdpa_scores_dq_scale: A.Buffer((1, 1, 1, 1), "float32x64", scope=A.DDR),
            sdpa_weights_q_scale: A.Buffer((1, 1, 1, 1), "float32x64", scope=A.DDR),
            sdpa_output_dq_scale: A.Buffer((1, 1, 1, 1), "float32x64", scope=A.DDR),
            attn_mask: A.Buffer((1, 1, seq_len, context_len // 64), "int8x64", scope=A.DDR),
            o_qparams: A.Buffer((2 * q_out_dim_pw,), "float16x64", scope=A.DDR),
            o_proj_w: A.Buffer((hidden_dim_pw, q_out_dim), "int8x64", scope=A.DDR),
            o_proj_dq: A.Buffer((1, 1, 2 * hidden_dim_pw), "float32x64", scope=A.DDR),
            residual_in: A.Buffer((1, seq_len, hidden_dim_pw), "float32x64", scope=A.DDR),
            rms2_consts: A.Buffer((4,), "float32x64", scope=A.DDR),
            up_proj_qparams: A.Buffer((2 * hidden_dim_pw,), "float32x64", scope=A.DDR),
            # The MLP in two halves of the intermediate dim (see MLP_PARTS): up_proj split
            # along N, down_proj along K, each half its own weight + dequant buffers.
            up_proj_lo_w: A.Buffer((part_pw, hidden_dim), "int8x64", scope=A.DDR),
            up_proj_lo_dq: A.Buffer((1, 1, 2 * part_pw), "float32x64", scope=A.DDR),
            up_proj_hi_w: A.Buffer((part_pw, hidden_dim), "int8x64", scope=A.DDR),
            up_proj_hi_dq: A.Buffer((1, 1, 2 * part_pw), "float32x64", scope=A.DDR),
            down_proj_lo_qparams: A.Buffer((2 * part_pw,), "float32x64", scope=A.DDR),
            down_proj_hi_qparams: A.Buffer((2 * part_pw,), "float32x64", scope=A.DDR),
            # [alpha_p, alpha_n, beta], each broadcast over a PWORD; the g(m) table chain.
            xielu_coef: A.Buffer((XIELU_COEF_ROWS,), "float32x64", scope=A.DDR),
            xielu_lut: A.Buffer((XIELU_LUT_ROWS,), "float32x64", scope=A.DDR),
            down_proj_lo_w: A.Buffer((hidden_dim_pw, part_dim), "int8x64", scope=A.DDR),
            down_proj_lo_dq: A.Buffer((1, 1, 2 * hidden_dim_pw), "float32x64", scope=A.DDR),
            down_proj_hi_w: A.Buffer((hidden_dim_pw, part_dim), "int8x64", scope=A.DDR),
            down_proj_hi_dq: A.Buffer((1, 1, 2 * hidden_dim_pw), "float32x64", scope=A.DDR),
        ):  # noqa: F821
            # SDPA zero-points + int8 range are zero-pinned constants; the dequant scales
            # arrive as per-layer input buffers.
            sdpa_weights_q_zp = A.alloc_constant(
                np.array([0] * 64).reshape((1, 1, 1, 1, 64)), "float32x64", scope=A.DDR
            )
            sdpa_weights_q_qmin = A.alloc_constant(
                np.array([-128] * 64).reshape((1, 1, 1, 1, 64)), "float32x64", scope=A.DDR
            )
            sdpa_weights_q_qmax = A.alloc_constant(
                np.array([127] * 64).reshape((1, 1, 1, 1, 64)), "float32x64", scope=A.DDR
            )
            sdpa_query_q_zp = A.alloc_constant(
                np.array([0] * 64).reshape((1, 1, 1, 1, 64)), "float32x64", scope=A.DDR
            )
            sdpa_value_q_zp = A.alloc_constant(
                np.array([0] * 64).reshape((1, 1, 1, 1, 64)), "float32x64", scope=A.DDR
            )

            # attention_layernorm (weight folded into q_proj_qparams) -> int8, shared by q/k/v.
            qkv_q = hidden_norm_quant(hidden_states, rms1_consts, q_proj_qparams, d, ATTN_NORM_FOLD)

            q5 = quantized_matmul_activation(
                qkv_q, q_proj_w, q_proj_dq, output=BufferInfo(dtype="float16x64")
            )
            q_heads = _to_heads(q5, num_q_heads)

            # The K chain is authored BEFORE the V projection (Domyn's order) so k_heads is
            # consumed and freed before V stages its input/weight/output tiles.
            k5 = quantized_matmul_activation(
                qkv_q, k_proj_w, k_proj_dq, output=BufferInfo(dtype="float16x64")
            )
            k_heads = _to_heads(k5, num_kv_heads)
            k_update = _norm_rope_quant(
                k_heads, k_norm_consts, k_norm_w, k_cache_qparams, rope_trig, num_kv_heads
            )

            v5 = quantized_matmul_activation(
                qkv_q, v_proj_w, v_proj_dq, output=BufferInfo(dtype="float16x64")
            )
            v_heads = _to_heads(v5, num_kv_heads)
            v_update = quantize_activation(
                v_heads, v_qparams, (num_kv_heads, seq_len), head_dim_pw
            )

            k_out, v_out = axol.core.update_kv_cache(k_update, v_update, k_cache, v_cache, seq_pos)

            q_attn = _norm_rope_quant(
                q_heads, q_norm_consts, q_norm_w, q_attn_qparams, rope_trig, num_q_heads
            )

            attn_out = axol.core.scaled_dot_product_attention(
                q_attn,
                k_out,
                v_out,
                sdpa_scores_dq_scale,
                sdpa_exp_lut,
                sdpa_inv_lut,
                sdpa_weights_q_scale,
                sdpa_weights_q_zp,
                sdpa_weights_q_qmin,
                sdpa_weights_q_qmax,
                sdpa_output_dq_scale,
                sdpa_query_q_zp,
                sdpa_value_q_zp,
                attn_mask,
                enable_gqa=True,
                use_newton_raphson=False,
                scale=1.0 / (head_dim**0.5),
                neg_exp_method=ChainedLUTMethod.ADDITIVE,
                inv_method=ChainedLUTMethod.ADDITIVE,
                # The o_proj-input quantize fused into SDPA's P@V write -> int8 out.
                attention_output_quantizer=o_qparams,
                **_sdpa_cores_kw,
            )

            attn_t = axol.core.transpose(attn_out, dims=(0, 2, 1, 3))
            attn_flat = axol.atomic.reshape(attn_t, shape=(1, seq_len, q_out_dim_pw))

            o6 = quantized_matmul_activation(
                attn_flat, o_proj_w, o_proj_dq, output=BufferInfo(dtype="float32x64")
            )

            residual = axol.atomic.add(residual_in, o6)

            # feedforward_layernorm (weight folded into up_proj_qparams) -> int8.
            up_q = hidden_norm_quant(residual, rms2_consts, up_proj_qparams, d, FFN_NORM_FOLD)

            # The MLP in two halves of the intermediate dim. quantized_matmul_activation holds a
            # whole K stripe in IMC (1 MiB per core = 256 int8 K-blocks = K 16384; Domyn's
            # down_proj is exactly at that limit), so down_proj's K = 21504 cannot be one
            # matmul. Each half is up_proj (N-slice, float16 out -- xielu reads
            # it in float18, so fp16 is not the precision limit, and an fp32 output tile made
            # Domyn's prefill infeasible) -> xielu_quantize -> down_proj (K-slice, float32
            # out). The halves' dequant tables split dz, so lo + hi is the full product.
            up_lo = _mlp_matmul(
                up_q, up_proj_lo_w, up_proj_lo_dq, **_mlp_cores_kw, output=BufferInfo(dtype="float16x64")
            )
            mlp_lo_2d = xielu_quantize(
                axol.atomic.reshape(up_lo, shape=(seq_len, part_pw)),
                down_proj_lo_qparams,
                xielu_coef,
                xielu_lut,
            )
            mlp_lo = axol.atomic.reshape(mlp_lo_2d, shape=(1, seq_len, part_pw))
            d6_lo = _mlp_matmul(
                mlp_lo, down_proj_lo_w, down_proj_lo_dq, **_mlp_cores_kw, output=BufferInfo(dtype="float32x64")
            )
            residual_lo = axol.atomic.add(residual, d6_lo)

            up_hi = _mlp_matmul(
                up_q, up_proj_hi_w, up_proj_hi_dq, **_mlp_cores_kw, output=BufferInfo(dtype="float16x64")
            )
            mlp_hi_2d = xielu_quantize(
                axol.atomic.reshape(up_hi, shape=(seq_len, part_pw)),
                down_proj_hi_qparams,
                xielu_coef,
                xielu_lut,
            )
            mlp_hi = axol.atomic.reshape(mlp_hi_2d, shape=(1, seq_len, part_pw))
            d6_hi = _mlp_matmul(
                mlp_hi, down_proj_hi_w, down_proj_hi_dq, **_mlp_cores_kw, output=BufferInfo(dtype="float32x64")
            )

            A.output(axol.atomic.add(residual_lo, d6_hi))

    return ApertusLayerFunctional


# Per-layer constant binding

#: Attention projections: kernel buffer prefixes and torch leaf names alike. The MLP's two
#: projections are packed per part (``up_proj_lo_w`` ...), see _mlp_part_buffers.
_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")
#: LUT buffer -> the RangeObserver site whose recorded range the table is fit over.
_LUT_SITES = {
    "sdpa_exp_lut": "self_attn.sdpa.exp_range",
    "sdpa_inv_lut": "self_attn.sdpa.reciprocal_range",
}
#: SDPA per-tensor scales the kernel takes as broadcast single-PWORD inputs.
_SDPA_SCALARS = {
    "sdpa_scores_dq_scale": "scores_dq_scale",  # q_scale * k_scale
    "sdpa_weights_q_scale": "qs_w",  # softmax-weight quant scale
    "sdpa_output_dq_scale": "output_dq_scale",  # w_scale * v_scale
}
#: Per-head QK-norm weight buffers (D-major, dpu_reduce) -> the norm's module path.
_QK_NORM_WEIGHTS = {"q_norm_w": "self_attn.q_norm", "k_norm_w": "self_attn.k_norm"}
#: Host-tensor names of the per-layer xIELU constants, in xielu_coef row order.
XIELU_COEF_HOST_TENSORS = ("xielu_alpha_p", "xielu_alpha_n", "xielu_beta")


def _nk(w):
    """[out, in] int8 -> NK (out//64, in, 64): [nb, k, lane] = w[nb*64 + lane, k]."""
    import torch

    out, inn = w.shape
    return w.reshape(out // PWORD, PWORD, inn).permute(0, 2, 1).contiguous().to(torch.int8)


def _dequant_table(dz, ds):
    """Interleaved [dz, ds] per output block, shape (1, 1, 2*out//64, 64)."""
    import torch

    dz = torch.as_tensor(dz).to(torch.float32).reshape(-1, PWORD)
    ds = torch.as_tensor(ds).to(torch.float32).reshape(-1, PWORD)
    dq = torch.empty((2 * dz.shape[0], PWORD), dtype=torch.float32)
    dq[0::2] = dz
    dq[1::2] = ds
    return dq.reshape(1, 1, 2 * dz.shape[0], PWORD).contiguous()


def xielu_coef_rows(bundle, layer: int):
    """The (3, 64) ``[alpha_p, alpha_n, beta]`` broadcast rows for one layer."""
    import torch

    rows = []
    for name in XIELU_COEF_HOST_TENSORS:
        if name not in bundle.host_tensors:
            raise KeyError(
                f"bundle has no host tensor {name!r}; it was exported by an architecture "
                "that does not carry the per-layer xIELU constants"
            )
        value = torch.as_tensor(bundle.host_tensors[name]).to(torch.float32).reshape(-1)[layer]
        rows.append(value.expand(PWORD))
    return torch.stack(rows).contiguous()


def mlp_part_tensors(up: dict, down: dict, part: int, part_dim: int) -> dict:
    """One MLP part's slice of the exported up_proj / down_proj params.

    up_proj is cut along its OUTPUT channels (rows): weight, dz and ds slice directly.
    down_proj is cut along its INPUT channels: the weight's columns and the activation qparams
    (qs, qz) slice; ds is per output channel and shared; dz = qz * sum_k w[:, k] is re-summed
    over this part's columns, so the parts' dz add up to the full dz.
    """
    import torch

    lo, hi = part * part_dim, (part + 1) * part_dim
    w_down = down["w"][:, lo:hi]
    qz = torch.as_tensor(down["qz"]).reshape(-1)
    if not torch.all(qz == qz[0]):
        raise ValueError("down_proj's activation zero point must be per-tensor to split dz")
    return {
        "up_w": up["w"][lo:hi],
        "up_dz": torch.as_tensor(up["dz"]).reshape(-1)[lo:hi],
        "up_ds": torch.as_tensor(up["ds"]).reshape(-1)[lo:hi],
        "down_qs": torch.as_tensor(down["qs"]).reshape(-1)[lo:hi],
        "down_qz": qz[lo:hi],
        "down_w": w_down,
        "down_dz": qz[0].to(torch.float64) * w_down.to(torch.int64).sum(dim=1).to(torch.float64),
        "down_ds": torch.as_tensor(down["ds"]).reshape(-1),
    }


def apertus_layer_buffers(bundle, spec, layer):
    """Bind one Apertus layer's constants into the kernel's buffers.

    Weights in NK layout ``(out//64, in, 64)``; output dequant as interleaved ``[dz, ds]``;
    activation quant as interleaved ``[scale, zp]`` with each hidden RMSNorm's weight
    folded into the scale that follows it; per-head QK-norm weights D-major with their
    norm consts; SDPA LUTs fit over the recorded ranges; the xIELU coefficients from the
    bundle's host tensors and the calibration-free g(m) table.
    """
    import torch

    from ...layout import fit_lut, layer_linears, layer_sdpa, layer_subset, pack
    from .xielu_lut import xielu_lut_table

    lin = layer_linears(bundle.linear, layer)
    sdpa = layer_sdpa(bundle.sdpa, layer)
    norm = layer_subset(bundle.norm, layer)
    rng = layer_subset(bundle.lut_ranges, layer)
    buffers: dict = {}

    for proj in _PROJECTIONS:
        p = lin[proj]
        if f"{proj}_w" in spec:
            buffers[f"{proj}_w"] = pack(_nk(p["w"]), spec[f"{proj}_w"])
        if f"{proj}_dq" in spec:
            buffers[f"{proj}_dq"] = pack(_dequant_table(p["dz"], p["ds"]), spec[f"{proj}_dq"])

    part_dim, _ = mlp_part_dims(int(bundle.config["intermediate_size"]))
    for i, part in enumerate(MLP_PARTS):
        t = mlp_part_tensors(lin["up_proj"], lin["down_proj"], i, part_dim)
        named = {
            f"up_proj_{part}_w": lambda t=t: _nk(t["up_w"]),
            f"up_proj_{part}_dq": lambda t=t: _dequant_table(t["up_dz"], t["up_ds"]),
            f"down_proj_{part}_w": lambda t=t: _nk(t["down_w"]),
            f"down_proj_{part}_dq": lambda t=t: _dequant_table(t["down_dz"], t["down_ds"]),
        }
        for name, make in named.items():
            if name in spec:
                buffers[name] = pack(make(), spec[name])
        name = f"down_proj_{part}_qparams"
        if name in spec:
            buffers[name] = interleave_qparams(t["down_qs"], t["down_qz"], spec[name])

    for buf, field_name in _SDPA_SCALARS.items():
        if buf in spec:
            buffers[buf] = pack(torch.as_tensor(sdpa[field_name]), spec[buf])
    for buf, site in _LUT_SITES.items():
        if buf in spec:
            buffers[buf] = fit_lut(rng[site], spec[buf])

    for buf, site in _QK_NORM_WEIGHTS.items():
        if buf in spec:
            buffers[buf] = pack(torch.as_tensor(norm[site]).float(), spec[buf])

    hidden_dim = int(bundle.config["hidden_size"])
    head_dim = int(bundle.config["head_dim"])
    add_norm_consts(
        buffers,
        spec,
        bundle,
        layer,
        {
            "rms1_consts": ("attention_layernorm.inv_sqrt_range", hidden_dim),
            "rms2_consts": ("feedforward_layernorm.inv_sqrt_range", hidden_dim),
            "q_norm_consts": ("self_attn.q_norm.inv_sqrt_range", head_dim),
            "k_norm_consts": ("self_attn.k_norm.inv_sqrt_range", head_dim),
        },
    )

    # Interleaved [scale, zp] activation-quant tables. Matmul-input quants carry the
    # SmoothQuant zero point; the SDPA quants are zero-pinned (qz=None).
    qparam_sites = {
        "q_proj_qparams": (lin["q_proj"]["qs"], lin["q_proj"]["qz"]),
        "up_proj_qparams": (lin["up_proj"]["qs"], lin["up_proj"]["qz"]),
        "o_qparams": (lin["o_proj"]["qs"], lin["o_proj"]["qz"]),
        "q_attn_qparams": (sdpa["qs_q"], None),
        "k_cache_qparams": (sdpa["qs_k"], None),
        "v_qparams": (sdpa["qs_v"], None),
    }
    folded = {f.qparams: f.site for f in APERTUS_NORM_FOLDS}
    for name, (qs, qz) in qparam_sites.items():
        if name in spec:
            site = folded.get(name)
            buffers[name] = interleave_qparams(
                qs, qz, spec[name], norm_weight=norm[site] if site else None
            )

    if "xielu_coef" in spec:
        buffers["xielu_coef"] = pack(xielu_coef_rows(bundle, layer), spec["xielu_coef"])
    if "xielu_lut" in spec:
        n_luts = spec["xielu_lut"][0][0]
        buffers["xielu_lut"] = pack(torch.from_numpy(xielu_lut_table(n_luts)), spec["xielu_lut"])

    check_norm_folds(buffers, APERTUS_NORM_FOLDS)
    return buffers


# Device head at the pruned vocabulary


def make_pruned_lm_head_kernel(
    config, ctx_len, phase, chunk_len=None, cores=1, generation=None, *, spec
):
    """The shared device head, sized by ``head_vocab_size`` instead of ``vocab_size``.

    ``make_lm_head_kernel`` reads the head's row count from ``config.vocab_size``, which
    for Apertus is the 266,752-row input embedding; the checkpoint's head has 131,072 rows. The explicit ``chunk_len`` / ``cores`` / ``generation`` parameters are
    load-bearing: ``_compile_phase`` passes each only if the builder's signature names it.
    """
    from ..lm_head_kernel import make_lm_head_kernel

    head_config = dataclasses.replace(config, vocab_size=config.head_vocab_size)
    return make_lm_head_kernel(
        head_config, ctx_len, phase, chunk_len, cores, generation, spec=spec
    )


class ApertusMetisImplementation(MulticoreDecoderImplementation):
    """Apertus 1.5 on Metis: host embed -> N device decoder layers -> device or host head."""

    name = "apertus-metis"
    architecture = "apertus"
    spec = APERTUS_SPEC
    layer_build = staticmethod(make_apertus_layer_kernel)
    layer_buffers_fn = staticmethod(apertus_layer_buffers)

    def tail_kernel(self, config):
        kernel = super().tail_kernel(config)
        if kernel is None:
            return None
        return dataclasses.replace(kernel, build=partial(make_pruned_lm_head_kernel, spec=self.spec))

    def _trig(self, positions, host_params):
        """Precombined sin||cos RoPE tables from the exported llama3 frequencies.

        ``rope_inv_freq`` (head_dim/2 values) is the checkpoint's own rescaled set -- theta
        4e6, factor 32, low/high freq factors 1/4, original context 8192 -- computed by the
        torch reference, so host and reference cannot disagree on the scaling.
        """
        head_dim = int(host_params["head_dim"])
        inv_freq = np.asarray(host_params["rope_inv_freq"], dtype=np.float64).reshape(-1)
        half = head_dim // 2
        if inv_freq.size != half:
            raise ValueError(
                f"rope_inv_freq has {inv_freq.size} frequencies; a head_dim={head_dim} "
                f"trig table needs {half}"
            )
        pos = np.atleast_1d(np.asarray(positions, dtype=np.float64))
        angles = pos[:, None] * inv_freq[None, :]
        rope_pw = half // PWORD
        cos = np.cos(angles).astype(np.float32).reshape(1, pos.size, rope_pw, PWORD)
        sin = np.sin(angles).astype(np.float32).reshape(1, pos.size, rope_pw, PWORD)
        return np.ascontiguousarray(np.concatenate([sin, cos], axis=2))


register_implementation(ApertusMetisImplementation())
