# Copyright Axelera AI, 2026 (structure follows models/domyn/domyn.py and models/qwen3/qwen3.py)
# SPDX-License-Identifier: Apache-2.0
"""Self-contained Apertus 1.5 text decoder reference in PyTorch.

A decoder-only transformer built from torch / torch.nn and this toolkit only -- it does NOT
import ``transformers`` (the Apertus 1.5 modelling code lives in the swiss-ai fork,
checked at 5.18.0.dev0, which pins a different ``transformers`` than the SDK). Matches
``Apertus1p5TextForCausalLM`` / the text backbone of ``Apertus1p5ForConditionalGeneration``
(https://huggingface.co/swiss-ai, ``model_type: "apertus1p5"``).

What sets it apart from the references beside it, each a place a shared kernel does not
carry over unchanged:

* **xIELU MLP, not SwiGLU.** ``down_proj(xielu(up_proj(x)))``: one up projection, no gate,
  so the MLP's SmoothQuant groups are two singletons (as Domyn's squared-ReLU). The
  activation has per-layer trainable constants (``alpha_p``, ``alpha_n`` stored before
  softplus; ``beta`` and ``eps`` as buffers). On device only ``g(m) = expm1(m) - m`` goes
  through a table -- see ``xielu_lut.py``.
* **QK-Norm** (per-head RMSNorm on q and k before RoPE), as Qwen3.
* **llama3 RoPE** with theta 4e6 and factor 32 (``llama3_inv_freq``).
* **Pruned, untied head.** The input embedding has ``vocab_size`` (266,752: text, image
  and audio tokens) rows; ``lm_head`` has ``head_vocab_size`` (131,072, the text prefix)
  rows, so a retained output id equals its token id.
* Hidden norms are named ``attention_layernorm`` / ``feedforward_layernorm``, not
  ``input_layernorm`` / ``post_attention_layernorm``, so the shared ``NormFold`` constants do
  not apply.

Module paths are ``model.layers.N.*``, ``model.norm``, ``lm_head`` -- the shared device-head
packer hard-codes ``model.norm`` and ``lm_head`` -- so the composite checkpoint's
``model.language_model.*`` keys are renamed on load (``checkpoint_name``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...modules import QuantSDPA, RMSNorm, apply_rotary_pos_emb, repeat_kv
from ...quant import ActivationObserver, QuantLinear, RangeObserver
from ...quant import quantize_model as _quantize_model
from ...quant import record_lut_ranges
from ..architecture import LLMArchitecture, register_architecture
from ..llama.llama import llama3_inv_freq
from .xielu_lut import XIELU_RANGE_FUNCTION, apply_xielu_lut_torch


def _quant_linear_kwargs(config: "ApertusConfig") -> dict:
    """The W8A8 keyword arguments shared by every quantizable linear."""
    return {
        "act_bits": config.activation_quant_bits,
        "weight_bits": config.weight_quant_bits,
        "act_symmetric": config.activation_quant_symmetric,
    }


# Configuration


@dataclass
class ApertusConfig:
    """Hyperparameters for an Apertus 1.5 text decoder. Defaults match Apertus-v1.5-8B."""

    #: Input-embedding rows: text + image + audio token ids.
    vocab_size: int = 266752
    #: lm_head rows (the checkpoint's ``output_vocab_size``): the text prefix of the vocabulary.
    head_vocab_size: int = 131072
    hidden_size: int = 4096
    intermediate_size: int = 21504
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    max_position_embeddings: int = 262144
    rms_norm_eps: float = 1e-5
    rope_theta: float = 4_000_000.0
    #: llama3 wavelength rescaling ({factor, low_freq_factor, high_freq_factor,
    #: original_max_position_embeddings}), or None for plain RoPE.
    rope_scaling: dict | None = field(
        default_factory=lambda: {
            "factor": 32.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
        }
    )
    tie_word_embeddings: bool = False
    # W8A8: per-tensor asymmetric activations, per-channel symmetric weights.
    activation_quant_bits: int = 8
    weight_quant_bits: int = 8
    activation_quant_symmetric: bool = False
    #: When enabled, Q/K/V and softmax weights are fake-quantized.
    quantize_attention: bool = False
    sdpa_quant_bits: int = 8
    #: When enabled lm_head is a QuantLinear (int8 weight + qparams, runs on device).
    quantize_lm_head: bool = False

    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @classmethod
    def from_hf_dict(cls, cfg: dict) -> "ApertusConfig":
        """Builds an ApertusConfig from a HuggingFace config.json dict.

        Accepts the composite ``apertus1p5`` config (the text decoder is ``text_config``) or a
        bare text config. Rejects what this reference does not model rather than loading a
        silently-wrong graph: projection biases (QuantLinear is bias-free), a post-norm
        variant, a model without QK-norm, another activation, or another RoPE type.
        """
        text = cfg.get("text_config", cfg)
        act = text.get("hidden_act", "xielu")
        if act != "xielu":
            raise ValueError(f"ApertusForCausalLM models the xIELU MLP; got hidden_act={act!r}")
        for key in ("attention_bias", "mlp_bias", "post_norm"):
            if text.get(key, False):
                raise ValueError(f"{key}=True is not modelled by this reference")
        if not text.get("qk_norm", True):
            raise ValueError("qk_norm=False is not modelled (Apertus 1.5 has QK-norm)")
        rope = dict(text.get("rope_parameters") or text.get("rope_scaling") or {})
        rope_theta = rope.pop("rope_theta", text.get("rope_theta"))
        rope_type = rope.pop("rope_type", rope.pop("type", "default"))
        if rope_type == "llama3":
            scaling = {
                k: rope[k]
                for k in (
                    "factor",
                    "low_freq_factor",
                    "high_freq_factor",
                    "original_max_position_embeddings",
                )
            }
        elif rope_type == "default":
            scaling = None
        else:
            raise ValueError(f"rope_type={rope_type!r} is not modelled (llama3 or default)")
        head_dim = text.get("head_dim") or text["hidden_size"] // text["num_attention_heads"]
        return cls(
            vocab_size=text["vocab_size"],
            head_vocab_size=text.get("output_vocab_size") or text["vocab_size"],
            hidden_size=text["hidden_size"],
            intermediate_size=text["intermediate_size"],
            num_hidden_layers=text["num_hidden_layers"],
            num_attention_heads=text["num_attention_heads"],
            num_key_value_heads=text["num_key_value_heads"],
            head_dim=head_dim,
            max_position_embeddings=text["max_position_embeddings"],
            rms_norm_eps=text["rms_norm_eps"],
            rope_theta=float(rope_theta),
            rope_scaling=scaling,
            tie_word_embeddings=text.get("tie_word_embeddings", False),
        )


# xIELU


def _softplus_inverse(y: float) -> float:
    return float(np.log(np.expm1(y)))


class XIELU(nn.Module):
    """xIELU activation (https://arxiv.org/abs/2411.13010), Apertus's MLP nonlinearity.

    Parameter names and storage match the checkpoint: ``alpha_p`` / ``alpha_n`` are raw
    (pre-softplus) ``(1,)`` parameters, ``beta`` / ``eps`` persistent scalar buffers.

    ``forward`` is the reference's own two-branch form. With ``"xielu"`` in ``lut_sites``
    it switches to the split form the device computes,
    ``alpha_p * relu(x)**2 + beta * x + alpha_n * lut(min(x, 0))``, with the device table
    modelled (``xielu_lut.apply_xielu_lut_torch``), so the fake-quant reference can
    separate table error from device bugs.
    """

    def __init__(self, alpha_p: float = 0.8, alpha_n: float = 0.8, beta: float = 0.5, eps=-1e-6):
        super().__init__()
        self.alpha_p = nn.Parameter(torch.tensor([_softplus_inverse(alpha_p)]))
        self.alpha_n = nn.Parameter(torch.tensor([_softplus_inverse(alpha_n - beta)]))
        self.register_buffer("beta", torch.tensor(float(beta)))
        self.register_buffer("eps", torch.tensor(float(eps)))
        self.lut_sites: frozenset = frozenset()

    def coefficients(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The effective ``(alpha_p, alpha_n, beta)`` the device multiplies by (float32)."""
        beta = self.beta.detach().float()
        alpha_p = F.softplus(self.alpha_p.detach().float()).reshape(())
        alpha_n = (beta + F.softplus(self.alpha_n.detach().float())).reshape(())
        return alpha_p, alpha_n, beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha_p = F.softplus(self.alpha_p)
        alpha_n = self.beta + F.softplus(self.alpha_n)
        if "xielu" in self.lut_sites:
            r = F.relu(x)
            m = x - r
            return alpha_p * r * r + self.beta * x + alpha_n * apply_xielu_lut_torch(m)
        return torch.where(
            x > 0,
            alpha_p * x * x + self.beta * x,
            (torch.expm1(torch.min(x, self.eps)) - x) * alpha_n + self.beta * x,
        )


class XIELUMLP(nn.Module):
    """``down_proj(xielu(up_proj(x)))``: Apertus's un-gated MLP."""

    def __init__(self, hidden_size: int, intermediate_size: int, **quant_kwargs) -> None:
        super().__init__()
        self.up_proj = QuantLinear(hidden_size, intermediate_size, **quant_kwargs)
        self.down_proj = QuantLinear(intermediate_size, hidden_size, **quant_kwargs)
        self.act_fn = XIELU()
        # The xIELU input range. Diagnostics only -- the device table is calibration-free --
        # but it is the number that says how far into the asymptote bin the model reaches.
        self.xielu_range = RangeObserver(XIELU_RANGE_FUNCTION)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up = self.xielu_range(self.up_proj(x))
        return self.down_proj(self.act_fn(up))


# Decoder


class ApertusRotaryEmbedding(nn.Module):
    """RoPE cos/sin tables from the llama3-rescaled inverse frequencies."""

    def __init__(self, head_dim: int, theta: float, scaling: dict | None) -> None:
        super().__init__()
        self.register_buffer("inv_freq", llama3_inv_freq(head_dim, theta, scaling), persistent=False)

    @torch.no_grad()
    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.inv_freq[None, :, None].float()
        positions = position_ids[:, None, :].float()
        freqs = (inv_freq @ positions).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


class ApertusAttention(nn.Module):
    """Grouped-query attention with QK-Norm and RoPE (no projection bias)."""

    def __init__(self, config: ApertusConfig) -> None:
        super().__init__()
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = config.num_key_value_groups
        qkw = _quant_linear_kwargs(config)
        self.q_proj = QuantLinear(config.hidden_size, self.num_heads * self.head_dim, **qkw)
        self.k_proj = QuantLinear(config.hidden_size, self.num_kv_heads * self.head_dim, **qkw)
        self.v_proj = QuantLinear(config.hidden_size, self.num_kv_heads * self.head_dim, **qkw)
        self.o_proj = QuantLinear(self.num_heads * self.head_dim, config.hidden_size, **qkw)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.sdpa = QuantSDPA(
            self.head_dim, quantize=config.quantize_attention, bits=config.sdpa_quant_bits
        )

    def forward(self, hidden_states, cos, sin, attention_mask):
        batch, seq_len, _ = hidden_states.shape
        hidden_shape = (batch, seq_len, -1, self.head_dim)
        q = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        k = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        k = repeat_kv(k, self.num_kv_groups)
        v = repeat_kv(v, self.num_kv_groups)
        out = self.sdpa(q, k, v, attention_mask)
        return self.o_proj(out.transpose(1, 2).reshape(batch, seq_len, -1))


class ApertusDecoderLayer(nn.Module):
    """Pre-norm block: RMSNorm -> attention -> residual, RMSNorm -> xIELU MLP -> residual."""

    def __init__(self, config: ApertusConfig) -> None:
        super().__init__()
        self.self_attn = ApertusAttention(config)
        self.mlp = XIELUMLP(
            config.hidden_size, config.intermediate_size, **_quant_linear_kwargs(config)
        )
        self.attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, cos, sin, attention_mask):
        residual = hidden_states
        hidden_states = self.self_attn(self.attention_layernorm(hidden_states), cos, sin, attention_mask)
        hidden_states = residual + hidden_states
        residual = hidden_states
        return residual + self.mlp(self.feedforward_layernorm(hidden_states))


class ApertusModel(nn.Module):
    """The trunk: embeddings + decoder layers + final RMSNorm."""

    def __init__(self, config: ApertusConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [ApertusDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = ApertusRotaryEmbedding(
            config.head_dim, config.rope_theta, config.rope_scaling
        )

    @staticmethod
    def _causal_mask(seq_len: int, dtype, device) -> torch.Tensor:
        mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min, dtype=dtype, device=device)
        return torch.triu(mask, diagonal=1)[None, None]

    def forward(self, input_ids, position_ids=None):
        seq_len = input_ids.shape[1]
        hidden_states = self.embed_tokens(input_ids)
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        cos, sin = (t.to(hidden_states.dtype) for t in self.rotary_emb(position_ids))
        mask = self._causal_mask(seq_len, hidden_states.dtype, hidden_states.device)
        for layer in self.layers:
            hidden_states = layer(hidden_states, cos, sin, mask)
        return self.norm(hidden_states)


# Checkpoint keys

#: The composite checkpoint's text-decoder prefix, renamed to this reference's ``model.``.
_TEXT_PREFIX = "model.language_model."
#: Checkpoint subtrees this reference does not load: the image and audio encoders run on
#: the host GPU, outside the device graph, and only their token ids reach the decoder.
_ENCODER_PREFIXES = ("model.vision_tokenizer.", "model.audio_tokenizer.")


def checkpoint_name(hf_key: str) -> str | None:
    """This reference's state-dict name for a checkpoint key, or None to skip it.

    ``model.language_model.X`` -> ``model.X`` (composite checkpoint); the encoders are
    skipped; everything else (``lm_head.weight``, a bare text checkpoint's ``model.X``)
    passes through and must then exist in the model, or loading fails.
    """
    if hf_key.startswith(_TEXT_PREFIX):
        return "model." + hf_key[len(_TEXT_PREFIX) :]
    if hf_key.startswith(_ENCODER_PREFIXES):
        return None
    return hf_key


def stream_checkpoint_into(model: nn.Module, repo_id: str, *, dtype=torch.float32):
    """Copy the text decoder's tensors into an allocated model, one tensor at a time.

    ``checkpoint.stream_hf_safetensors_into`` cannot be used: it has no key-renaming hook
    and loads whole shards, including the image and audio encoders. This
    reads only the shards that hold text-decoder keys, one tensor at a time through
    ``safe_open``, so the peak is the model plus one tensor. Returns ``(missing,
    unexpected)`` with ``load_state_dict(strict=False)`` meanings.
    """
    from safetensors import safe_open

    from ...checkpoint import repo_file

    index = repo_file(repo_id, "model.safetensors.index.json", optional=True)
    if index is not None:
        with open(index) as f:
            weight_map = json.load(f)["weight_map"]
    else:
        single = repo_file(repo_id, "model.safetensors")
        with safe_open(single, framework="pt") as f:
            weight_map = {k: "model.safetensors" for k in f.keys()}

    slots = dict(model.state_dict(keep_vars=True))
    by_shard: dict[str, list[tuple[str, str]]] = {}
    unexpected: list[str] = []
    for hf_key, shard in weight_map.items():
        name = checkpoint_name(hf_key)
        if name is None:
            continue
        if name not in slots:
            unexpected.append(hf_key)
            continue
        by_shard.setdefault(shard, []).append((hf_key, name))

    seen: set[str] = set()
    with torch.no_grad():
        for shard, pairs in sorted(by_shard.items()):
            with safe_open(repo_file(repo_id, shard), framework="pt", device="cpu") as f:
                for hf_key, name in pairs:
                    tensor = f.get_tensor(hf_key)
                    target = slots[name]
                    if tuple(target.shape) != tuple(tensor.shape):
                        raise RuntimeError(
                            f"{hf_key}: checkpoint shape {tuple(tensor.shape)} != model shape "
                            f"{tuple(target.shape)} ({name})"
                        )
                    target.copy_(tensor.to(dtype))
                    seen.add(name)
    missing = [name for name in slots if name not in seen]
    return missing, unexpected


class ApertusForCausalLM(nn.Module):
    """The Apertus 1.5 text decoder with its (pruned) language-modelling head."""

    def __init__(self, config: ApertusConfig) -> None:
        super().__init__()
        self.config = config
        self.model = ApertusModel(config)
        if config.tie_word_embeddings:
            raise ValueError("tied embeddings are not modelled (Apertus 1.5 is untied)")
        if config.quantize_lm_head:
            self.lm_head = QuantLinear(
                config.hidden_size, config.head_vocab_size, **_quant_linear_kwargs(config)
            )
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.head_vocab_size, bias=False)

    def forward(self, input_ids, position_ids=None):
        return self.lm_head(self.model(input_ids, position_ids))

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        *,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
        **config_overrides,
    ) -> "ApertusForCausalLM":
        """Load the text decoder from a HF checkpoint (local directory or hub id).

        ``config_overrides`` replace fields of the derived config (the build passes
        ``quantize_attention`` and, for a device-head build, ``quantize_lm_head``).
        Every model tensor must be supplied (observer statistics excepted) and every
        text-decoder checkpoint key must have a slot: a wrong mapping fails here rather than
        surfacing later as quantization loss.
        """
        import dataclasses

        from ...checkpoint import repo_file

        with open(repo_file(repo_id, "config.json")) as f:
            config = ApertusConfig.from_hf_dict(json.load(f))
        if config_overrides:
            config = dataclasses.replace(config, **config_overrides)
        model = cls(config).to(device=device or "cpu", dtype=dtype)
        missing, unexpected = stream_checkpoint_into(model, repo_id, dtype=dtype)
        allowed_missing = {
            f"{name}.{buf}"
            for name, m in model.named_modules()
            if isinstance(m, ActivationObserver)
            for buf, _ in m.named_buffers()
        }
        missing = [k for k in missing if k not in allowed_missing]
        if unexpected:
            raise RuntimeError(f"Unexpected keys in checkpoint: {unexpected}")
        if missing:
            raise RuntimeError(f"Missing keys after loading checkpoint: {missing}")
        return model.eval()


# Quantization + export


def smoothquant_groups(module: nn.Module) -> list[list[QuantLinear]]:
    """Groups of QuantLinear that share an input tensor.

    q/k/v share the attention_layernorm output. The xIELU MLP has a single up_proj, so
    up_proj and down_proj are singletons, as is o_proj. A quantized lm_head smooths as a
    singleton (per-tensor head qparams breach the perplexity gate on Qwen3).
    """
    groups: list[list[QuantLinear]] = []
    for layer in module.modules():
        if isinstance(layer, ApertusAttention):
            groups.append([layer.q_proj, layer.k_proj, layer.v_proj])
            groups.append([layer.o_proj])
        elif isinstance(layer, XIELUMLP):
            groups.append([layer.up_proj])
            groups.append([layer.down_proj])
    head = getattr(module, "lm_head", None)
    if isinstance(head, QuantLinear):
        groups.append([head])
    return groups


def quantize_model(module, calibration, *, smooth_alpha=0.8, qk_alpha=0.5) -> None:
    """Calibrate + enable W8A8 (+ SmoothQuant, + QK-smoothing as for QK-normed Qwen3)."""
    _quantize_model(
        module,
        calibration,
        groups=smoothquant_groups(module),
        smooth_alpha=smooth_alpha,
        qk_alpha=qk_alpha,
    )


def set_lut_modeling(module: nn.Module, sites) -> None:
    """Model the device tables at ``sites`` ("xielu", "exp", "reciprocal") in the reference."""
    sites = frozenset(sites)
    for m in module.modules():
        if hasattr(m, "lut_sites"):
            m.lut_sites = sites


def xielu_coefficients(model: nn.Module) -> dict[str, np.ndarray]:
    """Per-layer effective xIELU constants as ``[num_layers]`` float32 arrays."""
    rows = [layer.mlp.act_fn.coefficients() for layer in model.model.layers]
    eps = [float(layer.mlp.act_fn.eps) for layer in model.model.layers]
    return {
        "xielu_alpha_p": np.array([float(r[0]) for r in rows], dtype=np.float32),
        "xielu_alpha_n": np.array([float(r[1]) for r in rows], dtype=np.float32),
        "xielu_beta": np.array([float(r[2]) for r in rows], dtype=np.float32),
        "xielu_eps": np.array(eps, dtype=np.float32),
    }


class ApertusArchitecture(LLMArchitecture):
    """LLMArchitecture for Apertus 1.5: load + quantize the torch reference and export it."""

    name = "apertus"

    def load_pretrained(self, repo_id, **kwargs):
        return ApertusForCausalLM.from_pretrained(repo_id, **kwargs)

    def quantize(self, model, calibration, *, smooth_alpha=0.8, qk_alpha=0.5):
        quantize_model(model, calibration, smooth_alpha=smooth_alpha, qk_alpha=qk_alpha)

    def record_lut_ranges(self, model, calibration):
        record_lut_ranges(model, calibration)

    def num_layers(self, model):
        return model.config.num_hidden_layers

    def config_to_dict(self, model_config):
        return asdict(model_config)

    def config_from_dict(self, data):
        return ApertusConfig(**data)

    def export_host_io(self, model):
        """Host-side params for the embedding / final-norm / RoPE / head boundary.

        Beyond the shared contract: ``rope_inv_freq`` (the llama3-rescaled frequencies, read
        by ``ApertusMetisImplementation._trig`` -- the Llama host tables hard-code Llama 3.2's
        scaling); ``lm_head_weight`` for the host head (pruned rows, smoothing
        undone); and the per-layer xIELU constants as ``[num_layers]`` arrays. The bundle
        has no field for per-layer scalars, so they travel here and the layer
        packer reads them from ``bundle.host_tensors``; they also land in
        ``host_params.npz``, unread.
        """
        config = model.config
        host = {
            "embed_weight": model.model.embed_tokens.weight.detach().cpu().to(torch.float16).numpy(),
            "final_norm_weight": model.model.norm.weight.detach().cpu().to(torch.float32).numpy(),
            "rms_eps": np.float32(config.rms_norm_eps),
            "head_dim": np.int64(config.head_dim),
            "num_q_heads": np.int64(config.num_attention_heads),
            "rope_theta": np.float64(config.rope_theta),
            "rope_inv_freq": model.model.rotary_emb.inv_freq.detach().cpu().to(torch.float32).numpy(),
        }
        weight = model.lm_head.weight.detach().cpu().float()
        if getattr(model.lm_head, "smoothing_enabled", False):
            weight = weight / model.lm_head.smooth_scale.detach().cpu().float()
        host["lm_head_weight"] = weight.to(torch.float16).numpy()
        host.update(xielu_coefficients(model))
        return host


register_architecture(ApertusArchitecture())
