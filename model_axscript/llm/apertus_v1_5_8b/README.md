# Apertus-v1.5-8B (AxScript, Metis)

The text decoder of `swiss-ai/Apertus-v1.5-8B`, quantised to W8A8 and compiled for Metis.
The image (VQ) and audio (WavTokenizer) encoders are not part of this card: they run on
the host and pass integer token ids to the decoder.

**Status: experimental.** Built and run on one Metis chip (16 GiB) at context 1024, prefill
chunk 512, 4 cores, with the head on the device: 7.11 GiB of device memory, about 2.5 decode
tokens/s, coherent output in English, German, French, Italian, code and chat. Accuracy has
not been evaluated yet, and 2048 and 4096 have not been measured.

## Architecture

| Component | Value | On device |
|---|---|---|
| Layers | 32 | yes |
| Hidden size | 4096 | yes |
| Attention | 32 query, 8 KV heads, head dim 128, no bias | Llama-3.1-8B kernel shape |
| QK-norm | per-head RMSNorm before RoPE | Qwen3's fused `rms_norm_dpu_reduce` at decode |
| RoPE | llama3, theta 4,000,000, factor 32 | host tables from `rope_inv_freq` |
| MLP | `up_proj -> xIELU -> down_proj`, intermediate 21504 | `llm.xielu_quantize` |
| Norms | pre-norm RMSNorm, eps 1e-5 | folded into the following quantise |
| Embedding | 266,752 rows | host, fp16 |
| Head | 131,072 rows (pruned, untied) | device (default) or host |

## xIELU

`xielu(x) = alpha_p * relu(x)^2 + beta * x + alpha_n * g(min(x, 0))` with
`g(m) = expm1(m) - m`. The first two terms are exact on the DPU. Only `g` goes through a
2-table chained LUT, which is the same for every layer and needs no calibration
(`models/apertus/xielu_lut.py`). The per-layer constants (`alpha_p` from 0.5 to 20,
`alpha_n` from 0.9 to 11) are folded through softplus at build time.

## Build and run

See the repository README. In short:

```bash
export APERTUS_V1_5_8B_REPO=/path/to/Apertus-v1.5-8B   # same string wherever the artefact is used
python -m model_axscript.llm list                        # apertus-v1.5-8b should appear
python -m model_axscript.llm build apertus-v1.5-8b
python -m model_axscript.llm run apertus-v1.5-8b "The capital of Switzerland is"
```

To compile on a different host from the one that calibrated, copy the build's
`quantized_model/` directory and run `build apertus-v1.5-8b --quantized-model <dir>`.

## Measured sizes

| Context | Chunk | Cores | Head | Device memory | Run on chip |
|---|---|---|---|---|---|
| 1024 | 512 | 4 | device | 7.11 GiB | yes |
