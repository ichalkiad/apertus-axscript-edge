# Apertus on the edge - Axelera AI AxScript

Run [Apertus 1.5 8B](https://huggingface.co/swiss-ai) (Swiss AI Initiative) on an
[Axelera Metis](https://www.axelera.ai) NPU, using Axelera's Voyager SDK v1.9.0 and its
AxScript model toolkit (`model_axscript`).

The text decoder runs on the NPU, quantised to int8 (W8A8 with SmoothQuant). Tokenisation,
the embedding lookup and sampling run on the host. The image and audio tokenizers of
Apertus 1.5 are not included: this repository is text-only.

## Status

Experimental. Built and run on one 16 GiB Metis chip with Voyager SDK 1.9.0:

| | |
|---|---|
| Context length | 1024 tokens (prefill in 512-token chunks); 2048 and 4096 not yet measured |
| Cores | 4 (one model instance per chip) |
| Device memory | 7.11 GiB (int8 decoder weights, KV cache and the output head) |
| Host memory | ~3.1 GB at run time (fp16 embedding and host-side head) |
| Decode speed | ~2.5 tokens/s |
| Time to first token | ~1.7 s for a short prompt |
| Output | Coherent in English, German, French, Italian, code and chat |
| Accuracy | Not yet evaluated against the float model |

## Repository contents

```
model_axscript/                   added to the SDK's model_axscript (Apache-2.0)
  llm/apertus_v1_5_8b/            the model card: tag apertus-v1.5-8b, build/run config
  llm/models/apertus/apertus.py   PyTorch reference, checkpoint loader, quantisation, export
  llm/models/apertus/metis_impl.py  decoder-layer kernel, constant packing, host side
  llm/models/apertus/xielu_lut.py lookup table for the xIELU activation
  llm/ops/xielu_quantize.py       DPU op: xIELU fused with the int8 quantise
patches/register-apertus.patch    registers the model in two SDK lists
scripts/assemble_tree.sh          SDK model_axscript + this repository -> a working tree
scripts/run_apertus.py            generate text with a built artefact on a Metis device
```

How the port maps Apertus onto the SDK, in brief:

- **Attention:** the Llama-3.1-8B kernel shape (32 query and 8 KV heads, head dim 128),
  with Qwen3's QK-norm path, and llama3 RoPE tables (theta 4,000,000, factor 32) built on
  the host.
- **xIELU MLP:** `xielu(x) = alpha_p·relu(x)² + beta·x + alpha_n·(expm1(m) − m)` with
  `m = min(x, 0)`. The first two terms are computed exactly on the DPU; only the bounded
  `expm1(m) − m` goes through a 2-table lookup, which needs no calibration. The MLP runs in
  two halves of its 21504-wide intermediate dimension, because one SDK matmul can hold at
  most 16384 int8 input channels in the NPU's in-memory-compute array.
- **Output head:** pruned to the 131,072 text tokens, as in the checkpoint, and run on the
  device. The host head can be used instead on the same artefact.

## Requirements

**To build** (calibrate, quantise and compile; no NPU needed):

- Voyager SDK 1.9.0 Python packages, including the compiler. x86_64 and aarch64 both work.
- Python 3.10–3.13, about 35 GB of RAM, and a CUDA GPU for calibration (a CPU works, but
  slowly).
- The `swiss-ai/Apertus-v1.5-8B` checkpoint. It is gated on the Hugging Face Hub: accept
  its licence there, then download it or use the Hub id with a token.
- Calibration data: by default the SDK uses WikiText-2, GSM8K, HumanEval, Alpaca and MMLU
  from the Hugging Face Hub.

**To run:**

- A Metis device with 16 GiB, the Axelera driver (`metis-dkms` 1.7.16 is the version
  matched to SDK 1.9.0) and firmware 1.9.0 recommended.
- `axelera-runtime` 1.9.0, `numpy`, `torch` 2.9.1 (the CPU build is enough) and
  `transformers` 4.57.6.

These are the `torch` and `transformers` versions Axelera's own LLM package pins for SDK
1.9.0. Install the driver and firmware as described in Axelera's documentation.

## 1. Get the SDK source and assemble a tree

`model_axscript` ships in the Voyager SDK repository, not in a Python package:

```bash
git clone https://github.com/<you>/apertus-axscript-edge.git && cd apertus-axscript-edge
git clone --depth 1 --branch v1.9.0 https://github.com/axelera-ai-hub/voyager-sdk.git
scripts/assemble_tree.sh voyager-sdk tree
export PYTHONPATH=$PWD/tree
```

`tree/model_axscript` is now the SDK's toolkit with the Apertus model added.

## 2. Install the SDK packages

In a virtual environment, on the build host:

```bash
IDX=https://software.axelera.ai/artifactory/api/pypi/axelera-pypi/simple
pip install --extra-index-url $IDX \
    axelera-devkit==1.9.0 axelera-compiler==1.9.0 axelera-runtime==1.9.0 axelera-rt==1.9.0 \
    "torch==2.9.1" "transformers==4.57.6" "datasets>=4,<5" safetensors numpy
```

On a run-only host, `axelera-runtime==1.9.0`, `numpy`, `torch==2.9.1` and
`transformers==4.57.6` are enough.

Notes:

- **The SDK's licence:** the Axelera packages are covered by the Axelera EULA, which you
  accept by installing them.
- **On aarch64:** the compiler depends on `onnxoptimizer`, which has no aarch64 wheel and
  needs GCC 8 or newer to build. Set `CC=gcc-13 CXX=g++-13` (or another recent GCC) for the
  install. PyPI's aarch64 `torch` is CPU-only; for CUDA use PyTorch's index, for example
  `--index-url https://download.pytorch.org/whl/cu130 "torch==2.9.1+cu130"`.
- **`PATH`:** compiling for the device runs `axkernelcc` from the environment's `bin/`.
  Activate the environment, or put its `bin/` on `PATH`.
- **With uv:** add `--index-strategy unsafe-best-match`.

## 3. Build the artefact

```bash
export APERTUS_V1_5_8B_REPO=/path/to/Apertus-v1.5-8B     # or swiss-ai/Apertus-v1.5-8B
python -m model_axscript.llm list                         # apertus-v1.5-8b should appear
python -m model_axscript.llm build apertus-v1.5-8b \
    --output-dir build/apertus-v1.5-8b --cal-device cuda
```

The default calibration uses the SDK's full mix (about 4,200 samples). Add `--num-cal 64`
for a quick build: 64 samples took about 17 minutes in total on one GH200, including
compilation. The output directory is about 21 GB. Its `quantized_model/` subdirectory
(11 GB) is the calibrated model; it is only needed to recompile without recalibrating
(`build ... --quantized-model build/apertus-v1.5-8b/quantized_model`).

`APERTUS_V1_5_8B_REPO` is recorded in the artefact, and the SDK compares it as a plain
string later. Keep the same value for every later SDK command on that artefact.

## 4. Run on the NPU

With `scripts/run_apertus.py`, which works on any host with the runtime, wherever the
artefact was built:

```bash
python scripts/run_apertus.py --artifact-dir build/apertus-v1.5-8b \
    "The capital of Switzerland is"
python scripts/run_apertus.py --artifact-dir build/apertus-v1.5-8b --chat \
    "What is the capital of Switzerland? Answer in one sentence."
python scripts/run_apertus.py --artifact-dir build/apertus-v1.5-8b --benchmark \
    "The capital of Switzerland is"
```

Options:

| Option | Effect |
|---|---|
| `--device metis-0:1:0` (or an index) | Pick a chip |
| `--max-new-tokens N` | Token budget (default 256) |
| `--temperature T` | Sampling temperature (default 0, greedy) |
| `--host-head` | Run the output head on the CPU instead of the NPU |

`PYTHONPATH` must point at the assembled tree.

The SDK's own front door should also run it on a host where `APERTUS_V1_5_8B_REPO` holds the
same value as at build time and that path exists (not tested with this model yet):

```bash
python -m model_axscript.llm run apertus-v1.5-8b "The capital of Switzerland is" \
    --output-dir build/apertus-v1.5-8b --device metis-0:1:0
```

It loads the tokenizer from `APERTUS_V1_5_8B_REPO`, so on another host use
`run_apertus.py`, which reads the tokenizer shipped inside the artefact.

## Moving an artefact to another host

The run host needs the artefact without `quantized_model/`, plus the assembled tree:

```bash
tar -cf apertus-v1.5-8b.tar --exclude=quantized_model -C build apertus-v1.5-8b   # ~11 GB
tar -czf model_axscript-tree.tar.gz -C tree model_axscript
sha256sum apertus-v1.5-8b.tar model_axscript-tree.tar.gz > SHA256SUMS
```

On the run host, verify with `sha256sum -c SHA256SUMS`, unpack both, set `PYTHONPATH` to
the directory holding `model_axscript/`, and use `run_apertus.py`.

## Limitations

- **Context:** 1024 tokens only, until 2048 and 4096 have been measured on a chip.
- **Modalities:** text only. Image and audio input need the Apertus 1.5 encoders on the
  host, which this repository doesn't include.
- **Accuracy:** not yet measured. A quick build with few calibration samples is meant for
  trying the model out, not for evaluation.
- **One instance per chip:** a model instance uses all 4 cores and 7.1 GiB of the chip.
- **No Hugging Face wrapper:** `from_pretrained(<artefact>)` from the SDK's Hugging Face
  wrapper is unavailable, because `transformers` 4.57 does not know the `apertus1p5`
  architecture.
- **Not tuned for speed:** some of the SDK's weight-streaming optimisations are not yet
  used.

## Licence

Apache-2.0 (see [LICENSE](LICENSE) and [NOTICE](NOTICE)). The code in `model_axscript/`
and `patches/` adapts Axelera's Apache-2.0 `model_axscript`.

The Apertus 1.5 weights are licensed under Apache 2.0 together with the
[Apertus Acceptable Use Policy](https://github.com/swiss-ai/apertus-legal/blob/main/apertus_1.5/USAGE_POLICY.pdf),
which also applies to artefacts compiled from them. The Axelera SDK packages are under the
Axelera EULA and are not part of this repository.
