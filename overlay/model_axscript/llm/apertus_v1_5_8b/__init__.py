# SPDX-License-Identifier: Apache-2.0
"""Apertus-v1.5-8B on the AxScript path: the build+run config for this model.

The model code lives in ``model_axscript/llm/models/apertus/`` (torch reference, Metis
kernel, xIELU table) and ``model_axscript/llm/ops/xielu_quantize.py``; this package holds only
the per-model configuration ``__main__.py`` drives.

`swiss-ai/Apertus-v1.5-8B` is an ``Apertus1p5ForConditionalGeneration``: a 32-layer text
decoder (hidden 4096, 32 q / 8 kv heads, head_dim 128) plus a VQ image tokenizer and a
WavTokenizer audio tokenizer. Only the text decoder runs on Metis; the encoders run on the
host and hand it integer token ids (image ids from 131272, audio ids from 262344). Its
traits: QK-norm, llama3 RoPE (theta 4e6, factor 32), an un-gated xIELU MLP (intermediate
21504), and a pruned untied head (131072 rows) against a 266752-row input embedding.

Checkpoint: ``repo_id`` is read from ``$APERTUS_V1_5_8B_REPO`` (a local directory or a hub
id), defaulting to the hub id. The SDK records it in the artefact and COMPARES IT AS A
STRING: a build that reuses a calibrated ``quantized_model/`` (``build --quantized-model``),
and ``python -m model_axscript.llm run`` on a built artefact, both need the same string as
the original build, and ``run`` loads the tokenizer from it. To run an artefact on a host
where that path does not exist, use ``scripts/run_apertus.py``, which loads the tokenizer
shipped inside the artefact.
"""

import os
from pathlib import Path

MODEL_TAG = "apertus-v1.5-8b"

#: Environment variable naming the checkpoint (local directory or hub id).
REPO_ENV = "APERTUS_V1_5_8B_REPO"
DEFAULT_REPO = "swiss-ai/Apertus-v1.5-8B"


def make_config(artifact_dir, cal_device: str = ""):
    """Per-model build+run config (implementation, checkpoint, context length).

    Used by ``__main__.py`` for both build and run. ``cal_device`` is a compile-only knob
    (the torch device used for calibration); the runtime path ignores it.
    """
    from ..shared.config import GenerationConfig

    return GenerationConfig(
        implementation="apertus-metis",
        repo_id=os.environ.get(REPO_ENV, DEFAULT_REPO),
        artifact_dir=Path(artifact_dir),
        # 1024 has been built and run on a 16 GiB Metis chip (7.11 GiB of device memory) with
        # a 512-token prefill chunk. The MLP runs in two halves of the intermediate dim
        # (metis_impl.MLP_PARTS), which is what lets chunk 512 fit L1 and down_proj fit IMC.
        # 2048 and 4096 are added only after each has run on chip; widening this set is a
        # measurement, never an edit to make a build pass.
        context_len=1024,
        supported_context_lens=(1024,),
        chunk_lens=(512,),
        chunk_len=512,
        num_cores=4,
        # The device head at the checkpoint's pruned 131072 rows (~0.5 GiB int8). The front
        # door's GenerationConfig default is the HOST head while `python -m
        # model_axscript.llm.build` defaults to the implementation's device head, so
        # it is stated here. The host head stays usable on the same artefact
        # (`force_host_head`, `evaluation/chat.py --host-head`).
        device_lm_head=True,
        # Apertus stops on </s> (2) and the two end-of-turn ids of its chat template.
        stop_token_ids=(2, 68, 72),
        cal_device=cal_device,
    )
