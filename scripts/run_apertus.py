"""Generate text with a compiled apertus-v1.5-8b artefact on an Axelera Metis device.

Uses the SDK's own generation loop (the one `python -m model_axscript.llm run` drives), but
loads the tokenizer from the artefact directory. The SDK front door instead tokenises with
the checkpoint path recorded at build time and refuses any other, so it cannot run an
artefact on a machine where that path does not exist; this script can.

    PYTHONPATH=<assembled tree> python scripts/run_apertus.py \
        --artifact-dir build/apertus-v1.5-8b "The capital of Switzerland is"

    ... --chat "What is the capital of Switzerland?"        # one chat turn (chat template)
    ... --benchmark "The capital of Switzerland is"          # time to first token, tokens/s
    ... --device metis-0:1:0 --host-head --max-new-tokens 64 --temperature 0.7
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import sys
import time


def render_prompt(tokenizer, text: str, chat: bool) -> str:
    """The prompt as text. A chat turn is rendered with the checkpoint's template; its leading
    BOS is dropped because the tokenizer adds one itself."""
    if not chat:
        return text
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False
    )
    bos = tokenizer.bos_token or ""
    return rendered[len(bos):] if bos and rendered.startswith(bos) else rendered


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt", help="the prompt (or the user turn, with --chat)")
    ap.add_argument("--artifact-dir", type=Path, required=True, help="a built apertus-v1.5-8b artefact")
    ap.add_argument("--device", default=None, help="Metis device name (e.g. metis-0:1:0) or index; default: first")
    ap.add_argument("--chat", action="store_true", help="wrap the prompt in the chat template")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
    ap.add_argument("--host-head", action="store_true", help="run the lm_head on the host instead of the device")
    ap.add_argument("--benchmark", action="store_true", help="report time to first token and decode tokens/s")
    args = ap.parse_args()

    import numpy as np

    from model_axscript.llm.apertus_v1_5_8b import make_config

    # _compiled_run_config is how the SDK's own run path takes the context and chunk length
    # from the artefact rather than from the card's build intent.
    from model_axscript.llm.generate import (
        RuntimeModel,
        _compiled_run_config,
        benchmark_generation,
        load_tokenizer,
    )
    from model_axscript.llm.implementation import get_implementation

    art = args.artifact_dir.resolve()
    if not (art / "model.json").is_file():
        raise SystemExit(f"{art} is not a built artefact (no model.json)")
    pipeline = json.loads((art / "pipeline.json").read_text())
    config = dataclasses.replace(
        make_config(art),
        repo_id=str(art),
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    config = _compiled_run_config(config, pipeline)
    tokenizer = load_tokenizer(str(art))
    prompt = render_prompt(tokenizer, args.prompt, args.chat)

    t0 = time.time()
    runtime = RuntimeModel(art / "model.json", device=args.device)
    runtime.force_host_head = args.host_head
    print(f"[loaded in {time.time() - t0:.1f} s; context {config.context_len}, "
          f"{'host' if args.host_head else 'device'} head]", file=sys.stderr, flush=True)

    if args.benchmark:
        m = benchmark_generation(runtime, tokenizer, art, prompt, config)
        print(m["response"])
        print(f"[ttft {m['ttft_s']:.3f} s, decode {m['decode_tok_per_s']:.2f} tok/s, "
              f"{int(m['tokens'])} tokens]", file=sys.stderr)
        return 0

    host_params = dict(np.load(art / pipeline["host_params"]))
    impl = get_implementation(pipeline["implementation"])
    generated: list[int] = []
    shown = ""
    for token_id in impl.generate_token_ids(runtime, tokenizer, prompt, config, host_params=host_params):
        generated.append(token_id)
        text = tokenizer.decode(generated, skip_special_tokens=True)
        print(text[len(shown):], end="", flush=True)
        shown = text
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
