"""Load a trained tiny-lm checkpoint and complete configured prompts."""
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

from mamba_experiment import (
    CHECKPOINT_PATH as TRAINING_CHECKPOINT_PATH,
    build_generation_codec,
    generate_sequence,
    load_env_file,
    parameter_count,
    require_cuda,
)


CHECKPOINT_PATH = TRAINING_CHECKPOINT_PATH
PROMPTS = [
    "<*> Chapter 2: Seeing in the dark.\n "
    "Harry Potter's secret book had led them to the right place. Ron was brimming with excitement."
]

DTYPE = "float32"
CONTEXT_LENGTH = None
GENERATE_TOKENS = 200
TEMPERATURE = 0.0
TOP_K = 50
TOP_P = 0.95
REPETITION_PENALTY = 1.2
NO_REPEAT_NGRAM_SIZE = 4


def dtype_from_name(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported DTYPE: {name!r}")


def load_checkpoint(checkpoint_path: Path) -> dict[str, object]:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"{checkpoint_path} does not exist. Run examples/mamba_experiment.py "
            "first, or update CHECKPOINT_PATH at the top of this file."
        )

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint object: {type(checkpoint).__name__}")

    required_keys = {"model_config", "model_state_dict", "tokenizer", "context"}
    missing_keys = required_keys.difference(checkpoint)
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(f"Checkpoint is missing required keys: {missing}")

    return checkpoint


def load_trained_lm(
    checkpoint_path: Path,
    device: str,
    dtype: torch.dtype,
) -> tuple[torch.nn.Module, dict[str, object]]:
    checkpoint = load_checkpoint(checkpoint_path)
    config = MambaConfig(**checkpoint["model_config"])
    model = MambaLMHeadModel(config, device=device, dtype=dtype)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


@torch.inference_mode()
def run() -> None:
    if not PROMPTS:
        raise ValueError("Set at least one prompt in PROMPTS before running prediction.")

    require_cuda()
    dtype = dtype_from_name(DTYPE)
    model, checkpoint = load_trained_lm(
        CHECKPOINT_PATH,
        device="cuda",
        dtype=dtype,
    )
    tokenizer_valid_vocab, unit_name, encode_prompt, decode_ids = (
        build_generation_codec(checkpoint["tokenizer"])
    )

    context = int(checkpoint["context"]) if CONTEXT_LENGTH is None else CONTEXT_LENGTH
    valid_vocab = int(checkpoint.get("valid_vocab", tokenizer_valid_vocab))
    training = checkpoint.get("training", {})

    print(f"Loaded checkpoint: {Path(CHECKPOINT_PATH)}", flush=True)
    print(f"Parameters: {parameter_count(model):,}", flush=True)
    print(f"Context: {context:,} {unit_name}", flush=True)
    print(f"Generate: {GENERATE_TOKENS:,} {unit_name}", flush=True)
    print(f"Temperature: {TEMPERATURE:g}", flush=True)
    if isinstance(training, dict):
        checkpoint_step = training.get("checkpoint_step")
        checkpoint_selection = training.get("checkpoint_selection")
        checkpoint_loss = training.get("checkpoint_validation_loss")
        if checkpoint_step is not None and checkpoint_selection is not None:
            loss_text = (
                f", validation loss {checkpoint_loss:.4f}"
                if checkpoint_loss is not None
                else ""
            )
            print(
                f"Checkpoint weights: {checkpoint_selection} "
                f"at step {checkpoint_step}{loss_text}",
                flush=True,
            )
        if (
            training.get("best_step") is not None
            and training.get("best_validation_loss") is not None
        ):
            print(
                f"Best validation loss observed: "
                f"{training['best_validation_loss']:.4f} "
                f"at step {training['best_step']}",
                flush=True,
            )

    for prompt in PROMPTS:
        torch.cuda.synchronize()
        started = time.perf_counter()
        text = generate_sequence(
            model,
            prompt,
            encode_prompt,
            decode_ids,
            valid_vocab,
            context_length=context,
            generate_tokens=GENERATE_TOKENS,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        print(f"\nPrompt:\n{prompt}", flush=True)
        print(f"\nCompleted in {elapsed:.2f} seconds:\n", flush=True)
        print(text, flush=True)


def main() -> None:
    load_env_file()
    run()


if __name__ == "__main__":
    main()
