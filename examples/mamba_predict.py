"""Load a trained Mamba checkpoint and complete configured prompts."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
import os
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F

EXAMPLES_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXAMPLES_DIR.parent
ENV_PATH = PROJECT_ROOT / ".env"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from lm_experiment_utils import (  # noqa: E402
    build_generation_codec,
    dtype_from_name,
    load_env_file,
    parameter_count,
    require_cuda,
)


CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "mamba2_130m_lm.pt"
PROMPTS = [
    "<*> Chapter 2: Seeing in the dark.\n "
    "Harry Potter's secret book had led them to the right place. Ron was brimming with excitement."
]

DTYPE = "float32"
AMP_DTYPE = "bfloat16"
CONTEXT_LENGTH = None
GENERATE_TOKENS = 200
USE_INFERENCE_CACHE = True
TEMPERATURE = 0.0
TOP_K = 50
TOP_P = 0.95
REPETITION_PENALTY = 1.2
NO_REPEAT_NGRAM_SIZE = 4


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def autocast_context():
    dtype = dtype_from_name(AMP_DTYPE)
    enabled = dtype in {torch.float16, torch.bfloat16}
    return torch.autocast("cuda", dtype=dtype, enabled=enabled)


def active_forward_dtype(model: torch.nn.Module) -> torch.dtype:
    amp_dtype = dtype_from_name(AMP_DTYPE)
    if amp_dtype in {torch.float16, torch.bfloat16}:
        return amp_dtype
    return next(model.parameters()).dtype


@contextmanager
def skip_default_parameter_initialization():
    init_names = (
        "constant_",
        "kaiming_uniform_",
        "normal_",
        "ones_",
        "uniform_",
        "zeros_",
    )
    originals = {name: getattr(torch.nn.init, name) for name in init_names}

    def no_op(tensor, *args, **kwargs):
        return tensor

    try:
        for name in init_names:
            setattr(torch.nn.init, name, no_op)
        yield
    finally:
        for name, original in originals.items():
            setattr(torch.nn.init, name, original)


@dataclass
class InferenceParams:
    max_seqlen: int
    max_batch_size: int
    seqlen_offset: int = 0
    batch_size_offset: int = 0
    key_value_memory_dict: dict[int, object] = field(default_factory=dict)
    lengths_per_sample: torch.Tensor | None = None


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
    started = time.perf_counter()
    log(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = load_checkpoint(checkpoint_path)
    log(f"Checkpoint loaded in {time.perf_counter() - started:.1f} seconds")

    started = time.perf_counter()
    log("Importing Mamba model classes")
    os.environ["MAMBA_FAST_PREDICT"] = "1"
    from mamba_ssm.models.config_mamba import MambaConfig
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

    log(f"Mamba model classes imported in {time.perf_counter() - started:.1f} seconds")

    config = MambaConfig(**checkpoint["model_config"])
    state_dict = checkpoint["model_state_dict"]
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint model_state_dict must be a dictionary.")

    started = time.perf_counter()
    log("Instantiating Mamba model without random initialization")
    with skip_default_parameter_initialization():
        model = MambaLMHeadModel(config, dtype=dtype)
    log(f"Mamba model instantiated in {time.perf_counter() - started:.1f} seconds")

    started = time.perf_counter()
    log("Loading checkpoint weights into Mamba model")
    model.load_state_dict(state_dict)
    checkpoint.pop("model_state_dict", None)
    log(f"Checkpoint weights loaded in {time.perf_counter() - started:.1f} seconds")

    started = time.perf_counter()
    log(f"Moving Mamba model to {device}")
    model = model.to(device=device)
    log(f"Mamba model moved to {device} in {time.perf_counter() - started:.1f} seconds")
    model.eval()
    return model, checkpoint


def banned_ngram_tokens(token_ids: list[int], ngram_size: int) -> set[int]:
    if ngram_size <= 1 or len(token_ids) < ngram_size - 1:
        return set()

    prefix = tuple(token_ids[-(ngram_size - 1) :])
    banned = set()
    for index in range(len(token_ids) - ngram_size + 1):
        ngram = tuple(token_ids[index : index + ngram_size])
        if ngram[:-1] == prefix:
            banned.add(ngram[-1])
    return banned


def apply_history_constraints(
    logits: torch.Tensor,
    history_ids: list[int],
    repetition_penalty: float,
    no_repeat_ngram_size: int,
) -> torch.Tensor:
    if repetition_penalty > 1.0:
        token_ids = sorted(
            {token_id for token_id in history_ids if 0 <= token_id < logits.shape[-1]}
        )
        if token_ids:
            token_index = torch.tensor(token_ids, dtype=torch.long, device=logits.device)
            scores = logits[0, token_index]
            logits[0, token_index] = torch.where(
                scores < 0,
                scores * repetition_penalty,
                scores / repetition_penalty,
            )

    if no_repeat_ngram_size > 1:
        banned_tokens = [
            token_id
            for token_id in banned_ngram_tokens(history_ids, no_repeat_ngram_size)
            if 0 <= token_id < logits.shape[-1]
        ]
        if banned_tokens:
            token_index = torch.tensor(
                banned_tokens,
                dtype=torch.long,
                device=logits.device,
            )
            logits[0, token_index] = -1e9

    return logits


def sample_next_token(
    logits: torch.Tensor,
    history_ids: list[int],
    valid_vocab: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
) -> tuple[torch.Tensor, int]:
    greedy = temperature <= 0
    logits = logits[:, :valid_vocab].float()
    logits = torch.nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9)
    logits = apply_history_constraints(
        logits,
        history_ids,
        repetition_penalty,
        no_repeat_ngram_size,
    )
    if greedy:
        next_id = logits.argmax(dim=-1, keepdim=True)
        return next_id, int(next_id.item())

    logits = logits / temperature
    sample_vocab_size = valid_vocab if top_k <= 0 else min(top_k, valid_vocab)
    top_values, top_indices = torch.topk(logits, k=sample_vocab_size)
    sorted_values, sorted_positions = torch.sort(top_values, descending=True)
    if top_p < 1.0:
        sorted_probabilities = F.softmax(sorted_values, dim=-1)
        cumulative_probabilities = sorted_probabilities.cumsum(dim=-1)
        remove = cumulative_probabilities > top_p
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False
        sorted_values = sorted_values.masked_fill(remove, -1e9)

    sorted_values = sorted_values - sorted_values.max(dim=-1, keepdim=True).values
    probabilities = F.softmax(sorted_values, dim=-1)
    probabilities = torch.nan_to_num(
        probabilities,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).clamp_min(0.0)
    probabilities_sum = probabilities.sum(dim=-1, keepdim=True)
    if torch.any(probabilities_sum <= 0):
        next_id = logits.argmax(dim=-1, keepdim=True)
        return next_id, int(next_id.item())

    probabilities = probabilities / probabilities_sum
    sampled_position = torch.multinomial(probabilities, num_samples=1)
    original_position = sorted_positions.gather(1, sampled_position)
    next_id = top_indices.gather(1, original_position)
    return next_id, int(next_id.item())


@torch.inference_mode()
def generate_sequence_full_context(
    model: torch.nn.Module,
    prompt: str,
    encode_prompt: Callable[[str], list[int]],
    decode_ids: Callable[[list[int]], str],
    valid_vocab: int,
    context_length: int,
    generate_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
) -> str:
    model.eval()
    history_ids = encode_prompt(prompt)
    ids = torch.tensor([history_ids], dtype=torch.long, device="cuda")
    for _ in range(generate_tokens):
        with autocast_context():
            logits = model(ids[:, -context_length:], num_last_tokens=1).logits[:, -1]
        next_id, next_id_value = sample_next_token(
            logits,
            history_ids,
            valid_vocab,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            no_repeat_ngram_size,
        )
        history_ids.append(next_id_value)
        ids = torch.cat((ids, next_id), dim=1)
    return decode_ids(ids[0].tolist())


@torch.inference_mode()
def generate_sequence_cached(
    model: torch.nn.Module,
    prompt: str,
    encode_prompt: Callable[[str], list[int]],
    decode_ids: Callable[[list[int]], str],
    valid_vocab: int,
    context_length: int,
    generate_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
) -> str:
    model.eval()
    history_ids = encode_prompt(prompt)
    if generate_tokens <= 0:
        return decode_ids(history_ids)

    model_context_ids = history_ids[-context_length:]
    input_ids = torch.tensor([model_context_ids], dtype=torch.long, device="cuda")
    max_seqlen = input_ids.shape[1] + generate_tokens
    cache = model.allocate_inference_cache(
        batch_size=1,
        max_seqlen=max_seqlen,
        dtype=active_forward_dtype(model),
    )
    inference_params = InferenceParams(
        max_seqlen=max_seqlen,
        max_batch_size=1,
        key_value_memory_dict=cache,
    )

    with autocast_context():
        logits = model(
            input_ids,
            inference_params=inference_params,
            num_last_tokens=1,
        ).logits[:, -1]
    inference_params.seqlen_offset = input_ids.shape[1]

    for token_index in range(generate_tokens):
        next_id, next_id_value = sample_next_token(
            logits,
            history_ids,
            valid_vocab,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            no_repeat_ngram_size,
        )
        history_ids.append(next_id_value)
        if token_index == generate_tokens - 1:
            break

        with autocast_context():
            logits = model(
                next_id,
                inference_params=inference_params,
                num_last_tokens=1,
            ).logits[:, -1]
        inference_params.seqlen_offset += 1

    return decode_ids(history_ids)


@torch.inference_mode()
def generate_sequence(
    model: torch.nn.Module,
    prompt: str,
    encode_prompt: Callable[[str], list[int]],
    decode_ids: Callable[[list[int]], str],
    valid_vocab: int,
    context_length: int,
    generate_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
) -> str:
    if USE_INFERENCE_CACHE and hasattr(model, "allocate_inference_cache"):
        return generate_sequence_cached(
            model,
            prompt,
            encode_prompt,
            decode_ids,
            valid_vocab,
            context_length,
            generate_tokens,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            no_repeat_ngram_size,
        )

    return generate_sequence_full_context(
        model,
        prompt,
        encode_prompt,
        decode_ids,
        valid_vocab,
        context_length,
        generate_tokens,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        no_repeat_ngram_size,
    )


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
        build_generation_codec(checkpoint["tokenizer"], log_fn=log)
    )

    context = int(checkpoint["context"]) if CONTEXT_LENGTH is None else CONTEXT_LENGTH
    valid_vocab = int(checkpoint.get("valid_vocab", tokenizer_valid_vocab))
    training = checkpoint.get("training", {})

    print(f"Loaded checkpoint: {Path(CHECKPOINT_PATH)}", flush=True)
    print(f"Parameters: {parameter_count(model):,}", flush=True)
    print(f"Context: {context:,} {unit_name}", flush=True)
    print(f"Generate: {GENERATE_TOKENS:,} {unit_name}", flush=True)
    print(
        f"Inference cache: {'enabled' if USE_INFERENCE_CACHE else 'disabled'}",
        flush=True,
    )
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
    load_env_file(ENV_PATH)
    run()


if __name__ == "__main__":
    main()
