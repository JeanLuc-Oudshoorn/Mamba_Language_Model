"""Train a Mamba language model from pre-tokenized corpus files."""

from collections.abc import Callable
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

EXAMPLES_DIR = Path(__file__).resolve().parent
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from experiment_logging import ExperimentMetricsLogger


PROJECT_ROOT = EXAMPLES_DIR.parent
ENV_PATH = PROJECT_ROOT / ".env"

TRAIN_TOKEN_METADATA_PATH = (
    PROJECT_ROOT / "data" / "tokenized" / "mixed_gutenberg_fan_fiction_gpt_neox_20b_train_1.json"
)
VALIDATION_TOKEN_METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "tokenized"
    / "mixed_gutenberg_fan_fiction_gpt_neox_20b_validation.json"
)
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "mamba2_130m_lm.pt"
# To continue training on another tokenized dataset, update the train/validation
# metadata paths above and point this at a compatible previous checkpoint.
RESUME_CHECKPOINT_PATH: Path | None = None
CHECKPOINT_INTERVAL = 5000
PARAM_DTYPE = "bfloat16"
AMP_DTYPE = "bfloat16"
SAVE_BEST_WEIGHTS_FOR_EARLY_STOPPING = False
SAVE_OPTIMIZER_STATE = False

BLOCK = "mamba2"
MODEL_DIM = 768
LAYERS = 32

BATCH_SIZE = 8
CONTEXT_LENGTH = 1024
STEPS = 250_000
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.1
GRAD_CLIP_NORM = 1.0
CORPUS_REPEATS = 1
EVAL_BATCHES = 100
LOG_INTERVAL = 100
MIN_STEPS = 10_000
EARLY_STOP_PATIENCE = 200
EARLY_STOP_MIN_DELTA = 1e-6
SEED = 7

PROMPT = "<*> The King of England looked down from his castle. He knew the people were displeased."
GENERATE_TOKENS = 200
TEMPERATURE = 0.0
TOP_K = 50
TOP_P = 0.95
REPETITION_PENALTY = 1.05
NO_REPEAT_NGRAM_SIZE = 4

CHECKPOINT_FORMAT_VERSION = 4
TOKENIZED_FORMAT = "mamba_tokenized_corpus_v1"
TOKEN_DTYPE = np.int32
TOKEN_DTYPE_NAME = "int32"


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def format_bytes(byte_count: int) -> str:
    value = float(byte_count)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def memory_status() -> str | None:
    try:
        import psutil
    except ModuleNotFoundError:
        return None

    process = psutil.Process(os.getpid())
    rss = format_bytes(process.memory_info().rss)
    system = psutil.virtual_memory()
    return f"process RSS={rss}, system used={format_bytes(system.used)}"


def load_env_file(path: Path = ENV_PATH) -> bool:
    if not path.exists():
        return False

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value

    return True


def hf_auth_kwargs() -> dict[str, str]:
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    )
    return {"token": token} if token else {}


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this experiment.")


def dtype_from_name(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name!r}")


def autocast_context():
    dtype = dtype_from_name(AMP_DTYPE)
    enabled = dtype in {torch.float16, torch.bfloat16}
    return torch.autocast("cuda", dtype=dtype, enabled=enabled)


def cuda_memory_status() -> str:
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    peak = torch.cuda.max_memory_allocated()
    return (
        f"cuda allocated={format_bytes(allocated)}, "
        f"reserved={format_bytes(reserved)}, peak={format_bytes(peak)}"
    )


def lm_ssm_config(name: str) -> dict[str, object]:
    if name == "mamba1":
        return dict(layer="Mamba1", d_state=16, d_conv=4, expand=2)
    if name == "mamba2":
        return dict(layer="Mamba2", d_state=64, d_conv=4, expand=2, headdim=32)
    if name == "mamba3":
        return dict(layer="Mamba3", d_state=64, expand=2, headdim=32)
    raise ValueError(f"Unknown block: {name}")


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def copy_state_dict_to_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def tensor_tree_to_cpu(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: tensor_tree_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [tensor_tree_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(tensor_tree_to_cpu(item) for item in value)
    return value


def copy_optimizer_state_to_cpu(optimizer: torch.optim.Optimizer) -> dict[str, object]:
    return tensor_tree_to_cpu(optimizer.state_dict())  # type: ignore[return-value]


def maybe_copy_optimizer_state_to_cpu(
    optimizer: torch.optim.Optimizer,
) -> dict[str, object] | None:
    if not SAVE_OPTIMIZER_STATE:
        return None
    return copy_optimizer_state_to_cpu(optimizer)


def capture_rng_state(generator: torch.Generator) -> dict[str, object]:
    return {
        "torch": torch.random.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
        "batch_generator": generator.get_state(),
    }


def restore_rng_state(
    rng_state: object,
    generator: torch.Generator,
) -> None:
    if not isinstance(rng_state, dict):
        return

    torch_state = rng_state.get("torch")
    if isinstance(torch_state, torch.Tensor):
        torch.random.set_rng_state(torch_state.cpu())

    cuda_state = rng_state.get("cuda")
    if isinstance(cuda_state, list) and cuda_state:
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda_state])

    batch_generator_state = rng_state.get("batch_generator")
    if isinstance(batch_generator_state, torch.Tensor):
        generator.set_state(batch_generator_state.cpu())


def load_training_checkpoint(checkpoint_path: Path) -> dict[str, object]:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {checkpoint_path}")

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if not isinstance(checkpoint, dict):
        raise ValueError(
            f"Unsupported checkpoint object: {type(checkpoint).__name__}"
        )

    required_keys = {"model_config", "model_state_dict", "tokenizer"}
    missing_keys = required_keys.difference(checkpoint)
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(f"Checkpoint is missing required keys: {missing}")

    return checkpoint


def checkpoint_training_step(checkpoint: dict[str, object]) -> int:
    training = checkpoint.get("training")
    if not isinstance(training, dict):
        return 0

    for key in ("global_step", "checkpoint_step", "best_step"):
        value = training.get(key)
        if value is not None:
            return int(value)
    return 0


def validate_resume_checkpoint(
    checkpoint: dict[str, object],
    config: MambaConfig,
    tokenizer_meta: dict[str, object],
    vocab_size: int,
) -> None:
    checkpoint_config = checkpoint["model_config"]
    if not isinstance(checkpoint_config, dict):
        raise ValueError("Checkpoint model_config must be a dictionary.")

    expected_config = asdict(config)
    checked_config_keys = (
        "d_model",
        "d_intermediate",
        "n_layer",
        "vocab_size",
        "ssm_cfg",
        "attn_layer_idx",
        "attn_cfg",
        "rms_norm",
        "residual_in_fp32",
        "fused_add_norm",
        "pad_vocab_size_multiple",
        "tie_embeddings",
    )
    mismatches = [
        f"{key}: checkpoint={checkpoint_config.get(key)!r}, current={expected_config[key]!r}"
        for key in checked_config_keys
        if checkpoint_config.get(key) != expected_config[key]
    ]
    if mismatches:
        mismatch_text = "; ".join(mismatches)
        raise ValueError(
            "Resume checkpoint architecture does not match this run. "
            f"{mismatch_text}"
        )

    checkpoint_tokenizer = checkpoint.get("tokenizer")
    if checkpoint_tokenizer != tokenizer_meta:
        raise ValueError(
            "Resume checkpoint tokenizer does not match the current training "
            f"tokens: {checkpoint_tokenizer!r} vs {tokenizer_meta!r}"
        )

    checkpoint_vocab = int(checkpoint_config["vocab_size"])
    if checkpoint_vocab != vocab_size:
        raise ValueError(
            "Resume checkpoint vocabulary size does not match the current "
            f"training tokens: {checkpoint_vocab:,} vs {vocab_size:,}"
        )


def resolve_token_path(metadata_path: Path, metadata: dict[str, object]) -> Path:
    token_path = Path(str(metadata["token_path"]))
    if token_path.is_absolute():
        return token_path
    return metadata_path.parent / token_path


def load_tokenized_corpus(
    metadata_path: Path,
    label: str,
) -> tuple[torch.Tensor, dict[str, object], np.memmap]:
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"{metadata_path} does not exist. Run scripts/tokenize_training_corpus.py "
            "first, or update the metadata path constants at the top of this file."
        )

    log(f"{label}: loading metadata {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format") != TOKENIZED_FORMAT:
        raise ValueError(
            f"{label}: unsupported tokenized format {metadata.get('format')!r}"
        )
    if not metadata.get("complete"):
        raise ValueError(f"{label}: metadata is marked incomplete: {metadata_path}")
    if metadata.get("token_dtype") != TOKEN_DTYPE_NAME:
        raise ValueError(
            f"{label}: expected token dtype {TOKEN_DTYPE_NAME}, got "
            f"{metadata.get('token_dtype')!r}"
        )

    token_count = int(metadata["token_count"])
    token_path = resolve_token_path(metadata_path, metadata)
    expected_bytes = token_count * np.dtype(TOKEN_DTYPE).itemsize
    actual_bytes = token_path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"{label}: token file size mismatch for {token_path}; expected "
            f"{expected_bytes:,} bytes, found {actual_bytes:,} bytes"
        )

    array = np.memmap(token_path, dtype=TOKEN_DTYPE, mode="c", shape=(token_count,))
    tensor = torch.from_numpy(array)
    log(
        f"{label}: mapped {token_count:,} tokens from {token_path} "
        f"({format_bytes(actual_bytes)})"
    )
    return tensor, metadata, array


def tokenizer_metadata(metadata: dict[str, object]) -> dict[str, object]:
    tokenizer = metadata.get("tokenizer")
    if not isinstance(tokenizer, dict):
        raise ValueError("tokenized metadata is missing tokenizer details")
    if tokenizer.get("kind") != "hf":
        raise ValueError(f"unsupported tokenizer kind: {tokenizer.get('kind')!r}")
    return tokenizer


def build_generation_codec(
    tokenizer_meta: dict[str, object],
) -> tuple[int, str, Callable[[str], list[int]], Callable[[list[int]], str]]:
    from transformers import AutoTokenizer

    tokenizer_name = str(tokenizer_meta["name"])
    log(f"Loading tokenizer for prompt/generation: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **hf_auth_kwargs())
    tokenizer_size = len(tokenizer)

    def encode_prompt(prompt: str) -> list[int]:
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if not token_ids:
            token_ids = [tokenizer.eos_token_id or 0]
        return token_ids

    def decode_ids(ids: list[int]) -> str:
        return tokenizer.decode(ids, skip_special_tokens=True)

    return tokenizer_size, "tokens", encode_prompt, decode_ids


def validate_tokenizer_match(
    train_metadata: dict[str, object],
    validation_metadata: dict[str, object],
) -> None:
    train_tokenizer = tokenizer_metadata(train_metadata)
    validation_tokenizer = tokenizer_metadata(validation_metadata)
    if train_tokenizer != validation_tokenizer:
        raise ValueError(
            "training and validation token files were created with different "
            f"tokenizers: {train_tokenizer!r} vs {validation_tokenizer!r}"
        )


def effective_data_units(data: torch.Tensor, repeats: int) -> int:
    return data.numel() * max(1, repeats)


def training_window_count(
    data: torch.Tensor,
    context: int,
    repeats: int = 1,
) -> int:
    total_units = effective_data_units(data, repeats)
    if total_units <= context + 1:
        return 0
    return (total_units - 1) // context


def make_windows_batch(
    data: torch.Tensor,
    window_ids: torch.Tensor,
    context: int,
    repeats: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    _ = repeats
    starts = window_ids.long() * context
    offsets = torch.arange(context + 1)
    indices = (starts.unsqueeze(1) + offsets.unsqueeze(0)) % data.numel()
    batch = data[indices]
    x = batch[:, :-1]
    y = batch[:, 1:]
    return x.long().cuda(non_blocking=True), y.long().cuda(non_blocking=True)


class ShuffledWindowSampler:
    def __init__(
        self,
        data: torch.Tensor,
        batch_size: int,
        context: int,
        repeats: int,
        seed: int,
        state: object | None = None,
    ) -> None:
        self.data = data
        self.batch_size = batch_size
        self.context = context
        self.repeats = max(1, repeats)
        self.window_count = training_window_count(data, context, self.repeats)
        if self.window_count <= 0:
            raise ValueError(
                "Training corpus is too small for the configured context length. "
                "Use a larger corpus, increase CORPUS_REPEATS, or reduce "
                "CONTEXT_LENGTH."
            )

        self.generator = torch.Generator().manual_seed(seed)
        self.epoch = 0
        self.cursor = 0
        self.order = torch.empty(0, dtype=torch.long)
        if state is None:
            self._reshuffle()
        else:
            self.load_state_dict(state)

    def _reshuffle(self) -> None:
        self.order = torch.randperm(self.window_count, generator=self.generator)
        self.cursor = 0

    def load_state_dict(self, state: object) -> None:
        if not isinstance(state, dict):
            raise ValueError("training sampler state is not a dictionary")
        if int(state.get("window_count", -1)) != self.window_count:
            raise ValueError(
                "training sampler window count does not match the current "
                "training corpus"
            )
        if int(state.get("context", -1)) != self.context:
            raise ValueError("training sampler context length changed")
        if int(state.get("repeats", -1)) != self.repeats:
            raise ValueError("training sampler repeat count changed")

        order = state.get("order")
        if not isinstance(order, torch.Tensor):
            raise ValueError("training sampler state is missing its window order")
        if order.numel() != self.window_count:
            raise ValueError("training sampler window order has the wrong length")

        self.order = order.cpu().long()
        self.epoch = int(state.get("epoch", 0))
        self.cursor = int(state.get("cursor", 0))
        if not 0 <= self.cursor <= self.window_count:
            raise ValueError("training sampler cursor is outside the window order")

        generator_state = state.get("generator_state")
        if isinstance(generator_state, torch.Tensor):
            self.generator.set_state(generator_state.cpu())

    def state_dict(self) -> dict[str, object]:
        return {
            "format": "shuffled_window_sampler_v1",
            "epoch": self.epoch,
            "cursor": self.cursor,
            "window_count": self.window_count,
            "context": self.context,
            "repeats": self.repeats,
            "batch_size": self.batch_size,
            "order": self.order.detach().cpu().clone(),
            "generator_state": self.generator.get_state(),
        }

    def _next_window_ids(self, count: int) -> torch.Tensor:
        chunks: list[torch.Tensor] = []
        needed = count
        while needed > 0:
            if self.cursor >= self.window_count:
                self.epoch += 1
                self._reshuffle()

            remaining = self.window_count - self.cursor
            take = min(needed, remaining)
            chunks.append(self.order[self.cursor : self.cursor + take])
            self.cursor += take
            needed -= take

        if len(chunks) == 1:
            return chunks[0]
        return torch.cat(chunks)

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        window_ids = self._next_window_ids(self.batch_size)
        return make_windows_batch(
            self.data,
            window_ids,
            self.context,
            repeats=self.repeats,
        )


def make_random_batch(
    data: torch.Tensor,
    batch_size: int,
    context: int,
    generator: torch.Generator,
    repeats: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    repeats = max(1, repeats)
    total_units = effective_data_units(data, repeats)
    if total_units <= context + 1:
        raise ValueError(
            "Corpus is too small for the configured context length. "
            "Use a larger corpus, increase repeats, or reduce CONTEXT_LENGTH."
        )

    starts = torch.randint(
        0,
        total_units - context - 1,
        (batch_size,),
        generator=generator,
    )
    if repeats == 1:
        x = torch.stack(
            [
                data[int(start) : int(start) + context]
                for start in starts
            ]
        )
        y = torch.stack(
            [
                data[int(start) + 1 : int(start) + context + 1]
                for start in starts
            ]
        )
        return x.long().cuda(non_blocking=True), y.long().cuda(non_blocking=True)

    offsets = torch.arange(context + 1)
    indices = (starts.unsqueeze(1) + offsets.unsqueeze(0)) % data.numel()
    batch = data[indices]
    x = batch[:, :-1]
    y = batch[:, 1:]
    return x.long().cuda(non_blocking=True), y.long().cuda(non_blocking=True)


@torch.inference_mode()
def estimate_loss(
    model: torch.nn.Module,
    data: torch.Tensor,
    generator: torch.Generator,
    batches: int,
    repeats: int = 1,
) -> float:
    model.eval()
    losses = []
    for _ in range(batches):
        x, y = make_random_batch(
            data,
            BATCH_SIZE,
            CONTEXT_LENGTH,
            generator,
            repeats=repeats,
        )
        with autocast_context():
            logits = model(x).logits
            loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def apply_repetition_penalty(
    logits: torch.Tensor,
    ids: torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    if penalty <= 1.0:
        return logits

    for batch_index in range(logits.shape[0]):
        token_ids = set(ids[batch_index].tolist())
        for token_id in token_ids:
            if token_id >= logits.shape[-1]:
                continue
            score = logits[batch_index, token_id]
            logits[batch_index, token_id] = (
                score * penalty if score < 0 else score / penalty
            )
    return logits


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


def apply_no_repeat_ngram(
    logits: torch.Tensor,
    ids: torch.Tensor,
    ngram_size: int,
) -> torch.Tensor:
    if ngram_size <= 1:
        return logits

    for batch_index in range(logits.shape[0]):
        banned_tokens = banned_ngram_tokens(ids[batch_index].tolist(), ngram_size)
        if banned_tokens:
            valid_banned_tokens = [
                token_id for token_id in banned_tokens if token_id < logits.shape[-1]
            ]
            if valid_banned_tokens:
                logits[batch_index, valid_banned_tokens] = -1e9
    return logits


@torch.inference_mode()
def generate_sequence(
    model: torch.nn.Module,
    prompt: str,
    encode_prompt: Callable[[str], list[int]],
    decode_ids: Callable[[list[int]], str],
    valid_vocab: int,
    context_length: int | None = None,
    generate_tokens: int | None = None,
    temperature: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float | None = None,
    no_repeat_ngram_size: int | None = None,
) -> str:
    model.eval()
    context_length = CONTEXT_LENGTH if context_length is None else context_length
    generate_tokens = GENERATE_TOKENS if generate_tokens is None else generate_tokens
    temperature = TEMPERATURE if temperature is None else temperature
    top_k = TOP_K if top_k is None else top_k
    top_p = TOP_P if top_p is None else top_p
    repetition_penalty = (
        REPETITION_PENALTY if repetition_penalty is None else repetition_penalty
    )
    no_repeat_ngram_size = (
        NO_REPEAT_NGRAM_SIZE
        if no_repeat_ngram_size is None
        else no_repeat_ngram_size
    )
    greedy = temperature <= 0
    ids = torch.tensor(
        [encode_prompt(prompt)],
        dtype=torch.long,
        device="cuda",
    )
    for _ in range(generate_tokens):
        with autocast_context():
            logits = model(ids[:, -context_length:], num_last_tokens=1).logits[:, -1]
        logits = logits[:, :valid_vocab].float()
        logits = torch.nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9)
        logits = apply_repetition_penalty(logits, ids, repetition_penalty)
        logits = apply_no_repeat_ngram(logits, ids, no_repeat_ngram_size)
        if greedy:
            next_id = logits.argmax(dim=-1, keepdim=True)
            ids = torch.cat((ids, next_id), dim=1)
            continue

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
        prob_sum = probabilities.sum(dim=-1, keepdim=True)
        probabilities = torch.where(
            prob_sum > 0,
            probabilities / prob_sum.clamp_min(1e-12),
            torch.full_like(probabilities, 1.0 / probabilities.shape[-1]),
        )
        choice = torch.multinomial(probabilities.cpu(), num_samples=1).to(
            device=sorted_positions.device
        )
        top_choice = sorted_positions.gather(-1, choice)
        next_id = top_indices.gather(-1, top_choice)
        ids = torch.cat((ids, next_id), dim=1)
    return decode_ids(ids[0].tolist())


def checkpoint_metadata_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_name(f"{checkpoint_path.name}.metadata.json")


def train_corpus_progress_metadata(
    train_token_count: int,
    train_repeats: int,
    train_sampler_state: dict[str, object] | None,
) -> dict[str, object]:
    repeats = max(1, train_repeats)
    effective_token_count = train_token_count * repeats
    metadata: dict[str, object] = {
        "raw_token_count": train_token_count,
        "effective_token_count": effective_token_count,
        "repeats": repeats,
        "batch_size": BATCH_SIZE,
        "context_length": CONTEXT_LENGTH,
        "tokens_per_optimizer_step": BATCH_SIZE * CONTEXT_LENGTH,
    }

    if train_sampler_state is None:
        metadata["progress"] = {
            "source": "unavailable",
            "reason": "no training sampler state was saved for this checkpoint",
        }
        return metadata

    context = int(train_sampler_state.get("context", CONTEXT_LENGTH))
    window_count = int(train_sampler_state.get("window_count", 0))
    epoch = int(train_sampler_state.get("epoch", 0))
    cursor = int(train_sampler_state.get("cursor", 0))
    bounded_cursor = min(max(cursor, 0), max(window_count, 0))
    completed_windows = max(epoch, 0) * max(window_count, 0) + bounded_cursor
    scheduled_tokens_seen = completed_windows * context
    structured_epoch_tokens = max(window_count, 0) * context
    current_epoch_tokens_seen = bounded_cursor * context

    metadata["progress"] = {
        "source": "train_sampler_state",
        "completed_structured_epochs": epoch,
        "current_structured_epoch_window": cursor,
        "structured_epoch_window_count": window_count,
        "current_structured_epoch_fraction_seen": (
            bounded_cursor / window_count if window_count > 0 else 0.0
        ),
        "current_structured_epoch_tokens_seen": current_epoch_tokens_seen,
        "structured_epoch_tokens": structured_epoch_tokens,
        "scheduled_windows_seen": completed_windows,
        "scheduled_tokens_seen": scheduled_tokens_seen,
        "effective_corpus_passes_seen": (
            scheduled_tokens_seen / effective_token_count
            if effective_token_count > 0
            else 0.0
        ),
    }
    return metadata


def build_checkpoint_metadata(
    checkpoint_path: Path,
    config: MambaConfig,
    tokenizer_meta: dict[str, object],
    valid_vocab: int,
    unit_name: str,
    training_state: dict[str, object],
    train_token_count: int,
    train_repeats: int,
    validation_token_count: int,
    validation_repeats: int,
    train_sampler_state: dict[str, object] | None,
) -> dict[str, object]:
    global_step = int(training_state["global_step"])
    run_step = int(training_state["run_step"])
    checkpoint_step = int(training_state["checkpoint_step"])
    tokens_per_optimizer_step = BATCH_SIZE * CONTEXT_LENGTH
    config_dict = asdict(config)
    return {
        "format": "mamba_training_checkpoint_metadata_v1",
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "checkpoint_path": str(checkpoint_path),
        "saved_at_unix_seconds": time.time(),
        "saved_at_local_time": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "step": {
            "checkpoint": checkpoint_step,
            "global": global_step,
            "run": run_step,
            "resume_global": int(training_state["resume_step"]),
            "selection": training_state["checkpoint_selection"],
        },
        "optimizer_progress": {
            "tokens_per_optimizer_step": tokens_per_optimizer_step,
            "token_exposures_this_run": run_step * tokens_per_optimizer_step,
            "token_exposures_from_global_step_estimate": (
                global_step * tokens_per_optimizer_step
            ),
        },
        "loss": {
            "checkpoint_validation_loss": training_state[
                "checkpoint_validation_loss"
            ],
            "best_validation_loss": training_state["best_validation_loss"],
            "best_step": training_state["best_step"],
        },
        "training_corpus": train_corpus_progress_metadata(
            train_token_count,
            train_repeats,
            train_sampler_state,
        ),
        "validation_corpus": {
            "raw_token_count": validation_token_count,
            "effective_token_count": validation_token_count
            * max(1, validation_repeats),
            "repeats": max(1, validation_repeats),
            "source": training_state["validation_source"],
        },
        "paths": {
            "train_token_metadata": training_state["train_token_metadata_path"],
            "validation_token_metadata": training_state[
                "validation_token_metadata_path"
            ],
            "resume_checkpoint": training_state["resume_checkpoint_path"],
        },
        "model": {
            "block": BLOCK,
            "d_model": config_dict["d_model"],
            "n_layer": config_dict["n_layer"],
            "vocab_size": config_dict["vocab_size"],
            "valid_vocab": valid_vocab,
            "unit_name": unit_name,
            "context_length": CONTEXT_LENGTH,
        },
        "precision": {
            "param_dtype": PARAM_DTYPE,
            "amp_dtype": AMP_DTYPE,
            "save_optimizer_state": SAVE_OPTIMIZER_STATE,
            "save_best_weights_for_early_stopping": (
                SAVE_BEST_WEIGHTS_FOR_EARLY_STOPPING
            ),
        },
        "tokenizer": tokenizer_meta,
        "training": training_state,
    }


def write_checkpoint_metadata(
    checkpoint_path: Path,
    metadata: dict[str, object],
) -> Path:
    metadata_path = checkpoint_metadata_path(checkpoint_path)
    temp_metadata_path = metadata_path.with_name(f"{metadata_path.name}.tmp")
    temp_metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp_metadata_path.replace(metadata_path)
    return metadata_path


def save_training_checkpoint(
    checkpoint_path: Path,
    state_dict: dict[str, torch.Tensor],
    optimizer_state_dict: dict[str, object] | None,
    config: MambaConfig,
    tokenizer_meta: dict[str, object],
    valid_vocab: int,
    unit_name: str,
    best_validation_loss: float,
    best_step: int,
    checkpoint_validation_loss: float,
    checkpoint_step: int,
    global_step: int,
    run_step: int,
    checkpoint_selection: str,
    train_metadata_path: Path,
    validation_metadata_path: Path | None,
    validation_source: str,
    train_token_count: int,
    train_repeats: int,
    validation_token_count: int,
    validation_repeats: int,
    elapsed: float,
    rng_state: dict[str, object] | None,
    train_sampler_state: dict[str, object] | None,
    resume_checkpoint_path: Path | None,
    resume_step: int,
) -> Path:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    training_state = {
        "best_step": best_step,
        "best_validation_loss": best_validation_loss,
        "checkpoint_step": checkpoint_step,
        "global_step": global_step,
        "run_step": run_step,
        "resume_step": resume_step,
        "checkpoint_validation_loss": checkpoint_validation_loss,
        "checkpoint_selection": checkpoint_selection,
        "elapsed_seconds": elapsed,
        "train_token_metadata_path": str(train_metadata_path),
        "validation_token_metadata_path": (
            str(validation_metadata_path) if validation_metadata_path else None
        ),
        "validation_source": validation_source,
        "corpus_repeats": max(1, CORPUS_REPEATS),
        "steps_requested": STEPS,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "resume_checkpoint_path": (
            str(resume_checkpoint_path) if resume_checkpoint_path else None
        ),
        "seed": SEED,
    }
    checkpoint_metadata = build_checkpoint_metadata(
        checkpoint_path,
        config,
        tokenizer_meta,
        valid_vocab,
        unit_name,
        training_state,
        train_token_count,
        train_repeats,
        validation_token_count,
        validation_repeats,
        train_sampler_state,
    )
    checkpoint = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_config": asdict(config),
        "model_state_dict": state_dict,
        "optimizer_state_dict": optimizer_state_dict,
        "tokenizer": tokenizer_meta,
        "valid_vocab": valid_vocab,
        "unit_name": unit_name,
        "context": CONTEXT_LENGTH,
        "block": BLOCK,
        "precision": {
            "param_dtype": PARAM_DTYPE,
            "amp_dtype": AMP_DTYPE,
            "save_optimizer_state": SAVE_OPTIMIZER_STATE,
            "save_best_weights_for_early_stopping": (
                SAVE_BEST_WEIGHTS_FOR_EARLY_STOPPING
            ),
        },
        "training": training_state,
        "metadata": checkpoint_metadata,
        "rng_state": rng_state,
        "train_sampler_state": train_sampler_state,
        "generation_defaults": {
            "generate": GENERATE_TOKENS,
            "temperature": TEMPERATURE,
            "top_k": TOP_K,
            "top_p": TOP_P,
            "repetition_penalty": REPETITION_PENALTY,
            "no_repeat_ngram_size": NO_REPEAT_NGRAM_SIZE,
        },
    }
    temp_checkpoint_path = checkpoint_path.with_name(f"{checkpoint_path.name}.tmp")
    torch.save(checkpoint, temp_checkpoint_path)
    temp_checkpoint_path.replace(checkpoint_path)
    write_checkpoint_metadata(checkpoint_path, checkpoint_metadata)
    return checkpoint_path


def print_training_configuration(
    train_data: torch.Tensor,
    validation_data: torch.Tensor,
    validation_source: str,
    train_repeats: int,
    validation_repeats: int,
    vocab_size: int,
    unit_name: str,
    model: torch.nn.Module,
    resume_checkpoint_path: Path | None,
    resume_step: int,
    train_sampler: ShuffledWindowSampler,
) -> None:
    planned_training_passes = (
        STEPS
        * BATCH_SIZE
        * CONTEXT_LENGTH
        / max(effective_data_units(train_data, train_repeats), 1)
    )
    log("Training configuration")
    print(f"  Block: {BLOCK}", flush=True)
    print(f"  Layers: {LAYERS}", flush=True)
    print(f"  Model dim: {MODEL_DIM}", flush=True)
    print(f"  Parameter dtype: {PARAM_DTYPE}", flush=True)
    print(f"  Autocast dtype: {AMP_DTYPE}", flush=True)
    print(f"  Batch size: {BATCH_SIZE}", flush=True)
    print(f"  Context length: {CONTEXT_LENGTH}", flush=True)
    print(f"  Steps: {STEPS:,}", flush=True)
    if resume_checkpoint_path is not None:
        print(f"  Resume checkpoint: {resume_checkpoint_path}", flush=True)
        print(f"  Resume global step: {resume_step:,}", flush=True)
        print(f"  Target global step: {resume_step + STEPS:,}", flush=True)
    print(f"  Checkpoint interval: {CHECKPOINT_INTERVAL:,}", flush=True)
    print(
        f"  Checkpoint metadata: {checkpoint_metadata_path(CHECKPOINT_PATH)}",
        flush=True,
    )
    print(f"  Save optimizer state: {SAVE_OPTIMIZER_STATE}", flush=True)
    print(
        f"  Keep best early-stop weights in RAM: "
        f"{SAVE_BEST_WEIGHTS_FOR_EARLY_STOPPING}",
        flush=True,
    )
    print(f"  Learning rate: {LEARNING_RATE:g}", flush=True)
    print(f"  Corpus repeats: {CORPUS_REPEATS:,} (virtual)", flush=True)
    print(f"  Training {unit_name}: {train_data.numel():,}", flush=True)
    print(f"  Validation {unit_name}: {validation_data.numel():,}", flush=True)
    print(f"  Validation source: {validation_source}", flush=True)
    print(
        f"  Effective training {unit_name}: "
        f"{effective_data_units(train_data, train_repeats):,}",
        flush=True,
    )
    print(
        f"  Effective validation {unit_name}: "
        f"{effective_data_units(validation_data, validation_repeats):,}",
        flush=True,
    )
    print(
        f"  Training windows per structured epoch: "
        f"{train_sampler.window_count:,}",
        flush=True,
    )
    print(
        f"  Optimizer steps per structured epoch: "
        f"{math.ceil(train_sampler.window_count / BATCH_SIZE):,}",
        flush=True,
    )
    print(
        f"  Current sampler epoch/cursor: "
        f"{train_sampler.epoch:,}/{train_sampler.cursor:,}",
        flush=True,
    )
    print(
        f"  Planned structured training-{unit_name} passes: "
        f"{planned_training_passes:.2f}",
        flush=True,
    )
    print(f"  Vocabulary: {vocab_size:,} {unit_name}", flush=True)
    print(f"  Parameters: {parameter_count(model):,}", flush=True)
    memory = memory_status()
    if memory:
        print(f"  Memory: {memory}", flush=True)
    print(f"  CUDA memory: {cuda_memory_status()}", flush=True)


def train_mamba_lm() -> None:
    load_env_file()
    require_cuda()
    torch.manual_seed(SEED)
    eval_generator = torch.Generator().manual_seed(SEED + 1)
    resume_checkpoint_path = RESUME_CHECKPOINT_PATH
    resume_checkpoint: dict[str, object] | None = None
    resume_step = 0

    log("Starting Mamba LM training")
    log(f"CUDA device: {torch.cuda.get_device_name(0)}")
    memory = memory_status()
    if memory:
        log(f"Initial memory: {memory}")

    train_data, train_metadata, train_array = load_tokenized_corpus(
        TRAIN_TOKEN_METADATA_PATH,
        "train",
    )
    validation_array = None
    validation_metadata_path: Path | None = None
    if VALIDATION_TOKEN_METADATA_PATH.exists():
        validation_data, validation_metadata, validation_array = load_tokenized_corpus(
            VALIDATION_TOKEN_METADATA_PATH,
            "validation",
        )
        validate_tokenizer_match(train_metadata, validation_metadata)
        validation_source = str(VALIDATION_TOKEN_METADATA_PATH)
        validation_metadata_path = VALIDATION_TOKEN_METADATA_PATH
        train_repeats = max(1, CORPUS_REPEATS)
        validation_repeats = max(1, CORPUS_REPEATS)
    else:
        log(
            "validation: tokenized validation metadata not found; using final 10% "
            "of training tokens as a fallback"
        )
        split = int(0.9 * train_data.numel())
        validation_data = train_data[split:]
        train_data = train_data[:split]
        validation_source = "last 10% of tokenized training corpus (fallback)"
        train_repeats = max(1, CORPUS_REPEATS)
        validation_repeats = 1

    if (
        effective_data_units(train_data, train_repeats) <= CONTEXT_LENGTH + 1
        or effective_data_units(validation_data, validation_repeats)
        <= CONTEXT_LENGTH + 1
    ):
        raise ValueError(
            "Training or validation token data is too small for CONTEXT_LENGTH."
        )

    tokenizer_meta = tokenizer_metadata(train_metadata)
    valid_vocab, unit_name, encode_prompt, decode_ids = build_generation_codec(
        tokenizer_meta,
    )
    vocab_size = int(train_metadata.get("vocab_size", valid_vocab))
    if vocab_size != valid_vocab:
        log(
            f"Metadata vocabulary size ({vocab_size:,}) differs from tokenizer "
            f"size ({valid_vocab:,}); using metadata value for the model."
        )

    log("Building model configuration")
    config = MambaConfig(
        d_model=MODEL_DIM,
        n_layer=LAYERS,
        vocab_size=vocab_size,
        ssm_cfg=lm_ssm_config(BLOCK),
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        pad_vocab_size_multiple=8,
    )

    if resume_checkpoint_path is not None:
        log(f"Loading resume checkpoint: {resume_checkpoint_path}")
        resume_checkpoint = load_training_checkpoint(resume_checkpoint_path)
        validate_resume_checkpoint(
            resume_checkpoint,
            config,
            tokenizer_meta,
            vocab_size,
        )
        resume_step = checkpoint_training_step(resume_checkpoint)

    param_dtype = dtype_from_name(PARAM_DTYPE)
    log(f"Allocating model on CUDA with parameter dtype {PARAM_DTYPE}")
    model = MambaLMHeadModel(config, device="cuda", dtype=param_dtype)
    if resume_checkpoint is not None:
        log("Loading model weights from resume checkpoint")
        model.load_state_dict(resume_checkpoint["model_state_dict"])

    log("Creating optimizer")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    if resume_checkpoint is not None:
        optimizer_state_dict = resume_checkpoint.get("optimizer_state_dict")
        if isinstance(optimizer_state_dict, dict):
            log("Loading optimizer state from resume checkpoint")
            optimizer.load_state_dict(optimizer_state_dict)
            for state in optimizer.state.values():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to("cuda")
        else:
            log("Resume checkpoint has no optimizer state; optimizer starts fresh")
        restore_rng_state(resume_checkpoint.get("rng_state"), eval_generator)

    train_sampler_state = (
        resume_checkpoint.get("train_sampler_state")
        if resume_checkpoint is not None
        else None
    )
    if resume_checkpoint is not None and train_sampler_state is not None:
        checkpoint_training = resume_checkpoint.get("training")
        checkpoint_train_metadata_path = (
            checkpoint_training.get("train_token_metadata_path")
            if isinstance(checkpoint_training, dict)
            else None
        )
        if checkpoint_train_metadata_path != str(TRAIN_TOKEN_METADATA_PATH):
            log(
                "Resume checkpoint uses a different training token metadata "
                "path; starting a fresh training sampler for this corpus"
            )
            train_sampler_state = None

    try:
        train_sampler = ShuffledWindowSampler(
            train_data,
            batch_size=BATCH_SIZE,
            context=CONTEXT_LENGTH,
            repeats=train_repeats,
            seed=SEED,
            state=train_sampler_state,
        )
    except ValueError as exc:
        log(f"Starting a fresh training sampler: {exc}")
        train_sampler = ShuffledWindowSampler(
            train_data,
            batch_size=BATCH_SIZE,
            context=CONTEXT_LENGTH,
            repeats=train_repeats,
            seed=SEED,
        )

    parameters = parameter_count(model)
    metrics_logger = ExperimentMetricsLogger(
        PROJECT_ROOT,
        "mamba_experiment",
        {
            "block": BLOCK,
            "model_dim": MODEL_DIM,
            "layers": LAYERS,
            "vocab_size": vocab_size,
            "valid_vocab": valid_vocab,
            "unit_name": unit_name,
            "batch_size": BATCH_SIZE,
            "context_length": CONTEXT_LENGTH,
            "steps": STEPS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "eval_batches": EVAL_BATCHES,
            "log_interval": LOG_INTERVAL,
            "checkpoint_interval": CHECKPOINT_INTERVAL,
            "checkpoint_path": CHECKPOINT_PATH,
            "resume_checkpoint_path": resume_checkpoint_path,
            "train_token_metadata_path": TRAIN_TOKEN_METADATA_PATH,
            "validation_source": validation_source,
            "parameters": parameters,
            "seed": SEED,
        },
    )
    log(f"Metrics log: {metrics_logger.path}")

    print_training_configuration(
        train_data=train_data,
        validation_data=validation_data,
        validation_source=validation_source,
        train_repeats=train_repeats,
        validation_repeats=validation_repeats,
        vocab_size=vocab_size,
        unit_name=unit_name,
        model=model,
        resume_checkpoint_path=resume_checkpoint_path,
        resume_step=resume_step,
        train_sampler=train_sampler,
    )

    log(f"Estimating initial validation loss over {EVAL_BATCHES:,} batches")
    initial_loss = estimate_loss(
        model,
        validation_data,
        eval_generator,
        batches=EVAL_BATCHES,
        repeats=validation_repeats,
    )
    print(f"Initial validation loss: {initial_loss:.4f}", flush=True)

    log("Beginning optimizer steps")
    model.train()
    started = time.perf_counter()
    best_validation_loss = initial_loss
    best_step = resume_step
    last_validation_loss = initial_loss
    metrics_logger.write_metrics(
        run_step=0,
        global_step=resume_step,
        elapsed_seconds=0.0,
        train_loss=None,
        validation_loss=initial_loss,
        perplexity=math.exp(min(initial_loss, 20)),
        memory=memory_status(),
        cuda_memory=cuda_memory_status(),
        best_validation_loss=best_validation_loss,
        best_step=best_step,
    )
    completed_run_step = 0
    completed_global_step = resume_step
    early_stopping_enabled = EARLY_STOP_PATIENCE > 0
    best_state_dict = (
        copy_state_dict_to_cpu(model)
        if early_stopping_enabled and SAVE_BEST_WEIGHTS_FOR_EARLY_STOPPING
        else None
    )
    checks_without_improvement = 0
    stopped_early = False

    def write_checkpoint(
        state_dict: dict[str, torch.Tensor],
        optimizer_state_dict: dict[str, object] | None,
        checkpoint_validation_loss: float,
        checkpoint_step: int,
        global_step: int,
        run_step: int,
        checkpoint_selection: str,
        include_train_sampler_state: bool = True,
    ) -> Path:
        return save_training_checkpoint(
            CHECKPOINT_PATH,
            state_dict,
            optimizer_state_dict,
            config,
            tokenizer_meta,
            valid_vocab,
            unit_name,
            best_validation_loss,
            best_step,
            checkpoint_validation_loss,
            checkpoint_step,
            global_step,
            run_step,
            checkpoint_selection,
            TRAIN_TOKEN_METADATA_PATH,
            validation_metadata_path,
            validation_source,
            train_data.numel(),
            train_repeats,
            validation_data.numel(),
            validation_repeats,
            time.perf_counter() - started,
            capture_rng_state(eval_generator),
            train_sampler.state_dict() if include_train_sampler_state else None,
            resume_checkpoint_path,
            resume_step,
        )

    for run_step in range(1, STEPS + 1):
        global_step = resume_step + run_step
        completed_run_step = run_step
        completed_global_step = global_step
        x, y = train_sampler.next_batch()
        optimizer.zero_grad(set_to_none=True)
        with autocast_context():
            logits = model(x).logits
            loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()

        if run_step == 1 or run_step % LOG_INTERVAL == 0 or run_step == STEPS:
            validation_loss = estimate_loss(
                model,
                validation_data,
                eval_generator,
                batches=EVAL_BATCHES,
                repeats=validation_repeats,
            )
            last_validation_loss = validation_loss
            perplexity = math.exp(min(validation_loss, 20))
            cuda_memory = cuda_memory_status()
            print(
                f"step {run_step:>4}/{STEPS} "
                f"(global {global_step:,}): train loss={loss.item():.4f}, "
                f"validation loss={validation_loss:.4f}, "
                f"perplexity={perplexity:.2f}, {cuda_memory}",
                flush=True,
            )
            if validation_loss < best_validation_loss - EARLY_STOP_MIN_DELTA:
                best_validation_loss = validation_loss
                best_step = global_step
                if early_stopping_enabled and SAVE_BEST_WEIGHTS_FOR_EARLY_STOPPING:
                    best_state_dict = copy_state_dict_to_cpu(model)
                checks_without_improvement = 0
            elif early_stopping_enabled:
                checks_without_improvement += 1
                if (
                    run_step >= MIN_STEPS
                    and checks_without_improvement >= EARLY_STOP_PATIENCE
                ):
                    print(
                        "Early stopping: validation loss did not improve by at "
                        f"least {EARLY_STOP_MIN_DELTA:g} for "
                        f"{EARLY_STOP_PATIENCE} validation checks. "
                        f"Best validation loss was {best_validation_loss:.4f} "
                        f"at global step {best_step}.",
                        flush=True,
                    )
                    stopped_early = True
            metrics_logger.write_metrics(
                run_step=run_step,
                global_step=global_step,
                elapsed_seconds=time.perf_counter() - started,
                train_loss=float(loss.detach().cpu()),
                validation_loss=validation_loss,
                perplexity=perplexity,
                memory=memory_status(),
                cuda_memory=cuda_memory,
                best_validation_loss=best_validation_loss,
                best_step=best_step,
            )
            if stopped_early:
                break

        if (
            CHECKPOINT_INTERVAL > 0
            and run_step % CHECKPOINT_INTERVAL == 0
            and run_step != STEPS
        ):
            log(f"Saving resumable checkpoint at global step {global_step:,}")
            write_checkpoint(
                copy_state_dict_to_cpu(model),
                maybe_copy_optimizer_state_to_cpu(optimizer),
                last_validation_loss,
                global_step,
                global_step,
                run_step,
                "latest_step",
            )
            model.train()

    elapsed = time.perf_counter() - started
    if stopped_early:
        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)
            checkpoint_state_dict = best_state_dict
            checkpoint_global_step = best_step
            checkpoint_selection = "best_validation_loss_after_early_stopping"
        else:
            checkpoint_state_dict = copy_state_dict_to_cpu(model)
            checkpoint_global_step = completed_global_step
            checkpoint_selection = "early_stopping_final_step"
        checkpoint_optimizer_state_dict = None
        checkpoint_validation_loss = last_validation_loss
        checkpoint_step = checkpoint_global_step
    else:
        checkpoint_state_dict = copy_state_dict_to_cpu(model)
        checkpoint_optimizer_state_dict = maybe_copy_optimizer_state_to_cpu(optimizer)
        checkpoint_validation_loss = last_validation_loss
        checkpoint_step = completed_global_step
        checkpoint_global_step = completed_global_step
        checkpoint_selection = "final_step"

    model.eval()
    log("Saving checkpoint")
    include_checkpoint_sampler_state = (
        checkpoint_selection != "best_validation_loss_after_early_stopping"
    )
    checkpoint_path = write_checkpoint(
        checkpoint_state_dict,
        checkpoint_optimizer_state_dict,
        checkpoint_validation_loss,
        checkpoint_step,
        checkpoint_global_step,
        max(0, checkpoint_global_step - resume_step),
        checkpoint_selection,
        include_train_sampler_state=include_checkpoint_sampler_state,
    )

    log("Generating sample")
    sample = generate_sequence(
        model,
        PROMPT,
        encode_prompt,
        decode_ids,
        valid_vocab,
    )
    print(f"\nTraining time: {elapsed:.1f} seconds", flush=True)
    if stopped_early:
        if SAVE_BEST_WEIGHTS_FOR_EARLY_STOPPING:
            print(
                f"Restored best model weights from global step {best_step} "
                "before generation.",
                flush=True,
            )
        else:
            print(
                "Early stopping triggered. Best weights were not kept in RAM, "
                "so generation uses current final-step weights.",
                flush=True,
            )
    else:
        print(
            f"Completed {completed_run_step:,} optimizer steps this run "
            f"({completed_global_step:,} global). "
            "Using final-step weights for generation and checkpointing.",
            flush=True,
        )
        print(
            f"Best validation loss observed: {best_validation_loss:.4f} "
            f"at global step {best_step}; "
            f"final validation loss: {last_validation_loss:.4f}",
            flush=True,
        )
    print(f"Saved checkpoint: {checkpoint_path}", flush=True)
    print(
        f"Saved checkpoint metadata: {checkpoint_metadata_path(checkpoint_path)}",
        flush=True,
    )
    print("\nGenerated sample:\n", flush=True)
    print(sample, flush=True)
    metrics_logger.write_event(
        "run_complete",
        completed_run_step=completed_run_step,
        completed_global_step=completed_global_step,
        elapsed_seconds=elapsed,
        best_validation_loss=best_validation_loss,
        best_step=best_step,
        final_validation_loss=last_validation_loss,
        checkpoint_path=checkpoint_path,
    )
    metrics_logger.close()

    _ = train_array, validation_array


def main() -> None:
    train_mamba_lm()


if __name__ == "__main__":
    main()
