"""Train an RWKV-7 language model on the existing tokenized corpus files.

This script mirrors the workflow in examples/mamba_experiment.py, but it
instantiates the RWKV-7 implementation vendored from blinkdl/rwkv-lm under
third_party/rwkv-lm/RWKV-v7/train_temp.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time
from types import ModuleType, SimpleNamespace

import torch
import torch.nn.functional as F

EXAMPLES_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXAMPLES_DIR.parent
RWKV_TRAIN_DIR = PROJECT_ROOT / "third_party" / "rwkv-lm" / "RWKV-v7" / "train_temp"
ENV_PATH = PROJECT_ROOT / ".env"

if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from lm_experiment_utils import (  # noqa: E402
    ShuffledWindowSampler,
    apply_no_repeat_ngram,
    apply_repetition_penalty,
    build_generation_codec,
    capture_rng_state,
    cuda_memory_status,
    dtype_from_name,
    effective_data_units,
    format_bytes,
    load_env_file,
    load_tokenized_corpus,
    make_random_batch,
    memory_status,
    parameter_count,
    require_cuda,
    restore_rng_state,
    tokenizer_metadata,
    validate_tokenizer_match,
)
from experiment_logging import ExperimentMetricsLogger  # noqa: E402


TRAIN_TOKEN_METADATA_PATH = (
    PROJECT_ROOT / "data" / "tokenized" / "mixed_gutenberg_fan_fiction_gpt_neox_20b_train_1.json"
)
VALIDATION_TOKEN_METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "tokenized"
    / "mixed_gutenberg_fan_fiction_gpt_neox_20b_validation.json"
)
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "rwkv7_150m_lm.pt"
RESUME_CHECKPOINT_PATH: Path | None = None

MODEL_DIM = 768
LAYERS = 10
HEAD_SIZE = 64
HEAD_CHUNK = 0
RWKV_KERNEL = "@rwkv3"
RWKV_PRECISION = "bf16"
RWKV_INFERENCE_ONLY = False
RWKV_PREFER_CACHED_EXTENSIONS = False
PARAM_DTYPE = "bfloat16"
AMP_DTYPE = "bfloat16"

BATCH_SIZE = 8
CONTEXT_LENGTH = 1024
STEPS = 250_000
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.1
BETA1 = 0.9
BETA2 = 0.99
ADAM_EPS = 1e-18
GRAD_CLIP_NORM = 1.0
GRADIENT_CHECKPOINTING = False
CORPUS_REPEATS = 1
EVAL_BATCHES = 100
LOG_INTERVAL = 100
CHECKPOINT_INTERVAL = 5000
MIN_STEPS = 10_000
EARLY_STOP_PATIENCE = 200
EARLY_STOP_MIN_DELTA = 1e-6
SAVE_BEST_WEIGHTS_FOR_EARLY_STOPPING = False
SAVE_OPTIMIZER_STATE = False
USE_RWKV_CUDA_CROSS_ENTROPY = True
SEED = 7

PROMPT = "<*> The King of England looked down from his castle. He knew the people were displeased."
GENERATE_TOKENS = 200
TEMPERATURE = 0.0
TOP_K = 50
TOP_P = 0.95
REPETITION_PENALTY = 1.05
NO_REPEAT_NGRAM_SIZE = 4

CHECKPOINT_FORMAT_VERSION = 1
CHECKPOINT_METADATA_FORMAT = "rwkv7_training_checkpoint_metadata_v1"


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def autocast_context():
    # RWKV-7 bf16 kernels require bf16 activations; CUDA autocast promotes
    # LayerNorm outputs to fp32 in this torch build.
    return nullcontext()


def checkpoint_metadata_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_name(f"{checkpoint_path.name}.metadata.json")


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



def copy_state_dict_to_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def maybe_copy_optimizer_state_to_cpu(
    optimizer: torch.optim.Optimizer,
) -> dict[str, object] | None:
    if not SAVE_OPTIMIZER_STATE:
        return None
    return tensor_tree_to_cpu(optimizer.state_dict())  # type: ignore[return-value]


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
    if checkpoint.get("architecture") != "rwkv7":
        raise ValueError(
            f"Checkpoint architecture is {checkpoint.get('architecture')!r}, "
            "expected 'rwkv7'."
        )
    if "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint is missing model_state_dict")
    return checkpoint


def checkpoint_training_step(checkpoint: dict[str, object]) -> int:
    training = checkpoint.get("training")
    if isinstance(training, dict):
        return int(training.get("global_step", training.get("checkpoint_step", 0)))
    return int(checkpoint.get("step", 0))


def prepend_env_path(name: str, path: Path) -> None:
    path_value = str(path)
    current = os.environ.get(name)
    if not current:
        os.environ[name] = path_value
        return

    entries = current.split(os.pathsep)
    if path_value not in entries:
        os.environ[name] = os.pathsep.join([path_value, *entries])


def ninja_safe_overlay_root() -> Path:
    override = os.environ.get("MAMBA_CPP_BUILD_OVERLAY_DIR")
    candidates = [Path(override)] if override else []
    if os.name == "posix":
        candidates.append(Path.home() / ".cache" / "mamba_cpp_build_overlays")
        candidates.append(Path("/tmp"))
    else:
        candidates.append(PROJECT_ROOT / ".cache" / "cpp_build_overlays")

    for candidate in candidates:
        if any(character.isspace() for character in str(candidate)):
            continue
        try:
            candidate.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
        return candidate

    raise RuntimeError(
        "Could not create a no-space overlay directory for PyTorch/Ninja builds."
    )


def ninja_safe_build_path(path: Path, prefix: str, description: str) -> Path:
    path = path.resolve()
    if os.name != "posix" or not any(character.isspace() for character in str(path)):
        return path

    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:12]
    link_path = ninja_safe_overlay_root() / f"{prefix}-{digest}"
    if link_path.exists() or link_path.is_symlink():
        if link_path.resolve() == path:
            return link_path
        raise RuntimeError(
            f"Cannot use {description}={path} because the path contains spaces "
            f"and {link_path} already points somewhere else."
        )

    try:
        link_path.symlink_to(path, target_is_directory=True)
    except OSError as exc:
        raise RuntimeError(
            f"Cannot use {description}={path} because the path contains spaces "
            "and creating a no-space symlink for PyTorch/Ninja failed. Move the "
            "project to a path without spaces."
        ) from exc
    return link_path


def ninja_safe_cuda_home(cuda_home: Path) -> Path:
    cuda_home = cuda_home.resolve()
    if os.name != "posix":
        return cuda_home

    has_spaces = any(character.isspace() for character in str(cuda_home))
    needs_cudart_alias = (
        not (cuda_home / "lib" / "libcudart.so").exists()
        and any((cuda_home / "lib").glob("libcudart.so.*"))
    )
    if not has_spaces and not needs_cudart_alias:
        return cuda_home

    digest = hashlib.sha1(str(cuda_home).encode("utf-8")).hexdigest()[:12]
    overlay_path = ninja_safe_overlay_root() / f"mamba-cuda-build-{digest}"
    overlay_path.mkdir(parents=True, exist_ok=True)

    for name in ("bin", "include", "nvvm"):
        source = cuda_home / name
        if source.exists():
            ensure_symlink(overlay_path / name, source)

    source_lib = cuda_home / "lib"
    overlay_lib = overlay_path / "lib"
    overlay_lib.mkdir(exist_ok=True)
    for source in source_lib.iterdir():
        ensure_symlink(overlay_lib / source.name, source)
    ensure_versioned_library_alias(overlay_lib, "libcudart.so")
    return overlay_path


def ensure_symlink(link_path: Path, target_path: Path) -> None:
    if link_path.exists() or link_path.is_symlink():
        if link_path.resolve() == target_path.resolve():
            return
        raise RuntimeError(
            f"Cannot use {link_path} for the RWKV CUDA build because it already "
            "points somewhere else."
        )
    link_path.symlink_to(target_path, target_is_directory=target_path.is_dir())


def ensure_versioned_library_alias(lib_dir: Path, unversioned_name: str) -> None:
    unversioned_path = lib_dir / unversioned_name
    if unversioned_path.exists() or unversioned_path.is_symlink():
        return

    matches = sorted(lib_dir.glob(f"{unversioned_name}.*"))
    if not matches:
        return
    unversioned_path.symlink_to(matches[-1])


def ninja_safe_torch_lib_path(torch_lib_path: Path) -> Path:
    return ninja_safe_build_path(
        torch_lib_path,
        "mamba-torch-lib",
        "PyTorch library path",
    )


def require_cuda_cccl_headers(cuda_home: Path) -> None:
    if (cuda_home / "include" / "nv" / "target").exists():
        return

    raise RuntimeError(
        "CUDA CCCL headers are missing from the CUDA toolkit used for the RWKV "
        f"extension build: {cuda_home / 'include' / 'nv' / 'target'} was not "
        "found. Install a CUDA CCCL package matching the WSL venv's "
        "nvidia-cuda-nvcc version."
    )


def configure_torch_cpp_extension_paths(cuda_home: Path) -> None:
    import torch.utils.cpp_extension as cpp_extension

    torch_lib_path = ninja_safe_torch_lib_path(Path(cpp_extension.TORCH_LIB_PATH))
    cpp_extension.CUDA_HOME = str(cuda_home)
    cpp_extension.TORCH_LIB_PATH = str(torch_lib_path)
    prepend_env_path("LD_LIBRARY_PATH", torch_lib_path)


def configure_cuda_home() -> None:
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home and (Path(cuda_home) / "bin" / "nvcc").exists():
        cuda_home_path = ninja_safe_cuda_home(Path(cuda_home))
        require_cuda_cccl_headers(cuda_home_path)
        os.environ["CUDA_HOME"] = str(cuda_home_path)
        prepend_env_path("PATH", cuda_home_path / "bin")
        prepend_env_path("LD_LIBRARY_PATH", cuda_home_path / "lib")
        configure_torch_cpp_extension_paths(cuda_home_path)
        return

    python_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    candidates = [
        Path(sys.prefix)
        / "lib"
        / python_version
        / "site-packages"
        / "nvidia"
        / "cu13",
        PROJECT_ROOT
        / ".venv-wsl"
        / "lib"
        / python_version
        / "site-packages"
        / "nvidia"
        / "cu13",
    ]
    for candidate in candidates:
        if (candidate / "bin" / "nvcc").exists():
            cuda_home_path = ninja_safe_cuda_home(candidate)
            require_cuda_cccl_headers(cuda_home_path)
            os.environ["CUDA_HOME"] = str(cuda_home_path)
            prepend_env_path("PATH", cuda_home_path / "bin")
            prepend_env_path("LD_LIBRARY_PATH", cuda_home_path / "lib")
            configure_torch_cpp_extension_paths(cuda_home_path)
            return

    raise RuntimeError(
        "Could not find nvcc for the RWKV CUDA extension build. Install a CUDA "
        "toolkit in the WSL venv or set CUDA_HOME to a toolkit directory that "
        "contains bin/nvcc."
    )


def configure_rwkv_environment() -> None:
    if not RWKV_TRAIN_DIR.exists():
        raise FileNotFoundError(
            f"{RWKV_TRAIN_DIR} does not exist. Pull in blinkdl/rwkv-lm first."
        )
    if CONTEXT_LENGTH % 16 != 0:
        raise ValueError("RWKV-7 requires CONTEXT_LENGTH to be divisible by 16.")
    if HEAD_SIZE != 64:
        raise ValueError("The vendored RWKV-7 x070 CUDA kernel expects HEAD_SIZE=64.")
    if HEAD_CHUNK != 0:
        raise ValueError(
            "This comparison wrapper expects HEAD_CHUNK=0 because generation "
            "needs logits from model.forward()."
        )
    if GRADIENT_CHECKPOINTING and importlib.util.find_spec("deepspeed") is None:
        raise RuntimeError(
            "GRADIENT_CHECKPOINTING=True requires deepspeed. Install deepspeed "
            "or set GRADIENT_CHECKPOINTING=False."
        )

    configure_cuda_home()
    os.environ["RWKV_MY_TESTING"] = "x070"
    os.environ["RWKV_KERNEL"] = RWKV_KERNEL
    os.environ["RWKV_CTXLEN"] = str(CONTEXT_LENGTH)
    os.environ["RWKV_HEAD_SIZE"] = str(HEAD_SIZE)
    os.environ["RWKV_HEAD_L2WRAP_CE_CHUNK"] = str(HEAD_CHUNK)
    os.environ["RWKV_FLOAT_MODE"] = RWKV_PRECISION
    os.environ["RWKV_JIT_ON"] = "0"
    if RWKV_INFERENCE_ONLY:
        os.environ["RWKV_INFERENCE_ONLY"] = "1"
    else:
        os.environ.pop("RWKV_INFERENCE_ONLY", None)
    if RWKV_PREFER_CACHED_EXTENSIONS:
        os.environ["RWKV_PREFER_CACHED_EXTENSIONS"] = "1"
    else:
        os.environ.pop("RWKV_PREFER_CACHED_EXTENSIONS", None)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.backends.cuda.matmul.allow_tf32 = True


def import_rwkv_model_module() -> ModuleType:
    configure_rwkv_environment()
    original_cwd = Path.cwd()
    train_dir_string = str(RWKV_TRAIN_DIR)
    if train_dir_string not in sys.path:
        sys.path.insert(0, train_dir_string)

    os.chdir(RWKV_TRAIN_DIR)
    try:
        return importlib.import_module("src.model")
    finally:
        os.chdir(original_cwd)


def build_rwkv_args(vocab_size: int) -> SimpleNamespace:
    dim_ffn = int((MODEL_DIM * 3.5) // 32 * 32)
    return SimpleNamespace(
        accelerator="gpu",
        adam_eps=ADAM_EPS,
        betas=(BETA1, BETA2),
        ctx_len=CONTEXT_LENGTH,
        dim_att=MODEL_DIM,
        dim_ffn=dim_ffn,
        grad_cp=1 if GRADIENT_CHECKPOINTING else 0,
        head_size=HEAD_SIZE,
        lr_init=LEARNING_RATE,
        my_testing="x070",
        n_embd=MODEL_DIM,
        n_layer=LAYERS,
        vocab_size=vocab_size,
        weight_decay=WEIGHT_DECAY,
    )


def rwkv_args_to_dict(args: SimpleNamespace) -> dict[str, object]:
    return {
        key: value
        for key, value in vars(args).items()
        if isinstance(value, (str, int, float, bool, tuple))
    }


def validate_resume_checkpoint(
    checkpoint: dict[str, object],
    args: SimpleNamespace,
    tokenizer_meta: dict[str, object],
) -> None:
    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("Resume checkpoint is missing model_config")

    expected = rwkv_args_to_dict(args)
    for key in ("ctx_len", "head_size", "n_embd", "n_layer", "vocab_size"):
        if int(model_config.get(key, -1)) != int(expected[key]):
            raise ValueError(
                f"Resume checkpoint model_config[{key!r}] is "
                f"{model_config.get(key)!r}, expected {expected[key]!r}."
            )

    if checkpoint.get("tokenizer") != tokenizer_meta:
        raise ValueError("Resume checkpoint tokenizer does not match the corpus.")


def build_rwkv_model(
    rwkv_module: ModuleType,
    args: SimpleNamespace,
    resume_checkpoint: dict[str, object] | None,
) -> torch.nn.Module:
    param_dtype = dtype_from_name(PARAM_DTYPE)
    log("Instantiating RWKV-7 model")
    model = rwkv_module.RWKV(args)
    if resume_checkpoint is not None:
        log("Loading model weights from resume checkpoint")
        model.load_state_dict(resume_checkpoint["model_state_dict"])
    else:
        log("Initializing RWKV-7 weights with the official initializer")
        model.load_state_dict(model.generate_init_weight())

    log(f"Moving model to CUDA with parameter dtype {PARAM_DTYPE}")
    return model.to(device="cuda", dtype=param_dtype)


def create_rwkv_optimizer(
    model: torch.nn.Module,
) -> torch.optim.Optimizer:
    decay_names: list[str] = []
    lr_1x_names: list[str] = []
    lr_2x_names: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "att.w0" in name:
            lr_2x_names.append(name)
        elif (
            len(parameter.squeeze().shape) >= 2
            and WEIGHT_DECAY > 0
            and ".weight" in name
        ):
            decay_names.append(name)
        else:
            lr_1x_names.append(name)

    param_dict = {name: parameter for name, parameter in model.named_parameters()}
    groups: list[dict[str, object]] = []
    if lr_1x_names:
        groups.append(
            {
                "params": [param_dict[name] for name in sorted(lr_1x_names)],
                "weight_decay": 0.0,
                "lr": LEARNING_RATE,
            }
        )
    if lr_2x_names:
        groups.append(
            {
                "params": [param_dict[name] for name in sorted(lr_2x_names)],
                "weight_decay": 0.0,
                "lr": LEARNING_RATE * 2,
            }
        )
    if decay_names:
        groups.append(
            {
                "params": [param_dict[name] for name in sorted(decay_names)],
                "weight_decay": WEIGHT_DECAY,
                "lr": LEARNING_RATE,
            }
        )

    print(f"RWKV optimizer groups: {len(lr_1x_names):,} 1x, {len(lr_2x_names):,} 2x, {len(decay_names):,} decay", flush=True)
    return torch.optim.AdamW(
        groups,
        lr=LEARNING_RATE,
        betas=(BETA1, BETA2),
        eps=ADAM_EPS,
    )


def rwkv_training_loss(
    rwkv_module: ModuleType,
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    if USE_RWKV_CUDA_CROSS_ENTROPY and hasattr(rwkv_module, "l2wrap_cross_entropy"):
        return rwkv_module.l2wrap_cross_entropy(logits, targets)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1))


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
            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y.reshape(-1),
            )
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


@torch.inference_mode()
def generate_sequence(
    model: torch.nn.Module,
    prompt: str,
    encode_prompt: Callable[[str], list[int]],
    decode_ids: Callable[[list[int]], str],
    valid_vocab: int,
    context_length: int = CONTEXT_LENGTH,
    generate_tokens: int = GENERATE_TOKENS,
    temperature: float = TEMPERATURE,
    top_k: int = TOP_K,
    top_p: float = TOP_P,
    repetition_penalty: float = REPETITION_PENALTY,
    no_repeat_ngram_size: int = NO_REPEAT_NGRAM_SIZE,
) -> str:
    model.eval()
    if context_length % 16 != 0:
        raise ValueError("RWKV-7 generation context must be divisible by 16.")

    greedy = temperature <= 0
    prompt_ids = encode_prompt(prompt)
    ids = torch.tensor([prompt_ids], dtype=torch.long, device="cuda")
    pad_id = 0
    for _ in range(generate_tokens):
        window = ids[:, -context_length:]
        if window.shape[1] < context_length:
            padding = torch.full(
                (window.shape[0], context_length - window.shape[1]),
                pad_id,
                dtype=torch.long,
                device=window.device,
            )
            window = torch.cat((padding, window), dim=1)

        with autocast_context():
            logits = model(window)[:, -1]
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


def build_checkpoint_metadata(
    checkpoint_path: Path,
    model_config: dict[str, object],
    tokenizer_meta: dict[str, object],
    valid_vocab: int,
    unit_name: str,
    training_state: dict[str, object],
    train_token_count: int,
    validation_token_count: int,
) -> dict[str, object]:
    global_step = int(training_state["global_step"])
    run_step = int(training_state["run_step"])
    tokens_per_optimizer_step = BATCH_SIZE * CONTEXT_LENGTH
    return {
        "format": CHECKPOINT_METADATA_FORMAT,
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "checkpoint_path": str(checkpoint_path),
        "saved_at_unix_seconds": time.time(),
        "saved_at_local_time": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "architecture": "rwkv7",
        "step": {
            "checkpoint": int(training_state["checkpoint_step"]),
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
        "training_corpus": {
            "raw_token_count": train_token_count,
            "effective_token_count": train_token_count * max(1, CORPUS_REPEATS),
            "repeats": max(1, CORPUS_REPEATS),
            "batch_size": BATCH_SIZE,
            "context_length": CONTEXT_LENGTH,
            "tokens_per_optimizer_step": tokens_per_optimizer_step,
        },
        "validation_corpus": {
            "raw_token_count": validation_token_count,
            "source": training_state["validation_source"],
        },
        "paths": {
            "train_token_metadata": training_state["train_token_metadata_path"],
            "validation_token_metadata": training_state[
                "validation_token_metadata_path"
            ],
            "resume_checkpoint": training_state["resume_checkpoint_path"],
            "rwkv_source": str(RWKV_TRAIN_DIR),
        },
        "model": {
            **model_config,
            "valid_vocab": valid_vocab,
            "unit_name": unit_name,
            "parameter_count_estimate": training_state["parameter_count"],
        },
        "precision": {
            "param_dtype": PARAM_DTYPE,
            "amp_dtype": AMP_DTYPE,
            "rwkv_precision": RWKV_PRECISION,
            "rwkv_kernel": RWKV_KERNEL,
            "use_rwkv_cuda_cross_entropy": USE_RWKV_CUDA_CROSS_ENTROPY,
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
    model_config: dict[str, object],
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
    validation_token_count: int,
    elapsed: float,
    rng_state: dict[str, object] | None,
    train_sampler_state: dict[str, object] | None,
    resume_checkpoint_path: Path | None,
    resume_step: int,
    parameters: int,
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
        "parameter_count": parameters,
    }
    checkpoint_metadata = build_checkpoint_metadata(
        checkpoint_path,
        model_config,
        tokenizer_meta,
        valid_vocab,
        unit_name,
        training_state,
        train_token_count,
        validation_token_count,
    )
    checkpoint = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "architecture": "rwkv7",
        "model_config": model_config,
        "model_state_dict": state_dict,
        "optimizer_state_dict": optimizer_state_dict,
        "tokenizer": tokenizer_meta,
        "valid_vocab": valid_vocab,
        "unit_name": unit_name,
        "context": CONTEXT_LENGTH,
        "precision": {
            "param_dtype": PARAM_DTYPE,
            "amp_dtype": AMP_DTYPE,
            "rwkv_precision": RWKV_PRECISION,
            "rwkv_kernel": RWKV_KERNEL,
            "use_rwkv_cuda_cross_entropy": USE_RWKV_CUDA_CROSS_ENTROPY,
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
    vocab_size: int,
    unit_name: str,
    model: torch.nn.Module,
    resume_checkpoint_path: Path | None,
    resume_step: int,
    train_sampler: ShuffledWindowSampler,
) -> None:
    print("\nRWKV-7 LM configuration", flush=True)
    print(f"  Source: {RWKV_TRAIN_DIR}", flush=True)
    print(f"  Train metadata: {TRAIN_TOKEN_METADATA_PATH}", flush=True)
    print(f"  Validation source: {validation_source}", flush=True)
    print(f"  Train tokens: {train_data.numel():,}", flush=True)
    print(f"  Validation tokens: {validation_data.numel():,}", flush=True)
    print(f"  Vocabulary: {vocab_size:,} {unit_name}", flush=True)
    print(f"  Layers: {LAYERS}", flush=True)
    print(f"  Model dim: {MODEL_DIM}", flush=True)
    print(f"  Context length: {CONTEXT_LENGTH}", flush=True)
    print(f"  Batch size: {BATCH_SIZE}", flush=True)
    print(f"  Steps this run: {STEPS:,}", flush=True)
    print(f"  Learning rate: {LEARNING_RATE:g}", flush=True)
    print(f"  Weight decay: {WEIGHT_DECAY:g}", flush=True)
    print(f"  RWKV kernel: {RWKV_KERNEL or 'default'}", flush=True)
    print(f"  Gradient checkpointing: {GRADIENT_CHECKPOINTING}", flush=True)
    print(f"  Resume checkpoint: {resume_checkpoint_path}", flush=True)
    print(f"  Resume global step: {resume_step:,}", flush=True)
    print(f"  Training windows per structured epoch: {train_sampler.window_count:,}", flush=True)
    print(f"  Parameters: {parameter_count(model):,}", flush=True)
    memory = memory_status()
    if memory:
        print(f"  Memory: {memory}", flush=True)
    print(f"  CUDA memory: {cuda_memory_status()}", flush=True)


def train_rwkv7_lm() -> None:
    load_env_file(ENV_PATH)
    require_cuda()
    configure_rwkv_environment()
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    eval_generator = torch.Generator().manual_seed(SEED + 1)
    resume_checkpoint_path = RESUME_CHECKPOINT_PATH
    resume_checkpoint: dict[str, object] | None = None
    resume_step = 0

    log("Starting RWKV-7 LM training")
    log(f"CUDA device: {torch.cuda.get_device_name(0)}")
    memory = memory_status()
    if memory:
        log(f"Initial memory: {memory}")

    log(
        "Importing RWKV-7 module and loading CUDA extensions "
        "(cached checks can take several minutes on /mnt paths)"
    )
    rwkv_module = import_rwkv_model_module()
    log("RWKV-7 module imported")

    train_data, train_metadata, train_array = load_tokenized_corpus(
        TRAIN_TOKEN_METADATA_PATH,
        "train",
        log_fn=log,
    )
    validation_array = None
    validation_metadata_path: Path | None = None
    if VALIDATION_TOKEN_METADATA_PATH.exists():
        validation_data, validation_metadata, validation_array = load_tokenized_corpus(
            VALIDATION_TOKEN_METADATA_PATH,
            "validation",
            log_fn=log,
        )
        validate_tokenizer_match(train_metadata, validation_metadata)
        validation_source = str(VALIDATION_TOKEN_METADATA_PATH)
        validation_metadata_path = VALIDATION_TOKEN_METADATA_PATH
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
    validation_repeats = max(1, CORPUS_REPEATS) if validation_metadata_path else 1
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
        log_fn=log,
    )
    vocab_size = int(train_metadata.get("vocab_size", valid_vocab))
    if vocab_size != valid_vocab:
        log(
            f"Metadata vocabulary size ({vocab_size:,}) differs from tokenizer "
            f"size ({valid_vocab:,}); using metadata value for the model."
        )

    rwkv_args = build_rwkv_args(vocab_size)
    model_config = rwkv_args_to_dict(rwkv_args)
    if resume_checkpoint_path is not None:
        log(f"Loading resume checkpoint: {resume_checkpoint_path}")
        resume_checkpoint = load_training_checkpoint(resume_checkpoint_path)
        validate_resume_checkpoint(resume_checkpoint, rwkv_args, tokenizer_meta)
        resume_step = checkpoint_training_step(resume_checkpoint)

    model = build_rwkv_model(rwkv_module, rwkv_args, resume_checkpoint)
    optimizer = create_rwkv_optimizer(model)
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
        "rwkv7_experiment",
        {
            "model_dim": MODEL_DIM,
            "layers": LAYERS,
            "head_size": HEAD_SIZE,
            "head_chunk": HEAD_CHUNK,
            "rwkv_kernel": RWKV_KERNEL,
            "rwkv_precision": RWKV_PRECISION,
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
            model_config,
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
            validation_data.numel(),
            time.perf_counter() - started,
            capture_rng_state(eval_generator),
            train_sampler.state_dict() if include_train_sampler_state else None,
            resume_checkpoint_path,
            resume_step,
            parameters,
        )

    for run_step in range(1, STEPS + 1):
        global_step = resume_step + run_step
        completed_run_step = run_step
        completed_global_step = global_step
        x, y = train_sampler.next_batch()
        optimizer.zero_grad(set_to_none=True)
        with autocast_context():
            logits = model(x)
            loss = rwkv_training_loss(rwkv_module, logits, y)
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
    print(
        f"Completed {completed_run_step:,} optimizer steps this run "
        f"({completed_global_step:,} global).",
        flush=True,
    )
    print(
        f"Best validation loss observed: {best_validation_loss:.4f} "
        f"at global step {best_step}; final validation loss: "
        f"{last_validation_loss:.4f}",
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
    train_rwkv7_lm()


if __name__ == "__main__":
    main()
