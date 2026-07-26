"""Train Mamba-2 on adaptive llm-abba pieces from synthetic time series.

This intentionally uses only llm-abba's adaptive compression step. It does not
call ABBA/XABBA digitization, aggregation, k-means, or symbol assignment.
"""

from collections.abc import Callable
from dataclasses import dataclass
import importlib.metadata as importlib_metadata
import importlib.util
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

# Mamba parameters copied locally from mamba_experiment.py.
BLOCK = "mamba2"
MODEL_DIM = 768
LAYERS = 32

BATCH_SIZE = 8
CONTEXT_LENGTH = 1024
STEPS = 500
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.1
GRAD_CLIP_NORM = 1.0
EVAL_BATCHES = 100
LOG_INTERVAL = 100
SEED = 7
PARAM_DTYPE = "bfloat16"
AMP_DTYPE = "bfloat16"

# Mamba-2's fused path requires causal-conv1d. Leave this false on Windows
# unless that package is installed and importable in the active environment.
USE_MEM_EFF_PATH = False

# Synthetic time-series and adaptive llm-abba preprocessing parameters.
TRAIN_SERIES = 512
SERIES_LENGTH = 256
PREFIX_LENGTH = 192
PREDICT_STEPS = 64
ABBA_TOL = 0.15
ABBA_MAX_LEN = 24
INCREMENT_BINS = 512
TEMPERATURE = 0.0

# Printed/plotted example of the adaptive compression step.
SHOW_ADAPTIVE_COMPRESSION_EXAMPLE = True
ADAPTIVE_EXAMPLE_SERIES_LENGTH = 64
ADAPTIVE_EXAMPLE_PLOT_PATH = PROJECT_ROOT / "examples" / "adaptive_compression_example.png"
ADAPTIVE_EXAMPLE_PRINT_PRECISION = 4

BOS_TOKEN = 0
EOS_TOKEN = 1


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def dtype_from_name(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name!r}")


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


@dataclass(frozen=True)
class AdaptivePiece:
    length: int
    increment: float


@dataclass(frozen=True)
class PieceTokenizer:
    max_len: int
    inc_bins: int
    inc_min: float
    inc_max: float

    @property
    def length_offset(self) -> int:
        return 2

    @property
    def increment_offset(self) -> int:
        return self.length_offset + self.max_len

    @property
    def vocab_size(self) -> int:
        return self.increment_offset + self.inc_bins

    def encode_length(self, length: int) -> int:
        length = int(np.clip(length, 1, self.max_len))
        return self.length_offset + length - 1

    def decode_length(self, token_id: int) -> int:
        return int(token_id - self.length_offset + 1)

    def encode_increment(self, increment: float) -> int:
        clipped = float(np.clip(increment, self.inc_min, self.inc_max))
        if self.inc_max <= self.inc_min:
            bin_id = 0
        else:
            position = (clipped - self.inc_min) / (self.inc_max - self.inc_min)
            bin_id = int(np.clip(round(position * (self.inc_bins - 1)), 0, self.inc_bins - 1))
        return self.increment_offset + bin_id

    def decode_increment(self, token_id: int) -> float:
        bin_id = int(np.clip(token_id - self.increment_offset, 0, self.inc_bins - 1))
        if self.inc_bins <= 1:
            return float(self.inc_min)
        position = bin_id / (self.inc_bins - 1)
        return float(self.inc_min + position * (self.inc_max - self.inc_min))

    def encode_pieces(self, pieces: list[AdaptivePiece]) -> list[int]:
        ids = [BOS_TOKEN]
        for piece in pieces:
            ids.append(self.encode_length(piece.length))
            ids.append(self.encode_increment(piece.increment))
        ids.append(EOS_TOKEN)
        return ids

    def decode_body(self, token_ids: list[int]) -> list[AdaptivePiece]:
        pieces: list[AdaptivePiece] = []
        body = [token_id for token_id in token_ids if token_id not in {BOS_TOKEN, EOS_TOKEN}]
        for index in range(0, len(body) - 1, 2):
            length_id = body[index]
            inc_id = body[index + 1]
            if not self.is_length_token(length_id) or not self.is_increment_token(inc_id):
                continue
            pieces.append(
                AdaptivePiece(
                    length=self.decode_length(length_id),
                    increment=self.decode_increment(inc_id),
                )
            )
        return pieces

    def is_length_token(self, token_id: int) -> bool:
        return self.length_offset <= token_id < self.increment_offset

    def is_increment_token(self, token_id: int) -> bool:
        return self.increment_offset <= token_id < self.vocab_size


def load_llmabba_adaptive_functions() -> tuple[Callable[..., list], Callable[..., list], str]:
    """Load llm-abba's small adaptive modules without importing its LLM stack."""

    try:
        distribution = importlib_metadata.distribution("llmabba")
    except importlib_metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "llmabba is not installed. Install the adaptive source with "
            "`python -m pip install llmabba==0.0.5 --no-deps`."
        ) from exc

    def load_module(module_name: str, relative_file: str):
        files = distribution.files or []
        matches = [
            item
            for item in files
            if str(item).replace("\\", "/") == relative_file
        ]
        if not matches:
            raise RuntimeError(f"Could not find {relative_file} in llmabba installation.")
        module_path = distribution.locate_file(matches[0])
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load llmabba module from {module_path}.")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module, str(module_path)

    comp_module, comp_path = load_module("llmabba_adaptive_comp", "llmabba/comp.py")
    inv_module, _ = load_module("llmabba_adaptive_inverse", "llmabba/inverse.py")
    return comp_module.compress, inv_module.inv_compress, comp_path


def generate_synthetic_series(
    count: int,
    length: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, length, dtype=np.float64)
    series = np.empty((count, length), dtype=np.float64)

    for row in range(count):
        seasonal = np.zeros_like(t)
        for _ in range(rng.integers(2, 5)):
            amplitude = rng.uniform(0.2, 1.2)
            frequency = rng.uniform(1.0, 8.0)
            phase = rng.uniform(0.0, 2.0 * math.pi)
            seasonal += amplitude * np.sin(2.0 * math.pi * frequency * t + phase)

        trend = rng.uniform(-1.0, 1.0) * t
        drift = rng.normal(0.0, 0.035, size=length).cumsum()
        shocks = np.zeros(length, dtype=np.float64)
        for point in rng.choice(np.arange(16, length - 16), size=3, replace=False):
            shocks[point:] += rng.normal(0.0, 0.35)

        noise = rng.normal(0.0, 0.04, size=length)
        y = seasonal + trend + drift + shocks + noise
        y = (y - y.mean()) / max(y.std(), 1e-6)
        series[row] = y

    return series


def adaptive_pieces_from_series(
    ts: np.ndarray,
    compress_fn: Callable[..., list],
    tol: float,
    max_len: int,
) -> list[AdaptivePiece]:
    raw_pieces = np.asarray(compress_fn(ts.astype(np.float64), tol=tol, max_len=max_len))
    if raw_pieces.ndim != 2 or raw_pieces.shape[1] < 3:
        raise ValueError(f"Unexpected llm-abba piece shape: {raw_pieces.shape}")

    pieces = [
        AdaptivePiece(length=max(1, int(round(row[0]))), increment=float(row[2]))
        for row in raw_pieces
    ]
    return pieces


def reconstruct_from_raw_llmabba_pieces(
    start_value: float,
    raw_pieces: np.ndarray,
    inv_compress_fn: Callable[..., list],
) -> np.ndarray:
    reconstructed = inv_compress_fn(raw_pieces[:, :2].astype(np.float64), start_value)
    return np.asarray(reconstructed, dtype=np.float64)


def rounded_values(values: np.ndarray) -> list[float]:
    return np.round(values.astype(np.float64), ADAPTIVE_EXAMPLE_PRINT_PRECISION).tolist()


def print_adaptive_compression_example(
    ts: np.ndarray,
    raw_pieces: np.ndarray,
    reconstructed: np.ndarray,
) -> None:
    log("Adaptive compression example")
    print(f"  Tolerance: {ABBA_TOL}", flush=True)
    print(f"  Max segment length: {ABBA_MAX_LEN}", flush=True)
    print(f"  Original points: {ts.shape[0]}", flush=True)
    print(f"  Straight-line pieces: {raw_pieces.shape[0]}", flush=True)
    print("  Original values:", rounded_values(ts), flush=True)
    print(
        "  Reconstructed values from adaptive pieces:",
        rounded_values(reconstructed),
        flush=True,
    )
    print(
        "  Adaptive representation rows: "
        "[piece, start_idx, end_idx, length, start_value, end_value, increment, error]",
        flush=True,
    )

    start_idx = 0
    start_value = float(ts[0])
    for piece_index, row in enumerate(raw_pieces):
        length = int(round(row[0]))
        end_idx = start_idx + length
        end_value = float(row[1])
        increment = float(row[2])
        error = float(row[3])
        formatted = [
            piece_index,
            start_idx,
            end_idx,
            length,
            round(start_value, ADAPTIVE_EXAMPLE_PRINT_PRECISION),
            round(end_value, ADAPTIVE_EXAMPLE_PRINT_PRECISION),
            round(increment, ADAPTIVE_EXAMPLE_PRINT_PRECISION),
            round(error, ADAPTIVE_EXAMPLE_PRINT_PRECISION),
        ]
        print(f"  {formatted}", flush=True)
        start_idx = end_idx
        start_value = end_value


def plot_adaptive_compression_example(
    ts: np.ndarray,
    reconstructed: np.ndarray,
    raw_pieces: np.ndarray,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print(
            "  Plot skipped: matplotlib is not installed in this Python environment.",
            flush=True,
        )
        return

    x = np.arange(ts.shape[0])
    boundary_x = [0]
    for length in raw_pieces[:, 0].astype(int):
        boundary_x.append(boundary_x[-1] + int(length))
    boundary_y = reconstructed[np.asarray(boundary_x)]

    ADAPTIVE_EXAMPLE_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(x, ts, color="#1f77b4", linewidth=1.5, label="Original series")
    ax.plot(
        x,
        reconstructed,
        color="#d62728",
        linewidth=2.0,
        label="Adaptive piecewise-linear reconstruction",
    )
    ax.scatter(boundary_x, boundary_y, color="#d62728", s=24, zorder=3)
    ax.set_title(
        f"llm-abba adaptive compression: {raw_pieces.shape[0]} straight-line pieces"
    )
    ax.set_xlabel("Time index")
    ax.set_ylabel("Value")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ADAPTIVE_EXAMPLE_PLOT_PATH, dpi=160)
    plt.close(fig)
    print(f"  Saved adaptive compression plot: {ADAPTIVE_EXAMPLE_PLOT_PATH}", flush=True)


def show_adaptive_compression_example(
    compress_fn: Callable[..., list],
    inv_compress_fn: Callable[..., list],
) -> None:
    example = generate_synthetic_series(
        1,
        ADAPTIVE_EXAMPLE_SERIES_LENGTH,
        SEED + 30_000,
    )[0]
    raw_pieces = np.asarray(
        compress_fn(example.astype(np.float64), tol=ABBA_TOL, max_len=ABBA_MAX_LEN),
        dtype=np.float64,
    )
    reconstructed = reconstruct_from_raw_llmabba_pieces(
        float(example[0]),
        raw_pieces,
        inv_compress_fn,
    )
    print_adaptive_compression_example(example, raw_pieces, reconstructed)
    plot_adaptive_compression_example(example, reconstructed, raw_pieces)


def fit_piece_tokenizer(
    pieces_by_series: list[list[AdaptivePiece]],
    max_len: int,
    inc_bins: int,
) -> PieceTokenizer:
    increments = np.asarray(
        [piece.increment for pieces in pieces_by_series for piece in pieces],
        dtype=np.float64,
    )
    inc_min, inc_max = np.quantile(increments, [0.001, 0.999])
    if inc_min == inc_max:
        inc_min -= 1.0
        inc_max += 1.0
    else:
        margin = 0.05 * (inc_max - inc_min)
        inc_min -= margin
        inc_max += margin
    return PieceTokenizer(
        max_len=max_len,
        inc_bins=inc_bins,
        inc_min=float(inc_min),
        inc_max=float(inc_max),
    )


def flatten_token_sequences(sequences: list[list[int]]) -> torch.Tensor:
    tokens = [token_id for sequence in sequences for token_id in sequence]
    return torch.tensor(tokens, dtype=torch.long)


def make_random_batch(
    data: torch.Tensor,
    batch_size: int,
    context_length: int,
    generator: torch.Generator,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if data.numel() <= context_length + 1:
        raise ValueError("Tokenized adaptive data is too small for the context length.")
    starts = torch.randint(
        0,
        data.numel() - context_length - 1,
        (batch_size,),
        generator=generator,
    )
    batch = torch.stack(
        [data[int(start) : int(start) + context_length + 1] for start in starts]
    )
    return batch[:, :-1].to(device), batch[:, 1:].to(device)


@torch.inference_mode()
def estimate_loss(
    model: torch.nn.Module,
    data: torch.Tensor,
    batch_size: int,
    context_length: int,
    generator: torch.Generator,
    batches: int,
    device: str,
    amp_dtype: torch.dtype,
) -> float:
    model.eval()
    losses: list[float] = []
    for _ in range(batches):
        x, y = make_random_batch(data, batch_size, context_length, generator, device)
        with torch.autocast(device_type=device, dtype=amp_dtype, enabled=device == "cuda"):
            logits = model(x).logits
            loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
        losses.append(float(loss.item()))
    model.train()
    return sum(losses) / len(losses)


def train_model(
    train_tokens: torch.Tensor,
    validation_tokens: torch.Tensor,
    tokenizer: PieceTokenizer,
) -> torch.nn.Module:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this Mamba-2 experiment.")

    device = "cuda"
    torch.manual_seed(SEED)
    generator = torch.Generator().manual_seed(SEED + 1)

    ssm_cfg = lm_ssm_config(BLOCK)
    if BLOCK != "mamba2":
        raise ValueError("This example is intended for Mamba-2 only.")
    if not USE_MEM_EFF_PATH:
        ssm_cfg["use_mem_eff_path"] = False

    config = MambaConfig(
        d_model=MODEL_DIM,
        n_layer=LAYERS,
        vocab_size=tokenizer.vocab_size,
        ssm_cfg=ssm_cfg,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        pad_vocab_size_multiple=8,
    )

    param_dtype = dtype_from_name(PARAM_DTYPE)
    amp_dtype = dtype_from_name(AMP_DTYPE)
    model = MambaLMHeadModel(config, device=device, dtype=param_dtype)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    log("Training configuration")
    print(f"  Block: {BLOCK}", flush=True)
    print(f"  Layers: {LAYERS}", flush=True)
    print(f"  Model dim: {MODEL_DIM}", flush=True)
    print(f"  Batch size: {BATCH_SIZE}", flush=True)
    print(f"  Context length: {CONTEXT_LENGTH}", flush=True)
    print(f"  Steps: {STEPS}", flush=True)
    print(f"  Parameter dtype: {PARAM_DTYPE}", flush=True)
    print(f"  Autocast dtype: {AMP_DTYPE}", flush=True)
    print(f"  Mamba-2 fused path: {USE_MEM_EFF_PATH}", flush=True)
    print(f"  Vocab size: {tokenizer.vocab_size}", flush=True)
    print(f"  Parameters: {parameter_count(model):,}", flush=True)

    validation_loss = estimate_loss(
        model,
        validation_tokens,
        BATCH_SIZE,
        CONTEXT_LENGTH,
        generator,
        EVAL_BATCHES,
        device,
        amp_dtype,
    )
    print(f"Initial validation loss: {validation_loss:.4f}", flush=True)

    started = time.perf_counter()
    model.train()
    for step in range(1, STEPS + 1):
        x, y = make_random_batch(
            train_tokens,
            BATCH_SIZE,
            CONTEXT_LENGTH,
            generator,
            device,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device, dtype=amp_dtype, enabled=True):
            logits = model(x).logits
            loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()

        if step == 1 or step % LOG_INTERVAL == 0 or step == STEPS:
            validation_loss = estimate_loss(
                model,
                validation_tokens,
                BATCH_SIZE,
                CONTEXT_LENGTH,
                generator,
                EVAL_BATCHES,
                device,
                amp_dtype,
            )
            print(
                f"step {step:>4}/{STEPS}: "
                f"train loss={loss.item():.4f}, validation loss={validation_loss:.4f}",
                flush=True,
            )

    torch.cuda.synchronize()
    print(f"Training time: {time.perf_counter() - started:.1f} seconds", flush=True)
    model.eval()
    return model


def allowed_next_token_mask(
    ids: list[int],
    tokenizer: PieceTokenizer,
    generated_body_tokens: int,
) -> torch.Tensor:
    mask = torch.full((tokenizer.vocab_size,), -1e9, device="cuda")
    expecting_length = generated_body_tokens % 2 == 0
    if expecting_length:
        mask[tokenizer.length_offset : tokenizer.increment_offset] = 0.0
    else:
        mask[tokenizer.increment_offset : tokenizer.vocab_size] = 0.0
    _ = ids
    return mask


@torch.inference_mode()
def generate_continuation_pieces(
    model: torch.nn.Module,
    prompt_ids: list[int],
    tokenizer: PieceTokenizer,
    context_length: int,
    target_steps: int,
    temperature: float,
) -> tuple[list[int], list[AdaptivePiece]]:
    ids = torch.tensor([prompt_ids[:-1]], dtype=torch.long, device="cuda")
    generated: list[int] = []
    decoded_steps = 0
    max_new_tokens = 2 * target_steps

    for _ in range(max_new_tokens):
        logits = model(ids[:, -context_length:], num_last_tokens=1).logits[:, -1].float()
        logits = logits[:, : tokenizer.vocab_size]
        logits = logits + allowed_next_token_mask(ids[0].tolist(), tokenizer, len(generated))
        if temperature <= 0.0:
            next_id = logits.argmax(dim=-1, keepdim=True)
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        token_id = int(next_id.item())
        generated.append(token_id)
        ids = torch.cat((ids, next_id), dim=1)

        pieces = tokenizer.decode_body(generated)
        decoded_steps = sum(piece.length for piece in pieces)
        if decoded_steps >= target_steps and len(generated) % 2 == 0:
            break

    return generated, tokenizer.decode_body(generated)


def reconstruct_from_increment_pieces(
    start_value: float,
    pieces: list[AdaptivePiece],
    inv_compress_fn: Callable[..., list],
) -> np.ndarray:
    if not pieces:
        return np.asarray([], dtype=np.float64)
    endpoints = []
    current = float(start_value)
    for piece in pieces:
        current += piece.increment
        endpoints.append(current)
    llmabba_pieces = np.asarray(
        [[piece.length, endpoint] for piece, endpoint in zip(pieces, endpoints)],
        dtype=np.float64,
    )
    reconstructed = inv_compress_fn(llmabba_pieces, start_value)
    return np.asarray(reconstructed[1:], dtype=np.float64)


def run() -> None:
    compress_fn, inv_compress_fn, comp_path = load_llmabba_adaptive_functions()
    log(f"Using llm-abba adaptive compressor: {comp_path}")
    if SHOW_ADAPTIVE_COMPRESSION_EXAMPLE:
        show_adaptive_compression_example(compress_fn, inv_compress_fn)

    train_series = generate_synthetic_series(
        TRAIN_SERIES,
        SERIES_LENGTH,
        SEED,
    )
    validation_series = generate_synthetic_series(
        max(4, TRAIN_SERIES // 8),
        SERIES_LENGTH,
        SEED + 10_000,
    )

    train_pieces = [
        adaptive_pieces_from_series(series, compress_fn, ABBA_TOL, ABBA_MAX_LEN)
        for series in train_series
    ]
    validation_pieces = [
        adaptive_pieces_from_series(series, compress_fn, ABBA_TOL, ABBA_MAX_LEN)
        for series in validation_series
    ]
    tokenizer = fit_piece_tokenizer(train_pieces, ABBA_MAX_LEN, INCREMENT_BINS)

    train_tokens = flatten_token_sequences(
        [tokenizer.encode_pieces(pieces) for pieces in train_pieces]
    )
    validation_tokens = flatten_token_sequences(
        [tokenizer.encode_pieces(pieces) for pieces in validation_pieces]
    )
    compression_ratio = train_tokens.numel() / train_series.size
    log("Adaptive preprocessing")
    print(f"  Train series: {train_series.shape}", flush=True)
    print(f"  Train adaptive pieces: {sum(len(p) for p in train_pieces):,}", flush=True)
    print(f"  Train model tokens: {train_tokens.numel():,}", flush=True)
    print(f"  Token/original-point ratio: {compression_ratio:.3f}", flush=True)
    print(
        f"  Increment quantizer: [{tokenizer.inc_min:.4f}, {tokenizer.inc_max:.4f}] "
        f"over {tokenizer.inc_bins} bins",
        flush=True,
    )

    model = train_model(train_tokens, validation_tokens, tokenizer)

    test_series = generate_synthetic_series(1, SERIES_LENGTH, SEED + 20_000)[0]
    prefix = test_series[:PREFIX_LENGTH]
    future = test_series[PREFIX_LENGTH : PREFIX_LENGTH + PREDICT_STEPS]
    prefix_pieces = adaptive_pieces_from_series(prefix, compress_fn, ABBA_TOL, ABBA_MAX_LEN)
    prompt_ids = tokenizer.encode_pieces(prefix_pieces)
    generated_ids, generated_pieces = generate_continuation_pieces(
        model,
        prompt_ids,
        tokenizer,
        CONTEXT_LENGTH,
        PREDICT_STEPS,
        TEMPERATURE,
    )
    prediction = reconstruct_from_increment_pieces(prefix[-1], generated_pieces, inv_compress_fn)
    prediction = prediction[: future.shape[0]]
    comparable = min(prediction.shape[0], future.shape[0])
    if comparable:
        error = prediction[:comparable] - future[:comparable]
        mae = float(np.mean(np.abs(error)))
        rmse = float(np.sqrt(np.mean(error**2)))
    else:
        mae = float("nan")
        rmse = float("nan")

    log("Out-of-sample forecast")
    print(f"  Prefix points: {prefix.shape[0]}", flush=True)
    print(f"  Target future points: {future.shape[0]}", flush=True)
    print(f"  Generated tokens: {generated_ids[:20]}{' ...' if len(generated_ids) > 20 else ''}", flush=True)
    print(f"  Generated adaptive pieces: {len(generated_pieces)}", flush=True)
    print(f"  Predicted points: {prediction.shape[0]}", flush=True)
    print(f"  MAE over comparable horizon: {mae:.4f}", flush=True)
    print(f"  RMSE over comparable horizon: {rmse:.4f}", flush=True)
    print("  First 12 actual future values:", np.round(future[:12], 4).tolist(), flush=True)
    print("  First 12 predicted values:", np.round(prediction[:12], 4).tolist(), flush=True)
def main() -> None:
    run()


if __name__ == "__main__":
    main()
