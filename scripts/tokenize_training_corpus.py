"""Tokenize text corpora incrementally for mamba_experiment.py."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"

TOKENIZER_NAME = "EleutherAI/gpt-neox-20b"

TEXT_CHUNK_CHARS = 50_000_000
MAX_TRAIN_CHARS = 0
MAX_VALIDATION_CHARS = 0

TRAIN_TEXT_PATH = PROJECT_ROOT / "data" / "fan_fiction_train.txt"
VALIDATION_TEXT_PATH = PROJECT_ROOT / "data" / "fan_fiction_validation.txt"
OUTPUT_DIR = PROJECT_ROOT / "data" / "tokenized"
OUTPUT_STEM = "fan_fiction_gpt_neox_20b"
TRAIN_TOKEN_PATH = OUTPUT_DIR / f"{OUTPUT_STEM}_train.int32.bin"
TRAIN_METADATA_PATH = OUTPUT_DIR / f"{OUTPUT_STEM}_train.json"
VALIDATION_TOKEN_PATH = OUTPUT_DIR / f"{OUTPUT_STEM}_validation.int32.bin"
VALIDATION_METADATA_PATH = OUTPUT_DIR / f"{OUTPUT_STEM}_validation.json"

TOKENIZE_VALIDATION = True
OVERWRITE_OUTPUT = True
LOG_EVERY_CHUNKS = 1

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


def normalized_char_limit(value: int) -> int | None:
    return None if value == 0 else value


def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp_path.replace(path)


def metadata_payload(
    label: str,
    source_text_path: Path,
    token_path: Path,
    tokenizer_name: str,
    vocab_size: int,
    max_chars: int | None,
    chars_read: int,
    tokens_written: int,
    chunks_written: int,
    truncated: bool,
    elapsed: float,
    complete: bool,
) -> dict[str, object]:
    return {
        "format": TOKENIZED_FORMAT,
        "complete": complete,
        "label": label,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_text_path": str(source_text_path),
        "source_char_limit": max_chars,
        "source_chars_read": chars_read,
        "source_truncated": truncated,
        "token_path": str(token_path),
        "token_dtype": TOKEN_DTYPE_NAME,
        "token_count": tokens_written,
        "chunks_written": chunks_written,
        "text_chunk_chars": TEXT_CHUNK_CHARS,
        "tokenizer": {"kind": "hf", "name": tokenizer_name},
        "vocab_size": vocab_size,
        "elapsed_seconds": elapsed,
    }


def last_whitespace_index(text: str) -> int:
    for index in range(len(text) - 1, -1, -1):
        if text[index].isspace():
            return index
    return -1


def prepare_output_paths(token_path: Path, metadata_path: Path) -> tuple[Path, Path]:
    token_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    temp_token_path = token_path.with_name(f"{token_path.name}.tmp")
    progress_metadata_path = metadata_path.with_name(f"{metadata_path.stem}.progress.json")

    existing_paths = [token_path, metadata_path, temp_token_path, progress_metadata_path]
    existing = [path for path in existing_paths if path.exists()]
    if existing and not OVERWRITE_OUTPUT:
        existing_text = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"output already exists: {existing_text}")
    for path in existing:
        path.unlink()
    return temp_token_path, progress_metadata_path


def write_token_chunk(
    writer,
    tokenizer,
    text: str,
    label: str,
) -> int:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        return 0
    max_token_id = max(token_ids)
    if max_token_id > np.iinfo(TOKEN_DTYPE).max:
        raise ValueError(
            f"{label} produced token id {max_token_id}, which does not fit in "
            f"{TOKEN_DTYPE_NAME}"
        )
    np.asarray(token_ids, dtype=TOKEN_DTYPE).tofile(writer)
    return len(token_ids)


def tokenize_text_file(
    label: str,
    source_text_path: Path,
    token_path: Path,
    metadata_path: Path,
    tokenizer,
    max_chars: int | None,
) -> None:
    if not source_text_path.exists():
        raise FileNotFoundError(f"{source_text_path} does not exist")

    temp_token_path, progress_metadata_path = prepare_output_paths(
        token_path,
        metadata_path,
    )
    source_size = source_text_path.stat().st_size
    log(f"{label}: source={source_text_path}")
    log(f"{label}: source size={format_bytes(source_size)}, max chars={max_chars or 'all'}")
    log(f"{label}: writing temporary token file={temp_token_path}")

    started = time.perf_counter()
    chars_read = 0
    tokens_written = 0
    chunks_written = 0
    carry = ""
    truncated = False
    remaining = max_chars

    with source_text_path.open("r", encoding="utf-8") as reader:
        with temp_token_path.open("wb") as writer:
            while True:
                if remaining is not None and remaining <= 0:
                    truncated = bool(reader.read(1))
                    break

                read_size = (
                    TEXT_CHUNK_CHARS
                    if remaining is None
                    else min(TEXT_CHUNK_CHARS, remaining)
                )
                raw_chunk = reader.read(read_size)
                if not raw_chunk:
                    break

                chars_read += len(raw_chunk)
                if remaining is not None:
                    remaining -= len(raw_chunk)

                buffer = carry + raw_chunk
                final_chunk = len(raw_chunk) < read_size or (
                    remaining is not None and remaining <= 0
                )
                if final_chunk:
                    text_chunk = buffer
                    carry = ""
                else:
                    split_index = last_whitespace_index(buffer)
                    if split_index <= 0:
                        text_chunk = buffer
                        carry = ""
                    else:
                        text_chunk = buffer[:split_index]
                        carry = buffer[split_index:]

                if not text_chunk:
                    continue

                token_count = write_token_chunk(writer, tokenizer, text_chunk, label)
                writer.flush()
                tokens_written += token_count
                chunks_written += 1

                elapsed = time.perf_counter() - started
                progress_payload = metadata_payload(
                    label=label,
                    source_text_path=source_text_path,
                    token_path=token_path,
                    tokenizer_name=TOKENIZER_NAME,
                    vocab_size=len(tokenizer),
                    max_chars=max_chars,
                    chars_read=chars_read,
                    tokens_written=tokens_written,
                    chunks_written=chunks_written,
                    truncated=truncated,
                    elapsed=elapsed,
                    complete=False,
                )
                atomic_write_json(progress_metadata_path, progress_payload)

                if chunks_written % LOG_EVERY_CHUNKS == 0:
                    rate = tokens_written / max(elapsed, 1e-9)
                    memory = memory_status()
                    suffix = f", {memory}" if memory else ""
                    log(
                        f"{label}: chunks={chunks_written:,}, "
                        f"chars={chars_read:,}, tokens={tokens_written:,}, "
                        f"rate={rate:,.0f} tokens/s{suffix}"
                    )

            if carry:
                token_count = write_token_chunk(writer, tokenizer, carry, label)
                writer.flush()
                tokens_written += token_count
                chunks_written += 1

    elapsed = time.perf_counter() - started
    temp_token_path.replace(token_path)
    final_payload = metadata_payload(
        label=label,
        source_text_path=source_text_path,
        token_path=token_path,
        tokenizer_name=TOKENIZER_NAME,
        vocab_size=len(tokenizer),
        max_chars=max_chars,
        chars_read=chars_read,
        tokens_written=tokens_written,
        chunks_written=chunks_written,
        truncated=truncated,
        elapsed=elapsed,
        complete=True,
    )
    atomic_write_json(metadata_path, final_payload)
    try:
        progress_metadata_path.unlink()
    except FileNotFoundError:
        pass

    log(
        f"{label}: complete, tokens={tokens_written:,}, "
        f"output size={format_bytes(token_path.stat().st_size)}, "
        f"elapsed={elapsed:.1f}s"
    )
    if truncated:
        log(f"{label}: source was truncated by the configured character limit")


def main() -> None:
    load_env_file()
    log("Loading tokenizer")
    log(f"Tokenizer: {TOKENIZER_NAME}")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, **hf_auth_kwargs())
    tokenizer.model_max_length = max(
        getattr(tokenizer, "model_max_length", 0) or 0,
        TEXT_CHUNK_CHARS,
    )
    log(f"Tokenizer vocabulary size: {len(tokenizer):,}")

    tokenize_text_file(
        label="train",
        source_text_path=TRAIN_TEXT_PATH,
        token_path=TRAIN_TOKEN_PATH,
        metadata_path=TRAIN_METADATA_PATH,
        tokenizer=tokenizer,
        max_chars=normalized_char_limit(MAX_TRAIN_CHARS),
    )
    if TOKENIZE_VALIDATION:
        if VALIDATION_TEXT_PATH.exists():
            tokenize_text_file(
                label="validation",
                source_text_path=VALIDATION_TEXT_PATH,
                token_path=VALIDATION_TOKEN_PATH,
                metadata_path=VALIDATION_METADATA_PATH,
                tokenizer=tokenizer,
                max_chars=normalized_char_limit(MAX_VALIDATION_CHARS),
            )
        else:
            log(f"validation: skipped because {VALIDATION_TEXT_PATH} does not exist")


if __name__ == "__main__":
    main()
