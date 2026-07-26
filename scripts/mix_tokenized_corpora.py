"""Mix tokenized corpora into two train shards and one validation file."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import time
from typing import Any, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOKENIZED_DIR = PROJECT_ROOT / "data" / "tokenized"
TOKENIZED_FORMAT = "mamba_tokenized_corpus_v1"
TOKEN_DTYPE = np.int32
TOKEN_DTYPE_NAME = "int32"
DEFAULT_BLOCK_TOKENS = 1_048_576
DEFAULT_CONTEXT_LENGTH = 1024
DEFAULT_SEED = 7

DEFAULT_GUTENBERG_TRAIN = (
    TOKENIZED_DIR / "project_gutenberg_en_gpt_neox_20b_train.json"
)
DEFAULT_GUTENBERG_VALIDATION = (
    TOKENIZED_DIR / "project_gutenberg_en_gpt_neox_20b_validation.json"
)
DEFAULT_FANFICTION_TRAIN = TOKENIZED_DIR / "fan_fiction_gpt_neox_20b_train.json"
DEFAULT_FANFICTION_VALIDATION = (
    TOKENIZED_DIR / "fan_fiction_gpt_neox_20b_validation.json"
)
DEFAULT_OUTPUT_STEM = "mixed_gutenberg_fan_fiction_gpt_neox_20b"


@dataclass(frozen=True)
class TokenSource:
    name: str
    metadata_path: Path
    metadata: dict[str, Any]
    token_path: Path
    token_count: int
    tokenizer: dict[str, Any]
    vocab_size: int


@dataclass(frozen=True)
class TokenBlock:
    source_index: int
    start: int
    end: int

    @property
    def token_count(self) -> int:
        return self.end - self.start


def import_tqdm() -> Any:
    try:
        from tqdm.auto import tqdm
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "tqdm is required for progress bars. Install it with: pip install tqdm"
        ) from exc
    return tqdm


def format_bytes(byte_count: int) -> str:
    value = float(byte_count)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def resolve_token_path(metadata_path: Path, metadata: dict[str, Any]) -> Path:
    raw_path = Path(str(metadata["token_path"]))
    candidates = [raw_path]
    if raw_path.is_absolute() and os.name == "nt":
        raw = str(raw_path).replace("\\", "/")
        parts = raw.split("/")
        if len(parts) > 3 and parts[1] == "mnt" and len(parts[2]) == 1:
            candidates.append(Path(f"{parts[2]}:/" + "/".join(parts[3:])))
    if raw_path.is_absolute() and os.name != "nt":
        drive = raw_path.drive.rstrip(":").lower()
        if drive:
            relative = str(raw_path)[len(raw_path.drive) :].lstrip("\\/")
            candidates.append(Path("/mnt") / drive / relative.replace("\\", "/"))
    if not raw_path.is_absolute():
        candidates.append(metadata_path.parent / raw_path)
    candidates.append(metadata_path.with_suffix(".int32.bin"))

    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"Could not find token file for {metadata_path}; searched: {searched}"
    )


def load_source(name: str, metadata_path: Path) -> TokenSource:
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        progress_path = metadata_path.with_name(f"{metadata_path.stem}.progress.json")
        hint = (
            f" Found progress file instead: {progress_path}. Wait for tokenization "
            "to finish before mixing."
            if progress_path.exists()
            else ""
        )
        raise FileNotFoundError(f"Metadata file does not exist: {metadata_path}.{hint}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format") != TOKENIZED_FORMAT:
        raise ValueError(
            f"{metadata_path}: unsupported format {metadata.get('format')!r}"
        )
    if not metadata.get("complete"):
        raise ValueError(f"{metadata_path}: metadata is not marked complete")
    if metadata.get("token_dtype") != TOKEN_DTYPE_NAME:
        raise ValueError(
            f"{metadata_path}: expected token dtype {TOKEN_DTYPE_NAME}, got "
            f"{metadata.get('token_dtype')!r}"
        )

    token_count = int(metadata["token_count"])
    token_path = resolve_token_path(metadata_path, metadata)
    expected_bytes = token_count * np.dtype(TOKEN_DTYPE).itemsize
    actual_bytes = token_path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"{metadata_path}: token file size mismatch for {token_path}; "
            f"expected {expected_bytes:,} bytes, found {actual_bytes:,}"
        )

    tokenizer = metadata.get("tokenizer")
    if not isinstance(tokenizer, dict):
        raise ValueError(f"{metadata_path}: missing tokenizer metadata")

    return TokenSource(
        name=name,
        metadata_path=metadata_path,
        metadata=metadata,
        token_path=token_path,
        token_count=token_count,
        tokenizer=tokenizer,
        vocab_size=int(metadata["vocab_size"]),
    )


def validate_sources(sources: Iterable[TokenSource]) -> tuple[dict[str, Any], int]:
    sources = list(sources)
    if not sources:
        raise ValueError("At least one token source is required")

    tokenizer = sources[0].tokenizer
    vocab_size = sources[0].vocab_size
    for source in sources[1:]:
        if source.tokenizer != tokenizer:
            raise ValueError(
                "Tokenizer mismatch: "
                f"{sources[0].name}={tokenizer!r}, {source.name}={source.tokenizer!r}"
            )
        if source.vocab_size != vocab_size:
            raise ValueError(
                "Vocabulary size mismatch: "
                f"{sources[0].name}={vocab_size:,}, "
                f"{source.name}={source.vocab_size:,}"
            )
    return tokenizer, vocab_size


def source_arrays(sources: list[TokenSource]) -> list[np.memmap]:
    return [
        np.memmap(
            source.token_path,
            dtype=TOKEN_DTYPE,
            mode="r",
            shape=(source.token_count,),
        )
        for source in sources
    ]


def aligned_split(token_count: int, context_length: int) -> int:
    midpoint = token_count // 2
    aligned = (midpoint // context_length) * context_length
    if aligned <= 0 or aligned >= token_count:
        return midpoint
    return aligned


def blocks_for_range(
    source_index: int,
    start: int,
    end: int,
    block_tokens: int,
) -> list[TokenBlock]:
    blocks: list[TokenBlock] = []
    position = start
    while position < end:
        next_position = min(position + block_tokens, end)
        blocks.append(TokenBlock(source_index, position, next_position))
        position = next_position
    return blocks


def train_blocks(
    sources: list[TokenSource],
    shard_index: int,
    block_tokens: int,
    context_length: int,
) -> list[TokenBlock]:
    blocks: list[TokenBlock] = []
    for source_index, source in enumerate(sources):
        split = aligned_split(source.token_count, context_length)
        if shard_index == 1:
            start, end = 0, split
        elif shard_index == 2:
            start, end = split, source.token_count
        else:
            raise ValueError("shard_index must be 1 or 2")
        blocks.extend(blocks_for_range(source_index, start, end, block_tokens))
    return blocks


def validation_blocks(
    sources: list[TokenSource],
    block_tokens: int,
) -> list[TokenBlock]:
    blocks: list[TokenBlock] = []
    for source_index, source in enumerate(sources):
        blocks.extend(blocks_for_range(source_index, 0, source.token_count, block_tokens))
    return blocks


def shuffled_blocks(
    blocks: list[TokenBlock],
    seed: int,
    label: str,
) -> list[TokenBlock]:
    shuffled = blocks[:]
    random.Random(f"{seed}:{label}").shuffle(shuffled)
    return shuffled


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp_path.replace(path)


def remove_existing_outputs(token_path: Path, metadata_path: Path, overwrite: bool) -> None:
    temp_token_path = token_path.with_name(f"{token_path.name}.tmp")
    existing = [
        path
        for path in (token_path, metadata_path, temp_token_path)
        if path.exists()
    ]
    if existing and not overwrite:
        existing_text = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"output already exists: {existing_text}")
    for path in existing:
        path.unlink()


def mix_metadata_payload(
    label: str,
    token_path: Path,
    sources: list[TokenSource],
    tokenizer: dict[str, Any],
    vocab_size: int,
    token_count: int,
    block_count: int,
    block_tokens: int,
    seed: int,
    elapsed: float,
) -> dict[str, Any]:
    return {
        "format": TOKENIZED_FORMAT,
        "complete": True,
        "label": label,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_text_path": "mixed tokenized corpora",
        "source_char_limit": None,
        "source_chars_read": None,
        "source_truncated": False,
        "token_path": str(token_path),
        "token_dtype": TOKEN_DTYPE_NAME,
        "token_count": token_count,
        "chunks_written": block_count,
        "text_chunk_chars": None,
        "tokenizer": tokenizer,
        "vocab_size": vocab_size,
        "elapsed_seconds": elapsed,
        "mix": {
            "block_tokens": block_tokens,
            "seed": seed,
            "sources": [
                {
                    "name": source.name,
                    "metadata_path": str(source.metadata_path),
                    "token_path": str(source.token_path),
                    "token_count": source.token_count,
                }
                for source in sources
            ],
        },
    }


def write_mixed_file(
    label: str,
    token_path: Path,
    metadata_path: Path,
    sources: list[TokenSource],
    arrays: list[np.memmap],
    blocks: list[TokenBlock],
    tokenizer: dict[str, Any],
    vocab_size: int,
    block_tokens: int,
    seed: int,
    overwrite: bool,
) -> None:
    tqdm = import_tqdm()
    remove_existing_outputs(token_path, metadata_path, overwrite=overwrite)
    token_path.parent.mkdir(parents=True, exist_ok=True)
    temp_token_path = token_path.with_name(f"{token_path.name}.tmp")
    token_count = sum(block.token_count for block in blocks)

    print(
        f"{label}: writing {token_count:,} tokens from {len(blocks):,} mixed blocks "
        f"to {token_path}",
        flush=True,
    )
    started = time.perf_counter()
    written = 0
    with temp_token_path.open("wb") as writer:
        progress = tqdm(blocks, desc=label, unit="block")
        for block in progress:
            arrays[block.source_index][block.start : block.end].tofile(writer)
            written += block.token_count
            progress.set_postfix(tokens=f"{written:,}")

    temp_token_path.replace(token_path)
    elapsed = time.perf_counter() - started
    metadata = mix_metadata_payload(
        label=label,
        token_path=token_path,
        sources=sources,
        tokenizer=tokenizer,
        vocab_size=vocab_size,
        token_count=token_count,
        block_count=len(blocks),
        block_tokens=block_tokens,
        seed=seed,
        elapsed=elapsed,
    )
    atomic_write_json(metadata_path, metadata)
    print(
        f"{label}: complete, tokens={token_count:,}, "
        f"size={format_bytes(token_path.stat().st_size)}, elapsed={elapsed:.1f}s",
        flush=True,
    )


def output_paths(output_dir: Path, output_stem: str) -> dict[str, tuple[Path, Path]]:
    return {
        "train_1": (
            output_dir / f"{output_stem}_train_1.int32.bin",
            output_dir / f"{output_stem}_train_1.json",
        ),
        "train_2": (
            output_dir / f"{output_stem}_train_2.int32.bin",
            output_dir / f"{output_stem}_train_2.json",
        ),
        "validation": (
            output_dir / f"{output_stem}_validation.int32.bin",
            output_dir / f"{output_stem}_validation.json",
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Mix Project Gutenberg and fan-fiction tokenized corpora into two "
            "train shards and one validation token file."
        )
    )
    parser.add_argument("--gutenberg-train", type=Path, default=DEFAULT_GUTENBERG_TRAIN)
    parser.add_argument(
        "--gutenberg-validation",
        type=Path,
        default=DEFAULT_GUTENBERG_VALIDATION,
    )
    parser.add_argument("--fanfiction-train", type=Path, default=DEFAULT_FANFICTION_TRAIN)
    parser.add_argument(
        "--fanfiction-validation",
        type=Path,
        default=DEFAULT_FANFICTION_VALIDATION,
    )
    parser.add_argument("--output-dir", type=Path, default=TOKENIZED_DIR)
    parser.add_argument("--output-stem", default=DEFAULT_OUTPUT_STEM)
    parser.add_argument(
        "--block-tokens",
        type=int,
        default=DEFAULT_BLOCK_TOKENS,
        help="Token block size used before shuffling blocks.",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=DEFAULT_CONTEXT_LENGTH,
        help="Used to align the train split between the two train shards.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Fail if any mixed output already exists.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.block_tokens <= 0:
        raise SystemExit("--block-tokens must be greater than 0")
    if args.context_length <= 0:
        raise SystemExit("--context-length must be greater than 0")

    train_sources = [
        load_source("project_gutenberg_train", args.gutenberg_train),
        load_source("fan_fiction_train", args.fanfiction_train),
    ]
    validation_sources = [
        load_source("project_gutenberg_validation", args.gutenberg_validation),
        load_source("fan_fiction_validation", args.fanfiction_validation),
    ]
    tokenizer, vocab_size = validate_sources(train_sources + validation_sources)
    output_map = output_paths(args.output_dir, args.output_stem)
    overwrite = not args.no_overwrite

    print("Loaded token sources:", flush=True)
    for source in train_sources + validation_sources:
        print(
            f"  {source.name}: {source.token_count:,} tokens "
            f"from {source.metadata_path}",
            flush=True,
        )

    train_arrays = source_arrays(train_sources)
    validation_arrays = source_arrays(validation_sources)
    train_1_blocks = shuffled_blocks(
        train_blocks(
            train_sources,
            shard_index=1,
            block_tokens=args.block_tokens,
            context_length=args.context_length,
        ),
        seed=args.seed,
        label="train_1",
    )
    train_2_blocks = shuffled_blocks(
        train_blocks(
            train_sources,
            shard_index=2,
            block_tokens=args.block_tokens,
            context_length=args.context_length,
        ),
        seed=args.seed,
        label="train_2",
    )
    mixed_validation_blocks = shuffled_blocks(
        validation_blocks(validation_sources, block_tokens=args.block_tokens),
        seed=args.seed,
        label="validation",
    )

    write_mixed_file(
        label="train_1",
        token_path=output_map["train_1"][0],
        metadata_path=output_map["train_1"][1],
        sources=train_sources,
        arrays=train_arrays,
        blocks=train_1_blocks,
        tokenizer=tokenizer,
        vocab_size=vocab_size,
        block_tokens=args.block_tokens,
        seed=args.seed,
        overwrite=overwrite,
    )
    write_mixed_file(
        label="train_2",
        token_path=output_map["train_2"][0],
        metadata_path=output_map["train_2"][1],
        sources=train_sources,
        arrays=train_arrays,
        blocks=train_2_blocks,
        tokenizer=tokenizer,
        vocab_size=vocab_size,
        block_tokens=args.block_tokens,
        seed=args.seed,
        overwrite=overwrite,
    )
    write_mixed_file(
        label="validation",
        token_path=output_map["validation"][0],
        metadata_path=output_map["validation"][1],
        sources=validation_sources,
        arrays=validation_arrays,
        blocks=mixed_validation_blocks,
        tokenizer=tokenizer,
        vocab_size=vocab_size,
        block_tokens=args.block_tokens,
        seed=args.seed,
        overwrite=overwrite,
    )

    print("\nUse these metadata files for sequential training:", flush=True)
    print(f"  train 1: {output_map['train_1'][1]}", flush=True)
    print(f"  train 2: {output_map['train_2'][1]}", flush=True)
    print(f"  validation: {output_map['validation'][1]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
