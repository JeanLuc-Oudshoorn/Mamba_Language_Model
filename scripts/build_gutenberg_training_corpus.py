"""Build cleaned train/validation corpora from Project Gutenberg on HuggingFace."""

from __future__ import annotations

import argparse
from collections import Counter
from itertools import islice
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterable

from build_training_corpus import (

    CHAPTER_MARKER_LINE_RE,
    PAGE_NUMBER_LINE_RE,
    MULTIPLE_BLANK_LINES_RE,
    ROMAN_NUMERAL_LINE_RE,
    SPECIAL_SEPARATOR_RE,
    TRAILING_BARE_NUMBER_RE,
    WHITESPACE_RE,
    edge_line_indices,
    join_wrapped_lines,
    normalize_text,
    repeated_edge_lines,
    remove_malformed_segments,
    should_remove_line,
    split_document_text,
    validation_fraction_arg,
    validation_output_path_for,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_NAME = "manu/project_gutenberg"
DEFAULT_SPLIT = "en"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "project_gutenberg_en.txt"
DEFAULT_VALIDATION_FRACTION = 0.10
DEFAULT_MAX_TRAIN_CHARS = 25_000_000_000
DEFAULT_DOCUMENT_BOUNDARY = "<*>"
DEFAULT_EDGE_LINES = 3
DEFAULT_MIN_REPEATED_PAGES = 4
EXPECTED_COLUMNS = {"id", "text"}
DOTENV_KEYS = {"HF_TOKEN", "HF_HOME", "HF_HUB_CACHE"}
DOTENV_PATH_KEYS = {"HF_HOME", "HF_HUB_CACHE"}

GUTENBERG_START_RE = re.compile(
    r"(?im)^[^\S\n]*\*{3}\s*START OF\b.*?\*{3}[^\S\n]*$"
)
GUTENBERG_END_RE = re.compile(
    r"(?im)^[^\S\n]*\*{3}\s*END OF\b.*?\*{3}[^\S\n]*$"
)
WINDOWS_ABSOLUTE_PATH_RE = re.compile(r"^(?P<drive>[A-Za-z]):[\\/](?P<path>.*)$")
FIRST_CHAPTER_MARKER_DESCRIPTION = (
    "Chapter I / Chapter 1 / Chapter One / Book One / Prologue / Opening / Overture / Prelude / Introduction / Foreword /"
    " Book 1 / Book I / Part I / Part 1 / Part One / Volume 1 / Volume I / Scene 1 / Scene I / Act 1 / Act I / Section 1 /"
    " Section I / The Beginning / Letter 1 / Letter I / Letter One / Essay 1 / Essay I / Essay One / Introductory Note "
)
FIRST_CHAPTER_LINE_RE = re.compile(
    r"""
    ^\W*(?:
        chapter\s+(?:i|1|one)
        |book\s+(?:i|1|one)
        |part\s+(?:i|1|one)
        |volume\s+(?:i|1|one)
        |scene\s+(?:i|1|one)
        |act\s+(?:i|1|one)
        |section\s+(?:i|1|one)
        |letter\s+(?:i|1|one)
        |essay\s+(?:i|1|one)
        |the\s+beginning
        |introductory\s+note
        |prologue
        |opening
        |overture
        |prelude
        |introduction
        |foreword
    )\b
    """,
    re.I | re.X,
)
THE_END_LINE_RE = re.compile(r"^\W*the\s+end\b", re.I)


def parse_dotenv_line(line: str) -> tuple[str, str] | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if line.startswith("export "):
        line = line[len("export ") :].lstrip()
    if "=" not in line:
        return None

    key, value = line.split("=", 1)
    key = key.strip()
    if not key:
        return None

    value = value.strip()
    if value and value[0] in {"'", '"'}:
        quote = value[0]
        end_index = value.find(quote, 1)
        if end_index != -1:
            value = value[1:end_index]
        else:
            value = value[1:]
    else:
        value = value.split("#", 1)[0].strip()

    return key, value


def wsl_path_from_windows_path(value: str) -> str:
    match = WINDOWS_ABSOLUTE_PATH_RE.match(value)
    if not match or not sys.platform.startswith("linux"):
        return value

    drive = match.group("drive").lower()
    path = match.group("path").replace("\\", "/")
    return f"/mnt/{drive}/{path}"


def load_huggingface_env(dotenv_path: Path = PROJECT_ROOT / ".env") -> None:
    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8-sig").splitlines():
        parsed_line = parse_dotenv_line(raw_line)
        if parsed_line is None:
            continue

        key, value = parsed_line
        if key not in DOTENV_KEYS or os.environ.get(key):
            continue
        if key in DOTENV_PATH_KEYS:
            value = wsl_path_from_windows_path(value)
        os.environ[key] = value


class CorpusWriter:
    def __init__(
        self,
        path: Path,
        document_boundary: str = DEFAULT_DOCUMENT_BOUNDARY,
        max_characters: int | None = None,
    ) -> None:
        self.path = path
        self.document_boundary = document_boundary.strip()
        self.max_characters = max_characters
        suffix = path.suffix or ".txt"
        self.temp_path = path.with_name(f"{path.stem}{suffix}.tmp")
        self.document_count = 0
        self.character_count = 0
        self._handle = None

    def __enter__(self) -> CorpusWriter:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.temp_path.open("w", encoding="utf-8")
        return self

    def format_document(self, text: str) -> str:
        text = text.strip()
        if self.document_boundary:
            return f"{self.document_boundary}\n\n{text}"
        return text

    def projected_character_count(self, text: str) -> int:
        text = text.strip()
        if not text:
            return self.character_count
        separator_chars = 2 if self.document_count else 0
        return self.character_count + separator_chars + len(self.format_document(text))

    def can_write_document(self, text: str) -> bool:
        text = text.strip()
        if not text:
            return False
        if self.max_characters is None:
            return True
        # Account for the final newline written during commit.
        return self.projected_character_count(text) + 1 <= self.max_characters

    def write_document(self, text: str) -> bool:
        text = text.strip()
        if not text:
            return False
        if not self.can_write_document(text):
            return False
        if self._handle is None:
            raise RuntimeError("CorpusWriter is not open")
        formatted_text = self.format_document(text)
        if self.document_count:
            self._handle.write("\n\n")
            self.character_count += 2
        self._handle.write(formatted_text)
        self.document_count += 1
        self.character_count += len(formatted_text)
        return True

    def commit(self) -> None:
        if self._handle is None:
            raise RuntimeError("CorpusWriter is not open")
        if self.document_count:
            self._handle.write("\n")
            self.character_count += 1
        self._handle.close()
        self._handle = None
        self.temp_path.replace(self.path)

    def discard(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        try:
            self.temp_path.unlink()
        except FileNotFoundError:
            pass

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if exc_type is not None:
            self.discard()


def strip_gutenberg_boilerplate(text: str) -> tuple[str, bool, bool]:
    text = normalize_text(text)
    start_match = GUTENBERG_START_RE.search(text)
    body_start = start_match.end() if start_match else 0

    end_match = GUTENBERG_END_RE.search(text, body_start)
    body_end = end_match.start() if end_match else len(text)

    return (
        text[body_start:body_end].strip(),
        start_match is not None,
        end_match is not None,
    )


def normalized_line(line: str) -> str:
    return WHITESPACE_RE.sub(" ", line).strip()


def trim_to_latest_first_chapter(body: str) -> tuple[str, bool]:
    lines = body.splitlines()
    start_location: tuple[int, int] | None = None

    for index, line in enumerate(lines):
        match = FIRST_CHAPTER_LINE_RE.search(line)
        if match:
            start_location = (index, match.start())

    if start_location is None:
        return body.strip(), False

    start_line, start_char = start_location
    lines = lines[start_line:]
    lines[0] = lines[0][start_char:].lstrip()
    return "\n".join(lines).strip(), True


def trim_after_the_end(body: str) -> tuple[str, bool]:
    lines = body.splitlines()
    for index, line in enumerate(lines):
        match = THE_END_LINE_RE.search(line)
        if not match:
            continue
        lines = lines[: index + 1]
        lines[index] = line[: match.end()]
        return "\n".join(lines).strip(), True
    return body.strip(), False


def split_gutenberg_pages(body: str, min_repeated_pages: int) -> list[list[str]]:
    if "\f" in body:
        return [page.splitlines() for page in body.split("\f")]

    lines = body.splitlines()
    page_number_indices = [
        index
        for index, line in enumerate(lines)
        if PAGE_NUMBER_LINE_RE.fullmatch(normalized_line(line))
    ]
    if len(page_number_indices) < min_repeated_pages:
        return [lines]

    pages: list[list[str]] = []
    current_page: list[str] = []
    for line in lines:
        if PAGE_NUMBER_LINE_RE.fullmatch(normalized_line(line)):
            if current_page:
                pages.append(current_page)
                current_page = []
            continue
        current_page.append(line)
    if current_page:
        pages.append(current_page)
    return pages or [lines]


def clean_gutenberg_line(line: str) -> str | None:
    line = normalize_text(line)
    line = SPECIAL_SEPARATOR_RE.sub(" ", line)
    line = normalized_line(line)
    if not line:
        return ""
    if PAGE_NUMBER_LINE_RE.fullmatch(line):
        return None
    if ROMAN_NUMERAL_LINE_RE.fullmatch(line):
        return None
    if CHAPTER_MARKER_LINE_RE.fullmatch(line):
        return None

    trailing_number = TRAILING_BARE_NUMBER_RE.fullmatch(line)
    if trailing_number:
        body = trailing_number.group("body").rstrip()
        if len(body.split()) >= 5 or any(mark in body for mark in ".,;:!?"):
            line = body
    return line


def clean_gutenberg_pages(
    body: str,
    edge_lines: int,
    min_repeated_pages: int,
) -> str:
    raw_pages = split_gutenberg_pages(body, min_repeated_pages)
    pages = [
        [clean_gutenberg_line(line) for line in page_lines]
        for page_lines in raw_pages
    ]
    (
        repeated_exact_lines,
        repeated_uppercase_lines,
        repeated_numbered_lines,
    ) = repeated_edge_lines(
        pages,
        edge_lines=edge_lines,
        min_pages=min_repeated_pages,
    )

    cleaned_lines: list[str] = []
    for page_index, page_lines in enumerate(pages):
        page_edge_indices = edge_line_indices(page_lines, edge_lines)
        if page_index:
            cleaned_lines.append("")
        for line_index, line in enumerate(page_lines):
            if line is None:
                continue
            if not line:
                cleaned_lines.append("")
                continue
            if should_remove_line(
                line,
                line_index=line_index,
                page_edge_indices=page_edge_indices,
                repeated_exact_lines=repeated_exact_lines,
                repeated_uppercase_lines=repeated_uppercase_lines,
                repeated_numbered_lines=repeated_numbered_lines,
            ):
                continue
            cleaned_lines.append(line)

    cleaned_text = join_wrapped_lines(cleaned_lines)
    return MULTIPLE_BLANK_LINES_RE.sub("\n\n", cleaned_text).strip()


def clean_gutenberg_text(
    text: str,
    trim_chapter_start: bool,
    edge_lines: int,
    min_repeated_pages: int,
) -> tuple[str, bool, bool, bool]:
    body, found_start_marker, found_end_marker = strip_gutenberg_boilerplate(text)
    found_chapter_start = False
    if trim_chapter_start:
        body, found_chapter_start = trim_to_latest_first_chapter(body)
    body, _found_the_end = trim_after_the_end(body)
    cleaned_text = clean_gutenberg_pages(
        body,
        edge_lines=edge_lines,
        min_repeated_pages=min_repeated_pages,
    )
    return (
        cleaned_text.strip(),
        found_start_marker,
        found_end_marker,
        found_chapter_start,
    )


def validate_record_columns(record: dict[str, Any], strict_columns: bool) -> None:
    columns = set(record)
    missing_columns = EXPECTED_COLUMNS - columns
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"missing required column(s): {missing}")
    if strict_columns and columns != EXPECTED_COLUMNS:
        extra = ", ".join(sorted(columns - EXPECTED_COLUMNS))
        raise ValueError(f"unexpected column(s): {extra}")


def maybe_limited(
    dataset: Iterable[dict[str, Any]],
    max_documents: int | None,
) -> Iterable[dict[str, Any]]:
    if max_documents is None:
        return dataset
    return islice(dataset, max_documents)


def format_counter(counter: Counter[str]) -> str:
    return ", ".join(f"{count:,} {reason}" for reason, count in counter.items())


def build_corpus(
    dataset_name: str,
    split: str,
    output_path: Path,
    validation_output_path: Path | None,
    validation_fraction: float,
    max_train_chars: int | None,
    document_boundary: str,
    trim_chapter_start: bool,
    edge_lines: int,
    min_repeated_pages: int,
    streaming: bool,
    require_gutenberg_markers: bool,
    strict_columns: bool,
    dedupe_ids: bool,
    max_documents: int | None,
) -> int:
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "datasets is required to stream the HuggingFace dataset. "
            "Install it with: pip install datasets"
        ) from exc

    try:
        from tqdm.auto import tqdm
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "tqdm is required for the progress bar. Install it with: pip install tqdm"
        ) from exc

    dataset = load_dataset(dataset_name, split=split, streaming=streaming)
    total = max_documents
    if total is None:
        try:
            total = len(dataset)  # type: ignore[arg-type]
        except TypeError:
            total = None

    stats: Counter[str] = Counter()
    removed_segments: Counter[str] = Counter()
    seen_ids: set[str] = set()

    validation_writer_context: CorpusWriter | None = None
    train_writer = CorpusWriter(
        output_path,
        document_boundary=document_boundary,
        max_characters=max_train_chars,
    )
    if validation_output_path is not None:
        validation_writer_context = CorpusWriter(
            validation_output_path,
            document_boundary=document_boundary,
        )

    with train_writer:
        if validation_writer_context is None:
            validation_writer = None
        else:
            validation_writer = validation_writer_context.__enter__()

        try:
            progress = tqdm(
                maybe_limited(dataset, max_documents),
                desc=f"{dataset_name}:{split}",
                total=total,
                unit="book",
            )
            for record in progress:
                stats["seen"] += 1
                validate_record_columns(record, strict_columns=strict_columns)

                document_id = str(record["id"]).strip()
                if dedupe_ids and document_id in seen_ids:
                    stats["skipped duplicate id"] += 1
                    continue

                raw_text = record["text"]
                if not isinstance(raw_text, str) or not raw_text.strip():
                    stats["skipped empty text"] += 1
                    continue

                (
                    text,
                    found_start_marker,
                    found_end_marker,
                    found_chapter_start,
                ) = clean_gutenberg_text(
                    raw_text,
                    trim_chapter_start=trim_chapter_start,
                    edge_lines=edge_lines,
                    min_repeated_pages=min_repeated_pages,
                )
                if require_gutenberg_markers and (
                    not found_start_marker or not found_end_marker
                ):
                    stats["skipped missing Gutenberg marker"] += 1
                    continue
                if not found_start_marker:
                    stats["missing start marker"] += 1
                if not found_end_marker:
                    stats["missing end marker"] += 1
                if trim_chapter_start and not found_chapter_start:
                    stats["missing first chapter marker"] += 1
                if not text:
                    stats["skipped empty after boilerplate removal"] += 1
                    continue

                text, document_removed_segments = remove_malformed_segments(text)
                removed_segments.update(document_removed_segments)
                if not text:
                    stats["skipped empty after malformed cleanup"] += 1
                    continue

                train_text, validation_text = split_document_text(
                    text,
                    validation_fraction,
                )
                if train_text:
                    if not train_writer.can_write_document(train_text):
                        stats["stopped max training characters"] += 1
                        progress.set_postfix(
                            kept=f"{stats['kept']:,}",
                            chars=f"{train_writer.character_count:,}",
                        )
                        break
                    train_writer.write_document(train_text)
                if validation_writer is not None and validation_text:
                    validation_writer.write_document(validation_text)

                stats["kept"] += 1
                seen_ids.add(document_id)
                progress.set_postfix(
                    kept=f"{stats['kept']:,}",
                    skipped=f"{stats['seen'] - stats['kept']:,}",
                    chars=f"{train_writer.character_count:,}",
                )

            if not train_writer.document_count:
                raise RuntimeError("No Project Gutenberg text could be extracted.")

            train_writer.commit()
            if validation_writer is not None:
                if validation_writer.document_count:
                    validation_writer.commit()
                else:
                    validation_writer.discard()
        except Exception:
            train_writer.discard()
            if validation_writer is not None:
                validation_writer.discard()
            raise
        finally:
            if (
                validation_writer_context is not None
                and validation_writer_context._handle is not None
            ):
                validation_writer_context.discard()

    print(f"\nWrote training corpus: {output_path}")
    if validation_writer_context is not None and validation_writer_context.document_count:
        print(f"Wrote validation corpus: {validation_output_path}")
    print(f"Input rows seen: {stats['seen']:,}")
    print(f"Documents kept: {stats['kept']:,}")
    print(f"Training characters: {train_writer.character_count:,}")
    if validation_writer_context is not None:
        print(f"Validation characters: {validation_writer_context.character_count:,}")
    if max_train_chars is not None:
        print(f"Training character cap: {max_train_chars:,}")
    if document_boundary:
        print(f"Document boundary marker: {document_boundary!r}")
    if trim_chapter_start:
        print(
            "Chapter-start trim: latest "
            f"{FIRST_CHAPTER_MARKER_DESCRIPTION} marker when found"
        )
    print(
        "Repeated edge-line filter: "
        f"edge_lines={edge_lines:,} min_repeated_pages={min_repeated_pages:,}"
    )
    if stats["stopped max training characters"]:
        print("Stopped before adding the next document because the cap was reached.")
    skipped = stats["seen"] - stats["kept"]
    if skipped:
        skip_reasons = Counter(
            {
                reason: count
                for reason, count in stats.items()
                if reason.startswith("skipped") or reason.startswith("stopped")
            }
        )
        print(f"Skipped rows: {skipped:,} ({format_counter(skip_reasons)})")
    if removed_segments:
        print(f"Removed malformed text spans: {format_counter(removed_segments)}")
    if stats["missing start marker"] or stats["missing end marker"]:
        print(
            "Rows with missing Gutenberg markers kept: "
            f"start={stats['missing start marker']:,} end={stats['missing end marker']:,}"
        )
    if stats["missing first chapter marker"]:
        print(
            "Rows with no first-chapter marker kept: "
            f"{stats['missing first chapter marker']:,}"
        )
    return 0


def positive_int_arg(raw_value: str) -> int:
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return value


def nonnegative_int_arg(raw_value: str) -> int:
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if value < 0:
        raise argparse.ArgumentTypeError("must be greater than or equal to 0")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stream the English split of manu/project_gutenberg and write cleaned "
            "train/validation .txt corpora."
        )
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET_NAME,
        help="HuggingFace dataset name.",
    )
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        help="Dataset split to read. The default uses the English books.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output training .txt corpus path.",
    )
    parser.add_argument(
        "--validation-output",
        type=Path,
        default=None,
        help=(
            "Output validation .txt corpus path. Defaults to "
            "<output stem>_validation.txt."
        ),
    )
    parser.add_argument(
        "--validation-fraction",
        type=validation_fraction_arg,
        default=DEFAULT_VALIDATION_FRACTION,
        help=(
            "Fraction of each extracted book reserved for validation. "
            "Set to 0 to write only the training corpus."
        ),
    )
    parser.add_argument(
        "--max-train-chars",
        type=nonnegative_int_arg,
        default=DEFAULT_MAX_TRAIN_CHARS,
        help=(
            "Maximum characters to write to the training corpus. "
            "The default is 200,000,000. Use 0 for no cap."
        ),
    )
    parser.add_argument(
        "--document-boundary",
        default=DEFAULT_DOCUMENT_BOUNDARY,
        help=(
            "Marker written before each source document so the model can learn "
            "where one text ends and another begins. Use an empty value to disable."
        ),
    )
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Download/cache the split locally instead of streaming it.",
    )
    parser.add_argument(
        "--allow-missing-chapter-start",
        action="store_true",
        help=(
            "Deprecated: books with no "
            f"{FIRST_CHAPTER_MARKER_DESCRIPTION} marker are now always kept."
        ),
    )
    parser.add_argument(
        "--no-chapter-start-trim",
        action="store_true",
        help=(
            "Do not trim to the latest "
            f"{FIRST_CHAPTER_MARKER_DESCRIPTION} marker when one is found."
        ),
    )
    parser.add_argument(
        "--edge-lines",
        type=int,
        default=DEFAULT_EDGE_LINES,
        help=(
            "Number of first/last inferred page lines checked for repeated "
            "headers/footers, matching build_training_corpus.py."
        ),
    )
    parser.add_argument(
        "--min-repeated-pages",
        type=int,
        default=DEFAULT_MIN_REPEATED_PAGES,
        help="Remove edge lines repeated on at least this many inferred pages.",
    )
    parser.add_argument(
        "--allow-missing-markers",
        action="store_true",
        help=(
            "Deprecated: Project Gutenberg START/END markers are now optional "
            "by default."
        ),
    )
    parser.add_argument(
        "--require-gutenberg-markers",
        action="store_true",
        help=(
            "Skip rows missing Project Gutenberg START/END markers. By default, "
            "rows are kept and cleaned with the remaining preprocessing rules."
        ),
    )
    parser.add_argument(
        "--allow-extra-columns",
        action="store_true",
        help="Ignore columns other than id and text if the hosted dataset schema changes.",
    )
    parser.add_argument(
        "--keep-duplicate-ids",
        action="store_true",
        help="Keep later rows with an id that has already been processed.",
    )
    parser.add_argument(
        "--max-documents",
        type=positive_int_arg,
        default=None,
        help="Optional cap for smoke tests or partial corpus builds.",
    )
    return parser


def main() -> int:
    load_huggingface_env()
    args = build_parser().parse_args()
    validation_output_path = None
    if args.validation_fraction > 0:
        validation_output_path = args.validation_output or validation_output_path_for(
            args.output
        )

    try:
        return build_corpus(
            dataset_name=args.dataset,
            split=args.split,
            output_path=args.output,
            validation_output_path=validation_output_path,
            validation_fraction=args.validation_fraction,
            max_train_chars=(
                None if args.max_train_chars == 0 else args.max_train_chars
            ),
            document_boundary=args.document_boundary,
            trim_chapter_start=not args.no_chapter_start_trim,
            edge_lines=args.edge_lines,
            min_repeated_pages=args.min_repeated_pages,
            streaming=not args.no_streaming,
            require_gutenberg_markers=args.require_gutenberg_markers,
            strict_columns=not args.allow_extra_columns,
            dedupe_ids=not args.keep_duplicate_ids,
            max_documents=args.max_documents,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
