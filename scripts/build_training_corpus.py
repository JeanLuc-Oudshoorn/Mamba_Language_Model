"""Build cleaned train/validation corpora from every PDF in the data folder."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import re
import sys
import unicodedata


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_PATH = DEFAULT_DATA_DIR / "nineteenth_century_english_literature.txt"
DEFAULT_VALIDATION_FRACTION = 0.10

PAGE_NUMBER_CORE_PATTERN = r"(?:page\s+)?\d+(?:\s*(?:of|/)\s*\d+)?"
PAGE_NUMBER_BRACKETED_PATTERN = (
    rf"(?:\[\s*{PAGE_NUMBER_CORE_PATTERN}\s*\]"
    rf"|\{{\s*{PAGE_NUMBER_CORE_PATTERN}\s*\}})"
)
PAGE_NUMBER_REF_PATTERN = (
    rf"(?:{PAGE_NUMBER_CORE_PATTERN}|{PAGE_NUMBER_BRACKETED_PATTERN})"
)
PAGE_NUMBER_LINE_RE = re.compile(rf"^\s*{PAGE_NUMBER_REF_PATTERN}\s*$", re.I)
EDGE_PAGE_NUMBER_PREFIX_RE = re.compile(rf"^\s*{PAGE_NUMBER_REF_PATTERN}\W+", re.I)
EDGE_PAGE_NUMBER_SUFFIX_RE = re.compile(rf"\W+{PAGE_NUMBER_REF_PATTERN}\s*$", re.I)
TRAILING_BARE_NUMBER_RE = re.compile(r"^(?P<body>.*?)[ \t]+(?P<number>\d+)[ \t]*$")
WHITESPACE_RE = re.compile(r"[ \t]+")
MULTIPLE_BLANK_LINES_RE = re.compile(r"\n{3,}")
PARAGRAPH_SPLIT_RE = re.compile(r"\n{2,}")
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")
LETTER_WORD_RE = re.compile(r"[^\W\d_]+")
ROMAN_NUMERAL_PATTERN = (
    r"(?=[MDCLXVI])M{0,4}(?:CM|CD|D?C{0,3})"
    r"(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})"
)
ROMAN_NUMERAL_LINE_RE = re.compile(rf"^\s*(?:{ROMAN_NUMERAL_PATTERN})[.)]?\s*$", re.I)
CHAPTER_MARKER_LINE_RE = re.compile(
    rf"^\s*chapter\s+(?:\d+|{ROMAN_NUMERAL_PATTERN})[.)]?\s*$",
    re.I,
)
SOURCE_START_RE = re.compile(r"^\W*chapter\s+(?:i|1)\b", re.I)
SOURCE_END_RE = re.compile(r"^\W*the\s+end\b", re.I)
SPECIAL_SEPARATOR_CHARS = (
    "\u2666\u25c6\u25c7\u25ca\u25cf\u25cb\u2022\u25e6"
    "\u25aa\u25ab\u25a0\u25a1\u25a2\u25a3\u2b29\u2756"
    "\u2726\u2727\u203b\u2042\u2766\u2767"
)
SPECIAL_SEPARATOR_RE = re.compile(f"[{re.escape(SPECIAL_SEPARATOR_CHARS)}]")
RUNNING_HEAD_EDGE_CHARS = " .,:;|/-_()[]{}'\""
LONG_WORD_LENGTH = 40
MIN_CONSECUTIVE_SINGLE_CHAR_WORDS = 8


def validation_output_path_for(output_path: Path) -> Path:
    suffix = output_path.suffix or ".txt"
    if output_path.stem.endswith("_train"):
        stem = f"{output_path.stem[:-len('_train')]}_validation"
    else:
        stem = f"{output_path.stem}_validation"
    return output_path.with_name(f"{stem}{suffix}")


def validation_fraction_arg(raw_value: str) -> float:
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a floating-point number") from exc

    if not 0 <= value < 1:
        raise argparse.ArgumentTypeError("must be >= 0 and < 1")
    return value


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\ufeff", "")
    text = text.replace("\u00ad", "")
    text = text.replace("\u200b", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text


def trim_source_boundaries(raw_pages: list[list[str]]) -> list[list[str]]:
    pages = [page_lines[:] for page_lines in raw_pages]

    start_location: tuple[int, int, int] | None = None
    for page_index, page_lines in enumerate(pages):
        for line_index, line in enumerate(page_lines):
            match = SOURCE_START_RE.search(line)
            if match:
                start_location = (page_index, line_index, match.start())

    if start_location is not None:
        start_page, start_line, start_char = start_location
        for page_index in range(start_page):
            pages[page_index] = []
        pages[start_page] = pages[start_page][start_line:]
        pages[start_page][0] = pages[start_page][0][start_char:].lstrip()

    for page_index, page_lines in enumerate(pages):
        for line_index, line in enumerate(page_lines):
            match = SOURCE_END_RE.search(line)
            if not match:
                continue
            pages[page_index] = page_lines[: line_index + 1]
            pages[page_index][line_index] = line[: match.end()]
            for following_page_index in range(page_index + 1, len(pages)):
                pages[following_page_index] = []
            return pages

    return pages


def clean_line(line: str) -> str | None:
    line = normalize_text(line)
    line = SPECIAL_SEPARATOR_RE.sub(" ", line)
    line = WHITESPACE_RE.sub(" ", line).strip()
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


def line_key(line: str) -> str:
    return WHITESPACE_RE.sub(" ", line.strip()).casefold()


def is_mostly_uppercase(line: str) -> bool:
    letters = [char for char in line if char.isalpha()]
    if len(letters) < 3:
        return False
    return sum(char.isupper() for char in letters) / len(letters) >= 0.8


def uppercase_running_head_key(line: str) -> str | None:
    line = EDGE_PAGE_NUMBER_PREFIX_RE.sub("", line)
    line = EDGE_PAGE_NUMBER_SUFFIX_RE.sub("", line)
    line = WHITESPACE_RE.sub(" ", line).strip(RUNNING_HEAD_EDGE_CHARS)
    if not line or not is_mostly_uppercase(line):
        return None
    return line_key(line)


def numbered_running_head_key(line: str) -> str | None:
    line, prefix_count = EDGE_PAGE_NUMBER_PREFIX_RE.subn("", line, count=1)
    line, suffix_count = EDGE_PAGE_NUMBER_SUFFIX_RE.subn("", line, count=1)
    if prefix_count + suffix_count == 0:
        return None
    line = WHITESPACE_RE.sub(" ", line).strip(RUNNING_HEAD_EDGE_CHARS)
    if not line:
        return None
    return line_key(line)


def edge_line_indices(page_lines: list[str | None], edge_lines: int) -> set[int]:
    if edge_lines <= 0:
        return set()
    nonempty_indices = [index for index, line in enumerate(page_lines) if line]
    return set(nonempty_indices[:edge_lines] + nonempty_indices[-edge_lines:])


def repeated_edge_lines(
    pages: list[list[str | None]],
    edge_lines: int,
    min_pages: int,
) -> tuple[set[str], set[str], set[str]]:
    exact_counts: Counter[str] = Counter()
    uppercase_counts: Counter[str] = Counter()
    numbered_counts: Counter[str] = Counter()
    for page_lines in pages:
        exact_candidates: set[str] = set()
        uppercase_candidates: set[str] = set()
        numbered_candidates: set[str] = set()
        for index in edge_line_indices(page_lines, edge_lines):
            line = page_lines[index]
            if line is None:
                continue
            exact_candidates.add(line_key(line))
            uppercase_key = uppercase_running_head_key(line)
            if uppercase_key:
                uppercase_candidates.add(uppercase_key)
            numbered_key = numbered_running_head_key(line)
            if numbered_key:
                numbered_candidates.add(numbered_key)
        exact_counts.update(exact_candidates)
        uppercase_counts.update(uppercase_candidates)
        numbered_counts.update(numbered_candidates)

    exact_lines = {
        key
        for key, count in exact_counts.items()
        if (
            count >= min_pages
            and len(key) <= 120
            and not PAGE_NUMBER_LINE_RE.fullmatch(key)
        )
    }
    uppercase_lines = {
        key
        for key, count in uppercase_counts.items()
        if (
            count >= min_pages
            and len(key) <= 120
            and not PAGE_NUMBER_LINE_RE.fullmatch(key)
        )
    }
    numbered_lines = {
        key
        for key, count in numbered_counts.items()
        if (
            count >= min_pages
            and len(key) <= 120
            and not PAGE_NUMBER_LINE_RE.fullmatch(key)
        )
    }
    return exact_lines, uppercase_lines, numbered_lines


def should_remove_line(
    line: str,
    line_index: int,
    page_edge_indices: set[int],
    repeated_exact_lines: set[str],
    repeated_uppercase_lines: set[str],
    repeated_numbered_lines: set[str],
) -> bool:
    if line_key(line) in repeated_exact_lines:
        return True
    if line_index not in page_edge_indices:
        return False
    uppercase_key = uppercase_running_head_key(line)
    if uppercase_key is not None and uppercase_key in repeated_uppercase_lines:
        return True
    numbered_key = numbered_running_head_key(line)
    return numbered_key is not None and numbered_key in repeated_numbered_lines


def join_wrapped_lines(lines: list[str]) -> str:
    paragraphs: list[str] = []
    current: list[str] = []

    def flush_current() -> None:
        if current:
            paragraphs.append(" ".join(current))
            current.clear()

    for line in lines:
        if not line:
            flush_current()
            continue

        if current and current[-1].endswith("-"):
            current[-1] = current[-1][:-1] + line
        else:
            current.append(line)

    flush_current()
    return "\n\n".join(paragraphs)


def long_words(text: str) -> list[str]:
    return [
        word
        for word in LETTER_WORD_RE.findall(text)
        if len(word) > LONG_WORD_LENGTH
    ]


def max_consecutive_single_char_words(sentence: str) -> int:
    longest_run = 0
    current_run = 0
    for word in LETTER_WORD_RE.findall(sentence):
        if len(word) == 1:
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 0
    return longest_run


def malformed_segment_reason(segment: str) -> str | None:
    if long_words(segment):
        return f"word longer than {LONG_WORD_LENGTH} characters"
    if (
        max_consecutive_single_char_words(segment)
        >= MIN_CONSECUTIVE_SINGLE_CHAR_WORDS
    ):
        return "letter-spaced OCR"
    return None


def remove_malformed_segments(text: str) -> tuple[str, Counter[str]]:
    removed: Counter[str] = Counter()
    cleaned_paragraphs: list[str] = []

    for paragraph in PARAGRAPH_SPLIT_RE.split(text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        kept_segments: list[str] = []
        for segment in SENTENCE_BOUNDARY_RE.split(paragraph):
            segment = segment.strip()
            if not segment:
                continue
            reason = malformed_segment_reason(segment)
            if reason:
                removed[reason] += 1
                continue
            kept_segments.append(segment)

        if kept_segments:
            cleaned_paragraphs.append(" ".join(kept_segments))

    cleaned_text = "\n\n".join(cleaned_paragraphs)
    cleaned_text = MULTIPLE_BLANK_LINES_RE.sub("\n\n", cleaned_text)
    return cleaned_text.strip(), removed


def split_text_at_word_boundary(
    text: str,
    validation_fraction: float,
) -> tuple[str, str]:
    if validation_fraction <= 0 or len(text) < 2:
        return text.strip(), ""

    split_index = round(len(text) * (1 - validation_fraction))
    split_index = max(1, min(len(text) - 1, split_index))
    window = max(1, min(len(text) // 10, 1000))
    search_start = max(1, split_index - window)
    search_end = min(len(text) - 1, split_index + window)
    boundary_candidates = [
        index for index in range(search_start, search_end) if text[index].isspace()
    ]
    if boundary_candidates:
        split_index = min(
            boundary_candidates,
            key=lambda index: abs(index - split_index),
        )

    return text[:split_index].strip(), text[split_index:].strip()


def split_document_text(
    text: str,
    validation_fraction: float,
) -> tuple[str, str]:
    text = text.strip()
    if validation_fraction <= 0 or len(text) < 2:
        return text, ""

    paragraphs = [
        paragraph.strip()
        for paragraph in PARAGRAPH_SPLIT_RE.split(text)
        if paragraph.strip()
    ]
    if len(paragraphs) < 2:
        return split_text_at_word_boundary(text, validation_fraction)

    total_chars = sum(len(paragraph) for paragraph in paragraphs)
    target_validation_chars = total_chars * validation_fraction
    train_paragraphs: list[str] = []
    validation_paragraphs: list[str] = []
    validation_chars = 0
    consumed_chars = 0

    for paragraph in paragraphs:
        paragraph_len = len(paragraph)
        desired_validation_chars = (consumed_chars + paragraph_len) * validation_fraction
        if (
            validation_chars + paragraph_len / 2 <= desired_validation_chars
            and validation_chars < target_validation_chars
        ):
            validation_paragraphs.append(paragraph)
            validation_chars += paragraph_len
        else:
            train_paragraphs.append(paragraph)
        consumed_chars += paragraph_len

    train_text = "\n\n".join(train_paragraphs).strip()
    validation_text = "\n\n".join(validation_paragraphs).strip()
    if not train_text or not validation_text:
        return split_text_at_word_boundary(text, validation_fraction)
    return train_text, validation_text


def join_documents(documents: list[str]) -> str:
    return (
        "\n\n".join(document.strip() for document in documents if document.strip()).strip()
        + "\n"
    )


def extract_pdf_text(path: Path, edge_lines: int, min_repeated_pages: int) -> tuple[str, int]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"could not decrypt encrypted PDF: {exc}") from exc

    raw_pages: list[list[str]] = []
    for page in reader.pages:
        raw_text = page.extract_text() or ""
        raw_pages.append(normalize_text(raw_text).splitlines())

    pages = [
        [clean_line(line) for line in page_lines]
        for page_lines in trim_source_boundaries(raw_pages)
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
    for page_lines in pages:
        page_edge_indices = edge_line_indices(page_lines, edge_lines)
        for index, line in enumerate(page_lines):
            if line is None:
                continue
            if not line:
                cleaned_lines.append("")
                continue
            if should_remove_line(
                line,
                line_index=index,
                page_edge_indices=page_edge_indices,
                repeated_exact_lines=repeated_exact_lines,
                repeated_uppercase_lines=repeated_uppercase_lines,
                repeated_numbered_lines=repeated_numbered_lines,
            ):
                continue
            cleaned_lines.append(line)

    text = join_wrapped_lines(cleaned_lines)
    text = MULTIPLE_BLANK_LINES_RE.sub("\n\n", text)
    return text.strip(), len(reader.pages)


def build_corpus(
    data_dir: Path,
    output_path: Path,
    validation_output_path: Path | None,
    validation_fraction: float,
    edge_lines: int,
    min_repeated_pages: int,
) -> int:
    try:
        import pypdf  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError("pypdf is required to extract PDF text.") from exc

    pdf_paths = sorted(data_dir.glob("*.pdf"), key=lambda path: path.name.casefold())
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {data_dir}")

    train_documents: list[str] = []
    validation_documents: list[str] = []
    for pdf_path in pdf_paths:
        print(f"Extracting {pdf_path.name}...")
        try:
            text, page_count = extract_pdf_text(
                pdf_path,
                edge_lines=edge_lines,
                min_repeated_pages=min_repeated_pages,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  skipped: {exc}", file=sys.stderr)
            continue

        if not text:
            print("  skipped: no extractable text found", file=sys.stderr)
            continue

        text, removed_segments = remove_malformed_segments(text)
        if removed_segments:
            removed_summary = ", ".join(
                f"{count:,} {reason}" for reason, count in removed_segments.items()
            )
            print(f"  removed malformed text spans: {removed_summary}")

        if not text:
            print(
                "  skipped: no extractable text remained after OCR cleanup",
                file=sys.stderr,
            )
            continue

        train_text, validation_text = split_document_text(text, validation_fraction)
        if train_text:
            train_documents.append(train_text)
        if validation_text:
            validation_documents.append(validation_text)
        print(
            f"  pages={page_count:,} chars={len(text):,} "
            f"train={len(train_text):,} validation={len(validation_text):,}"
        )

    if not train_documents:
        raise RuntimeError("No PDF text could be extracted.")

    corpus = join_documents(train_documents)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(corpus, encoding="utf-8")

    validation_corpus = ""
    if validation_output_path is not None and validation_documents:
        validation_corpus = join_documents(validation_documents)
        validation_output_path.parent.mkdir(parents=True, exist_ok=True)
        validation_output_path.write_text(validation_corpus, encoding="utf-8")

    print(f"\nWrote training corpus: {output_path}")
    if validation_corpus:
        print(f"Wrote validation corpus: {validation_output_path}")
    print(f"Documents: {len(train_documents):,}")
    print(f"Training characters: {len(corpus):,}")
    if validation_corpus:
        print(f"Validation characters: {len(validation_corpus):,}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract and clean every PDF in data/ into train/validation corpora."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Folder containing PDF files.",
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
            "Fraction of each extracted PDF document reserved for validation. "
            "Set to 0 to write only the training corpus."
        ),
    )
    parser.add_argument(
        "--edge-lines",
        type=int,
        default=3,
        help=(
            "Number of first/last page lines checked for repeated headers/footers, "
            "including all-caps running heads with page numbers."
        ),
    )
    parser.add_argument(
        "--min-repeated-pages",
        type=int,
        default=4,
        help="Remove edge lines repeated on at least this many pages.",
    )

    return parser


def main() -> int:
    args = build_parser().parse_args()
    validation_output_path = None
    if args.validation_fraction > 0:
        validation_output_path = args.validation_output or validation_output_path_for(
            args.output
        )
    return build_corpus(
        data_dir=args.data_dir,
        output_path=args.output,
        validation_output_path=validation_output_path,
        validation_fraction=args.validation_fraction,
        edge_lines=args.edge_lines,
        min_repeated_pages=args.min_repeated_pages,
    )


if __name__ == "__main__":
    raise SystemExit(main())
