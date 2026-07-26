"""Download small text samples from Library of Congress digitized books.

The Library of Congress data package used here is documented at:
https://libraryofcongress.github.io/data-exploration/Data%20Packages/digitized-books.html
"""

import argparse
import csv
import json
from pathlib import Path
import re
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_URL = "https://data.labs.loc.gov/digitized-books/"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "library_of_congress"
DEFAULT_PER_CATEGORY = 3
DEFAULT_CATEGORIES = (
    "history",
    "united states",
    "description and travel",
    "politics and government",
    "biography",
    "world war",
    "civil war",
    "poetry",
    "education",
    "grammar",
)
METADATA_CACHE_NAME = "metadata.json"
MANIFEST_CACHE_NAME = "manifest.json"
TEXT_SUFFIX = ".txt"


def slugify(value: str) -> str:
    value = value.casefold().strip()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "uncategorized"


def safe_filename(value: str, max_length: int = 100) -> str:
    value = re.sub(r"[^\w .,'()-]+", " ", value, flags=re.ASCII)
    value = re.sub(r"\s+", " ", value).strip(" .")
    return (value[:max_length].rstrip(" .") or "untitled") + ".txt"


def import_requests() -> Any:
    try:
        import requests
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "requests is required. Install it with: pip install requests"
        ) from exc
    return requests


def fetch_json(url: str) -> Any:
    requests = import_requests()
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    return response.json()


def cached_json(cache_path: Path, url: str, refresh: bool) -> Any:
    if cache_path.exists() and not refresh:
        return json.loads(cache_path.read_text(encoding="utf-8"))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    data = fetch_json(url)
    temp_path = cache_path.with_name(f"{cache_path.name}.tmp")
    temp_path.write_text(json.dumps(data), encoding="utf-8")
    temp_path.replace(cache_path)
    return data


def rows_from_manifest(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    columns = manifest["cols"]
    return [dict(zip(columns, row)) for row in manifest["rows"]]


def rows_from_metadata(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    return list(metadata.values())


def normalized_values(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        values = value
    else:
        values = [value]
    return [str(item).casefold().strip() for item in values if str(item).strip()]


def is_english_book(row: dict[str, Any]) -> bool:
    languages = normalized_values(row.get("language"))
    return not languages or "english" in languages or "en" in languages


def item_subjects(row: dict[str, Any]) -> set[str]:
    return set(normalized_values(row.get("subject")))


def text_files_by_item(manifest_rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    text_files: dict[str, dict[str, Any]] = {}
    for row in manifest_rows:
        object_key = str(row.get("object_key", ""))
        item_id = str(row.get("item_id", "")).strip()
        if item_id and object_key.endswith(TEXT_SUFFIX):
            text_files.setdefault(item_id, row)
    return text_files


def selected_items_for_category(
    category: str,
    metadata_rows: Iterable[dict[str, Any]],
    text_files: dict[str, dict[str, Any]],
    per_category: int,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    normalized_category = category.casefold().strip()

    for row in metadata_rows:
        item_id = str(row.get("id", "")).strip()
        if not item_id or item_id not in text_files:
            continue
        if normalized_category not in item_subjects(row):
            continue
        if not is_english_book(row):
            continue
        selected.append((row, text_files[item_id]))
        if len(selected) >= per_category:
            break

    return selected


def download_text_file(url: str, path: Path, refresh: bool) -> None:
    if path.exists() and not refresh:
        return

    requests = import_requests()
    path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    temp_path = path.with_name(f"{path.name}.tmp")
    temp_path.write_text(response.text, encoding="utf-8", errors="replace")
    temp_path.replace(path)


def book_output_path(
    output_dir: Path,
    category: str,
    index: int,
    metadata_row: dict[str, Any],
) -> Path:
    title = str(metadata_row.get("title") or "untitled")
    item_id = slugify(str(metadata_row.get("id") or f"book-{index}"))
    filename = f"{index:02d}-{item_id}-{safe_filename(title)}"
    return output_dir / slugify(category) / filename


def text_metrics(path: Path) -> dict[str, int]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return {
        "bytes": path.stat().st_size,
        "characters": len(text),
        "lines": len(text.splitlines()),
        "words": len(re.findall(r"\S+", text)),
    }


def download_samples(
    output_dir: Path,
    categories: list[str],
    per_category: int,
    refresh_metadata: bool,
    refresh_books: bool,
) -> list[dict[str, Any]]:
    cache_dir = output_dir / "_cache"
    metadata = cached_json(
        cache_dir / METADATA_CACHE_NAME,
        f"{DATA_URL}metadata.json",
        refresh=refresh_metadata,
    )
    manifest = cached_json(
        cache_dir / MANIFEST_CACHE_NAME,
        f"{DATA_URL}manifest.json",
        refresh=refresh_metadata,
    )

    metadata_rows = rows_from_metadata(metadata)
    text_files = text_files_by_item(rows_from_manifest(manifest))
    summary_rows: list[dict[str, Any]] = []

    print(
        f"Loaded {len(metadata_rows):,} metadata records and "
        f"{len(text_files):,} text files."
    )

    for category in categories:
        selected = selected_items_for_category(
            category,
            metadata_rows,
            text_files,
            per_category,
        )
        print(f"{category}: selected {len(selected):,} text files")

        for index, (metadata_row, file_row) in enumerate(selected, start=1):
            object_key = str(file_row["object_key"])
            file_url = f"https://{object_key}"
            text_path = book_output_path(output_dir, category, index, metadata_row)
            download_text_file(file_url, text_path, refresh=refresh_books)
            preview_path = text_path.with_suffix(".preview.txt")
            if preview_path.exists():
                preview_path.unlink()
            metrics = text_metrics(text_path)

            summary_rows.append(
                {
                    "category": category,
                    "item_id": metadata_row.get("id", ""),
                    "title": metadata_row.get("title", ""),
                    "date": metadata_row.get("date", ""),
                    "language": "; ".join(normalized_values(metadata_row.get("language"))),
                    "subjects": "; ".join(sorted(item_subjects(metadata_row))),
                    "loc_url": metadata_row.get("url", ""),
                    "text_url": file_url,
                    "text_path": str(text_path),
                    **metrics,
                }
            )

    return summary_rows


def write_summary(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "sample_manifest.json"
    csv_path = output_dir / "sample_manifest.csv"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    if not rows:
        csv_path.write_text("", encoding="utf-8")
        return

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote review manifest: {json_path}")
    print(f"Wrote review manifest: {csv_path}")


def category_arg(raw_value: str) -> list[str]:
    categories = [item.strip().casefold() for item in raw_value.split(",")]
    return [item for item in categories if item]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download a few English .txt books from selected Library of "
            "Congress digitized-books subject categories."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Folder where category samples and review manifests are written.",
    )
    parser.add_argument(
        "--categories",
        type=category_arg,
        default=list(DEFAULT_CATEGORIES),
        help=(
            "Comma-separated exact LOC subject categories. Defaults to the top "
            "subjects shown in the LOC tutorial."
        ),
    )
    parser.add_argument(
        "--per-category",
        type=int,
        default=DEFAULT_PER_CATEGORY,
        help="Number of books to download per category.",
    )
    parser.add_argument(
        "--refresh-metadata",
        action="store_true",
        help="Re-download metadata.json and manifest.json instead of using cache.",
    )
    parser.add_argument(
        "--refresh-books",
        action="store_true",
        help="Re-download book text files even when local copies already exist.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.per_category <= 0:
        raise SystemExit("--per-category must be greater than 0")

    rows = download_samples(
        output_dir=args.output_dir,
        categories=args.categories,
        per_category=args.per_category,
        refresh_metadata=args.refresh_metadata,
        refresh_books=args.refresh_books,
    )
    write_summary(args.output_dir, rows)
    print(f"Downloaded {len(rows):,} sample books into {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
