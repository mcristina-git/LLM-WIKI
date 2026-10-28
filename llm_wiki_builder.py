from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

BUILDER_VERSION = "3.2"

OBSIDIAN_DIR = Path(
    os.environ.get("OBSIDIAN_DIR", Path(__file__).resolve().parent)
)

RAW_DIR = OBSIDIAN_DIR / "Clippings"
WIKI_DIR = OBSIDIAN_DIR / "AI Wiki"

SOURCES_DIR = WIKI_DIR / "sources"
REQUIREMENTS_DIR = WIKI_DIR / "requirements"

INDEX_FILE = WIKI_DIR / "index.md"
CATALOG_FILE = WIKI_DIR / "catalog.json"
BUILD_REPORT_FILE = WIKI_DIR / "build_report.json"

CHUNK_MAX_CHARS = 6500

TRACKED_DELETION_MARKER = (
    "Tracked deletion in source"
)

VOID_RE = re.compile(
    r"^\*{0,4}\s*Void\s*\*{0,4}$",
    re.IGNORECASE,
)


# ============================================================
# DATA
# ============================================================

@dataclass
class SourceChunk:
    source_path: Path
    source_title: str
    source_slug: str
    chunk_number: int
    content: str
    headings: list[str] = field(default_factory=list)
    requirement_ids: list[str] = field(default_factory=list)


@dataclass
class RequirementRecord:
    requirement_id: str
    document: str
    source_slug: str
    chunk_number: int
    text: str
    status: str
    source_md: str
    chunk_path: str
    requirement_path: str


# ============================================================
# BASIC HELPERS
# ============================================================

def read_text(path: Path) -> str:
    if not path.exists():
        return ""

    return path.read_text(
        encoding="utf-8"
    )


def write_text(
    path: Path,
    content: str
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    path.write_text(
        content.strip() + "\n",
        encoding="utf-8"
    )


def slugify(value: str) -> str:
    value = re.sub(
        r"[^a-z0-9]+",
        "-",
        value.lower().strip()
    )

    return value.strip("-") or "untitled"


def normalize_requirement_id(
    value: str
) -> str:
    match = re.search(
        r"R?\s*(\d+(?:\.\d+)+)",
        value,
        re.IGNORECASE
    )

    if not match:
        return value.strip()

    return f"R {match.group(1)}"


def markdown_heading_level(
    line: str
) -> int | None:
    match = re.match(
        r"^(#{1,6})\s+(.+)$",
        line
    )

    if not match:
        return None

    return len(
        match.group(1)
    )


def clean_heading_text(
    line: str
) -> str:
    return re.sub(
        r"^#{1,6}\s*",
        "",
        line
    ).strip()


def extract_requirement_id_from_heading(
    heading: str
) -> str | None:
    match = re.match(
        r"^R\s*(\d+(?:\.\d+)+)\b",
        heading,
        re.IGNORECASE
    )

    if not match:
        return None

    return normalize_requirement_id(
        match.group(1)
    )


def build_wiki_link_target(
    path: Path
) -> str:
    return str(
        path.relative_to(
            WIKI_DIR
        ).with_suffix("")
    ).replace("\\", "/")


def build_wiki_link(
    path: Path
) -> str:
    return (
        f"[[{build_wiki_link_target(path)}]]"
    )


# ============================================================
# INPUT DISCOVERY
# ============================================================

def list_raw_files() -> list[Path]:
    """
    The stable ingestion input is Markdown only.

    Files such as:
      *.conversion.json
    are intentionally ignored.
    """
    if not RAW_DIR.exists():
        return []

    return sorted(
        path
        for path in RAW_DIR.rglob("*.md")
        if path.is_file()
    )


# ============================================================
# STRUCTURAL SPLITTING
# ============================================================

def split_markdown_into_units(
    content: str
) -> list[str]:
    """
    Split on Markdown headings.

    A requirement generated as:

        #### R 5.1.14
        UE SHOULD ...

    remains an atomic structural unit.
    """
    lines = content.splitlines()

    units: list[str] = []
    current: list[str] = []

    for line in lines:
        if (
            markdown_heading_level(line)
            is not None
        ):
            if (
                current
                and any(
                    item.strip()
                    for item in current
                )
            ):
                units.append(
                    "\n".join(
                        current
                    ).strip()
                )

            current = [line]
        else:
            current.append(line)

    if (
        current
        and any(
            item.strip()
            for item in current
        )
    ):
        units.append(
            "\n".join(
                current
            ).strip()
        )

    return [
        unit
        for unit in units
        if unit
    ]


def split_oversized_unit(
    unit: str,
    max_chars: int
) -> list[str]:
    """
    Split very large non-requirement sections conservatively.

    No content is summarized or deleted.
    """
    if len(unit) <= max_chars:
        return [unit]

    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(
            r"\n\s*\n",
            unit
        )
        if paragraph.strip()
    ]

    pieces: list[str] = []
    current = ""

    for paragraph in paragraphs:
        candidate = (
            paragraph
            if not current
            else current
            + "\n\n"
            + paragraph
        )

        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            pieces.append(current)
            current = ""

        if len(paragraph) <= max_chars:
            current = paragraph
            continue

        # Last resort: split by lines.
        line_buffer = ""

        for line in paragraph.splitlines():
            candidate = (
                line
                if not line_buffer
                else line_buffer
                + "\n"
                + line
            )

            if len(candidate) <= max_chars:
                line_buffer = candidate
            else:
                if line_buffer:
                    pieces.append(
                        line_buffer
                    )

                line_buffer = line

        if line_buffer:
            current = line_buffer

    if current:
        pieces.append(current)

    return pieces


def extract_headings(
    text: str
) -> list[str]:
    return [
        clean_heading_text(line)
        for line in text.splitlines()
        if (
            markdown_heading_level(line)
            is not None
        )
    ]


def extract_requirement_ids(
    text: str
) -> list[str]:
    ids = []

    for heading in extract_headings(
        text
    ):
        req_id = (
            extract_requirement_id_from_heading(
                heading
            )
        )

        if req_id:
            ids.append(req_id)

    return ids


def build_chunks_for_source(
    source_path: Path
) -> list[SourceChunk]:
    content = read_text(
        source_path
    )

    source_title = (
        source_path.stem
    )

    source_slug = slugify(
        source_title
    )

    raw_units = split_markdown_into_units(
        content
    )

    units: list[str] = []

    for unit in raw_units:
        # Requirement blocks should remain atomic whenever possible.
        first_line = (
            unit.splitlines()[0]
            if unit.splitlines()
            else ""
        )

        heading = clean_heading_text(
            first_line
        )

        is_requirement = (
            extract_requirement_id_from_heading(
                heading
            )
            is not None
        )

        if (
            is_requirement
            and len(unit) <= CHUNK_MAX_CHARS
        ):
            units.append(unit)
        else:
            units.extend(
                split_oversized_unit(
                    unit,
                    CHUNK_MAX_CHARS
                )
            )

    chunks: list[SourceChunk] = []
    current_units: list[str] = []
    current_length = 0
    chunk_number = 1

    def flush():
        nonlocal current_units
        nonlocal current_length
        nonlocal chunk_number

        if not current_units:
            return

        text = "\n\n".join(
            current_units
        ).strip()

        chunks.append(
            SourceChunk(
                source_path=source_path,
                source_title=source_title,
                source_slug=source_slug,
                chunk_number=chunk_number,
                content=text,
                headings=extract_headings(text),
                requirement_ids=(
                    extract_requirement_ids(text)
                ),
            )
        )

        chunk_number += 1
        current_units = []
        current_length = 0

    for unit in units:
        extra = len(unit)

        if current_units:
            extra += 2

        if (
            current_units
            and current_length
            + extra
            > CHUNK_MAX_CHARS
        ):
            flush()

        current_units.append(unit)
        current_length += extra

    flush()

    return chunks


# ============================================================
# GENERATED PATHS
# ============================================================

def get_document_dir(
    source_slug: str
) -> Path:
    return (
        SOURCES_DIR
        / source_slug
    )


def get_document_index_path(
    source_slug: str
) -> Path:
    return (
        get_document_dir(
            source_slug
        )
        / "index.md"
    )


def get_chunk_path(
    source_slug: str,
    chunk_number: int
) -> Path:
    return (
        get_document_dir(
            source_slug
        )
        / f"chunk-{chunk_number:03d}.md"
    )


def get_requirement_path(
    source_slug: str,
    requirement_id: str
) -> Path:
    return (
        REQUIREMENTS_DIR
        / source_slug
        / (
            slugify(
                requirement_id
            )
            + ".md"
        )
    )


# ============================================================
# REQUIREMENT EXTRACTION
# ============================================================

def extract_requirement_blocks(
    content: str
) -> list[tuple[str, str]]:
    """
    Extract requirement blocks.

    A block starts at:
        #### R x.y.z

    and ends at the next Markdown heading of any level.

    The converter v3.2 may intentionally add:
        ##### Context after R x.y.z

    That context therefore does NOT become part of the requirement.
    """
    lines = content.splitlines()

    results: list[
        tuple[str, str]
    ] = []

    current_req: str | None = None
    current_lines: list[str] = []

    def flush():
        nonlocal current_req
        nonlocal current_lines

        if current_req:
            text = "\n".join(
                current_lines
            ).strip()

            results.append(
                (
                    current_req,
                    text
                )
            )

        current_req = None
        current_lines = []

    for line in lines:
        level = (
            markdown_heading_level(
                line
            )
        )

        if level is not None:
            heading = clean_heading_text(
                line
            )

            req_id = (
                extract_requirement_id_from_heading(
                    heading
                )
            )

            if req_id:
                flush()

                current_req = req_id
                current_lines = [line]

            elif current_req:
                # Any non-requirement heading closes the requirement.
                flush()

        elif current_req:
            current_lines.append(line)

    flush()

    return results


def requirement_body_only(
    requirement_text: str
) -> str:
    lines = requirement_text.splitlines()

    if (
        lines
        and markdown_heading_level(
            lines[0]
        )
        is not None
    ):
        lines = lines[1:]

    return "\n".join(
        lines
    ).strip()


def classify_requirement(
    requirement_text: str
) -> str:
    """
    Possible status values:
      active
      void
      tracked_deleted
      empty
    """
    body = requirement_body_only(
        requirement_text
    )

    if (
        TRACKED_DELETION_MARKER.lower()
        in body.lower()
    ):
        return "tracked_deleted"

    normalized_lines = [
        re.sub(
            r"[*_>`\[\]!-]",
            "",
            line
        ).strip()
        for line in body.splitlines()
        if line.strip()
    ]

    if (
        len(normalized_lines) == 1
        and VOID_RE.match(
            normalized_lines[0]
        )
    ):
        return "void"

    if not body:
        return "empty"

    return "active"


# ============================================================
# WRITE CHUNKS
# ============================================================

def write_chunk_page(
    chunk: SourceChunk
) -> Path:
    path = get_chunk_path(
        chunk.source_slug,
        chunk.chunk_number
    )

    headings = "\n".join(
        f"- {heading}"
        for heading
        in chunk.headings
    )

    requirements = "\n".join(
        f"- {req_id}"
        for req_id
        in chunk.requirement_ids
    )

    content = "\n\n".join(
        [
            (
                f"# {chunk.source_title} — "
                f"Chunk {chunk.chunk_number:03d}"
            ),
            (
                "**Original source:** "
                f"`{chunk.source_path.relative_to(OBSIDIAN_DIR)}`"
            ),
            (
                f"**Chunk:** "
                f"{chunk.chunk_number:03d}"
            ),
            (
                "## Headings in this chunk\n"
                + (
                    headings
                    or "- None"
                )
            ),
            (
                "## Requirements in this chunk\n"
                + (
                    requirements
                    or "- None"
                )
            ),
            (
                "## Source Content — Authoritative\n\n"
                + chunk.content
            ),
        ]
    )

    write_text(
        path,
        content
    )

    return path


# ============================================================
# WRITE REQUIREMENT PAGES / CATALOG
# ============================================================

def write_requirement_pages(
    chunk: SourceChunk
) -> list[RequirementRecord]:
    records: list[
        RequirementRecord
    ] = []

    chunk_path = get_chunk_path(
        chunk.source_slug,
        chunk.chunk_number
    )

    for (
        req_id,
        req_text
    ) in extract_requirement_blocks(
        chunk.content
    ):
        status = classify_requirement(
            req_text
        )

        req_path = get_requirement_path(
            chunk.source_slug,
            req_id
        )

        content = "\n\n".join(
            [
                f"# {req_id}",
                (
                    f"**Document:** "
                    f"[[sources/{chunk.source_slug}/index]]"
                ),
                (
                    f"**Chunk:** "
                    f"{build_wiki_link(chunk_path)}"
                ),
                (
                    f"**Status:** `{status}`"
                ),
                (
                    "## Requirement — Authoritative\n\n"
                    + req_text
                ),
            ]
        )

        write_text(
            req_path,
            content
        )

        records.append(
            RequirementRecord(
                requirement_id=req_id,
                document=chunk.source_title,
                source_slug=chunk.source_slug,
                chunk_number=chunk.chunk_number,
                text=req_text,
                status=status,
                source_md=str(
                    chunk.source_path.relative_to(
                        OBSIDIAN_DIR
                    )
                ).replace("\\", "/"),
                chunk_path=build_wiki_link_target(
                    chunk_path
                ),
                requirement_path=(
                    build_wiki_link_target(
                        req_path
                    )
                ),
            )
        )

    return records


# ============================================================
# DOCUMENT INDEX
# ============================================================

def write_document_index(
    source_path: Path,
    chunks: list[SourceChunk],
    records: list[RequirementRecord]
) -> Path:
    source_title = (
        source_path.stem
    )

    source_slug = slugify(
        source_title
    )

    path = get_document_index_path(
        source_slug
    )

    chunk_lines = []

    for chunk in chunks:
        chunk_path = get_chunk_path(
            source_slug,
            chunk.chunk_number
        )

        headings = (
            ", ".join(
                chunk.headings[:4]
            )
        )

        suffix = (
            f" — {headings}"
            if headings
            else ""
        )

        chunk_lines.append(
            f"- {build_wiki_link(chunk_path)}"
            f"{suffix}"
        )

    counts = {
        "active": 0,
        "void": 0,
        "tracked_deleted": 0,
        "empty": 0,
    }

    for record in records:
        counts[record.status] = (
            counts.get(
                record.status,
                0
            )
            + 1
        )

    content = "\n\n".join(
        [
            f"# {source_title}",
            (
                "**Original source:** "
                f"`{source_path.relative_to(OBSIDIAN_DIR)}`"
            ),
            (
                f"**Chunks generated:** "
                f"{len(chunks)}"
            ),
            (
                f"**Requirements:** "
                f"{len(records)} "
                f"(active {counts['active']}, "
                f"void {counts['void']}, "
                f"tracked-deleted "
                f"{counts['tracked_deleted']}, "
                f"empty {counts['empty']})"
            ),
            (
                "## Chunks\n"
                + (
                    "\n".join(
                        chunk_lines
                    )
                    or "- None"
                )
            ),
        ]
    )

    write_text(
        path,
        content
    )

    return path


# ============================================================
# GLOBAL INDEX / JSON CATALOG
# ============================================================

def records_to_json(
    records: list[RequirementRecord]
) -> list[dict]:
    return [
        {
            "requirement_id": record.requirement_id,
            "document": record.document,
            "source_slug": record.source_slug,
            "chunk": record.chunk_number,
            "status": record.status,
            "source_md": record.source_md,
            "chunk_path": record.chunk_path,
            "requirement_path": (
                record.requirement_path
            ),
            "text": record.text,
        }
        for record in records
    ]


def write_global_index(
    document_pages: list[Path],
    records: list[RequirementRecord]
) -> None:
    documents = "\n".join(
        f"- {build_wiki_link(path)}"
        for path in document_pages
    )

    req_lines = []

    for record in sorted(
        records,
        key=lambda item: (
            item.document.lower(),
            [
                int(part)
                for part
                in re.findall(
                    r"\d+",
                    item.requirement_id
                )
            ]
        )
    ):
        req_lines.append(
            f"- [[{record.requirement_path}]] "
            f"— `{record.status}` "
            f"— {record.document}"
        )

    active_count = sum(
        record.status == "active"
        for record in records
    )

    void_count = sum(
        record.status == "void"
        for record in records
    )

    deleted_count = sum(
        record.status
        == "tracked_deleted"
        for record in records
    )

    empty_count = sum(
        record.status == "empty"
        for record in records
    )

    content = "\n\n".join(
        [
            "# Wiki Index",
            (
                f"**Builder version:** "
                f"{BUILDER_VERSION}"
            ),
            (
                f"**Requirement totals:** "
                f"{len(records)} "
                f"(active {active_count}, "
                f"void {void_count}, "
                f"tracked-deleted "
                f"{deleted_count}, "
                f"empty {empty_count})"
            ),
            (
                "## Source Documents\n"
                + (
                    documents
                    or "- None"
                )
            ),
            (
                "## Requirements\n"
                + (
                    "\n".join(
                        req_lines
                    )
                    or "- None"
                )
            ),
        ]
    )

    write_text(
        INDEX_FILE,
        content
    )

    CATALOG_FILE.write_text(
        json.dumps(
            records_to_json(records),
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )


def write_build_report(
    source_count: int,
    total_chunks: int,
    records: list[RequirementRecord]
) -> None:
    status_counts = {}

    for record in records:
        status_counts[record.status] = (
            status_counts.get(
                record.status,
                0
            )
            + 1
        )

    duplicates = {}

    for record in records:
        key = (
            record.source_slug,
            record.requirement_id
        )

        duplicates[key] = (
            duplicates.get(
                key,
                0
            )
            + 1
        )

    duplicate_requirements = [
        {
            "document": source_slug,
            "requirement_id": req_id,
            "count": count,
        }
        for (
            source_slug,
            req_id
        ), count in duplicates.items()
        if count > 1
    ]

    report = {
        "builder_version": BUILDER_VERSION,
        "source_count": source_count,
        "chunk_count": total_chunks,
        "requirement_count": len(records),
        "status_counts": status_counts,
        "duplicate_requirements": (
            duplicate_requirements
        ),
    }

    BUILD_REPORT_FILE.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )


# ============================================================
# CLEAN GENERATED WIKI
# ============================================================

def clear_generated_wiki() -> None:
    """
    Rebuild AI Wiki from Clippings.

    Clippings is NEVER modified.
    """
    if WIKI_DIR.exists():
        shutil.rmtree(
            WIKI_DIR
        )

    WIKI_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print(
        f"LLM Wiki Builder v{BUILDER_VERSION}"
    )

    print(
        "Deterministic ingestion / no LLM calls"
    )

    print(
        f"Clippings directory: {RAW_DIR}"
    )

    print(
        f"Chunk target size: "
        f"{CHUNK_MAX_CHARS} chars"
    )

    print()

    source_paths = list_raw_files()

    if not source_paths:
        print(
            "Processed 0 sources."
        )

        print(
            f"No .md files found in: "
            f"{RAW_DIR}"
        )

        return

    clear_generated_wiki()

    document_pages: list[Path] = []
    all_records: list[
        RequirementRecord
    ] = []

    total_chunks = 0

    for source_index, source_path in enumerate(
        source_paths,
        start=1
    ):
        print(
            f"[{source_index}/"
            f"{len(source_paths)}] "
            f"{source_path.name}"
        )

        chunks = build_chunks_for_source(
            source_path
        )

        print(
            f"  -> {len(chunks)} chunk(s)"
        )

        source_records: list[
            RequirementRecord
        ] = []

        for chunk in chunks:
            print(
                f"     Writing chunk "
                f"{chunk.chunk_number:03d}/"
                f"{len(chunks):03d}"
            )

            write_chunk_page(
                chunk
            )

            records = (
                write_requirement_pages(
                    chunk
                )
            )

            source_records.extend(
                records
            )

            all_records.extend(
                records
            )

            total_chunks += 1

        document_page = (
            write_document_index(
                source_path,
                chunks,
                source_records
            )
        )

        document_pages.append(
            document_page
        )

        print(
            f"  -> {len(source_records)} "
            f"requirement(s)"
        )

        print()

    write_global_index(
        document_pages,
        all_records
    )

    write_build_report(
        len(source_paths),
        total_chunks,
        all_records
    )

    print("=" * 60)

    print(
        f"Processed "
        f"{len(source_paths)} source(s), "
        f"{total_chunks} chunk(s), "
        f"{len(all_records)} requirement(s)."
    )

    status_counts = {}

    for record in all_records:
        status_counts[
            record.status
        ] = (
            status_counts.get(
                record.status,
                0
            )
            + 1
        )

    print(
        "Requirement status: "
        + ", ".join(
            f"{key}={value}"
            for key, value
            in sorted(
                status_counts.items()
            )
        )
    )

    print(
        f"AI Wiki: {WIKI_DIR}"
    )

    print(
        "No LLM calls were used "
        "during ingestion."
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
