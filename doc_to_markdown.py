from __future__ import annotations

import argparse
import json
import os
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from docx import Document
from docx.document import Document as _Document
from docx.table import Table, _Cell
from docx.text.paragraph import Paragraph
from lxml import etree


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(
    os.environ.get("OBSIDIAN_DIR", Path(__file__).resolve().parent)
)

DEFAULT_OUTPUT_DIR = BASE_DIR / "Clippings"

REQUIREMENT_ID_RE = re.compile(
    r"(?<![A-Za-z0-9])R\s*(\d+(?:\.\d+)+)\b",
    re.IGNORECASE,
)

NUMBERED_HEADING_RE = re.compile(
    r"^\s*(\d+(?:\.\d+){0,2})\s+(.+?)\s*$"
)

ALL_CAPS_HEADING_RE = re.compile(
    r"^[A-Z][A-Z0-9 /&(),+\-–—]{2,}$"
)

NORMATIVE_RE = re.compile(
    r"^(MUST NOT|SHALL NOT|SHOULD NOT|NOT RECOMMENDED|"
    r"MUST|REQUIRED|SHALL|SHOULD|RECOMMENDED|MAY|OPTIONAL)\s*:\s*(.*)$",
    re.IGNORECASE,
)

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W_NS}


# ============================================================
# DATA
# ============================================================

@dataclass
class RevisionInfo:
    requirement_id: str
    active_text: str = ""
    deleted_text: str = ""
    inserted_text: str = ""

    @property
    def has_deleted_body(self) -> bool:
        return bool(self.deleted_text.strip())

    @property
    def has_active_body(self) -> bool:
        body = remove_requirement_prefix(
            self.active_text,
            self.requirement_id
        )
        return bool(body.strip())


@dataclass
class ConversionStats:
    requirements_found: int = 0
    multi_requirement_paragraphs: int = 0
    requirement_rows_from_tables: int = 0
    markdown_tables: int = 0

    tracked_deleted_body_requirements: list[str] = field(
        default_factory=list
    )
    tracked_inserted_requirements: list[str] = field(
        default_factory=list
    )
    suspicious_empty_requirements: list[str] = field(
        default_factory=list
    )


# ============================================================
# BASIC CLEANING
# ============================================================

def clean_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = text.replace("\u200b", "")
    text = text.replace("\ufeff", "")
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def clean_inline(text: str) -> str:
    return re.sub(r"\s+", " ", clean_text(text)).strip()


def escape_markdown_cell(text: str) -> str:
    text = clean_text(text)
    text = text.replace("|", r"\|")
    return text.replace("\n", "<br>")


def normalize_requirement_id(number_or_id: str) -> str:
    value = clean_inline(number_or_id)

    match = re.search(
        r"R?\s*(\d+(?:\.\d+)+)",
        value,
        re.IGNORECASE
    )

    if not match:
        return value

    return f"R {match.group(1)}"


def remove_requirement_prefix(
    text: str,
    requirement_id: str
) -> str:
    number = re.escape(
        requirement_id.replace("R ", "")
    )

    return re.sub(
        rf"^\s*R\s*{number}\s*",
        "",
        clean_inline(text),
        count=1,
        flags=re.IGNORECASE,
    ).strip()


# ============================================================
# DOCX STRUCTURE ITERATION
# ============================================================

def iter_block_items(parent):
    """
    Yield top-level Paragraph and Table objects in document order.
    """
    if isinstance(parent, _Document):
        parent_elm = parent.element.body
    elif isinstance(parent, _Cell):
        parent_elm = parent._tc
    else:
        raise ValueError("Unsupported parent type")

    for child in parent_elm.iterchildren():
        if child.tag.endswith("}p"):
            yield Paragraph(child, parent)
        elif child.tag.endswith("}tbl"):
            yield Table(child, parent)


# ============================================================
# REVISION-AWARE XML INSPECTION
# ============================================================

def get_document_xml(docx_path: Path):
    with zipfile.ZipFile(docx_path) as archive:
        raw = archive.read("word/document.xml")

    return etree.fromstring(raw)


def element_text_excluding_deleted(element) -> str:
    """
    Return active Word text (w:t), excluding every node inside w:del.
    Inserted text in w:ins is included because it is part of the active
    revision state.
    """
    pieces = []

    for node in element.xpath(".//w:t", namespaces=NS):
        if node.xpath("ancestor::w:del", namespaces=NS):
            continue

        if node.text:
            pieces.append(node.text)

    return clean_inline("".join(pieces))


def element_deleted_text(element) -> str:
    pieces = element.xpath(
        ".//w:delText/text()",
        namespaces=NS
    )
    return clean_inline("".join(pieces))


def element_inserted_text(element) -> str:
    pieces = []

    for ins in element.xpath(
        ".//w:ins",
        namespaces=NS
    ):
        pieces.extend(
            ins.xpath(
                ".//w:t/text()",
                namespaces=NS
            )
        )

    return clean_inline("".join(pieces))


def build_revision_map(
    docx_path: Path
) -> dict[str, RevisionInfo]:
    """
    Inspect Word XML and identify requirement paragraphs affected by
    tracked changes.

    Important policy:
    - w:t = active/current text
    - w:ins/w:t = active inserted text
    - w:delText = deleted historical text, NOT authoritative active text

    Deleted content is never restored into the Markdown body.
    """
    root = get_document_xml(docx_path)
    revision_map: dict[str, RevisionInfo] = {}

    for paragraph in root.xpath(
        ".//w:p",
        namespaces=NS
    ):
        active = element_text_excluding_deleted(
            paragraph
        )
        deleted = element_deleted_text(
            paragraph
        )
        inserted = element_inserted_text(
            paragraph
        )

        combined_for_id = " ".join(
            x for x in (active, deleted) if x
        )

        ids = [
            normalize_requirement_id(match.group(1))
            for match in REQUIREMENT_ID_RE.finditer(
                combined_for_id
            )
        ]

        if not ids:
            continue

        for req_id in ids:
            info = revision_map.get(
                req_id,
                RevisionInfo(requirement_id=req_id)
            )

            # Prefer the longest active/deleted view found for an ID.
            if len(active) > len(info.active_text):
                info.active_text = active

            if len(deleted) > len(info.deleted_text):
                info.deleted_text = deleted

            if len(inserted) > len(info.inserted_text):
                info.inserted_text = inserted

            revision_map[req_id] = info

    return revision_map


# ============================================================
# REQUIREMENTS
# ============================================================

def split_requirements(text: str) -> list[tuple[str, str]]:
    text = clean_text(text)
    matches = list(REQUIREMENT_ID_RE.finditer(text))

    if not matches:
        return []

    results = []

    for i, match in enumerate(matches):
        start = match.end()
        end = (
            matches[i + 1].start()
            if i + 1 < len(matches)
            else len(text)
        )

        req_id = normalize_requirement_id(
            match.group(1)
        )

        body = clean_text(
            text[start:end]
        )

        results.append(
            (req_id, body)
        )

    return results


def render_requirement_blocks(
    text: str,
    stats: ConversionStats
) -> str | None:
    requirements = split_requirements(text)

    if not requirements:
        return None

    if len(requirements) > 1:
        stats.multi_requirement_paragraphs += 1

    blocks = []

    for req_id, body in requirements:
        stats.requirements_found += 1

        blocks.append(
            f"#### {req_id}"
            + (f"\n{body}" if body else "")
        )

    return "\n\n".join(blocks)


# ============================================================
# HEADINGS
# ============================================================

def paragraph_style_name(
    paragraph: Paragraph
) -> str:
    try:
        return paragraph.style.name or ""
    except Exception:
        return ""


def heading_level_from_style(
    paragraph: Paragraph
) -> int | None:
    style = paragraph_style_name(
        paragraph
    ).lower()

    match = re.search(
        r"heading\s*(\d+)",
        style
    )

    if match:
        return max(
            1,
            min(int(match.group(1)), 6)
        )

    if style == "title":
        return 1

    return None


def looks_like_false_heading(
    text: str
) -> bool:
    text = clean_inline(text)

    patterns = [
        r"^\d{2}\.\d{3}\b",
        r"^\d+(?:\.\d+){3,}\b",
        r"^\d+(?:,\d+)*\s*MHz\b",
        r"^[<>≤≥±+\-]?\s*\[?-?\d",
        r"^Band\s+\d+\b",
        r"^Table\s+\d+\b",
        r"^Figure\s+\d+\b",
    ]

    return any(
        re.search(
            pattern,
            text,
            re.IGNORECASE
        )
        for pattern in patterns
    )


def detect_heading(
    text: str,
    paragraph: Paragraph
) -> tuple[int, str] | None:
    text = clean_text(text)

    if not text:
        return None

    if REQUIREMENT_ID_RE.search(text):
        return None

    if looks_like_false_heading(text):
        return None

    style_level = heading_level_from_style(
        paragraph
    )

    if style_level:
        return style_level, text

    match = NUMBERED_HEADING_RE.match(
        text
    )

    if match:
        number = match.group(1)
        title = match.group(2)

        level = min(
            number.count(".") + 2,
            6
        )

        return level, f"{number} {title}"

    if (
        len(text) <= 90
        and ALL_CAPS_HEADING_RE.match(text)
        and not text.endswith(".")
    ):
        return 2, text

    return None


# ============================================================
# TABLES
# ============================================================

def cell_lines(
    cell: _Cell
) -> list[str]:
    lines = []

    for paragraph in cell.paragraphs:
        text = clean_text(
            paragraph.text
        )

        if text:
            lines.extend(
                part.strip()
                for part in text.splitlines()
                if part.strip()
            )

    return lines


def extract_req_ids_from_cell(
    cell: _Cell
) -> list[str]:
    ids = []

    for line in cell_lines(cell):
        for match in REQUIREMENT_ID_RE.finditer(
            line
        ):
            ids.append(
                normalize_requirement_id(
                    match.group(1)
                )
            )

    return ids


def try_requirement_table(
    table: Table,
    stats: ConversionStats
) -> str | None:
    blocks = []
    converted_any = False

    for row in table.rows:
        row_ids = []
        body_candidates = []

        for cell in row.cells:
            ids = extract_req_ids_from_cell(
                cell
            )

            if ids:
                row_ids.extend(ids)
            else:
                text = clean_text(cell.text)

                if text:
                    body_candidates.append(
                        text
                    )

        if not row_ids:
            continue

        converted_any = True

        body = "\n".join(
            body_candidates
        ).strip()

        body_lines = [
            clean_inline(item)
            for item in re.split(
                r"\n+",
                body
            )
            if clean_inline(item)
        ]

        if (
            len(row_ids) == len(body_lines)
            and len(row_ids) > 1
        ):
            for req_id, statement in zip(
                row_ids,
                body_lines
            ):
                stats.requirements_found += 1
                stats.requirement_rows_from_tables += 1

                blocks.append(
                    f"#### {req_id}\n{statement}"
                )

        elif len(row_ids) == 1:
            stats.requirements_found += 1
            stats.requirement_rows_from_tables += 1

            blocks.append(
                f"#### {row_ids[0]}"
                + (f"\n{body}" if body else "")
            )

        else:
            # Preserve every ID without guessing the mapping.
            for i, req_id in enumerate(row_ids):
                stats.requirements_found += 1
                stats.requirement_rows_from_tables += 1

                if i == 0 and body:
                    blocks.append(
                        f"#### {req_id}\n{body}"
                    )
                else:
                    blocks.append(
                        f"#### {req_id}"
                    )

    if not converted_any:
        return None

    return "\n\n".join(blocks)


def table_to_markdown(
    table: Table,
    stats: ConversionStats
) -> str:
    requirement_form = try_requirement_table(
        table,
        stats
    )

    if requirement_form:
        return requirement_form

    rows = []

    for row in table.rows:
        values = [
            escape_markdown_cell(
                cell.text
            )
            for cell in row.cells
        ]

        if any(values):
            rows.append(values)

    if not rows:
        return ""

    stats.markdown_tables += 1

    width = max(
        len(row)
        for row in rows
    )

    rows = [
        row + [""] * (width - len(row))
        for row in rows
    ]

    header = rows[0]

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(
            ["---"] * width
        ) + " |",
    ]

    for row in rows[1:]:
        lines.append(
            "| " + " | ".join(row) + " |"
        )

    return "\n".join(lines)


# ============================================================
# PARAGRAPHS
# ============================================================

def paragraph_to_markdown(
    paragraph: Paragraph,
    stats: ConversionStats
) -> str:
    text = clean_text(
        paragraph.text
    )

    if not text:
        return ""

    req_md = render_requirement_blocks(
        text,
        stats
    )

    if req_md:
        return req_md

    heading = detect_heading(
        text,
        paragraph
    )

    if heading:
        level, title = heading
        return (
            f"{'#' * level} {title}"
        )

    style = paragraph_style_name(
        paragraph
    ).lower()

    if "list bullet" in style:
        return f"- {text}"

    if "list number" in style:
        return f"1. {text}"

    normative = NORMATIVE_RE.match(
        text
    )

    if normative:
        return (
            f"**{normative.group(1).upper()}:** "
            f"{normative.group(2)}"
        )

    return text


# ============================================================
# POST-PROCESSING
# ============================================================

def postprocess_markdown(
    markdown: str
) -> str:
    lines = markdown.splitlines()
    cleaned = []

    for line in lines:
        stripped = line.strip()

        if stripped in {
            "", "•", "▪", "◦"
        }:
            continue

        if re.match(
            r"^#{1,6}\s+",
            line
        ):
            title = re.sub(
                r"^#{1,6}\s+",
                "",
                line
            ).strip()

            if looks_like_false_heading(
                title
            ):
                line = title

        cleaned.append(line)

    text = "\n".join(cleaned)

    text = re.sub(
        r"\n{4,}",
        "\n\n\n",
        text
    )

    return text.strip() + "\n"


# ============================================================
# REVISION-AWARE MARKDOWN VALIDATION
# ============================================================

def split_markdown_requirement_blocks(
    markdown: str
):
    """
    Return tuples:
        (start_index, end_index, req_id)

    The end is the next Markdown heading of any level.
    """
    lines = markdown.splitlines()

    req_heading_re = re.compile(
        r"^####\s+(R\s+\d+(?:\.\d+)+)\s*$",
        re.IGNORECASE
    )

    blocks = []

    for i, line in enumerate(lines):
        match = req_heading_re.match(
            line.strip()
        )

        if not match:
            continue

        req_id = normalize_requirement_id(
            match.group(1)
        )

        end = len(lines)

        for j in range(i + 1, len(lines)):
            if re.match(
                r"^#{1,6}\s+",
                lines[j].strip()
            ):
                end = j
                break

        blocks.append(
            (i, end, req_id)
        )

    return lines, blocks


def active_requirement_body_from_block(
    lines: list[str],
    start: int,
    end: int
) -> str:
    body_lines = [
        line
        for line in lines[start + 1:end]
        if line.strip()
    ]

    return clean_inline(
        " ".join(body_lines)
    )


def apply_revision_diagnostics(
    markdown: str,
    revision_map: dict[str, RevisionInfo],
    stats: ConversionStats
) -> str:
    """
    Mark requirements whose body exists only as tracked-deleted Word text.

    We intentionally DO NOT restore w:delText because it is not part of the
    current active revision.

    If ordinary prose follows such an empty requirement before the next
    requirement/section, insert a Context heading so that the maintainer
    cannot accidentally treat that prose as the deleted requirement body.
    """
    lines, blocks = split_markdown_requirement_blocks(
        markdown
    )

    block_by_start = {
        start: (end, req_id)
        for start, end, req_id in blocks
    }

    output = []
    i = 0
    suspicious = []

    while i < len(lines):
        if i not in block_by_start:
            output.append(lines[i])
            i += 1
            continue

        end, req_id = block_by_start[i]
        info = revision_map.get(req_id)

        output.append(lines[i])

        raw_body_lines = lines[i + 1:end]
        visible_body = active_requirement_body_from_block(
            lines,
            i,
            end
        )

        tracked_deleted_only = (
            info is not None
            and info.has_deleted_body
            and not info.has_active_body
        )

        if tracked_deleted_only:
            stats.tracked_deleted_body_requirements.append(
                req_id
            )

            output.append("")
            output.append(
                "> [!warning] Tracked deletion in source"
            )
            output.append(
                "> The requirement body exists only as deleted "
                "tracked-change text in the DOCX and is therefore "
                "not treated as active authoritative content."
            )

            # If the converter had attached following prose to the
            # requirement, separate it explicitly as context.
            if visible_body:
                output.append("")
                output.append(
                    f"##### Context after {req_id}"
                )
                output.extend(raw_body_lines)

        else:
            output.extend(raw_body_lines)

            if info and info.inserted_text:
                stats.tracked_inserted_requirements.append(
                    req_id
                )

            if not visible_body:
                suspicious.append(req_id)

        i = end

    stats.tracked_deleted_body_requirements = list(
        dict.fromkeys(
            stats.tracked_deleted_body_requirements
        )
    )

    stats.tracked_inserted_requirements = list(
        dict.fromkeys(
            stats.tracked_inserted_requirements
        )
    )

    stats.suspicious_empty_requirements = list(
        dict.fromkeys(suspicious)
    )

    return postprocess_markdown(
        "\n".join(output)
    )


# ============================================================
# DOCUMENT
# ============================================================

def extract_core_properties(
    document: Document
) -> list[str]:
    props = document.core_properties
    lines = []

    if props.title:
        lines.append(
            f"**Document title:** "
            f"{clean_text(props.title)}"
        )

    if props.subject:
        lines.append(
            f"**Subject:** "
            f"{clean_text(props.subject)}"
        )

    return lines


def convert_docx_to_markdown(
    docx_path: Path
) -> tuple[str, ConversionStats]:
    document = Document(docx_path)
    stats = ConversionStats()

    revision_map = build_revision_map(
        docx_path
    )

    output = [
        f"# {docx_path.stem}",
        "",
        f"**Original file:** `{docx_path.name}`",
    ]

    metadata = extract_core_properties(
        document
    )

    if metadata:
        output.extend(
            [""] + metadata
        )

    output.extend(
        ["", "---", ""]
    )

    for block in iter_block_items(
        document
    ):
        if isinstance(
            block,
            Paragraph
        ):
            md = paragraph_to_markdown(
                block,
                stats
            )
        else:
            md = table_to_markdown(
                block,
                stats
            )

        if md:
            output.extend(
                [md, ""]
            )

    markdown = postprocess_markdown(
        "\n".join(output)
    )

    markdown = apply_revision_diagnostics(
        markdown,
        revision_map,
        stats
    )

    return markdown, stats


# ============================================================
# OUTPUT
# ============================================================

def output_name_same_as_original(
    docx_path: Path
) -> str:
    return docx_path.with_suffix(
        ".md"
    ).name


def write_report(
    output_path: Path,
    stats: ConversionStats
) -> Path:
    report_path = output_path.with_suffix(
        ".conversion.json"
    )

    report = {
        "converter_version": "3.2",
        "markdown_file": output_path.name,
        "requirements_found": (
            stats.requirements_found
        ),
        "multi_requirement_paragraphs": (
            stats.multi_requirement_paragraphs
        ),
        "requirement_rows_from_tables": (
            stats.requirement_rows_from_tables
        ),
        "markdown_tables": (
            stats.markdown_tables
        ),
        "tracked_deleted_body_requirements": (
            stats.tracked_deleted_body_requirements
        ),
        "tracked_inserted_requirements": (
            stats.tracked_inserted_requirements
        ),
        "suspicious_empty_requirements": (
            stats.suspicious_empty_requirements
        ),
    }

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8",
    )

    return report_path


def convert_file(
    docx_path: Path,
    output_dir: Path
) -> tuple[Path, Path]:
    if not docx_path.exists():
        raise FileNotFoundError(
            f"File not found: {docx_path}"
        )

    if docx_path.suffix.lower() != ".docx":
        raise ValueError(
            "This converter accepts DOCX files only."
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    markdown, stats = convert_docx_to_markdown(
        docx_path
    )

    output_path = (
        output_dir
        / output_name_same_as_original(
            docx_path
        )
    )

    output_path.write_text(
        markdown,
        encoding="utf-8"
    )

    report_path = write_report(
        output_path,
        stats
    )

    return output_path, report_path


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert a DOCX technical specification into "
            "revision-aware Obsidian Markdown."
        )
    )

    parser.add_argument(
        "docx",
        type=Path,
        help="Path to source .docx",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Default: {DEFAULT_OUTPUT_DIR}",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    output_path, report_path = convert_file(
        args.docx,
        args.output_dir,
    )

    print("Conversion completed.")
    print("Converter version: 3.2")
    print(f"Markdown file: {output_path}")
    print(f"Quality report: {report_path}")


if __name__ == "__main__":
    main()
