"""
Adaptive PDF→Markdown converter using Docling (RapidOCR-only) with selective full-page OCR.

Summary
- Public API: `convert_pdf_to_markdown(pdf_path, out_file, device="auto", heuristics=None, logger=None)`.
- One pass to parse; then score pages; re-run **only suspect pages** with **full-page OCR** (RapidOCR) and merge.
- **No DocTags** usage and **no Tesseract** fallback. Output is a **single Markdown file**.
- Atomic write to the target path.

Heuristics (per page)
- Low absolute/relative text length.
- High weird-character ratio (garbled text layers).
- Presence of image tokens in the initial Markdown (indicates likely raster content) → triggers full-page OCR.
- If many pages are suspect, OCR the whole document for efficiency.

Requirements
- `docling` with RapidOCR support (`rapidocr_onnxruntime` extras) and a recent version exposing AcceleratorOptions.
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# --- Docling imports ---------------------------------------------------------
try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
except Exception as exc:
    raise RuntimeError("Docling is required: pip install docling") from exc

try:
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        RapidOcrOptions,
    )
except Exception as exc:
    raise RuntimeError(
        "Docling build must include RapidOCR support (rapidocr_onnxruntime)."
    ) from exc

try:
    from docling.datamodel.accelerator_options import (
        AcceleratorOptions,
        AcceleratorDevice,
    )
except Exception as exc:
    raise RuntimeError(
        "Your Docling version lacks AcceleratorOptions; update to a newer release."
    ) from exc


# --- Heuristic configuration -------------------------------------------------
@dataclass
class SuspectHeuristics:
    """Thresholds for deciding which pages require forced full-page OCR.

    Attributes
    ----------
    min_chars_abs : int
        Absolute minimum characters on a page. Below this, the page is treated as effectively empty and OCR is forced.
    rel_vs_median : float
        Relative threshold vs the document median page length. If a page length is below (rel_vs_median * median_length), OCR is forced.
    weird_char_ratio : float
        Maximum tolerated share of non-basic ASCII characters. If the ratio of non-ASCII exceeds this value, OCR is forced.
    suspect_bulk_ratio : float
        If the fraction of suspect pages across the document exceeds this ratio, the entire document is re-processed with full-page OCR.
    """

    min_chars_abs: int = 200
    rel_vs_median: float = 0.20
    weird_char_ratio: float = 0.25
    suspect_bulk_ratio: float = 0.50


# --- Small utilities ---------------------------------------------------------

def _to_intervals(pages: Iterable[int]) -> List[Tuple[int, int]]:
    """Group 1-based page numbers into half-open intervals (start, end_exclusive)."""
    pages = sorted(set(pages))
    if not pages:
        return []
    out: List[Tuple[int, int]] = []
    s = e = pages[0]
    for p in pages[1:]:
        if p == e + 1:
            e = p
        else:
            out.append((s, e + 1))
            s = e = p
    out.append((s, e + 1))
    return out


def _ascii_share(text: str) -> float:
    """Share of characters within basic printable ASCII range."""
    if not text:
        return 0.0
    ok = sum(1 for ch in text if (" " <= ch <= "~"))
    return ok / len(text)


def _count_image_tokens(md: str) -> int:
    """Approximate count of image placeholders in Markdown output."""
    return len(re.findall(r"!\[|\[Image[:\]]", md))


# --- Page scoring ------------------------------------------------------------

def _score_suspects(document, heur: SuspectHeuristics) -> List[int]:
    """Return 1-based page numbers that should be re-OCR'ed.

    Signals used:
    - Absolute low text length.
    - Relative low text vs document median.
    - Garbled text layer (low ASCII share → high weird-char share).
    - Presence of image tokens in the Markdown.
    """
    per_len: Dict[int, int] = {}
    per_md: Dict[int, str] = {}

    for p in document.pages:
        md = p.export_to_markdown() or ""
        per_md[p.page_no] = md
        per_len[p.page_no] = len(md.strip())

    lengths = [l for _, l in sorted(per_len.items())]
    if lengths:
        mid = len(lengths) // 2
        median_len = lengths[mid] if len(lengths) % 2 else (lengths[mid - 1] + lengths[mid]) // 2
    else:
        median_len = 0

    suspects: List[int] = []
    for page_no, md in per_md.items():
        n = per_len[page_no]
        abs_low = n < heur.min_chars_abs
        rel_low = median_len > 0 and (n < max(heur.min_chars_abs, int(heur.rel_vs_median * median_len)))
        ascii_share = _ascii_share(md)
        garbled = (n > 0) and (ascii_share < (1.0 - heur.weird_char_ratio))
        img_tokens = _count_image_tokens(md)
        has_image = img_tokens > 0  # explicit trigger, per user requirement

        if abs_low or rel_low or garbled or has_image:
            suspects.append(page_no)

    return sorted(set(suspects))


# --- Core conversion ---------------------------------------------------------

def _make_accel(device: str) -> AcceleratorOptions:
    """Map device string to AcceleratorOptions."""
    dev = {
        "auto": AcceleratorDevice.AUTO,
        "cpu": AcceleratorDevice.CPU,
        "cuda": AcceleratorDevice.CUDA,
        "mps": AcceleratorDevice.MPS,
    }.get(device.lower())
    if dev is None:
        raise ValueError("Unknown device. Use auto|cpu|cuda|mps.")
    return AcceleratorOptions(device=dev)


def convert_pdf_to_markdown(
    pdf_path: Path | str,
    out_file: Path | str,
    device: str = "auto",
    heuristics: Optional[SuspectHeuristics] = None,
    logger: Optional[logging.Logger] = None,
) -> Path:
    """Convert a single PDF to a Markdown file using RapidOCR and selective full-page OCR.

    Parameters
    ----------
    pdf_path : Path | str
        Input PDF path.
    out_file : Path | str
        Exact output path for the Markdown file. The directory is created if missing. Atomic write is used (tmp + rename).
    device : str, optional
        Accelerator selection: "auto" | "cuda" | "mps" | "cpu". Default is "auto".
    heuristics : SuspectHeuristics | None, optional
        Thresholds controlling when to force full-page OCR on pages. If None, sensible defaults are used.
    logger : logging.Logger | None, optional
        Logger. If None, the module logger is used.

    Returns
    -------
    Path
        Path to the written Markdown file.
    """
    log = logger or logging.getLogger(__name__)
    pdf = Path(pdf_path)
    if not pdf.exists():
        raise FileNotFoundError(pdf)

    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    heur = heuristics or SuspectHeuristics()

    accel = _make_accel(device)

    # Pass 1: normal conversion (do_ocr=True, no force)
    pipeline = PdfPipelineOptions(do_ocr=True, ocr_options=RapidOcrOptions())
    converter = DocumentConverter(
        accelerator_options=accel,
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline)},
    )

    log.info("Pass 1: %s", pdf.name)
    res1 = converter.convert(pdf)
    doc = res1.document

    suspects = _score_suspects(doc, heur)
    total = len(doc.pages)

    if suspects:
        run_whole = total > 0 and (len(suspects) / total) >= heur.suspect_bulk_ratio
        intervals = [(1, total + 1)] if run_whole else _to_intervals(suspects)

        force_pipeline = PdfPipelineOptions(do_ocr=True, ocr_options=RapidOcrOptions())
        force_converter = DocumentConverter(
            accelerator_options=accel,
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=force_pipeline)},
        )

        # Re-convert only suspect intervals, forcing full-page OCR by rendering as image via RapidOCR path
        # (RapidOCR treats page images; Docling will ignore embedded text effectively in this forced pass.)
        for (s, e) in intervals:
            log.info("Pass 2 (force OCR): pages %d..%d", s, e - 1)
            res_fix = force_converter.convert(pdf, page_range=(s, e), force_full_page_ocr=True)
            for page in res_fix.document.pages:
                doc.pages[page.page_no - 1] = page

    # Export and atomically write Markdown
    md_text = doc.export_to_markdown()
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_path.write_text(md_text, encoding="utf-8")
    os.replace(tmp_path, out_path)
    log.info("Wrote %s", out_path)
    return out_path


# --- CLI --------------------------------------------------------------------

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RapidOCR Docling PDF→Markdown converter (selective full-page OCR)")
    p.add_argument("pdf", type=Path, help="Input PDF path")
    p.add_argument("out_file", type=Path, help="Output Markdown path (.md)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Accelerator device")

    # Heuristic tuning (advanced)
    p.add_argument("--min-chars-abs", type=int, default=SuspectHeuristics.min_chars_abs)
    p.add_argument("--rel-vs-median", type=float, default=SuspectHeuristics.rel_vs_median)
    p.add_argument("--weird-char-ratio", type=float, default=SuspectHeuristics.weird_char_ratio)
    p.add_argument("--suspect-bulk-ratio", type=float, default=SuspectHeuristics.suspect_bulk_ratio)

    p.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s:%(name)s:%(message)s")

    heur = SuspectHeuristics(
        min_chars_abs=args.min_chars_abs,
        rel_vs_median=args.rel_vs_median,
        weird_char_ratio=args.weird_char_ratio,
        suspect_bulk_ratio=args.suspect_bulk_ratio,
    )

    try:
        out = convert_pdf_to_markdown(
            pdf_path=args.pdf,
            out_file=args.out_file,
            device=args.device,
            heuristics=heur,
        )
        logging.getLogger("docling-pipeline").info("Done: %s", out)
        return 0
    except Exception as ex:
        logging.getLogger("docling-pipeline").error("Conversion failed: %s", ex)
        return 2


if __name__ == "__main__":
    sys.exit(main())
