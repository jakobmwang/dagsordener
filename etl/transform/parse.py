"""
Granite Docling PDF→Markdown converter.

This module wraps Docling's VLM pipeline around the
`ibm-granite/granite-docling-258M` model. The focus is on GPU-first execution
for speed: we build a single VLM pipeline, run it once per document, and export
DocTags to Markdown.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import ConversionStatus, InputFormat
    from docling.datamodel.pipeline_options import VlmPipelineOptions
    from docling.datamodel.pipeline_options_vlm_model import (
        InlineVlmOptions,
        InferenceFramework,
        ResponseFormat,
        TransformersModelType,
    )
    from docling.datamodel.accelerator_options import (
        AcceleratorDevice,
        AcceleratorOptions,
    )
    from docling.pipeline.vlm_pipeline import VlmPipeline
except Exception as exc:  # pragma: no cover - import guard
    raise RuntimeError("Docling with VLM support is required: pip install docling") from exc


LOGGER = logging.getLogger(__name__)

MODEL_REPO = "ibm-granite/granite-docling-258M"
PROMPT = "Convert this page to DocTags. Only output DocTags markup."  # model-specific prompt
STOP_STRINGS = ["</doctag>", "<end_of_utterance>"]


def _parse_device(device: str) -> AcceleratorDevice:
    mapping = {
        "auto": AcceleratorDevice.AUTO,
        "cpu": AcceleratorDevice.CPU,
        "cuda": AcceleratorDevice.CUDA,
        "mps": AcceleratorDevice.MPS,
    }
    try:
        return mapping[device.lower()]
    except KeyError as err:
        raise ValueError("Unknown device. Use auto|cpu|cuda|mps.") from err


def _build_converter(device: AcceleratorDevice) -> DocumentConverter:
    accelerator = AcceleratorOptions(device=device)

    supported_devices = [AcceleratorDevice.CPU, AcceleratorDevice.CUDA, AcceleratorDevice.MPS]
    if device != AcceleratorDevice.AUTO and device not in supported_devices:
        supported_devices.append(device)

    vlm_options = InlineVlmOptions(
        repo_id=MODEL_REPO,
        prompt=PROMPT,
        response_format=ResponseFormat.DOCTAGS,
        inference_framework=InferenceFramework.TRANSFORMERS,
        transformers_model_type=TransformersModelType.AUTOMODEL_IMAGETEXTTOTEXT,
        torch_dtype="bfloat16",
        load_in_8bit=False,
        supported_devices=supported_devices,
        stop_strings=STOP_STRINGS,
    )

    pipeline_options = VlmPipelineOptions(
        accelerator_options=accelerator,
        vlm_options=vlm_options,
        images_scale=2.0,
        generate_page_images=False,
        generate_picture_images=False,
    )

    pdf_option = PdfFormatOption(
        pipeline_cls=VlmPipeline,
        pipeline_options=pipeline_options,
    )

    return DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={InputFormat.PDF: pdf_option},
    )


def convert_pdf_to_markdown(
    pdf_path: Path | str,
    out_file: Path | str,
    device: str = "cuda",
    logger: Optional[logging.Logger] = None,
) -> Path:
    """Convert a PDF into Markdown using the Granite Docling VLM."""

    log = logger or LOGGER
    src = Path(pdf_path)
    if not src.exists():
        raise FileNotFoundError(src)

    dst = Path(out_file)
    dst.parent.mkdir(parents=True, exist_ok=True)

    accelerator_device = _parse_device(device)
    converter = _build_converter(accelerator_device)

    log.info("Granite Docling pass (device=%s): %s", accelerator_device.value, src.name)
    result = converter.convert(src)
    if result.status not in {ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS}:
        raise RuntimeError(f"Conversion failed with status {result.status} for {src}")

    markdown = result.document.export_to_markdown()
    tmp_path = dst.with_suffix(dst.suffix + ".tmp")
    tmp_path.write_text(markdown, encoding="utf-8")
    os.replace(tmp_path, dst)
    log.info("Wrote %s", dst)
    return dst


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a PDF to Markdown using the Granite Docling model",
    )
    parser.add_argument("--pdf", type=Path, required=True, help="Input PDF path")
    parser.add_argument("--out", type=Path, required=True, help="Output Markdown path")
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Accelerator to use (default: cuda)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s:%(name)s:%(message)s")

    try:
        convert_pdf_to_markdown(
            pdf_path=args.pdf,
            out_file=args.out,
            device=args.device,
        )
        return 0
    except Exception as exc:  # pragma: no cover - cli guard
        LOGGER.error("Conversion failed: %s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
