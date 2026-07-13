"""
OCR extraction for photographed ingredient labels.

Default: local Tesseract via pytesseract (no external calls, works offline,
fine for an MVP/demo). For production accuracy on curved/glossy packaging,
swap this for a Nebius-hosted vision-language endpoint (e.g. Qwen2-VL) by
setting OCR_BACKEND=nebius-vlm and implementing `_extract_via_nebius_vlm`.
"""
from __future__ import annotations

import io
import logging

from PIL import Image, ImageOps

logger = logging.getLogger("foodlens.ocr")

try:
    import pytesseract
    _TESSERACT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TESSERACT_AVAILABLE = False
    logger.warning("pytesseract not installed - OCR endpoint will return an error until it is.")


def _preprocess(image: Image.Image) -> Image.Image:
    """Light preprocessing to improve OCR accuracy on food labels."""
    image = ImageOps.exif_transpose(image)  # respect phone camera orientation
    image = image.convert("L")  # grayscale
    # Upscale small images - tesseract does better with more pixels
    if max(image.size) < 1500:
        scale = 1500 / max(image.size)
        image = image.resize((int(image.width * scale), int(image.height * scale)), Image.LANCZOS)
    image = ImageOps.autocontrast(image)
    return image


def extract_text_from_image(image_bytes: bytes) -> tuple[str, float]:
    if not _TESSERACT_AVAILABLE:
        raise RuntimeError(
            "pytesseract/tesseract-ocr is not installed. Install the 'tesseract-ocr' system "
            "package (see docker/Dockerfile.api) or switch OCR_BACKEND to a Nebius VLM endpoint."
        )

    image = Image.open(io.BytesIO(image_bytes))
    processed = _preprocess(image)

    data = pytesseract.image_to_data(processed, output_type=pytesseract.Output.DICT)
    words = [w for w in data["text"] if w.strip()]
    confidences = [int(c) for c, w in zip(data["conf"], data["text"]) if w.strip() and c != "-1"]

    text = " ".join(words)
    avg_confidence = (sum(confidences) / len(confidences) / 100) if confidences else 0.0
    return text, round(avg_confidence, 2)
