from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.models.schemas import OcrResponse
from app.services import ocr_service

router = APIRouter(prefix="/api", tags=["ocr"])

_MAX_FILE_SIZE_MB = 10
_ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/heic"}


@router.post("/ocr", response_model=OcrResponse)
async def ocr(file: UploadFile = File(...)) -> OcrResponse:
    if file.content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail=f"Unsupported content type: {file.content_type}")

    image_bytes = await file.read()
    if len(image_bytes) > _MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds {_MAX_FILE_SIZE_MB}MB limit")

    try:
        text, confidence = ocr_service.extract_text_from_image(image_bytes)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if not text.strip():
        raise HTTPException(status_code=422, detail="No text could be extracted from the image")

    return OcrResponse(extracted_text=text, confidence=confidence)
