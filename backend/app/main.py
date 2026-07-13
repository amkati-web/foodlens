from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.routers import analyze, ocr
from app.services import nebius_client

logging.basicConfig(level=logging.INFO)
settings = get_settings()

app = FastAPI(
    title="FoodLens API",
    description=(
        "Intelligent food label analyzer: additive safety, NOVA ultra-processing "
        "classification, and personal allergen/dietary compatibility checks."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze.router)
app.include_router(ocr.router)


@app.get("/api/health")
async def health() -> dict:
    return {
        "status": "ok",
        "llm_available": nebius_client.is_available(),
        "active_model": settings.active_model,
    }


# Serve the static frontend (index.html/app.js/styles.css) at the root
app.mount("/", StaticFiles(directory="/app/frontend", html=True), name="frontend")
