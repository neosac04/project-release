from __future__ import annotations
import os
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import api_router
from app.config.settings import settings
from app.models.registry import ModelRegistry

log = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("startup", models_dir=settings.models_dir, device=settings.device)
    registry = ModelRegistry.get_instance()
    registry.load_all(settings.models_dir, settings.device)
    loaded = list(registry.all().keys())
    log.info("models_ready", loaded=loaded)
    yield
    log.info("shutdown")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Deepfake Detector API",
        description=(
            "Multi-model deepfake detection with explainability heatmaps, "
            "anatomical analysis, frequency forensics, and fake-type classification."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router, prefix="/api/v1")
    return app


app = create_app()
