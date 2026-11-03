"""
FastAPI Main Application Entry Point.

Initializes the FastAPI application, sets up logging, and registers core routes.
"""

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.core.logging import setup_logging

logger = logging.getLogger("call_audit.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan context manager for startup and shutdown events.
    """
    settings = get_settings()
    setup_logging(log_level=settings.LOG_LEVEL)
    logger.info("Starting %s in [%s] mode...", settings.APP_NAME, settings.APP_ENV)
    logger.info("Project Root: %s", settings.BASE_DIR)
    yield
    logger.info("Shutting down %s...", settings.APP_NAME)


def create_app() -> FastAPI:
    """
    Application factory for the Call Audit Automation FastAPI app.
    """
    settings = get_settings()

    application = FastAPI(
        title="Call Audit Automation API",
        description=(
            "AI-powered Call Audit Automation System for financial compliance, "
            "speaker diarization, and KPI policy validation."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    # Enable CORS for local development and future dashboard integration
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.get(
        "/health",
        summary="Service Health Check",
        description="Verify service availability and health status.",
        tags=["Health"],
    )
    async def health() -> Dict[str, Any]:
        """
        Basic health check endpoint returning service health status.
        """
        return {
            "status": "healthy",
            "service": "call-audit-automation",
        }

    @application.get(
        "/",
        summary="Root Welcome",
        description="Root metadata information.",
        tags=["General"],
    )
    async def root() -> Dict[str, Any]:
        """
        Root information endpoint.
        """
        return {
            "service": settings.APP_NAME,
            "version": "0.1.0",
            "status": "running",
            "documentation": "/docs",
        }

    return application


app = create_app()
