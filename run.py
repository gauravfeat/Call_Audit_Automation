"""
Application Runner Script.

Starts the Uvicorn ASGI server serving the FastAPI application.
"""

import sys
from pathlib import Path

# Add project root to Python search path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn
from app.core.config import get_settings


def main() -> None:
    """
    Entrypoint to start the Call Audit Automation FastAPI server.
    """
    settings = get_settings()

    print(f"Starting {settings.APP_NAME}...")
    print(f"Environment: {settings.APP_ENV}")
    print(f"Server URL:  http://{settings.HOST}:{settings.PORT}")
    print(f"Health Check: http://{settings.HOST}:{settings.PORT}/health")
    print(f"API Docs:     http://{settings.HOST}:{settings.PORT}/docs")

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        log_level=settings.LOG_LEVEL.lower(),
    )


if __name__ == "__main__":
    main()
