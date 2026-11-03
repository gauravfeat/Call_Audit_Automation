"""
Health Check and Base Configuration Test Suite.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from app.main import app
from app.core.config import get_settings


def test_health_endpoint():
    """
    Ensure GET /health returns HTTP 200 and expected payload.
    """
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data == {
        "status": "healthy",
        "service": "call-audit-automation",
    }


def test_root_endpoint():
    """
    Ensure GET / returns service information.
    """
    client = TestClient(app)
    response = client.get("/")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "running"
    assert "documentation" in data


def test_settings_initialization():
    """
    Verify application settings and directory resolution.
    """
    settings = get_settings()
    assert settings.APP_NAME == "Call Audit Automation"
    assert settings.PORT == 8000
    assert settings.recordings_dir.exists()
    assert settings.transcripts_dir.exists()
    assert settings.reports_dir.exists()
