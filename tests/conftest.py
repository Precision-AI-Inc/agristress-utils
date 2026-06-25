"""Shared pytest fixtures for the test suite."""

import pytest
from fastapi.testclient import TestClient

from pai.myproject.api.app import create_app


@pytest.fixture
def sample_name() -> str:
    """Return the canonical test name shared across the test suite."""
    return "World"


@pytest.fixture
def client() -> TestClient:
    """Return a configured test client for the FastAPI application."""
    return TestClient(create_app())
