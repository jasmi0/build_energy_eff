
import importlib
import pytest

def test_app_importable():
    try:
        importlib.import_module("app")
    except ModuleNotFoundError:
        pytest.skip("app.py not found in repo root; integrate upgrade pack first.")
