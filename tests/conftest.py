"""Pytest configuration for py-mkl-pardiso tests."""

import os

import pytest


def pytest_collection_modifyitems(config, items):
    """Under ASAN, skip tests explicitly marked as unsafe for exception paths."""
    if "ASAN_OPTIONS" not in os.environ:
        return
    skip = pytest.mark.skip(reason="C++ exception paths are disabled under ASAN")
    for item in items:
        if item.get_closest_marker("asan_unsafe"):
            item.add_marker(skip)
