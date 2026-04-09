"""Pytest configuration for py-mkl-pardiso tests."""

import inspect
import os

import pytest


def pytest_collection_modifyitems(config, items):
    """Under ASAN, skip tests that throw C++ exceptions.

    ASAN's __cxa_throw interceptor crashes when exceptions cross the
    boundary between an ASAN-compiled extension and non-ASAN Python.
    Detect these tests by checking for 'pytest.raises' in source.
    """
    if "ASAN_OPTIONS" not in os.environ:
        return
    skip = pytest.mark.skip(reason="C++ exceptions crash ASAN interceptor")
    for item in items:
        try:
            source = inspect.getsource(item.obj)
            if "pytest.raises" in source:
                item.add_marker(skip)
        except (TypeError, OSError):
            pass
