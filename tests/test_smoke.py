"""Smoke tests for package bootstrap."""

import multiverse_agent


def test_package_exposes_version() -> None:
    assert multiverse_agent.__version__ == "0.1.0"
