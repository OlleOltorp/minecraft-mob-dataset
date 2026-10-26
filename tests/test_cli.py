"""Tests for the command-line entry point."""

import importlib

import pytest

from mobdataset import __version__
from mobdataset.cli import CHECKS, COMMANDS, main


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == f"mobdataset {__version__}"


def test_no_command_lists_commands(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    out = capsys.readouterr().out
    assert all(name in out for name in COMMANDS)


def test_unknown_command_fails(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["nope"]) == 2
    assert "unknown command" in capsys.readouterr().err


@pytest.mark.parametrize("module", [m for m, _ in [*COMMANDS.values(), *CHECKS.values()]])
def test_every_command_has_main(module: str) -> None:
    heavy = ("model.", "game.view_windows", "game.window_capture")
    try:
        loaded = importlib.import_module(f"mobdataset.{module}")
    except ImportError as e:
        if module.startswith(heavy) or module.startswith("checks."):
            pytest.skip(f"optional extra missing: {e}")
        raise
    assert callable(loaded.main)
