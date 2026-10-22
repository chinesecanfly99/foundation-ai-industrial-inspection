"""The optional dependency groups in pyproject must match the install lists under requirements/.

Otherwise `pip install -e .[ui]` and `setup_*.sh|ps1` install two different version sets, and the demo behaves differently on different machines.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = PROJECT_ROOT / "requirements"


def _lines(path: Path) -> list[str]:
    """Expand a requirements file, dropping comments and pip option lines."""
    entries = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        entries.append(line)
    return entries


def _pins(path: Path) -> dict[str, str]:
    pins = {}
    for line in _lines(path):
        name, separator, version = line.partition("==")
        if separator:
            pins[name.strip().lower()] = version.strip()
    return pins


def _project() -> dict:
    return tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]


def test_core_dependencies_match_base_requirements():
    base = _lines(REQUIREMENTS / "base.txt")
    ranges = {line for line in base if "==" not in line and not line.lower().startswith("pytest")}
    assert set(_project()["dependencies"]) == ranges


def test_ui_extra_pins_match_base_requirements():
    base = _pins(REQUIREMENTS / "base.txt")
    ui = _project()["optional-dependencies"]["ui"]
    assert ui, "the ui optional dependency group must not be empty"
    for spec in ui:
        name, separator, version = spec.partition("==")
        assert separator, f"ui dependencies must pin exact versions: {spec}"
        assert base.get(name.lower()) == version, f"pyproject ui differs from base.txt: {spec}"


def test_deep_extra_pins_match_platform_torch_requirements():
    deep = _project()["optional-dependencies"]["deep"]
    assert deep, "the deep optional dependency group must not be empty"
    for platform in ("torch-macos.txt", "torch-cpu-windows.txt"):
        pins = _pins(REQUIREMENTS / platform)
        for spec in deep:
            name, separator, version = spec.partition("==")
            assert separator, f"deep dependencies must pin exact versions: {spec}"
            assert pins.get(name.lower()) == version, f"pyproject deep differs from {platform}: {spec}"


def test_test_extra_reuses_the_base_pytest_constraint():
    base = _lines(REQUIREMENTS / "base.txt")
    pytest_specs = [line for line in base if line.lower().startswith("pytest")]
    assert len(pytest_specs) == 1, base
    assert _project()["optional-dependencies"]["test"] == pytest_specs


def test_historical_windows_lock_is_archived_and_marked_unusable():
    # The old combination is history evidence only; it lives under history/ and its header says it must not be used for installation, so teammates do not build the old environment from it.
    assert not (REQUIREMENTS / "lock-windows-cpu.txt").exists()
    archived = REQUIREMENTS / "history" / "lock-windows-cpu-v0.1.txt"
    head = "".join(archived.read_text(encoding="utf-8").splitlines(keepends=True)[:2])
    assert "History only" in head and "do not use for installation" in head
    assert "torch==2.6.0+cpu" in archived.read_text(encoding="utf-8")
