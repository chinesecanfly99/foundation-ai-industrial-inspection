"""Select the record location while preserving the desktop storage layout."""
from contextvars import ContextVar
from pathlib import Path

active_results = ContextVar("inspection_results", default=None)


def results_dir(project_root: Path) -> Path:
    return active_results.get() or Path(project_root) / "results"
