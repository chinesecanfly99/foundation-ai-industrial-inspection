"""Project-local workaround for repeated Gradio 5.49.1 uploads on Windows.

Its upload route returns a cached URL before a background shutil.move finishes.
When Windows refuses os.rename onto an existing file, shutil.move truncates that
same file while FileResponse may be reading it. Reuse identical content, or
publish a fully written sibling atomically, so readers only see complete files.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import tempfile


def _same_contents(source: Path, destination: Path) -> bool:
    if not destination.is_file() or source.stat().st_size != destination.stat().st_size:
        return False
    with source.open("rb") as left, destination.open("rb") as right:
        return hashlib.file_digest(left, "sha256").digest() == hashlib.file_digest(right, "sha256").digest()


def move_uploaded_files_atomically(files: list[str], destinations: list[str]) -> None:
    """Move Gradio's temporary uploads without truncating a published cache file."""
    for file, dest in zip(files, destinations, strict=True):
        source, destination = Path(file), Path(dest)
        if source.resolve() == destination.resolve():
            continue
        if _same_contents(source, destination):
            source.unlink()
            continue
        staged = None
        try:
            with tempfile.NamedTemporaryFile(prefix=".upload-", suffix=".tmp", dir=destination.parent,
                                             delete=False) as output:
                staged = Path(output.name)
                with source.open("rb") as stream:
                    shutil.copyfileobj(stream, output)
            os.replace(staged, destination)
            source.unlink()
        finally:
            if staged is not None:
                staged.unlink(missing_ok=True)


def install_windows_upload_cache_fix() -> None:
    """Patch only this process and the exact pinned version with the known race."""
    if os.name != "nt":
        return
    import gradio
    if gradio.__version__ != "5.49.1":
        return
    import gradio.routes
    gradio.routes.move_uploaded_files_to_cache = move_uploaded_files_atomically
