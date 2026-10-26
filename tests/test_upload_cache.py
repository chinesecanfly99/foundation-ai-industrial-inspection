"""Repeated uploads must not truncate an image that the browser can already read."""
from pathlib import Path

import pytest

import app.upload_cache as upload_cache


def test_repeat_upload_keeps_existing_file_without_rewriting(tmp_path, monkeypatch):
    source, destination = tmp_path / "temporary", tmp_path / "good_000.png"
    contents = b"an existing complete image" * 100
    source.write_bytes(contents)
    destination.write_bytes(contents)
    before = destination.stat().st_mtime_ns
    monkeypatch.setattr(upload_cache.os, "replace", lambda *args: pytest.fail("Identical cache was rewritten"))
    upload_cache.move_uploaded_files_atomically([str(source)], [str(destination)])
    assert destination.read_bytes() == contents
    assert destination.stat().st_mtime_ns == before
    assert not source.exists()


def test_changed_cache_is_fully_staged_before_atomic_publication(tmp_path, monkeypatch):
    source, destination = tmp_path / "temporary", tmp_path / "good_000.png"
    source.write_bytes(b"new complete image" * 100)
    destination.write_bytes(b"old complete image")
    original_replace = upload_cache.os.replace
    observed = []
    def publish(staged, target):
        observed.append((Path(staged).read_bytes(), Path(target).read_bytes()))
        return original_replace(staged, target)
    monkeypatch.setattr(upload_cache.os, "replace", publish)
    upload_cache.move_uploaded_files_atomically([str(source)], [str(destination)])
    assert observed == [(b"new complete image" * 100, b"old complete image")]
    assert destination.read_bytes() == b"new complete image" * 100
    assert not source.exists()
    assert not list(tmp_path.glob(".upload-*.tmp"))


def test_failed_publication_keeps_existing_image_and_upload(tmp_path, monkeypatch):
    source, destination = tmp_path / "temporary", tmp_path / "good_000.png"
    source.write_bytes(b"new complete image")
    destination.write_bytes(b"old complete image")
    def refuse(*args):
        raise PermissionError("file is in use")
    monkeypatch.setattr(upload_cache.os, "replace", refuse)
    with pytest.raises(PermissionError, match="file is in use"):
        upload_cache.move_uploaded_files_atomically([str(source)], [str(destination)])
    assert destination.read_bytes() == b"old complete image"
    assert source.read_bytes() == b"new complete image"
    assert not list(tmp_path.glob(".upload-*.tmp"))


def test_same_source_and_destination_is_preserved(tmp_path):
    image = tmp_path / "good_000.png"
    image.write_bytes(b"complete image")
    upload_cache.move_uploaded_files_atomically([str(image)], [str(image)])
    assert image.read_bytes() == b"complete image"
