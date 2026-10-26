"""Download complete MVTec AD categories from a pinned, SHA256-checked mirror.

The mirror LFS hashes are checked, not the original publisher archive checksum.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import time
from urllib.parse import quote, urlparse

import requests

REPO = "foersben/mvtec-ad"
ORIGINAL = "https://www.mvtec.com/research-teaching/datasets/mvtec-ad"
REVISION = "c75b39616f84db43677bcc8228caaafaf5096d7f"
CATEGORIES = ("bottle", "metal_nut", "tile")
TIMEOUT = (20, 90)
ATTEMPTS = 4


def _request_json(url: str) -> tuple[list[dict], str | None]:
    for attempt in range(ATTEMPTS):
        try:
            response = requests.get(url, timeout=TIMEOUT)
            response.raise_for_status()
            value = response.json()
            if not isinstance(value, list):
                raise ValueError("Mirror tree response must be a list")
            return value, response.links.get("next", {}).get("url")
        except (requests.RequestException, ValueError):
            if attempt == ATTEMPTS - 1:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def _entries(category: str, revision: str) -> list[dict]:
    api_path = f"/api/datasets/{REPO}/tree/{revision}/{category}"
    url: str | None = f"https://huggingface.co{api_path}?recursive=true&limit=1000"
    pages: set[str] = set()
    files: dict[str, dict] = {}
    while url:
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.netloc != "huggingface.co" or parsed.path != api_path:
            raise ValueError("Unsafe mirror pagination URL")
        if url in pages:
            raise ValueError("Mirror pagination repeated a page")
        pages.add(url)
        entries, url = _request_json(url)
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("Malformed mirror tree entry")
            if entry.get("type") != "file":
                continue
            relative = entry.get("path", "")
            if not isinstance(relative, str):
                raise ValueError("Mirror path must be a string")
            path = PurePosixPath(relative)
            metadata = relative in {f"{category}/license.txt", f"{category}/readme.txt"}
            if ("\\" in relative or ":" in relative or path.is_absolute() or ".." in path.parts
                    or not path.parts or path.parts[0] != category
                    or (not metadata and (len(path.parts) < 3 or path.parts[1] not in {"train", "test", "ground_truth"}))):
                raise ValueError(f"Unsafe or unexpected mirror path: {relative!r}")
            expected = entry.get("lfs", {}).get("oid", "")
            size = entry.get("size")
            if not isinstance(size, int) or size <= 0:
                raise ValueError(f"Missing mirror file size: {relative}")
            if not metadata and not re.fullmatch(r"[a-f0-9]{64}", expected):
                raise ValueError(f"Missing mirror LFS SHA256 or size: {relative}")
            if metadata and not expected and not re.fullmatch(r"[a-f0-9]{40}", entry.get("oid", "")):
                raise ValueError(f"Missing metadata Git blob hash: {relative}")
            if entry.get("lfs", {}).get("size", size) != size:
                raise ValueError(f"Inconsistent mirror size: {relative}")
            if relative in files and files[relative] != entry:
                raise ValueError(f"Contradictory duplicate mirror entry: {relative}")
            files[relative] = entry
    if not files:
        raise ValueError(f"No files found for {category}")
    for split in ("train", "test", "ground_truth"):
        if not any(path.startswith(f"{category}/{split}/") for path in files):
            raise ValueError(f"Incomplete mirror category: missing {split}")
    return [files[path] for path in sorted(files)]


def _matches(path: Path, entry: dict) -> bool:
    lfs = entry.get("lfs", {}).get("oid")
    digest = hashlib.sha256() if lfs else hashlib.sha1(f"blob {entry['size']}\0".encode("ascii"))
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == (lfs or entry["oid"])


def download(root: Path, category: str = "bottle", workers: int = 6, revision: str = REVISION) -> dict:
    if category not in CATEGORIES:
        raise ValueError(f"Supported categories: {', '.join(CATEGORIES)}")
    if revision != REVISION:
        raise ValueError(f"This downloader requires pinned revision {REVISION}")
    if not 1 <= workers <= 16:
        raise ValueError("workers must be between 1 and 16")
    started = time.monotonic()
    entries = _entries(category, revision)
    root = Path(root).expanduser().resolve()
    category_root = (root / category).resolve()
    if not category_root.is_relative_to(root):
        raise ValueError("Unsafe category directory")
    category_root.mkdir(parents=True, exist_ok=True)
    # Resolve newly created Windows directories before starting worker threads.
    # Parent creation in a worker must not race another worker's path checks.
    category_root = category_root.resolve()
    destinations = {}
    for entry in entries:
        parent = (root / entry["path"]).parent.resolve()
        if not parent.is_relative_to(category_root):
            raise ValueError(f"Unsafe mirror parent directory: {parent}")
        parent.mkdir(parents=True, exist_ok=True)
        target = (root / entry["path"]).resolve()
        if not target.is_relative_to(category_root):
            raise ValueError(f"Unsafe mirror destination: {target}")
        destinations[entry["path"]] = target

    def fetch(entry: dict) -> tuple[str, bool]:
        target = destinations[entry["path"]]
        lfs = entry.get("lfs", {}).get("oid")
        expected = lfs or entry["oid"]
        if target.is_file() and target.stat().st_size == entry["size"] and _matches(target, entry):
            return entry["path"], True
        partial = target.with_name(target.name + ".partial")
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/{quote(entry['path'], safe='/')}?download=true"
        for attempt in range(ATTEMPTS):
            try:
                digest = hashlib.sha256() if lfs else hashlib.sha1(f"blob {entry['size']}\0".encode("ascii"))
                size = 0
                with requests.get(url, timeout=TIMEOUT, stream=True) as response:
                    response.raise_for_status()
                    with partial.open("wb") as handle:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                size += len(chunk)
                                if size > entry["size"]:
                                    raise ValueError(f"Mirror size exceeded: {entry['path']}")
                                digest.update(chunk)
                                handle.write(chunk)
                if size != entry["size"] or digest.hexdigest() != expected:
                    raise ValueError(f"Mirror LFS SHA256/size mismatch: {entry['path']}")
                partial.replace(target)
                return entry["path"], False
            except (requests.RequestException, ValueError, OSError):
                if partial.exists():
                    partial.unlink()
                if attempt == ATTEMPTS - 1:
                    raise
                time.sleep(2 ** attempt)
        raise AssertionError("unreachable")

    cached = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, entry) for entry in entries]
        for index, future in enumerate(as_completed(futures), 1):
            _, was_cached = future.result()
            cached += int(was_cached)
            if index % 40 == 0 or index == len(entries):
                print(f"{category}: verified {index}/{len(entries)} files", flush=True)
    source = {
        "dataset": "MVTec AD", "category": category, "original_url": ORIGINAL,
        "mirror": f"https://huggingface.co/datasets/{REPO}", "revision": revision,
        "license": "CC BY-NC-SA 4.0", "files": len(entries),
        "bytes": sum(entry["size"] for entry in entries),
        "integrity": "Every image and mask checked against mirror LFS SHA256 and size; license/readme checked against mirror Git blob SHA1 and size; original publisher archive checksum NOT checked",
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "cached_files": cached, "downloaded_files": len(entries) - cached,
        "file_manifest": [{"path": entry["path"], "size": entry["size"],
                           **({"sha256": entry["lfs"]["oid"]} if "lfs" in entry else {"git_blob_sha1": entry["oid"]})}
                          for entry in entries],
        "note": "Course research mirror. Use original train/good for train/calibration only; keep original test untouched.",
    }
    # Do not overwrite the original bottle-only root source.json.
    (category_root / "source.json").write_text(json.dumps(source, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in source.items() if key != "file_manifest"}, ensure_ascii=False, indent=2), flush=True)
    return source


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw/mvtec_ad"))
    parser.add_argument("--category", choices=CATEGORIES, default="bottle")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--revision", default=REVISION, help="Pinned revision; other revisions are rejected")
    arguments = parser.parse_args()
    download(arguments.root, arguments.category, arguments.workers, arguments.revision)
