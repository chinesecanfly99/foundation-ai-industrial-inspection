"""Fetch only MVTec AD bottle from a public mirror, verify mirror LFS hashes.

Original data: MVTec Software GmbH, CC BY-NC-SA 4.0.
This is NOT verification against the publisher's full archive checksum.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time

import requests

REPO = "foersben/mvtec-ad"
ORIGINAL = "https://www.mvtec.com/research-teaching/datasets/mvtec-ad"
REVISION = "c75b39616f84db43677bcc8228caaafaf5096d7f"


def download(root: Path, workers: int = 6, revision: str = REVISION):
    response = requests.get(f"https://huggingface.co/api/datasets/{REPO}/tree/{revision}/bottle?recursive=true&limit=1000", timeout=45)
    response.raise_for_status()
    entries = [x for x in response.json() if x["type"] == "file"]
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)

    def fetch(entry):
        target = (root / entry["path"]).resolve()
        if not target.is_relative_to(root):
            raise ValueError("Unsafe mirror path")
        expected = entry.get("lfs", {}).get("oid")
        if target.exists() and target.stat().st_size == entry["size"]:
            if not expected or hashlib.sha256(target.read_bytes()).hexdigest() == expected:
                return entry["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(3):
            try:
                r = requests.get(f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/{entry['path']}?download=true", timeout=90)
                r.raise_for_status()
                if len(r.content) != entry["size"]:
                    raise ValueError("Size mismatch")
                if expected and hashlib.sha256(r.content).hexdigest() != expected:
                    raise ValueError("Mirror LFS SHA256 mismatch")
                partial = target.with_suffix(target.suffix + ".partial")
                partial.write_bytes(r.content)
                partial.replace(target)
                return entry["path"]
            except (requests.RequestException, ValueError):
                if attempt == 2:
                    raise
                time.sleep(1 + attempt)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, entry) for entry in entries]
        for i, future in enumerate(as_completed(futures), 1):
            future.result()
            if i % 40 == 0 or i == len(entries):
                print(f"Verified {i}/{len(entries)} files", flush=True)
    source = {"dataset": "MVTec AD", "category": "bottle", "original_url": ORIGINAL,
              "mirror": f"https://huggingface.co/datasets/{REPO}", "revision": revision,
              "license": "CC BY-NC-SA 4.0", "files": len(entries), "bytes": sum(e["size"] for e in entries),
              "integrity": "Mirror LFS SHA256 and size checked; original publisher archive checksum NOT checked",
              "note": "Original legacy download URLs returned HTTP 404 on 2026-10-20. Mirror used for course prototype."}
    (root / "source.json").write_text(json.dumps(source, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(source, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw/mvtec_ad"))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--revision", default=REVISION)
    args = parser.parse_args()
    download(args.root, args.workers, args.revision)
