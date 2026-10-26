"""Backward-compatible entry point for the original bottle download command."""
from pathlib import Path
import argparse

try:  # Direct script and imported scripts module are both supported.
    from .download_mvtec import ORIGINAL, REPO, REVISION, download as download_category
except ImportError:
    from download_mvtec import ORIGINAL, REPO, REVISION, download as download_category


def download(root: Path, workers: int = 6, revision: str = REVISION):
    return download_category(root, category="bottle", workers=workers, revision=revision)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw/mvtec_ad"))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--revision", default=REVISION)
    args = parser.parse_args()
    download(args.root, args.workers, args.revision)
