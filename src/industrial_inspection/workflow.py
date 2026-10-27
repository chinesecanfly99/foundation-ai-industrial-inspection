"""Local inspection records, batches, comparisons and append-only reviews.

Model artifacts and calibration are never changed by workflow actions. Stored
recommendations are immutable; human reviews are separate, timestamped events.
"""
from __future__ import annotations

from contextlib import closing
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import uuid
import zipfile

from PIL import Image, ImageEnhance, ImageFilter, ImageOps, UnidentifiedImageError

from .catalog import PRODUCTS, REAL_PRODUCTS, product_id_from_card

MAX_BATCH_FILES = 20
MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_BATCH_BYTES = 100 * 1024 * 1024
DECISIONS = {"pass", "review", "reject", "undetermined"}
VERDICTS = {"pass", "review", "reject", "retake"}


class WorkflowError(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def finite(value):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise WorkflowError("invalid_report")
    return float(value)


def csv_cell(value):
    text = "" if value is None else str(value)
    # Exported user filenames and reviewer comments must remain text in Excel.
    if text.lstrip().startswith(("=", "+", "-", "@")) or text.startswith(("\t", "\r", "\n")):
        return "'" + text
    return value if isinstance(value, (int, float)) else text


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_cell(row.get(key)) for key in fields})
    return path


class InspectionStore:
    def __init__(self, project_root: Path):
        self.root = Path(project_root).resolve()
        self.folder = self.root / "results" / "workflow"
        self.folder.mkdir(parents=True, exist_ok=True)
        self.database = self.folder / "inspections.sqlite3"
        with closing(self.connect()) as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS inspections (
                    id TEXT PRIMARY KEY, created_at TEXT NOT NULL,
                    source TEXT NOT NULL, batch_id TEXT,
                    filename TEXT NOT NULL, product TEXT NOT NULL,
                    model_id TEXT NOT NULL, status TEXT NOT NULL,
                    decision TEXT, score REAL, threshold REAL,
                    report_json TEXT NOT NULL, error TEXT
                );
                CREATE TABLE IF NOT EXISTS reviews (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    inspection_id TEXT NOT NULL REFERENCES inspections(id),
                    created_at TEXT NOT NULL, reviewer TEXT NOT NULL,
                    verdict TEXT NOT NULL, comment TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_inspections_created ON inspections(created_at);
                CREATE INDEX IF NOT EXISTS idx_inspections_batch ON inspections(batch_id);
                CREATE INDEX IF NOT EXISTS idx_reviews_record ON reviews(inspection_id,event_id);
                PRAGMA user_version=1;
            """)
            connection.commit()

    def connect(self):
        connection = sqlite3.connect(self.database, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def assets(self, inspection_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{32}", str(inspection_id)):
            raise WorkflowError("record_missing")
        return self.folder / "records" / inspection_id

    def save(self, report: dict, image: Image.Image | None, overlay: Image.Image | None,
             *, source="single", batch_id=None, filename="uploaded_image.png", error=None) -> dict:
        if source not in {"single", "batch", "comparison", "challenge", "daily_case"}:
            raise WorkflowError("invalid_report")
        report = dict(report)
        decision = report.get("decision")
        if error is None and decision not in DECISIONS:
            raise WorkflowError("invalid_report")
        score, threshold = finite(report.get("score")), finite(report.get("image_threshold"))
        if threshold is not None and threshold <= 0:
            raise WorkflowError("invalid_report")
        inspection_id = uuid.uuid4().hex
        created_at = now()
        report.update(inspection_id=inspection_id, inspection_created_at=created_at,
                      inspection_source=source)
        folder = self.assets(inspection_id)
        folder.mkdir(parents=True, exist_ok=False)
        try:
            if image is not None:
                working = ImageOps.exif_transpose(image).convert("RGB")
                working.thumbnail((1024, 1024))
                working.save(folder / "input.png")
                report.setdefault("image_sha256", hashlib.sha256(working.tobytes()).hexdigest())
            if overlay is not None:
                overlay.save(folder / "overlay.png")
            encoded = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2)
            (folder / "report.json").write_text(encoded, encoding="utf-8")
            with closing(self.connect()) as connection:
                with connection:
                    connection.execute("INSERT INTO inspections VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                        inspection_id, created_at, source, batch_id, Path(str(filename)).name[:240],
                        str(report.get("product_category", "unknown")), str(report.get("model_id", "unknown")),
                        "failed" if error else "success", decision, score, threshold, encoded, error))
        except Exception:
            for name in ("input.png", "overlay.png", "report.json"):
                (folder / name).unlink(missing_ok=True)
            folder.rmdir()
            raise
        return report

    def query(self, *, product="all", review="all", source="all", limit=None) -> list[dict]:
        conditions, args = [], []
        if product != "all":
            conditions.append("i.product = ?"); args.append(product)
        if source != "all":
            conditions.append("i.source = ?"); args.append(source)
        if review == "unreviewed":
            conditions.append("r.event_id IS NULL AND i.status = 'success'")
        elif review == "reviewed":
            conditions.append("r.event_id IS NOT NULL")
        elif review != "all":
            raise WorkflowError("invalid_filter")
        sql = """SELECT i.*,r.reviewer,r.verdict,r.comment,r.created_at AS reviewed_at,
                    r.event_id AS review_event_id FROM inspections i LEFT JOIN reviews r
                    ON r.event_id=(SELECT MAX(event_id) FROM reviews WHERE inspection_id=i.id)"""
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY i.created_at DESC,i.id DESC"
        if limit is not None:
            sql += " LIMIT ?"; args.append(int(limit))
        with closing(self.connect()) as connection:
            return [dict(row) for row in connection.execute(sql, args)]

    def get(self, inspection_id: str) -> dict:
        self.assets(inspection_id)
        with closing(self.connect()) as connection:
            row = connection.execute("SELECT * FROM inspections WHERE id=?", (inspection_id,)).fetchone()
        if row is None:
            raise WorkflowError("record_missing")
        item = dict(row)
        item["report"] = json.loads(item.pop("report_json"))
        item["reviews"] = self.reviews(inspection_id)
        return item

    def reviews(self, inspection_id: str) -> list[dict]:
        with closing(self.connect()) as connection:
            return [dict(row) for row in connection.execute(
                "SELECT * FROM reviews WHERE inspection_id=? ORDER BY event_id", (inspection_id,))]

    def review(self, inspection_id: str, reviewer: str, verdict: str, comment: str) -> dict:
        reviewer, comment = str(reviewer or "").strip(), str(comment or "").strip()
        if not reviewer or len(reviewer) > 80 or len(comment) > 2000 or verdict not in VERDICTS:
            raise WorkflowError("invalid_review")
        item = self.get(inspection_id)
        if item["status"] != "success":
            raise WorkflowError("failed_record_review")
        with closing(self.connect()) as connection:
            with connection:
                connection.execute("INSERT INTO reviews(inspection_id,created_at,reviewer,verdict,comment) VALUES(?,?,?,?,?)",
                                   (inspection_id, now(), reviewer, verdict, comment))
        # Original report and previous reviews are deliberately not updated.
        return self.get(inspection_id)

    def export_history(self, *, product="all", review="all", source="all") -> Path:
        rows = self.query(product=product, review=review, source=source)
        return write_csv(self.folder / "exports" / uuid.uuid4().hex / "inspection_history.csv", rows,
                         ["id", "created_at", "source", "batch_id", "filename", "product", "model_id",
                          "status", "decision", "score", "threshold", "reviewer", "verdict", "comment", "reviewed_at", "error"])


def read_image(path: Path) -> Image.Image:
    path = Path(path)
    if not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
        raise WorkflowError("file_size")
    try:
        with Image.open(path) as source:
            if min(source.size) < 32 or source.width * source.height > 16_000_000 or max(source.size) > 6000:
                raise WorkflowError("image_size")
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.thumbnail((1024, 1024))
            return image.copy()
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError) as exc:
        if isinstance(exc, WorkflowError):
            raise
        raise WorkflowError("invalid_image") from exc


def validate_model(root: Path, product: str, method: str) -> str:
    if product not in REAL_PRODUCTS or method not in {"patch", "pixel"}:
        raise WorkflowError("invalid_model")
    model_id = f"{product}_{method}"
    directory = Path(root) / "models" / model_id
    try:
        card = json.loads((directory / "model_card.json").read_text(encoding="utf-8"))
        if not (directory / "model.npz").is_file() or product_id_from_card(card) != product:
            raise WorkflowError("invalid_model")
    except (OSError, ValueError) as exc:
        raise WorkflowError("invalid_model") from exc
    return model_id


def inspect_and_store(root: Path, image: Image.Image, product: str, method: str,
                      *, source="single", batch_id=None, filename="uploaded_image.png", metadata=None,
                      report_transform=None):
    from .pipeline import inspect_image
    if filename == 'uploaded_image.png' and isinstance(image, Image.Image):
        filename = image.info.get('inspection_filename', filename)
    image = transform_image(image, 1.0, 0.0)
    model_id = validate_model(root, product, method)
    overlay, report = inspect_image(image, Path(root) / "models" / model_id)
    if metadata:
        report = {**report, "experiment": metadata}
    if report_transform:
        report = report_transform(report)
    report = InspectionStore(root).save(report, image, overlay, source=source,
                                        batch_id=batch_id, filename=filename)
    return overlay, report


def run_batch(root: Path, paths, product: str, method="patch", *, progress=None, report_transform=None) -> dict:
    validate_model(root, product, method)
    paths = [Path(str(path)) for path in (paths or [])]
    if not 1 <= len(paths) <= MAX_BATCH_FILES:
        raise WorkflowError("batch_count")
    if sum(path.stat().st_size for path in paths if path.is_file()) > MAX_BATCH_BYTES:
        raise WorkflowError("batch_size")
    store = InspectionStore(root)
    batch_id, reports = uuid.uuid4().hex, []
    for index, path in enumerate(paths):
        try:
            image = read_image(path)
            _, report = inspect_and_store(root, image, product, method, source="batch",
                                           batch_id=batch_id, filename=path.name, report_transform=report_transform)
            status, error = "success", None
        except (WorkflowError, OSError, RuntimeError, ImportError) as exc:
            error = exc.code if isinstance(exc, WorkflowError) else "inference_failed"
            report = store.save({"product_category": product, "model_id": f"{product}_{method}",
                                 "error": error}, None, None, source="batch", batch_id=batch_id,
                                filename=path.name, error=error)
            status = "failed"
        reports.append({**report, "filename": path.name, "status": status, "error": error})
        if progress:
            progress((index + 1) / len(paths))
    folder = store.folder / "batches" / batch_id
    summary = write_csv(folder / "batch_summary.csv", reports,
                        ["inspection_id", "filename", "product_category", "model_id", "status", "decision", "score", "image_threshold", "error"])
    archive = folder / "batch_reports.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as package:
        package.write(summary, "batch_summary.csv")
        package.writestr("batch_manifest.json", json.dumps({"batch_id": batch_id,
            "product": product, "model_id": f"{product}_{method}", "count": len(reports),
            "success": sum(r["status"] == "success" for r in reports),
            "failed": sum(r["status"] == "failed" for r in reports),
            "inspection_ids": [r["inspection_id"] for r in reports]}, indent=2))
        for report in reports:
            for filename in ("input.png", "overlay.png", "report.json"):
                asset = store.assets(report["inspection_id"]) / filename
                if asset.is_file():
                    package.write(asset, f"records/{report['inspection_id']}/{filename}")
    return {"batch_id": batch_id, "reports": reports, "csv": str(summary), "zip": str(archive)}


def compare_models(root: Path, image: Image.Image, product: str, *, report_transform=None, filename='uploaded_image.png'):
    if image is None:
        raise WorkflowError("no_image")
    return [inspect_and_store(root, image, product, method, source="comparison",report_transform=report_transform,filename=filename) for method in ("pixel", "patch")]


def transform_image(image: Image.Image, brightness: float, blur: float) -> Image.Image:
    if image is None:
        raise WorkflowError("no_image")
    if not math.isfinite(float(brightness)) or not math.isfinite(float(blur)) or not .1 <= brightness <= 1.8 or not 0 <= blur <= 5:
        raise WorkflowError("invalid_transform")
    if min(image.size) < 32 or image.width * image.height > 16_000_000:
        raise WorkflowError("image_size")
    working = ImageOps.exif_transpose(image).convert("RGB")
    working.thumbnail((1024, 1024))
    return ImageEnhance.Brightness(working).enhance(brightness).filter(ImageFilter.GaussianBlur(radius=blur))


def quality_challenge(root: Path, image: Image.Image, product: str, brightness: float, blur: float, *, report_transform=None):
    changed = transform_image(image, brightness, blur)
    outputs = []
    for variant, picture in (("original", image), ("transformed", changed)):
        outputs.append(inspect_and_store(root, picture, product, "patch", source="challenge",report_transform=report_transform,
            metadata={"type": "quality_challenge", "variant": variant,
                      "brightness_multiplier": 1.0 if variant == "original" else brightness,
                      "blur_radius": 0.0 if variant == "original" else blur,
                      "benchmark_result": False}))
    return changed, outputs
