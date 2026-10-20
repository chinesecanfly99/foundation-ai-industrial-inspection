"""Normal-only image anomaly detectors with numeric, inspectable artifacts.

``PatchMemoryDetector`` is a deliberately small PatchCore-style teaching
implementation, not an implementation of the official PatchCore benchmark.
Torch and torchvision are imported lazily so the pixel baseline works offline.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageOps


@dataclass
class AnomalyPrediction:
    heatmap: np.ndarray
    score: float
    image_threshold: float
    pixel_threshold: float
    model_id: str
    metadata: dict[str, Any]


def _validate_rgb(rgb: np.ndarray) -> np.ndarray:
    array = np.asarray(rgb)
    if array.ndim != 3 or array.shape[2] != 3 or min(array.shape[:2]) < 2:
        raise ValueError("Expected an RGB image with shape (height, width, 3), at least 2×2.")
    if array.dtype.kind not in "uif" or not np.isfinite(array).all():
        raise ValueError("RGB pixels must be finite numeric values.")
    if array.min() < 0 or array.max() > 255:
        raise ValueError("RGB pixels must be in [0, 255] or floating point [0, 1].")
    if array.dtype.kind == "f" and array.max() <= 1:
        array = array * 255
    return np.rint(array).astype(np.uint8)


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(Path(path)) as image:
        return np.asarray(ImageOps.exif_transpose(image).convert("RGB")).copy()


def _resize_rgb(rgb: np.ndarray, image_size: int) -> np.ndarray:
    image = Image.fromarray(_validate_rgb(rgb))
    return np.asarray(image.resize((image_size, image_size), Image.Resampling.BILINEAR), dtype=np.float32) / 255


def _resize_map(heatmap: np.ndarray, height: int, width: int) -> np.ndarray:
    heatmap = np.asarray(heatmap, dtype=np.float32)
    if heatmap.shape == (height, width):
        return heatmap.copy()
    return np.asarray(
        Image.fromarray(heatmap).resize((width, height), Image.Resampling.BILINEAR),
        dtype=np.float32,
    ).copy()


def _score(heatmap: np.ndarray) -> float:
    """Mean of the highest 1% pixels: robust to one isolated noisy pixel."""
    values = np.asarray(heatmap, dtype=np.float32).ravel()
    count = max(1, int(np.ceil(values.size * 0.01)))
    return float(np.mean(np.partition(values, values.size - count)[-count:]))


def _validate_paths(train_paths: Sequence[Path], calibration_paths: Sequence[Path]) -> tuple[list[Path], list[Path]]:
    train, calibration = [Path(path) for path in train_paths], [Path(path) for path in calibration_paths]
    if not train:
        raise ValueError("At least one normal training image is required.")
    if not calibration:
        raise ValueError("An independent, non-empty normal calibration set is required.")
    if set(path.resolve() for path in train) & set(path.resolve() for path in calibration):
        raise ValueError("Training and calibration paths must be disjoint.")
    for path in train + calibration:
        if not path.is_file():
            raise FileNotFoundError(f"Image does not exist: {path}")
    return train, calibration


class _NormalDetector:
    method: str
    model_id: str

    def __init__(self, image_size: int, seed: int) -> None:
        if not isinstance(image_size, int) or not 32 <= image_size <= 1024:
            raise ValueError("image_size must be an integer between 32 and 1024.")
        self.image_size = image_size
        self.seed = int(seed)
        self.image_threshold = 0.0
        self.pixel_threshold = 0.0
        self.training_count = 0
        self.calibration_count = 0
        self._fitted = False

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Detector is not fitted; train it or load a saved model first.")

    def _raw_maps(self, images: Sequence[np.ndarray]) -> list[np.ndarray]:
        raise NotImplementedError

    def _calibrate(self, paths: Sequence[Path]) -> None:
        scores: list[float] = []
        pixels: list[np.ndarray] = []
        # Batch inference limits RAM and makes CNN calibration substantially faster.
        for start in range(0, len(paths), 16):
            originals = [_load_rgb(path) for path in paths[start : start + 16]]
            maps = self._raw_maps(originals)
            for rgb, heatmap in zip(originals, maps):
                resized = _resize_map(heatmap, *rgb.shape[:2])
                scores.append(_score(resized))
                # Calibrate in model coordinates so large input images do not
                # dominate the per-pixel quantile merely because of resolution.
                pixels.append(np.asarray(heatmap, dtype=np.float32).ravel())
        self.image_threshold = max(max(scores) * 1.05, 1e-6)
        self.pixel_threshold = max(float(np.quantile(np.concatenate(pixels), 0.995)) * 1.05, 1e-6)
        self.calibration_count = len(paths)

    def predict(self, rgb: np.ndarray) -> AnomalyPrediction:
        self._check_fitted()
        rgb = _validate_rgb(rgb)
        raw = self._raw_maps([rgb])[0]
        heatmap = _resize_map(raw, *rgb.shape[:2])
        return AnomalyPrediction(
            heatmap=heatmap,
            score=_score(heatmap),
            image_threshold=float(self.image_threshold),
            pixel_threshold=float(self.pixel_threshold),
            model_id=self.model_id,
            metadata={
                "method": self.method,
                "image_size": self.image_size,
                "training_count": self.training_count,
                "calibration_count": self.calibration_count,
                "score_definition": "mean_of_top_1_percent_pixels",
                "threshold_source": "held_out_normal_calibration_only",
            },
        )

    def _card(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "model_id": self.model_id,
            "method": self.method,
            "image_size": self.image_size,
            "seed": self.seed,
            "training_count": self.training_count,
            "calibration_count": self.calibration_count,
            "image_threshold": self.image_threshold,
            "pixel_threshold": self.pixel_threshold,
            "score_definition": "mean_of_top_1_percent_pixels",
            "calibration": {
                "source": "held_out_normal_images_only",
                "image_statistic": "maximum_calibration_image_score",
                "pixel_quantile": 0.995,
                "margin_multiplier": 1.05,
                "minimum_threshold": 1e-6,
                "limitation": "Normal-only thresholds do not establish defect recall or a production false-alarm guarantee.",
            },
            "artifact_format": "numeric_npz_allow_pickle_false",
        }

    def _save(self, model_dir: Path, arrays: dict[str, np.ndarray], extra: dict[str, Any]) -> None:
        self._check_fitted()
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        artifact = model_dir / "model.npz"
        np.savez_compressed(artifact, **arrays)
        card = self._card() | extra
        card["artifact_sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
        (model_dir / "model_card.json").write_text(json.dumps(card, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class PixelReferenceDetector(_NormalDetector):
    """Aligned-pixel mean/std residual baseline trained using normal images only."""

    method = "pixel_reference"
    model_id = "pixel_reference"
    std_floor = 0.035

    def __init__(self, image_size: int = 128, seed: int = 42) -> None:
        super().__init__(image_size, seed)
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, train_paths: Sequence[Path], calibration_paths: Sequence[Path]) -> PixelReferenceDetector:
        train, calibration = _validate_paths(train_paths, calibration_paths)
        total = np.zeros((self.image_size, self.image_size, 3), dtype=np.float64)
        total_squared = np.zeros_like(total)
        for path in train:
            array = _resize_rgb(_load_rgb(path), self.image_size).astype(np.float64)
            total += array
            total_squared += array * array
        self.mean = (total / len(train)).astype(np.float32)
        variance = np.maximum(total_squared / len(train) - np.square(total / len(train)), 0)
        self.std = np.maximum(np.sqrt(variance), self.std_floor).astype(np.float32)
        self.training_count = len(train)
        self._fitted = True
        self._calibrate(calibration)
        return self

    def _raw_maps(self, images: Sequence[np.ndarray]) -> list[np.ndarray]:
        self._check_fitted()
        assert self.mean is not None and self.std is not None
        maps = []
        for rgb in images:
            residual = (_resize_rgb(rgb, self.image_size) - self.mean) / self.std
            maps.append(np.sqrt(np.mean(np.square(residual), axis=2)).astype(np.float32))
        return maps

    def save(self, model_dir: Path) -> None:
        self._check_fitted()
        assert self.mean is not None and self.std is not None
        self._save(model_dir, {"mean": self.mean, "std": self.std}, {
            "std_floor": self.std_floor,
            "backbone": None,
            "source": "Project baseline: per-pixel RGB standardized residual.",
            "limitations": ["Requires reasonably aligned images and stable viewpoint.", "Texture or lighting shifts can produce false alarms."],
        })


class PatchMemoryDetector(_NormalDetector):
    """Simplified PatchCore-style local feature memory with pretrained ResNet18.

    Uses deterministic uniform sampling, not the official greedy coreset or
    neighborhood reweighting. No anomalous images are used to build the memory.
    """

    method = "patch_memory"
    model_id = "patch_memory_resnet18"
    weights_name = "ResNet18_Weights.IMAGENET1K_V1"

    def __init__(self, image_size: int = 128, memory_size: int = 2048, device: str = "cpu", seed: int = 42) -> None:
        super().__init__(image_size, seed)
        if not isinstance(memory_size, int) or not 1 <= memory_size <= 65536:
            raise ValueError("memory_size must be an integer between 1 and 65536.")
        if device != "cpu" and device != "mps" and device != "cuda" and not device.startswith("cuda:"):
            raise ValueError("device must be cpu, mps, cuda, or cuda:<index>.")
        self.memory_size = memory_size
        self.device = device
        self.memory: np.ndarray | None = None
        self._network: Any = None
        self._memory_tensor: Any = None

    def _ensure_network(self) -> None:
        if self._network is not None:
            return
        try:
            import torch
            from torchvision.models import ResNet18_Weights, resnet18
        except (ImportError, RuntimeError) as exc:
            raise RuntimeError("PatchMemory requires a compatible torch/torchvision installation. Use the documented environment setup.") from exc
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable. Select cpu or install a compatible GPU runtime.")
        if self.device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable. Select cpu on this computer.")
        cache_root = Path(os.environ.get("TORCH_HOME", Path(__file__).resolve().parents[2] / ".cache" / "torch"))
        torch.hub.set_dir(str(cache_root / "hub"))
        if self.device == "cpu":
            torch.set_num_threads(min(4, os.cpu_count() or 1))
        try:
            self._network = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval().to(self.device)
        except Exception as exc:
            raise RuntimeError("Unable to load official ResNet18 pretrained weights. Connect once to populate the torchvision cache, then retry.") from exc
        for parameter in self._network.parameters():
            parameter.requires_grad_(False)

    def _features(self, images: Sequence[np.ndarray]) -> tuple[Any, int, int]:
        self._ensure_network()
        import torch
        from torch.nn import functional as functional

        batch = np.stack([_resize_rgb(rgb, self.image_size).transpose(2, 0, 1) for rgb in images])
        tensor = torch.from_numpy(batch).to(self.device)
        mean = tensor.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = tensor.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        network = self._network
        with torch.inference_mode():
            tensor = (tensor - mean) / std
            tensor = network.maxpool(network.relu(network.bn1(network.conv1(tensor))))
            tensor = network.layer1(tensor)
            layer2 = network.layer2(tensor)
            layer3 = network.layer3(layer2)
            layer2 = functional.avg_pool2d(layer2, 3, stride=1, padding=1)
            layer3 = functional.avg_pool2d(layer3, 3, stride=1, padding=1)
            layer3 = functional.interpolate(layer3, size=layer2.shape[-2:], mode="bilinear", align_corners=False)
            features = torch.cat([layer2, layer3], dim=1)
            height, width = features.shape[-2:]
            features = features.permute(0, 2, 3, 1).reshape(len(images), height * width, -1)
        return features, height, width

    def fit(self, train_paths: Sequence[Path], calibration_paths: Sequence[Path]) -> PatchMemoryDetector:
        train, calibration = _validate_paths(train_paths, calibration_paths)
        rng = np.random.default_rng(self.seed)
        # Random priority sampling is a uniform sample without replacement and
        # bounds intermediate storage to one feature batch plus the memory cap.
        selected: np.ndarray | None = None
        priorities = np.empty(0, dtype=np.float64)
        for start in range(0, len(train), 16):
            originals = [_load_rgb(path) for path in train[start : start + 16]]
            features, _, _ = self._features(originals)
            features_np = features.detach().cpu().numpy().reshape(-1, features.shape[-1]).astype(np.float32)
            new_priorities = rng.random(len(features_np))
            selected = features_np if selected is None else np.concatenate([selected, features_np])
            priorities = np.concatenate([priorities, new_priorities])
            if len(selected) > self.memory_size:
                keep = np.argpartition(priorities, self.memory_size - 1)[: self.memory_size]
                # Sort priorities to make array order deterministic too.
                keep = keep[np.argsort(priorities[keep])]
                selected, priorities = selected[keep], priorities[keep]
        assert selected is not None
        self.memory = np.ascontiguousarray(selected, dtype=np.float32)
        self._memory_tensor = None
        self.training_count = len(train)
        self._fitted = True
        self._calibrate(calibration)
        return self

    def _raw_maps(self, images: Sequence[np.ndarray]) -> list[np.ndarray]:
        self._check_fitted()
        self._ensure_network()
        import torch
        from torch.nn import functional as functional

        assert self.memory is not None
        features, height, width = self._features(images)
        if self._memory_tensor is None:
            self._memory_tensor = torch.from_numpy(self.memory).to(self.device)
        batch_size, patch_count, dimension = features.shape
        flat = features.reshape(-1, dimension)
        minimum_distances = []
        with torch.inference_mode():
            # Limit distance matrix memory for both CPU and small laptop GPUs.
            for chunk in flat.split(512):
                distances = torch.cdist(chunk, self._memory_tensor, p=2)
                minimum_distances.append(distances.min(dim=1).values / np.sqrt(dimension))
            distance_map = torch.cat(minimum_distances).reshape(batch_size, 1, height, width)
            distance_map = functional.interpolate(distance_map, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return [array.copy() for array in distance_map[:, 0].cpu().numpy().astype(np.float32)]

    def save(self, model_dir: Path) -> None:
        self._check_fitted()
        assert self.memory is not None
        self._save(model_dir, {"memory": self.memory}, {
            "memory_size": self.memory_size,
            "actual_memory_size": len(self.memory),
            "backbone": "torchvision.resnet18",
            "weights": self.weights_name,
            "feature_layers": ["layer2", "layer3"],
            "feature_dimension": int(self.memory.shape[1]),
            "sampling": "seeded_uniform_random_priority_without_replacement",
            "distance": "euclidean_divided_by_sqrt_feature_dimension",
            "source": "https://github.com/amazon-science/patchcore-inspection",
            "implementation_note": "Simplified PatchCore-style teaching model; not an official PatchCore reproduction.",
            "limitations": ["Uses ResNet18 rather than the official benchmark backbone.", "Uses uniform sampling rather than greedy coreset selection.", "Omits official image-score neighborhood reweighting."],
        })


def load_detector(model_dir: Path, device: str = "cpu") -> PixelReferenceDetector | PatchMemoryDetector:
    """Load only numeric arrays; reject mismatched or incomplete model artifacts."""
    model_dir = Path(model_dir)
    card = json.loads((model_dir / "model_card.json").read_text(encoding="utf-8"))
    artifact = model_dir / "model.npz"
    if card.get("schema_version") != 1:
        raise ValueError("Unsupported model card schema.")
    if hashlib.sha256(artifact.read_bytes()).hexdigest() != card.get("artifact_sha256"):
        raise ValueError("Model artifact checksum does not match the model card.")
    detector: PixelReferenceDetector | PatchMemoryDetector
    method = card.get("method")
    with np.load(artifact, allow_pickle=False) as arrays:
        if method == "pixel_reference":
            detector = PixelReferenceDetector(int(card["image_size"]), int(card["seed"]))
            expected_shape = (detector.image_size, detector.image_size, 3)
            mean, std = arrays["mean"], arrays["std"]
            if mean.shape != expected_shape or std.shape != expected_shape or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
                raise ValueError("Invalid pixel-reference model arrays.")
            detector.mean, detector.std = mean.astype(np.float32), std.astype(np.float32)
            detector.std_floor = float(card.get("std_floor", 0.035))
        elif method == "patch_memory":
            detector = PatchMemoryDetector(int(card["image_size"]), int(card["memory_size"]), device, int(card["seed"]))
            memory = arrays["memory"]
            if memory.ndim != 2 or memory.shape[1] != 384 or not 1 <= memory.shape[0] <= detector.memory_size or not np.isfinite(memory).all():
                raise ValueError("Invalid patch memory array.")
            if card.get("weights") != detector.weights_name:
                raise ValueError("Unsupported pretrained backbone weights.")
            detector.memory = np.ascontiguousarray(memory, dtype=np.float32)
        else:
            raise ValueError(f"Unsupported detector method: {method}")
    detector.image_threshold = float(card["image_threshold"])
    detector.pixel_threshold = float(card["pixel_threshold"])
    if not np.isfinite([detector.image_threshold, detector.pixel_threshold]).all() or min(detector.image_threshold, detector.pixel_threshold) <= 0:
        raise ValueError("Model thresholds must be positive finite numbers.")
    detector.training_count = int(card["training_count"])
    detector.calibration_count = int(card["calibration_count"])
    detector.model_id = str(card["model_id"])
    detector._fitted = True
    return detector
