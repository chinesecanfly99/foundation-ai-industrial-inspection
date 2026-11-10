"""Run the unchanged feature memory and thresholds with a smaller CPU runtime."""
from pathlib import Path
from threading import Lock

import cv2
import numpy as np

from industrial_inspection.detectors import PatchMemoryDetector, _resize_rgb

_network_lock = Lock()
_network = None
_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def features(image):
    global _network
    tensor = ((image - _mean) / _std).transpose(2, 0, 1)[None]
    with _network_lock:
        if _network is None:
            model = Path(__file__).resolve().parents[1] / "feature_backbone.onnx"
            _network = cv2.dnn.readNetFromONNX(str(model))
        _network.setInput(np.ascontiguousarray(tensor))
        low, high = _network.forward(["layer2", "layer3"])
    low = cv2.blur(low[0].transpose(1, 2, 0), (3, 3), borderType=cv2.BORDER_CONSTANT)
    high = cv2.blur(high[0].transpose(1, 2, 0), (3, 3), borderType=cv2.BORDER_CONSTANT)
    height, width = low.shape[:2]
    high = cv2.resize(high, (width, height), interpolation=cv2.INTER_LINEAR)
    return np.concatenate((low, high), axis=2), height, width


def raw_maps(detector, images):
    detector._check_fitted()
    if detector.image_size != 128:
        raise ValueError("The classroom feature extractor uses 128-pixel model inputs.")
    memory = detector.memory
    memory_norm = np.einsum("ij,ij->i", memory, memory)
    output = []
    for rgb in images:
        grid, height, width = features(_resize_rgb(rgb, detector.image_size))
        flat = grid.reshape(height * width, -1)
        squared = np.einsum("ij,ij->i", flat, flat)[:, None] + memory_norm[None]
        squared -= 2 * (flat @ memory.T)
        distances = np.sqrt(np.maximum(squared.min(axis=1), 0) / flat.shape[1])
        heatmap = distances.reshape(height, width)
        output.append(cv2.resize(heatmap, (detector.image_size, detector.image_size), interpolation=cv2.INTER_LINEAR))
    return output


def use_small_cpu_backend():
    cv2.setNumThreads(1)
    PatchMemoryDetector._raw_maps = raw_maps
