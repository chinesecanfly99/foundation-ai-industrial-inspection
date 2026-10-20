import json
import numpy as np
import pytest
from PIL import Image
from industrial_inspection.pipeline import inspect_image


def test_no_image_or_tiny_input_fails_before_loading_model(tmp_path):
    with pytest.raises(ValueError, match="上传"):
        inspect_image(None, tmp_path, tmp_path / "none")
    with pytest.raises(ValueError, match="32"):
        inspect_image(Image.new("RGB", (20, 20)), tmp_path, tmp_path / "none")
