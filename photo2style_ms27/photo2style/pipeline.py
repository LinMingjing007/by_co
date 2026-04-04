"""Photo to style pipeline for MindSpore 2.7.0 + MindNLP 0.5.1."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np


def _load_mindspore():
    try:
        ms_mod = importlib.import_module("mindspore")
        return ms_mod, getattr(ms_mod, "Tensor"), getattr(ms_mod, "ops")
    except Exception:
        return None, None, None

ms, Tensor, ops = _load_mindspore()


STYLE_PRESETS: Dict[str, Dict[str, float]] = {
    "ghibli": {"saturation": 1.25, "contrast": 1.1, "gamma": 0.95, "bilateral": 2},
    "disney": {"saturation": 1.4, "contrast": 1.15, "gamma": 0.9, "bilateral": 3},
    "ink": {"saturation": 0.6, "contrast": 1.35, "gamma": 1.05, "bilateral": 1},
    "sketch": {"saturation": 0.25, "contrast": 1.45, "gamma": 1.0, "bilateral": 1},
}


@dataclass
class StyleResult:
    style_name: str
    image_rgb: np.ndarray


class Photo2StylePipeline:
    """Rule-based stylization pipeline that integrates MindSpore tensor ops."""

    def __init__(self, use_mindspore: bool = True, device_target: str = "CPU") -> None:
        self.backend = "numpy"
        self.ms, self.Tensor, self.ops = _load_mindspore() if use_mindspore else (None, None, None)
        if self.ms is not None:
            self.ms.set_context(mode=self.ms.PYNATIVE_MODE, device_target=device_target)
            self.backend = "mindspore"

    def stylize(self, image_bgr: np.ndarray, style_name: str = "ghibli") -> StyleResult:
        if style_name not in STYLE_PRESETS:
            raise ValueError(f"Unknown style: {style_name}. Available: {list(STYLE_PRESETS)}")

        cfg = STYLE_PRESETS[style_name]
        smoothed = image_bgr.copy()
        for _ in range(int(cfg["bilateral"])):
            smoothed = cv2.bilateralFilter(smoothed, d=7, sigmaColor=60, sigmaSpace=60)

        gray = cv2.cvtColor(smoothed, cv2.COLOR_BGR2GRAY)
        edges = cv2.adaptiveThreshold(
            cv2.medianBlur(gray, 5),
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            9,
            2,
        )
        cartoon = cv2.bitwise_and(smoothed, smoothed, mask=edges)
        rgb = cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB)
        rgb = self._tone_adjust(rgb, cfg)

        if style_name == "sketch":
            sketch = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            rgb = cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)

        return StyleResult(style_name=style_name, image_rgb=rgb)

    def stylize_path(self, image_path: str | Path, style_name: str = "ghibli") -> StyleResult:
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        return self.stylize(image, style_name)

    def _tone_adjust(self, image_rgb: np.ndarray, cfg: Dict[str, float]) -> np.ndarray:
        x = image_rgb.astype(np.float32) / 255.0
        gray = x.mean(axis=2, keepdims=True)
        x = gray + cfg["saturation"] * (x - gray)
        x = np.clip((x - 0.5) * cfg["contrast"] + 0.5, 0.0, 1.0)
        x = np.power(np.clip(x, 0.0, 1.0), cfg["gamma"])

        if self.backend == "mindspore":
            tx = self.Tensor(x, self.ms.float32)
            tx = self.ops.clip_by_value(tx, self.Tensor(0.0, self.ms.float32), self.Tensor(1.0, self.ms.float32))
            x = tx.asnumpy()

        return (x * 255.0).astype(np.uint8)


def parse_style_text(style_prompt: str) -> Tuple[str, str]:
    """Simple Chinese keyword parser for style control, compatible with MindNLP workflows."""
    prompt = style_prompt.strip().lower()
    mapping = {
        "吉卜力": "ghibli",
        "ghibli": "ghibli",
        "迪士尼": "disney",
        "disney": "disney",
        "古风": "ink",
        "水墨": "ink",
        "钢笔": "sketch",
        "素描": "sketch",
        "sketch": "sketch",
    }
    for key, value in mapping.items():
        if key in prompt:
            return value, f"命中关键词: {key}"
    return "ghibli", "未命中关键词，默认使用 ghibli 风格"
