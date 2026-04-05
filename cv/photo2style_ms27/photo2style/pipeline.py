"""真人照片转风格图像的轻量推理管线。

该模块提供：

- ``Photo2StylePipeline``: 输入图像并执行风格化
- ``StyleResult``: 返回风格名称和 RGB 结果图
- ``parse_style_text``: 将中文/英文提示词映射为内部风格名
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np


def _load_mindspore():
    """按需加载 MindSpore，避免在纯 NumPy 环境下导入失败。"""
    try:
        ms_mod = importlib.import_module("mindspore")
        return ms_mod, getattr(ms_mod, "Tensor"), getattr(ms_mod, "ops")
    except Exception:
        return None, None, None

ms, Tensor, ops = _load_mindspore()


def _load_mindnlp_tokenizer():
    """按需加载 MindNLP 分词器，作为风格提示词解析的标准化组件。"""
    candidates = (
        ("mindnlp.transforms", "BasicTokenizer"),
        ("mindnlp.dataset.transforms", "BasicTokenizer"),
        ("mindnlp.transforms.tokenizers", "BasicTokenizer"),
    )
    for module_name, class_name in candidates:
        try:
            module = importlib.import_module(module_name)
            tokenizer_cls = getattr(module, class_name)
            try:
                return tokenizer_cls(lower_case=True)
            except TypeError:
                return tokenizer_cls()
        except Exception:
            continue
    return None


STYLE_PRESETS: Dict[str, Dict[str, Any]] = {
    "ghibli": {
        "saturation": 1.18,
        "contrast": 1.08,
        "gamma": 0.96,
        "posterize_levels": 24,
        "quantize_passes": 1,
        "edge_strength": 0.22,
        "detail_boost": 0.14,
        "bilateral": 2,
        "color_temperature": 6.0,
        "clahe_clip": 1.8,
        "shadow_lift": 0.03,
        "skin_protect": 0.28,
        "highlight_protect": 0.20,
        "face_smooth": 0.22,
    },
    "disney": {
        "saturation": 1.32,
        "contrast": 1.12,
        "gamma": 0.92,
        "posterize_levels": 20,
        "quantize_passes": 2,
        "edge_strength": 0.18,
        "detail_boost": 0.18,
        "bilateral": 3,
        "color_temperature": 10.0,
        "clahe_clip": 2.2,
        "shadow_lift": 0.05,
        "skin_protect": 0.34,
        "highlight_protect": 0.24,
        "face_smooth": 0.28,
    },
    "ink": {
        "saturation": 0.72,
        "contrast": 1.28,
        "gamma": 1.03,
        "posterize_levels": 14,
        "quantize_passes": 1,
        "edge_strength": 0.34,
        "detail_boost": 0.08,
        "bilateral": 2,
        "color_temperature": -4.0,
        "clahe_clip": 1.4,
        "shadow_lift": 0.0,
        "skin_protect": 0.12,
        "highlight_protect": 0.12,
        "face_smooth": 0.10,
    },
    "sketch": {
        "saturation": 0.18,
        "contrast": 1.38,
        "gamma": 1.0,
        "posterize_levels": 10,
        "quantize_passes": 1,
        "edge_strength": 0.48,
        "detail_boost": 0.18,
        "bilateral": 1,
        "color_temperature": 0.0,
        "clahe_clip": 1.2,
        "shadow_lift": 0.0,
        "skin_protect": 0.10,
        "highlight_protect": 0.08,
        "face_smooth": 0.08,
    },
}


@dataclass
class StyleResult:
    """风格化结果。

    Attributes:
        style_name: 最终采用的内部风格名。
        image_rgb: 风格化后的 RGB 图像数组，类型为 ``np.uint8``。
    """

    style_name: str
    image_rgb: np.ndarray


class Photo2StylePipeline:
    """照片转风格图像的推理入口。

    该实现以 OpenCV 为主，MindSpore 仅用于演示张量后处理链路。

    Args:
        use_mindspore: 是否尝试启用 MindSpore 后端。
        device_target: MindSpore 运行设备，例如 ``CPU``、``GPU`` 或 ``Ascend``。
    """

    def __init__(self, use_mindspore: bool = True, device_target: str = "CPU") -> None:
        self.backend = "numpy"
        self.ms, self.Tensor, self.ops = _load_mindspore() if use_mindspore else (None, None, None)
        if self.ms is not None:
            self.ms.set_context(mode=self.ms.PYNATIVE_MODE, device_target=device_target)
            self.backend = "mindspore"

    def stylize(self, image_bgr: np.ndarray, style_name: str = "ghibli") -> StyleResult:
        """对单张 BGR 图像执行风格化。

        Args:
            image_bgr: OpenCV 读取后的 BGR 图像数组。
            style_name: 内部风格名，支持 ``ghibli``、``disney``、``ink``、``sketch``。

        Returns:
            ``StyleResult``，其中结果图为 RGB 排列。
        """
        if style_name not in STYLE_PRESETS:
            raise ValueError(f"Unknown style: {style_name}. Available: {list(STYLE_PRESETS)}")

        cfg = STYLE_PRESETS[style_name]
        original_size = (image_bgr.shape[1], image_bgr.shape[0])
        base = self._normalize_size(image_bgr)
        base = self._pre_enhance(base, cfg)
        smoothed = self._edge_preserving_smooth(base, cfg)
        quantized = self._quantize_colors(
            smoothed,
            levels=int(cfg["posterize_levels"]),
            passes=int(cfg["quantize_passes"]),
        )
        edges = self._extract_edges(base, edge_strength=float(cfg["edge_strength"]))
        blended = self._blend_edges(quantized, edges)
        rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
        rgb = self._tone_adjust(rgb, cfg)
        rgb = self._restore_local_detail(rgb, base, detail_boost=float(cfg["detail_boost"]))
        rgb = self._protect_portrait_regions(rgb, base, cfg)
        if (rgb.shape[1], rgb.shape[0]) != original_size:
            rgb = cv2.resize(rgb, original_size, interpolation=cv2.INTER_CUBIC)

        if style_name == "sketch":
            rgb = self._stylize_sketch(rgb)

        return StyleResult(style_name=style_name, image_rgb=rgb)

    def stylize_path(self, image_path: str | Path, style_name: str = "ghibli") -> StyleResult:
        """从图片路径读取文件并执行风格化。"""
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        return self.stylize(image, style_name)

    def _tone_adjust(self, image_rgb: np.ndarray, cfg: Dict[str, float]) -> np.ndarray:
        """执行饱和度、对比度和 gamma 调整。"""
        x = image_rgb.astype(np.float32) / 255.0
        gray = x.mean(axis=2, keepdims=True)
        x = gray + cfg["saturation"] * (x - gray)
        x = self._apply_color_temperature(x, float(cfg["color_temperature"]))
        x = np.clip(x + float(cfg["shadow_lift"]), 0.0, 1.0)
        x = np.clip((x - 0.5) * cfg["contrast"] + 0.5, 0.0, 1.0)
        x = np.power(np.clip(x, 0.0, 1.0), cfg["gamma"])

        if self.backend == "mindspore":
            tx = self.Tensor(x, self.ms.float32)
            tx = self.ops.clip_by_value(tx, self.Tensor(0.0, self.ms.float32), self.Tensor(1.0, self.ms.float32))
            x = tx.asnumpy()

        return (x * 255.0).astype(np.uint8)

    def _normalize_size(self, image_bgr: np.ndarray, max_side: int = 1280) -> np.ndarray:
        """限制超大输入尺寸，避免风格化时细节被过度磨平。"""
        height, width = image_bgr.shape[:2]
        longest = max(height, width)
        if longest <= max_side:
            return image_bgr

        scale = max_side / float(longest)
        new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        return cv2.resize(image_bgr, new_size, interpolation=cv2.INTER_AREA)

    def _pre_enhance(self, image_bgr: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
        """做轻量曝光和色彩校正，减少原图灰蒙或偏色带来的质量损失。"""
        balanced = self._gray_world_balance(image_bgr)
        lab = cv2.cvtColor(balanced, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=float(cfg["clahe_clip"]),
            tileGridSize=(8, 8),
        )
        l_channel = clahe.apply(l_channel)
        enhanced = cv2.merge((l_channel, a_channel, b_channel))
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    def _edge_preserving_smooth(self, image_bgr: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
        """使用多轮保边滤波和平滑，减少噪点同时保留主体轮廓。"""
        smoothed = image_bgr.copy()
        for _ in range(int(cfg["bilateral"])):
            smoothed = cv2.bilateralFilter(smoothed, d=9, sigmaColor=85, sigmaSpace=85)

        smoothed = cv2.edgePreservingFilter(smoothed, flags=1, sigma_s=60, sigma_r=0.35)
        return smoothed

    def _quantize_colors(self, image_bgr: np.ndarray, levels: int, passes: int) -> np.ndarray:
        """结合颜色聚类和分层压缩，得到更干净的插画色块。"""
        levels = max(2, levels)
        clustered = image_bgr
        for _ in range(max(1, passes)):
            clustered = self._kmeans_quantize(clustered, clusters=max(6, min(levels, 24)))

        step = 256.0 / levels
        posterized = np.floor(clustered.astype(np.float32) / step) * step + step * 0.5
        return np.clip(posterized, 0, 255).astype(np.uint8)

    def _kmeans_quantize(self, image_bgr: np.ndarray, clusters: int) -> np.ndarray:
        """使用 OpenCV k-means 压缩主色，减少颜色噪声。"""
        data = image_bgr.reshape((-1, 3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 12, 1.0)
        _compactness, labels, centers = cv2.kmeans(
            data,
            clusters,
            None,
            criteria,
            2,
            cv2.KMEANS_PP_CENTERS,
        )
        quantized = centers[labels.flatten()].reshape(image_bgr.shape)
        return np.clip(quantized, 0, 255).astype(np.uint8)

    def _extract_edges(self, image_bgr: np.ndarray, edge_strength: float) -> np.ndarray:
        """生成柔和的轮廓图，用于和色块结果融合。"""
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Laplacian(blurred, cv2.CV_32F, ksize=3)
        edges = cv2.convertScaleAbs(edges)
        edges = cv2.GaussianBlur(edges, (3, 3), 0)
        normalized = edges.astype(np.float32) / 255.0
        mask = 1.0 - np.clip(normalized * (1.4 + edge_strength), 0.0, 0.92)
        return (mask * 255.0).astype(np.uint8)

    def _blend_edges(self, image_bgr: np.ndarray, edge_mask: np.ndarray) -> np.ndarray:
        """将轮廓图压到色块结果上，得到更自然的卡通边线。"""
        mask = edge_mask.astype(np.float32) / 255.0
        blended = image_bgr.astype(np.float32) * mask[..., None]
        return np.clip(blended, 0, 255).astype(np.uint8)

    def _restore_local_detail(
        self,
        image_rgb: np.ndarray,
        source_bgr: np.ndarray,
        detail_boost: float,
    ) -> np.ndarray:
        """轻微回灌局部明暗细节，避免整体过糊。"""
        source_rgb = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB)
        base = image_rgb.astype(np.float32)
        base_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        source_lab = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        smooth_l = cv2.GaussianBlur(source_lab[..., 0], (0, 0), sigmaX=1.2)
        l_detail = source_lab[..., 0] - smooth_l
        base_lab[..., 0] = np.clip(base_lab[..., 0] + l_detail * (detail_boost * 0.85), 0.0, 255.0)
        restored = cv2.cvtColor(base_lab.astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)
        fine = cv2.GaussianBlur(source_rgb, (0, 0), sigmaX=1.0)
        chroma_detail = source_rgb.astype(np.float32) - fine.astype(np.float32)
        restored = restored + chroma_detail * (detail_boost * 0.25)
        return np.clip(restored, 0, 255).astype(np.uint8)

    def _stylize_sketch(self, image_rgb: np.ndarray) -> np.ndarray:
        """将彩色结果进一步转为更稳定的铅笔素描效果。"""
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        inverted = 255 - gray
        blurred = cv2.GaussianBlur(inverted, (0, 0), sigmaX=7)
        sketch = cv2.divide(gray, 255 - blurred, scale=256.0)
        sketch = cv2.equalizeHist(sketch)
        return cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)

    def _apply_color_temperature(self, image_rgb: np.ndarray, temperature_shift: float) -> np.ndarray:
        """对暖冷色做轻微偏移，让不同风格的色彩倾向更明显。"""
        if temperature_shift == 0.0:
            return image_rgb

        adjusted = image_rgb.copy()
        shift = temperature_shift / 255.0
        adjusted[..., 0] = np.clip(adjusted[..., 0] + shift, 0.0, 1.0)
        adjusted[..., 2] = np.clip(adjusted[..., 2] - shift, 0.0, 1.0)
        return adjusted

    def _gray_world_balance(self, image_bgr: np.ndarray) -> np.ndarray:
        """用灰世界假设做轻量白平衡，缓解偏黄或偏蓝输入。"""
        image = image_bgr.astype(np.float32)
        channel_means = image.mean(axis=(0, 1))
        mean_gray = float(channel_means.mean())
        scale = mean_gray / np.maximum(channel_means, 1.0)
        balanced = image * scale.reshape((1, 1, 3))
        return np.clip(balanced, 0, 255).astype(np.uint8)

    def _protect_portrait_regions(
        self,
        image_rgb: np.ndarray,
        source_bgr: np.ndarray,
        cfg: Dict[str, Any],
    ) -> np.ndarray:
        """对肤色、高光和人脸区域做保护，减少蜡像感和五官糊化。"""
        source_rgb = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB)
        skin_mask = self._build_skin_mask(source_bgr)
        highlight_mask = self._build_highlight_mask(source_bgr)
        face_mask = self._build_face_mask(source_bgr)

        result = image_rgb.astype(np.float32)
        source = source_rgb.astype(np.float32)

        if skin_mask is not None:
            result = self._blend_by_mask(
                result,
                source,
                skin_mask,
                strength=float(cfg["skin_protect"]),
            )

        if highlight_mask is not None:
            result = self._blend_by_mask(
                result,
                source,
                highlight_mask,
                strength=float(cfg["highlight_protect"]),
            )

        if face_mask is not None:
            softened = cv2.GaussianBlur(result.astype(np.uint8), (0, 0), sigmaX=1.1).astype(np.float32)
            smooth_strength = float(cfg["face_smooth"])
            result = self._blend_by_mask(
                result,
                softened,
                face_mask,
                strength=smooth_strength,
            )
            result = self._blend_by_mask(
                result,
                source,
                face_mask,
                strength=smooth_strength * 0.65,
            )

        return np.clip(result, 0, 255).astype(np.uint8)

    def _build_skin_mask(self, image_bgr: np.ndarray) -> np.ndarray:
        """在 YCrCb 空间粗略估计肤色区域。"""
        ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
        lower = np.array([0, 133, 77], dtype=np.uint8)
        upper = np.array([255, 173, 127], dtype=np.uint8)
        mask = cv2.inRange(ycrcb, lower, upper)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=3)
        return mask.astype(np.float32) / 255.0

    def _build_highlight_mask(self, image_bgr: np.ndarray) -> np.ndarray:
        """保护高光区域，避免额头、鼻梁等位置过度压平。"""
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        value = hsv[..., 2].astype(np.float32) / 255.0
        saturation = hsv[..., 1].astype(np.float32) / 255.0
        mask = np.clip((value - 0.72) / 0.28, 0.0, 1.0) * np.clip(1.0 - saturation * 0.7, 0.25, 1.0)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=4)
        return mask

    def _build_face_mask(self, image_bgr: np.ndarray) -> np.ndarray | None:
        """使用 OpenCV 级联检测人脸，并构造软遮罩。"""
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        if not cascade_path.exists():
            return None

        classifier = cv2.CascadeClassifier(str(cascade_path))
        if classifier.empty():
            return None

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        faces = classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(36, 36))
        if len(faces) == 0:
            return None

        mask = np.zeros(gray.shape, dtype=np.float32)
        for x, y, w, h in faces:
            pad_w = int(w * 0.18)
            pad_h = int(h * 0.22)
            x0 = max(0, x - pad_w)
            y0 = max(0, y - pad_h)
            x1 = min(mask.shape[1], x + w + pad_w)
            y1 = min(mask.shape[0], y + h + pad_h)
            mask[y0:y1, x0:x1] = 1.0

        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=9)
        return np.clip(mask, 0.0, 1.0)

    def _blend_by_mask(
        self,
        base: np.ndarray,
        source: np.ndarray,
        mask: np.ndarray,
        strength: float,
    ) -> np.ndarray:
        """按软遮罩混合两幅图像。"""
        alpha = np.clip(mask * strength, 0.0, 1.0)[..., None]
        return base * (1.0 - alpha) + source * alpha


def parse_style_text(style_prompt: str) -> Tuple[str, str]:
    """将自然语言风格提示词解析为内部风格名。

    Args:
        style_prompt: 用户输入的中文或英文风格描述。

    Returns:
        一个二元组 ``(style_name, message)``：
        ``style_name`` 为内部风格名，
        ``message`` 为解析说明，便于日志展示。
    """
    prompt = style_prompt.strip().lower()
    token_source = "规则回退"
    tokens = []
    tokenizer = _load_mindnlp_tokenizer()
    if tokenizer is not None:
        try:
            raw_tokens = tokenizer(style_prompt.strip())
            tokens = [str(token).lower() for token in raw_tokens if str(token).strip()]
            token_source = "MindNLP 分词"
        except Exception:
            tokens = []

    if not tokens:
        tokens = [segment for segment in prompt.replace("/", " ").replace("、", " ").split() if segment]

    normalized_prompt = " ".join(tokens) if tokens else prompt
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
        if key in prompt or key in normalized_prompt:
            return value, f"{token_source}命中关键词: {key}"
    return "ghibli", f"{token_source}未命中关键词，默认使用 ghibli 风格"
