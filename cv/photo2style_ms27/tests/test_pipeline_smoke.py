import cv2
import numpy as np

from photo2style.pipeline import Photo2StylePipeline, parse_style_text


def test_pipeline_smoke():
    """验证各风格最小推理链路能返回合法的 RGB 结果。"""
    pipe = Photo2StylePipeline(use_mindspore=False)
    gradient_x = np.tile(np.linspace(20, 220, 128, dtype=np.uint8), (128, 1))
    gradient_y = np.tile(np.linspace(40, 200, 128, dtype=np.uint8).reshape(128, 1), (1, 128))
    fake = np.stack((gradient_x, gradient_y, np.full((128, 128), 90, dtype=np.uint8)), axis=-1)

    for style_name in ("ghibli", "disney", "ink", "sketch"):
        result = pipe.stylize(fake, style_name=style_name)
        assert result.image_rgb.shape == fake.shape
        assert result.image_rgb.dtype == np.uint8
        assert not np.array_equal(result.image_rgb, fake[:, :, ::-1])


def test_pipeline_handles_portrait_like_input():
    """验证人像优先分支在类人脸输入下也能稳定输出。"""
    pipe = Photo2StylePipeline(use_mindspore=False)
    fake = np.full((160, 160, 3), 235, dtype=np.uint8)
    cv2.circle(fake, (80, 82), 42, (170, 185, 215), -1)
    cv2.circle(fake, (64, 74), 5, (60, 70, 90), -1)
    cv2.circle(fake, (96, 74), 5, (60, 70, 90), -1)
    cv2.ellipse(fake, (80, 98), (14, 8), 0, 0, 180, (80, 90, 120), 2)

    result = pipe.stylize(fake, style_name="ghibli")
    assert result.image_rgb.shape == fake.shape
    assert result.image_rgb.dtype == np.uint8


def test_parse_style_text():
    """验证中文提示词能稳定映射到内部风格名。"""
    style_name, message = parse_style_text("请生成迪士尼公主风")
    assert style_name == "disney"
    assert "迪士尼" in message
