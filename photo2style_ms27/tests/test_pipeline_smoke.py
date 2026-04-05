import numpy as np

from photo2style.pipeline import Photo2StylePipeline


def test_pipeline_smoke():
    """验证最小推理链路能返回合法的 RGB 结果。"""
    pipe = Photo2StylePipeline(use_mindspore=False)
    fake = np.zeros((128, 128, 3), dtype=np.uint8)
    fake[:, :, 0] = 220
    result = pipe.stylize(fake, style_name="ghibli")
    assert result.image_rgb.shape == fake.shape
    assert result.image_rgb.dtype == np.uint8
