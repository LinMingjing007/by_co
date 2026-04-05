from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from photo2style.pipeline import Photo2StylePipeline, parse_style_text


def main() -> None:
    """命令行单图推理入口。"""
    parser = argparse.ArgumentParser(description="Photo2Style 单张图片推理脚本")
    parser.add_argument("--input", required=True, help="输入照片路径")
    parser.add_argument("--output", required=True, help="输出图片路径")
    parser.add_argument("--style", default="吉卜力", help="风格提示词，例如：吉卜力/迪士尼/古风/钢笔")
    parser.add_argument("--device", default="CPU", help="MindSpore 设备目标，例如 CPU/GPU/Ascend")
    args = parser.parse_args()

    style_name, msg = parse_style_text(args.style)
    print(msg)

    pipe = Photo2StylePipeline(use_mindspore=True, device_target=args.device)
    result = pipe.stylize_path(args.input, style_name=style_name)

    bgr = cv2.cvtColor(result.image_rgb, cv2.COLOR_RGB2BGR)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), bgr)
    print(f"输出已保存: {out_path}")


if __name__ == "__main__":
    main()
