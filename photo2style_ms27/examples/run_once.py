from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from photo2style.pipeline import Photo2StylePipeline, parse_style_text


def main():
    parser = argparse.ArgumentParser(description="Photo2Style single-image runner")
    parser.add_argument("--input", required=True, help="Input photo path")
    parser.add_argument("--output", required=True, help="Output image path")
    parser.add_argument("--style", default="吉卜力", help="Style prompt, e.g. 吉卜力/迪士尼/古风/钢笔")
    parser.add_argument("--device", default="CPU", help="MindSpore device target")
    args = parser.parse_args()

    style_name, msg = parse_style_text(args.style)
    print(msg)

    pipe = Photo2StylePipeline(use_mindspore=True, device_target=args.device)
    result = pipe.stylize_path(args.input, style_name=style_name)

    bgr = cv2.cvtColor(result.image_rgb, cv2.COLOR_RGB2BGR)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), bgr)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

