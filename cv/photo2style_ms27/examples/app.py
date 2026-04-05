from __future__ import annotations

import cv2
import gradio as gr
import numpy as np

from photo2style.pipeline import Photo2StylePipeline, parse_style_text

pipe = Photo2StylePipeline(use_mindspore=True)


def run_demo(image: np.ndarray, style_prompt: str):
    """Gradio 推理回调函数。"""
    if image is None:
        raise gr.Error("请先上传一张真人照片")

    style_name, msg = parse_style_text(style_prompt)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    result = pipe.stylize(bgr, style_name=style_name)
    return result.image_rgb, f"{msg} -> 使用风格: {style_name}"


with gr.Blocks(title="MindSpore 真人照片转卡通") as demo:
    gr.Markdown("# MindSpore 2.7.0 真人照片转卡通 DEMO")
    gr.Markdown("输入提示词示例：吉卜力、迪士尼、古风、水墨、钢笔")

    with gr.Row():
        input_image = gr.Image(type="numpy", label="上传真人照片")
        output_image = gr.Image(type="numpy", label="风格化结果")

    style_prompt = gr.Textbox(value="吉卜力风格", label="风格提示词")
    status = gr.Textbox(label="运行日志", interactive=False)
    run_btn = gr.Button("开始生成")

    run_btn.click(run_demo, inputs=[input_image, style_prompt], outputs=[output_image, status])

if __name__ == "__main__":
    demo.launch()
