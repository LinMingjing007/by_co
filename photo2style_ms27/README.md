# 真人照片转卡通图像（MindSpore 2.7.0 + MindNLP 0.5.1）

本案例演示如何基于 MindSpore 和 MindNLP 构建一个“真人照片 -> 指定风格图像”的最小可运行应用。

## 目录结构

- `notebooks/photo2style_zh.ipynb`：中文教学版 Notebook（提交主文件）
- `photo2style/pipeline.py`：风格化推理管线
- `examples/app.py`：可交互 DEMO（Gradio）
- `examples/run_once.py`：单张图片推理脚本
- `docs/checklist.md`：合入前自检项
- `docs/pr_template_zh.md`：PR 文案模板
- `requirements.txt`：依赖清单

## 环境要求

- Python >= 3.9, < 3.12
- MindSpore == 2.7.0
- MindNLP == 0.5.1
- CANN >= 8.1.RC1（推荐 8.3.RC1）

## 快速开始

```bash
pip install -r requirements.txt
python examples/run_once.py --input input.jpg --output outputs/out.jpg --style 吉卜力
python examples/app.py
pytest -q tests/test_pipeline_smoke.py
```

## DEMO 链接（魔乐社区）

- 待上传后填写：`https://modelers.cn/spaces/<你的空间名>/photo2style-ms27`

## 合入说明

建议提交到 `mindspore-lab/applications` 的 `dev` 分支，目录名可使用：

- `applications/photo2style_ms27`

并同步更新对应路径 `README` 与效果图。
