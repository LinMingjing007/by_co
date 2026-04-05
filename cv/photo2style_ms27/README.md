# 真人照片转卡通图像（MindSpore 2.7.0 + MindNLP 0.5.1）

本案例位于 `cv/photo2style_ms27`，用于演示如何基于 MindSpore 2.7.0 和 MindNLP 0.5.1 实现“真人照片 -> 指定风格图像”的推理应用。

该版本已按 `mindspore-lab/applications` 的常见合入要求整理：

- 置于 `cv/` 视觉领域目录下
- 主 Notebook 使用推理类命名：`inference_photo2style_cartoon.ipynb`
- Notebook 内容为中文且不保留执行输出
- 同步提供 README、示例脚本、测试脚本和中文接口说明

面向中文读者的模块接口与使用说明见：`docs/module_interfaces_zh.md`

## 目录结构

- `notebooks/inference_photo2style_cartoon.ipynb`：主提交 Notebook
- `photo2style/pipeline.py`：风格化推理管线
- `examples/app.py`：可交互 DEMO（Gradio）
- `examples/run_once.py`：单张图片推理脚本
- `docs/checklist.md`：合入前自检项
- `docs/pr_template_zh.md`：PR 文案模板
- `docs/module_interfaces_zh.md`：模块接口与使用说明
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

## MindNLP 接入说明

本项目将 MindNLP 用于风格提示词解析链路：

- 优先尝试使用 MindNLP 分词器对中文/英文风格提示词做标准化切分
- 再将解析结果映射到内部风格标签 `ghibli`、`disney`、`ink`、`sketch`
- 当运行环境缺少可用的 MindNLP 分词组件时，自动回退到本地规则解析，保证推理脚本可运行

## DEMO 链接（魔乐社区）

- 待部署完成后回填真实地址：`https://modelers.cn/spaces/<你的空间名>/photo2style-ms27`

## 合入说明

目标仓库与分支：

- 仓库：`https://github.com/mindspore-lab/applications`
- 分支：`dev`
- 建议路径：`cv/photo2style_ms27`

提交前请同步更新域目录索引 README，并补充效果图与真实 DEMO 链接。
