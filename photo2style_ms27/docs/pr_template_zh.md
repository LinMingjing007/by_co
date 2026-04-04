# PR 标题建议

`feat(applications): add photo2style_ms27 demo with MindSpore 2.7.0 and MindNLP 0.5.1`

# 任务说明

- 目标：实现真人照片转特定风格图像（吉卜力/迪士尼/古风/钢笔）
- 版本：MindSpore 2.7.0，MindNLP 0.5.1
- 目录：`applications/photo2style_ms27`

# 变更内容

- 新增中文 Notebook：`notebooks/photo2style_zh.ipynb`
- 新增推理管线：`photo2style/pipeline.py`
- 新增交互 DEMO：`examples/app.py`
- 新增单图脚本：`examples/run_once.py`
- 新增 smoke test：`tests/test_pipeline_smoke.py`

# 运行与测试

- 环境安装：`pip install -r requirements.txt`
- 单图推理：`python examples/run_once.py --input input.jpg --output outputs/out.jpg --style 吉卜力`
- DEMO 启动：`python examples/app.py`

# DEMO 链接

- Modelers：`https://modelers.cn/spaces/<你的空间名>/photo2style-ms27`

# 风险与限制

- 当前为轻量风格化实现，复杂场景质量受输入照片质量影响
- 生产级效果可进一步替换为训练好的 MindSpore 生成模型

