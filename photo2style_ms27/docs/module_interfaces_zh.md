# 模块接口与使用说明

本文档面向中文读者，记录 `photo2style_ms27` 各模块的职责、公开接口和典型使用方式。

## 1. 包入口：`photo2style/__init__.py`

职责：

- 统一导出项目公开接口，供外部代码直接导入。

公开接口：

- `Photo2StylePipeline`
- `StyleResult`
- `parse_style_text`

推荐导入方式：

```python
from photo2style import Photo2StylePipeline, StyleResult, parse_style_text
```

## 2. 核心模块：`photo2style/pipeline.py`

职责：

- 提供照片风格化主流程。
- 维护风格预设参数。
- 支持中文风格提示词解析。
- 在可用时接入 MindSpore 张量后处理。

### 2.1 数据结构：`StyleResult`

用途：

- 封装风格化结果，作为 `stylize()` 和 `stylize_path()` 的返回值。

字段说明：

- `style_name: str`
  记录最终采用的内部风格名。
- `image_rgb: np.ndarray`
  记录风格化后的 RGB 图像，数据类型为 `np.uint8`。

使用示例：

```python
result = pipe.stylize(image_bgr, style_name="ghibli")
print(result.style_name)
print(result.image_rgb.shape)
```

### 2.2 类：`Photo2StylePipeline`

初始化方法：

```python
Photo2StylePipeline(use_mindspore: bool = True, device_target: str = "CPU")
```

参数说明：

- `use_mindspore`
  是否尝试启用 MindSpore。若本地未安装或导入失败，会自动退回 `numpy` 后端。
- `device_target`
  MindSpore 设备目标，可选值通常为 `CPU`、`GPU`、`Ascend`。

典型初始化：

```python
pipe = Photo2StylePipeline(use_mindspore=True, device_target="CPU")
```

#### 方法：`stylize`

接口：

```python
stylize(image_bgr: np.ndarray, style_name: str = "ghibli") -> StyleResult
```

输入说明：

- `image_bgr`
  使用 OpenCV 读取得到的 BGR 图像。
- `style_name`
  内部风格名，目前支持：
  `ghibli`、`disney`、`ink`、`sketch`

返回说明：

- 返回 `StyleResult`，其中 `image_rgb` 为 RGB 图像。

使用示例：

```python
import cv2
from photo2style import Photo2StylePipeline

pipe = Photo2StylePipeline(use_mindspore=False)
image_bgr = cv2.imread("input.jpg")
result = pipe.stylize(image_bgr, style_name="disney")
cv2.imwrite("output.jpg", cv2.cvtColor(result.image_rgb, cv2.COLOR_RGB2BGR))
```

#### 方法：`stylize_path`

接口：

```python
stylize_path(image_path: str | Path, style_name: str = "ghibli") -> StyleResult
```

输入说明：

- `image_path`
  输入图片路径。
- `style_name`
  内部风格名。

异常说明：

- 图片读取失败时抛出 `FileNotFoundError`。

使用示例：

```python
pipe = Photo2StylePipeline(use_mindspore=False)
result = pipe.stylize_path("input.jpg", style_name="ink")
```

### 2.3 函数：`parse_style_text`

接口：

```python
parse_style_text(style_prompt: str) -> tuple[str, str]
```

用途：

- 将中文或英文提示词解析为内部风格名，并返回解析日志。

当前支持的关键词：

- `吉卜力` / `ghibli` -> `ghibli`
- `迪士尼` / `disney` -> `disney`
- `古风` / `水墨` -> `ink`
- `钢笔` / `素描` / `sketch` -> `sketch`

返回值说明：

- 第一个值：内部风格名
- 第二个值：解析说明消息

使用示例：

```python
style_name, message = parse_style_text("请转成吉卜力风格")
print(style_name)  # ghibli
print(message)
```

## 3. 命令行脚本：`examples/run_once.py`

职责：

- 提供单张图片推理的命令行入口。

运行方式：

```bash
python examples/run_once.py --input input.jpg --output outputs/out.jpg --style 吉卜力
```

参数说明：

- `--input`
  输入照片路径，必填。
- `--output`
  输出图片路径，必填。
- `--style`
  中文或英文风格提示词，默认值为 `吉卜力`。
- `--device`
  MindSpore 设备目标，默认值为 `CPU`。

适用场景：

- 本地快速验证效果。
- 批量处理脚本开发前的单图调试。

## 4. Web 示例：`examples/app.py`

职责：

- 提供基于 Gradio 的交互式演示页面。

启动方式：

```bash
python examples/app.py
```

页面使用方法：

1. 上传一张真人照片。
2. 输入风格提示词，例如 `吉卜力风格`。
3. 点击“开始生成”。
4. 查看风格化图片和日志文本。

内部接口：

- `run_demo(image: np.ndarray, style_prompt: str)`
  接收 Gradio 输入，返回风格化后的 RGB 图像和状态文本。

## 5. 测试模块：`tests/test_pipeline_smoke.py`

职责：

- 提供最小冒烟测试，验证风格化主链路能正常返回图像结果。

运行方式：

```bash
pytest -q tests/test_pipeline_smoke.py
```

测试覆盖点：

- 输出图像尺寸与输入一致。
- 输出图像数据类型为 `np.uint8`。

## 6. 依赖说明：`requirements.txt`

核心依赖：

- `mindspore==2.7.0`
- `mindnlp==0.5.1`
- `numpy>=1.24`
- `opencv-python>=4.8`
- `gradio>=4.0`
- `pytest>=8.0`

安装方式：

```bash
pip install -r requirements.txt
```

## 7. 推荐调用顺序

如果你是模块使用者，推荐按以下顺序接入：

1. 使用 `parse_style_text()` 解析用户输入。
2. 创建 `Photo2StylePipeline` 实例。
3. 调用 `stylize()` 或 `stylize_path()` 获取 `StyleResult`。
4. 若需保存结果，用 OpenCV 将 `RGB` 转回 `BGR` 后写盘。

最小示例：

```python
import cv2
from photo2style import Photo2StylePipeline, parse_style_text

style_name, _ = parse_style_text("迪士尼风格")
pipe = Photo2StylePipeline(use_mindspore=False)
result = pipe.stylize_path("input.jpg", style_name=style_name)
cv2.imwrite("output.jpg", cv2.cvtColor(result.image_rgb, cv2.COLOR_RGB2BGR))
```
