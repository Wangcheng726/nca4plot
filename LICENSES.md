# 许可与第三方说明

## 本项目原创部分

- **源代码**：MIT License，版权归 Wang Cheng，完整文本见根目录 [`LICENSE`](LICENSE)。
- **训练权重**：Creative Commons Attribution 4.0 International（CC BY 4.0）。转载、改编或再训练时请注明作者 Wang Cheng、作品名 *永不落定的星空 / A Sky That Never Settles* 并链接到本项目。
- **作品说明、海报、过程图与展示视频**：CC BY 4.0，除非另有说明。

## 源艺术作品

Vincent van Gogh, *The Starry Night* (1889)。

- 来源：Wikimedia Commons / Google Art Project 公有领域扫描。
- 许可：Public domain（二维作品）。
- 具体来源 URL、扫描尺寸与 SHA256 见 [`data/exemplar_metadata.json`](data/exemplar_metadata.json)。
- 本作只使用避开月亮、柏树、村庄与山丘的纯天空局部作为训练 exemplar；网页不加载原画作为回退目标。

## 上游技术代码

训练路线来自 Google Research 的 `self-organising-systems` 与 Distill 文章 *Self-Organising Textures*。

- 上游仓库：<https://github.com/google-research/self-organising-systems>
- 上游许可：**Apache License 2.0**
- 本项目中的 PyTorch 模型、损失与训练循环是对公开 notebook 方法的最小化重写与工程化封装。对来自或改编自上游的部分，Apache-2.0 条件继续适用。
- Apache-2.0 完整文本：[`third_party/self-organising-systems-APACHE-2.0.txt`](third_party/self-organising-systems-APACHE-2.0.txt)

## ONNX Runtime Web

`web/assets/ort/` 内为 ONNX Runtime Web v1.31.1 的浏览器构建。

- 版权：Microsoft Corporation
- 许可：MIT License
- 项目：<https://github.com/microsoft/onnxruntime>
- 完整文本：[`third_party/onnxruntime-web-MIT.txt`](third_party/onnxruntime-web-MIT.txt)

## Python 依赖

Python 依赖及其版本记录在 `pyproject.toml` 与 `uv.lock` 中，未在源码树内重复分发。各依赖按其自身许可证使用，包括但不限于 PyTorch、TorchVision、ONNX、ONNX Runtime、NumPy、Pillow、Matplotlib、pytest、Playwright 与 imageio。

## 说明

本项目为学生的艺术与科学课程作品，与 Google Research、Microsoft、Wikimedia Commons 或 Van Gogh Museum / MoMA 无隶属、赞助或背书关系。引用的论文与作品仅用于学术和艺术语境说明。