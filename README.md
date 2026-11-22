# 复原力：一株会重新长成的植物（NCA）

艺术与科学课程小组作品。基于 Distill《Growing Neural Cellular Automata》(Mordvintsev et al., 2020) 的方法，训练一个约 1 万参数的神经细胞自动机：从一颗种子长成一株植物，被擦除后尝试重新长回。

> 2026-11-22：团队决定放弃此前的《星空》方向（复杂油画超出小模型能力，诚实记录见本地归档 `archive_starry/`），转向分工文档中的植物作品。当前处于清理＋计划阶段，实现即将开始。

- 详细实施计划：`docs/PLAN_plant.md`
- 核心代码：`src/nca4plot/`（model / losses / rollout / utils / export_onnx；train_plant 与 make_plant_target 即将新增）
- 网页引擎：`web/js/nca.js`（ONNX Runtime Web，WebGPU→WASM 回退）；极简 UI 即将重写
- 本地运行（实现后）：`.venv\Scripts\python.exe scripts\serve_web.py` → http://127.0.0.1:8765
- 参考与许可：`third_party/`、`LICENSES.md`
