# 复原力：一株会重新长成的植物（NCA）

艺术与科学课程小组作品。基于 Distill《Growing Neural Cellular Automata》(Mordvintsev et al., 2020) 的**官方训练配方**，训练一个 **8,336 参数**的神经细胞自动机：从一粒种子长成一株植物，擦掉一部分后自己长回来。

- 网页 demo（极简）：本地 `.venv\Scripts\python.exe scripts\serve_web.py` → http://127.0.0.1:8765 ；公网 https://wangcheng726.github.io/nca4plot/
- 制作记录（含星空方向的诚实失败记录）：`docs/制作记录.md`
- 实施计划：`docs/PLAN_plant.md`
- 核心代码：`src/nca4plot/`（model / train_plant / probe_plant / export_plant / make_plant_target / losses / rollout / utils / export_onnx）
- 网页引擎：`web/js/nca.js`（ONNX Runtime Web，WebGPU→WASM 回退）＋ `web/js/app.js`（极简 UI）
- 测试：`.venv\Scripts\python.exe -m pytest tests -q`
- 参考与许可：`third_party/`、`LICENSES.md`

> 2026-11-22：团队放弃《星空》方向（复杂油画超出小模型能力，失败证据归档于本地 `archive_starry/`），转向分工文档中的植物作品。**已完成**：M0 官方基线复现；植物 16000 步训练（含 rollout 窗口几何修正）；三损伤测试全部通过（2048 步稳态 MSE 7.3e-05；小损伤 32 步、大损伤 256 步恢复）；网页 headless 验收通过（鼠标误差 0.34 格、无频闪）；21 项测试全过。本地 demo：`scripts/serve_web.py` → http://127.0.0.1:8765 。公网 demo：https://wangcheng726.github.io/nca4plot/ （2026-11-22 经同意部署）。
