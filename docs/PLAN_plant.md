# 《复原力：一株会重新长成的植物》实施计划

v1 · 2026-11-22 · Pre 截止 11/26（周四）· 依据小组分工文档

## 0. 决策与教训

- **放弃星空**：512×288 多色油画远超 ~1 万参数 NCA 的记忆/再生能力（v3–v11 均未达标）。诚实记录已归档 `archive_starry/`（模型、训练日志、对比图、三模式录屏），可直接用于 Pre 的"制作记录"与失败讨论环节。
- **新方向**（分工文档硬要求）：小组原创植物目标图，轮廓清楚、颜色少；**必须同时具备"从单种子点生长"与"擦除后再生"**；三种损伤测试（①不擦 ②擦小块 ③擦大块）；失败也要诚实记录，不得把预期说成已实现。
- **核心教训**：瓶颈不是参数量，而是任务复杂度。官方 emoji（~40px、平色、少色）才是该方法的舒适区 → 植物目标按舒适区设计，**不盲目加大参数**。

## 1. 目标图（原创，程序化绘制）

- 新增 `src/nca4plot/make_plant_target.py`：用 PIL 矢量形状（茎=粗曲线、叶=椭圆/水滴、花=圆瓣）绘制后栅格化。保证原创、平色、边缘锐利、色板完全可控、可复现（不用 AI 生成图，避免渐变与杂色）。
- 默认规格（候选 A）：
  - 网格 **80×80**，RGBA（alpha 为第 4 通道，背景透明）；
  - 色板 **≤5 色**：茎深绿 #2F7D32、叶绿 #66BB6A、花瓣粉 #F48FB1、花心黄 #FBC02D（可选第二叶色）；
  - 形态：种子位于茎基部（约 x=40, y=62），茎微弯向上，4–6 片互生叶，顶端一朵 5 瓣花；植株占网格中央 ~70%，四周留 ≥8 格边距；
  - 无花盆、无渐变、无描边、无纹理。
- 同时产出候选 B（黄花/圆叶）、C（3 色极简芽苗）供全组选择；另备 **64×64 简化版**作为 80px 不收敛时的退路。
- 交付：`data/plant_target_80.png` + 放大预览图（幻灯片用）。

## 2. 模型（复用现有 CA，不打参数军备竞赛）

- `CA(chn=16, hidden_n=128, kernel_scale=1.0)`：RGBA＋12 隐藏通道；感知 = identity＋sobel_x＋sobel_y＋laplacian（现有实现，4 核×16=64 通道）；更新 = 64→128(ReLU)→16（零初始化）；alive 掩码（alpha>0.1）；update_rate=0.5 异步更新。
- 参数量 ≈ **10,368**，与星空 v10 同量级，但任务简单两个数量级。
- 与官方 growing 配方差异：官方 3 核（无 laplacian）48→128→16=8,320 参数。多一个 laplacian 属无害增强；若 M0 复现有出入，给 model.py 加开关退回 3 核。
- 单一模型，不做多档画质、不做大网格。

## 3. 训练（新写 train_plant.py，双阶段课程）

新写 ~300 行训练器（不再修补 64KB 的星空 train_target.py；其损伤课程仅作参考，已归档）。复用 model.py / rollout.py / utils.py。

### M0 官方基线复现（验证管线，~10 分钟）

- 先按官方 recipe 训练官方 emoji（40px）：pool=1024，batch=8，step_n∈[64,96]，Adam lr=2e-3（2000 步后减半），逐样本 RGBA MSE，每步把 batch 中 loss 最高的样本替换为种子，~2000–4000 iter。
- 种子约定按官方：单细胞 channels[3:]=1（alpha＋隐藏），RGB=0。植物种子放在茎基部（NCA 规则平移不变，图案相对种子锚定，绝对位置任意）。
- 验收：emoji 从单种子长出、擦除后稳定 → 证明 seed/padding/loss/pool 管线正确。同时满足分工文档"先原样运行原始示例"（本地等价复现；队内可另开官方 Colab 供演示）。

### Phase G 生长（植物，~20–40 分钟）

- 同 recipe，目标=植物 80px；2000–4000 iter；每 250 步存 checkpoint＋生长接触表。
- 门槛：固定种子 96 步内肉眼完整，终态 RGBA MSE ≤ 1e-3，再跑 512 步稳定不崩。
- 不达标：先简化目标（64px/更少叶/3 色），再调 lr/步数范围；**不加参数**。

### Phase R 再生课程（~20–40 分钟）

- 在 G 基础上续训 1000–3000 iter，混合三类 batch：
  - ~40% 纯生长（防遗忘，保留种子替换机制）；
  - ~40% 小损伤：对 pool 样本随机擦除圆盘 r∈[4,10] 或宽 3–5 的笔画（落在叶/茎上）；
  - ~20% 大损伤：圆盘 r∈[12,18]（较少，诚实记录恢复上限）。
  - 损伤后 rollout 64–96 步对目标算 MSE；写回 pool。
- 门槛：小损伤 ≤128 步恢复到 MSE ≤ 2e-3，肉眼无条纹/噪点/错位；大损伤如实记录（分工文档允许失败）。
- 长程探针：2048 步无 NaN、无漂移（persistent 行为是训练的自然副产品，不单独做模式）。

### 止损规则（与用户约定）

一轮完整训练后仍不达标 → 停下来汇报并重新评估目标/配方，不无限调参。GPU 总预算 ≤4 小时（分两天）；训练期间不并发其它 GPU 任务（含 headless WebGPU 验证）。

## 4. 导出与极简网页

- 导出：复用 export_onnx.py（CPU 导出，architecture＋kernel_scale 元数据），单模型 `web/models/plant_80.onnx`＋meta＋seed.png；PyTorch↔ORT 数值 parity ≤ 1e-3。
- **UI 重写为极度精简**（新 index.html / styles.css / js/app.js，合计 <400 行）：
  - 页面：一行标题＋居中画布（80×80 整数放大 ×6=480px，最近邻，平色像素干净利落）＋一行按钮＋一行说明。系统字体、纯色背景，无 hero/渐变/卡片/装饰动画。
  - 按钮：[重新生长] [暂停/继续] [笔刷 −/＋] [损伤测试：小块 | 大块]＋步数计数。损伤测试按钮=在固定位置（叶/茎）脚本化擦除，与分工文档三种测试一一对应，便于现场演示与录屏。
  - 行为：打开页面自动从种子生长；按住拖动擦除（letterbox 坐标映射）；CA 经 rAF 持续运行。**擦除即再生，无需模式切换**——官方 growing/regenerate/persistent 三种体验由"持续运行＋重新生长＋擦除"自然覆盖。
  - 渲染：**Canvas2D putImageData＋drawImage 放大**（替代 WebGL impasto 管线）。80×80 下开销可忽略，且从根上消除 WebGL 频闪/纹理上传一类 bug；nca.js（ORT WebGPU→WASM 回退引擎）原样复用。
  - 键盘：Space 暂停、R 重置；支持触摸。
- 本地：`scripts/serve_web.py` → http://127.0.0.1:8765。**公网不动**（Pages 仍是星空 v9），push 需另行明确同意。

## 5. 验收（映射分工文档交付物）

| 项 | 方法 | 通过标准 |
|---|---|---|
| 生长 | 固定种子探针，8/16/32/64/96/128 步帧 | 96 步内肉眼完整；MSE ≤ 1e-3 |
| ① 不擦 | 连续跑 2048 步 | 稳定、无 NaN、无漂移 |
| ② 擦小块 | 脚本＋手动，叶/茎 | ≤128 步恢复，肉眼无伪影 |
| ③ 擦大块 | 脚本＋手动 | 如实记录（部分恢复/失败均可，必须诚实） |
| 无频闪 | headless 帧差测试 | 连续帧无异常整帧跳变 |
| 指针对齐 | headless 点击已知像素 | 擦除细胞坐标一致 |
| 数值 parity | PyTorch vs ORT | ≤ 1e-3 |

- tests/ 重写为植物版（现有 5 个测试是星空专用）：test_model / test_rollout / test_onnx_export 改造，test_target / test_web 重写。
- 媒体交付：生长录屏、三种损伤测试录屏＋接触表、目标图预览、一页制作记录（docs/制作记录.md，含星空失败段落）、README。
- 验证工具：改造 scripts/probe_user_regen.py、verify_user_web.py、capture_web.py。

## 6. 时间表（今天 11/22 → 11/26 Pre）

- **11/22（今天）**：清理 ✅ → plan 确认 → M0 emoji 基线＋画 3 个植物候选给全组选。
- **11/23**：Phase G＋R 训练；ONNX 导出；极简网页；本地可演示。
- **11/24**：完整验收＋修复；录全部录屏/截图；制作记录初稿；全组第二次检查（三种损伤测试）。
- **11/25**：缓冲（必要时重训）；幻灯片素材定稿；公网部署决策（征求同意）；完整排练 28–30 分钟。
- **11/26**：Pre。
- **备用方案**（分工文档原文）：若 11/24 晚植物仍不达标 → 用"录屏＋原项目网页现场互动＋诚实失败讨论"展示，仍满足课程要求。

## 7. 风险与对策

1. 80px 不收敛 → 64px/3 色简化版（已预备）。
2. 再生条纹/噪点（星空教训）→ 平色＋小网格容易得多；必要时加轻量 sobel 一致性损失或延长 Phase R；失败则诚实记录。
3. 种子在底部 vs 官方中心 → NCA 平移不变，理论无差；M0 实测验证，不行就把种子放网格中心、目标相对布置。
4. WebGPU 算子/驱动问题 → WASM 回退在 80px 下足够快（nca.js 已实现）。
5. 时间 → 每轮训练 ≤1 小时＋止损规则；备用方案兜底。
6. WDDM GPU 并发 → 训练与 headless 验证严格串行。

## 8. 文件变更（本次清理已完成）

- **保留**：src/nca4plot/{model,losses,utils,rollout,export_onnx}.py、web/js/nca.js、web/assets/ort/*、scripts/{serve_web,capture_web,probe_user_regen,verify_user_web}.py、tests/（待重写）、.github/workflows/pages.yml、third_party 许可。
- **新增（实现阶段）**：src/nca4plot/{make_plant_target,train_plant}.py、scripts/probe_plant.py、web/{index.html,styles.css,js/app.js}、data/plant_target_*.png、docs/制作记录.md。
- **归档（仅本地，不入 git）**：archive_starry/ 18.9MB —— 星空模型 v3/v11＋训练日志、已部署 onnx、三模式录屏、对比图、旧 UI、旧训练机与一次性脚本、原始目标数据、旧 README/作品说明。
- **删除**：checkpoints 8.2GB（pool 状态/样本/诊断）、media/iterate 1.9GB、.cache 163MB、旧 web UI 文件，共 ~10.3GB。
- **git**：checkpoints/ 与 media/iterate 等解除跟踪并写入 .gitignore；本地 commit 记录转向，**不 push**。
