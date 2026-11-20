# 永不落定的星空 / A Sky That Never Settles

> 一片会流动、会受伤、会把伤口长回原画的《星空》。  
> 由作者亲手训练的 **两个各 5,856 参数** 的神经细胞自动机（NCA）驱动，全程在浏览器本地运行。

**在线作品：[https://wangcheng726.github.io/nca4plot/](https://wangcheng726.github.io/nca4plot/)**

![作品静帧](media/poster_sky_1920x1080.png)

## 作品一句话

观众拖拽撕开天空；被擦除的不是一层贴图，而是 NCA 的真实 12 通道状态。松手后，残余细胞依据同一套局部规则把缺口长回《星空》本来的笔触。艺术观念：**星空不是状态，而是过程**——可以被记住的过程、正在发生的过程、永不落定的过程。

## 三种时间观（对应 Distill Growing-CA 的三种演示）

| 模式 | 初始状态 | 擦除之后 | 模型 |
| --- | --- | --- | --- |
| **再生 regenerate**（默认） | 整幅《星空》本身（记忆启动） | 伤口从边缘长回原画的笔触与颜色 | 目标 NCA（Growing-CA 式） |
| **生长 growing** | 稀疏记忆星群（约 60% 像素携带原画色素） | 星点自组织、合并成整幅《星空》 | 目标 NCA（同一 256×144 权重） |
| **永动 persistent** | 零种子长出的天空纹理 | 伤口永久敞开、不愈合；其余天空持续流动 | 纹理 NCA（Self-Organising Textures 式）＋损伤锁定边界 |

再生模式的"记忆"不是贴图层：初始状态（RGBA=原画、隐通道=0）是被训练进规则里的不动点，其后的每一帧运动仍只来自 NCA 推理；渲染着色器只增强厚涂材质，不产生任何运动。

永动模式对应 Distill Growing-CA 的 **persistent 语义：损伤永不愈合**。实现上，笔刷核心（擦除后保留比例 <0.35 的细胞）被引擎记录为"损伤锁定"（damage-lock）边界条件，每次 NCA 推理后重新钉零；羽化边缘保留部分状态、继续流动，于是天空围绕一道永远敞开的伤口重新编织。**锁定是施加在状态上的边界条件，不是学习到的权重**——画面里所有运动仍 100% 来自 NCA 推理。重置会清除全部锁定，回到零种子重新生长。

## 操作

- **拖拽 / 触摸拖动**：擦除天空（清除笔刷内全部 12 通道）。
- **滚轮**：笔刷大小。
- **空格 / 暂停按钮**：暂停与继续。
- **R / 重置按钮**：按当前模式重置（再生=回到完整原画；生长=回到一片新的随机记忆星群；永动=零种子）。
- **模式按钮**：regenerate / growing / persistent 切换。
- **质感 128 / 256 / 512**：仅永动模式可选（`1/2/3` 键）。
- **作品说明 / 全屏**：观念与文献；无干扰观看。

支持 WebGPU 的浏览器优先 GPU 推理，否则自动回退 WASM。首次打开需下载 ONNX Runtime Web 后端（约 30–40 MB）。

## 技术与科学

### 两个小模型，同一套语法

| 项目 | 配置（两模型相同） |
| --- | --- |
| 参数 | 各 **5,856** 个可训练参数 |
| 状态 | 12 通道；目标模型前 4 通道为 RGBA∈[0,1]（零=透明=死亡），纹理模型前 3 通道 RGB |
| 感知 | identity / Sobel X / Sobel Y / Laplacian 四固定核，circular padding |
| 更新 | `Conv2d(48→96,1×1)`+ReLU → `Conv2d(96→12,1×1,bias=False)`，输出层零初始化 |
| 异步 | 每步约 50% 细胞随机更新 → 永不重复的流动 |
| 生命门 | 目标模型：3×3 max-pool alpha 门，死细胞不自发复活、活细胞不猝死 |

### 目标模型：把整幅画训练成不动点

目标为**整幅《星空》256×144**（含月亮、星晕、旋涡、柏树剪影；公有领域扫描，见 `data/target_metadata.json`），不是局部纹理裁剪。训练池 256 个状态，一半以"活着的画本身"启动（与网页再生模式启动状态逐像素一致），一半从单细胞种子、重度损伤与残岛状态启动；对池状态施加随机圆形损伤（1–3 个圆盘，半径 6–17% 短边）。

损失与课程（`src/nca4plot/train_target.py`）：

- **伤口加权 RGBA MSE**：损伤圆盘内权重 ×8。伤口只占画面 ~3%，普通全图 MSE 会让模型把"填平均色灰痂"学成稳定吸引子；加权后再生信号占损失的约四分之一。
- **Sobel 梯度 MSE（伤口同权 ×0.5）**：平涂没有边缘，该项迫使伤口长回笔触结构而不是灰雾。
- **VGG 感知锚（×0.15）**：VGG16 五层特征的逐像素 MSE，按层归一化、伤口区域下采样掩码同权 ×8。Sobel 只看 3×3 边缘，VGG 看五个尺度的笔触纹理；与 OT（分布匹配）不同，特征 MSE 把笔触约束在记忆中的位置。
- **overflow loss**：约束状态不越界。
- **去噪锚**：接近原画的池状态以 50% 概率加小噪声（RGBA σ∈[0.005,0.03]、隐通道 σ=0.05），训练"从扰动回到原画"，抑制长时程漂移。
- **长时程 rollout**：每 8 步一次 192–320 步稳定 rollout；种子样本 256–448 步生长 rollout；其余 64–96 步。网页观众一次看几十秒＝上千步，训练必须见过这个时程。
- Adam 2e-3，milestones 2000/4000/6000 ×0.3，累计约 8500 步（含断点续跑，最终 checkpoint step 8501）；梯度范数归一化；gradient checkpointing。

### 生长模式：记忆星群初始条件（为什么不是单细胞）

官方 emoji 演示从单细胞长出目标，但那是**扁平色块**目标、且传播距离短。诊断（v7–v12，见 `checkpoints/nca_grow*_training_log.jsonl`、`checkpoints/probe_*.png`）表明：整幅连续色调《星空》在 256×144 与 128×72 上、单种子或残岛出发的 frontier 传播都停在"平均色灰雾"吸引子（growth_mse≈0.017–0.02 不动，加长 rollout 反而漂远）；而**被活画包围的伤口**向内再生始终优秀（regen_mse≈0.0006）。5,856 参数记不住"上百格外的色素位置"，但记得住"邻域记忆如何补全"。

因此生长模式改用**稀疏记忆星群**初始条件：约 60% 像素随机携带原画 RGBA（隐通道 0），其余为死细胞。每个死细胞都紧邻活记忆，训练过的再生动力学于是把星群合并成整幅画——探针实测 60 步 mse 0.0033、240 步 0.0031（`checkpoints/probe_const60_nca_target.png`），与再生模式同权重、同网格。网页上生长模式以约 14 CA 步/秒的慢时钟运行（`stepsPerRun=1`），让观众看清"星点→整幅天空"的自组织；每次重置的星群随机分布不同，永不重复。

### 纹理模型：永不落定的天空物质

`src/nca4plot/train.py` 沿用官方 texture NCA：VGG16 层 `[1,6,11,18,25]` + sliced optimal transport + overflow，三块天空裁剪多 exemplar，无目标图、永不收敛。供永动模式使用。

### 为什么擦除后能长回原画

每个细胞只读近邻。擦除抹去局部记忆后，边界两侧的差异驱动更新向内传播；因为规则在训练里见过成千上万次"带伤口的整幅画→长回原画"的轨迹，再生是学出来的动力学，不是修复模块。生长与修复是同一过程的两面。

### 关于湍流，保持诚实

《星空》旋涡常被读作湍流直觉；NCA"局部规则→全局自组织"与湍流"局部作用→涡旋级联"构成结构同构。但《星空》亮度谱符合 Kolmogorov −5/3 律的 2024 年测量在 2025 年已有同行评论，解释仍有争议。**本作借用审美同构，不主张证明物理定律。**

## 环境与复现

已验证：Windows 11 · RTX 5070 Ti 16 GB（Blackwell sm_120）· Python 3.12 · PyTorch 2.9.1+cu128 · CUDA 12.8 · uv。Blackwell 需 CUDA 12.8+ 的 PyTorch；若 `torch.cuda.is_available()` 为 False，先查驱动与 CUDA 版本。

```powershell
uv sync
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"

# 1. 整幅目标（已随仓库提供 data/target_sky_256x144.png）
uv run python -m nca4plot.prepare_target

# 2. 目标模型训练（约 8500 步，约 1–2 小时，5070 Ti；可断点续跑）
uv run python -m nca4plot.train_target --steps 8500 --milestones 2000 4000 6000 --out checkpoints/nca_target.pt

# 3. 纹理模型（永动模式）
uv run python -m nca4plot.train --steps 6000 --out checkpoints/nca_starry.pt

# 4. 导出 ONNX 到 web/models（目标模型每次推理 2 步，与网页 masks 维度一致）
uv run python -m nca4plot.export_onnx --kind target --target-steps 2 --checkpoint checkpoints/nca_target.pt
uv run python -m nca4plot.export_onnx --kind texture --checkpoint checkpoints/nca_starry.pt

# 5. 本地运行（零构建静态站点）
python -m http.server 8765 --directory web --bind 127.0.0.1
# 打开 http://127.0.0.1:8765/

# 6. 测试与录屏
uv run pytest
uv run python scripts/capture_web.py --mode regenerate
uv run python scripts/capture_web.py --mode persistent
```

训练日志 `checkpoints/nca_target_training_log.jsonl`；每 500 步四联样本 `checkpoints/samples_nca_target/step_XXXXX_comparison.png`（TARGET / STABLE400 / GROWN400 / REGEN200）。

## 实测验收

验收命令：`python scripts/verify_local.py <url> <mode> <tag>`（Chrome headless · WebGPU · 1600×900），截图与报告见 `media/iterate/<tag>_*.png / .json`；本轮标签 `iter2002_regen / iter2002_growing / iter2002_persist`。mse＝RGBA［0,1］对 `data/target_sky_256x144.png` 的均方误差（隔 2 像素采样）；频闪＝暂停后相邻帧平均绝对差（0–255）；对齐＝擦除区质心与笔刷映射质心之差（cells）。

| 模式 | 网格 | 初始状态 | 擦除愈合后整幅 mse | 伤口区 mse | 暂停频闪 max | 笔刷对齐 |
| --- | --- | --- | --- | --- | --- | --- |
| 再生 regenerate | 256×144 | 即原画：mse 0.00011 | 0.0003 | 0.0052 | 0.0022 | 0.12 cells |
| 生长 growing | 256×144 | 记忆星群：step≈4 时约 60% 像素携带原画记忆 | 0.00381 | 0.0063 | 0.0026 | 0.12 cells |
| 永动 persistent | 256×256 | 无目标图，纹理从零生长 | —（无目标） | 伤口锁定保留＝0 | 0.0022 | 0.13 cells |

- **流动**：相邻帧差 regenerate 0.0455、growing 0.209（慢时钟自组织）、persistent 25.10（永不静止）；暂停后三模式帧差均 ≤0.0026，**无频闪**。
- **永动损伤锁定**：真实笔刷撕裂后继续运行 5 秒（约 600 CA 步），389 个锁定细胞的 12 通道 max|state|＝**0.000000**（伤口永久敞开、不愈合），其余天空帧差 25.10 持续流动；笔刷对齐 0.13 cells（`media/iterate/iter2002_persist_healed.png` 可见永久暗夜裂口）。
- **与旧版对比**：旧 growing（v6b，`media/iterate/v6b4001g_*`）愈合 mse 0.0213＝平均色灰雾；本轮 0.00381＝可辨认整幅原画，约 5.6× 改善。旧公网只有纹理模式、初始状态不是原画（`media/iterate/public_pre_stroke.png`）。
- **生长 boot 门**（`tests/test_web.py`）：alpha 在数步内铺满，故星群阶段按"每细胞保真度"度量——step≈6–14 暂停时钟，接近原画的细胞占比 ∈(0.35, 0.85) 且整幅 mse>0.006（仍是星群而非成品）；step≥240 要求 mse<0.012。实测 boot close≈0.69 / mse≈0.011，240 步 mse≈0.0035 / close≈0.90（`scripts/probe_boot_gates.py`）。
- **Python 侧**：模型恰好 5,856 参数；ONNX↔PyTorch 最大绝对差 2.09e-7（`checkpoints/export_target_report_256x144.json`）；星群探针 60 步 mse 0.0033、240 步 0.0031（`checkpoints/probe_const60_nca_target.png`）。
- **pytest 全绿**：参数量、生命门、ONNX parity、浏览器端到端（初始 mse<0.012、擦除 12 通道清零、伤口 mse<0.02、**永动伤口锁定保留＝核心 12 通道恒为 0**、三模式切换、移动端布局）。
- 演示视频：`media/nca4plot_demo.mp4 / .gif`（再生）、`media/growing/`（星群→整幅天空）、`media/persistent/`（永动），各含海报与过程接触表。

## 交付索引

- `web/`：零构建静态站点（index.html / js / styles / models / vendor），GitHub Pages 直接发布。
- `src/nca4plot/`：模型、损失、prepare_target / prepare_exemplar、train_target / train、export_onnx、rollout。
- `checkpoints/`：权重（.pt）、ONNX、训练日志、样本、eval json。
- `docs/artwork_statement.md`：作品说明（三种时间观＋科学边界＋文献）。
- `media/`：演示 MP4 / GIF / 海报 / 过程接触表 / 迭代验收截图。
- `tests/`：模型参数量、生命门、ONNX parity、浏览器端到端（初始 mse、擦除清零、伤口再生、三模式、移动端）。
- `scripts/verify_local.py`：三模式本地验收（频闪、笔刷对齐、mse、流动差分）。

## 已知限制

- **生长模式不是单细胞启动**：单细胞 frontier 在 5,856 参数下长不出连续色调原画（v7–v12 诊断在案）；生长模式以稀疏记忆星群为初始条件，自组织结果是印象式近似（全图 mse≈0.003），笔触细节略软于再生模式。
- **再生是局部记忆重建**：伤口长回原画的笔触与颜色分布，极大伤口或反复撕裂处可能有轻微笔触差异，不是像素级完美修复。
- WebGPU 不可用时回退 WASM，速度明显变慢；512 档仅建议桌面 GPU。
- 画面是实时随机动力学，无法逐像素复现某一历史帧；权重、种子逻辑与训练过程可复现。
- 笔触质感是实时渲染材质，不是物理颜料，无实体输出。

## 文献与先例

- Mordvintsev, Randazzo, Niklasson & Levin, [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/), Distill, 2020.（regenerate / growing / persistent 三模式与损伤再生的直接先例）
- Mordvintsev et al., [Self-Organising Textures](https://distill.pub/selforg/2021/textures/), Distill, 2021.
- Google Research, [self-organising-systems](https://github.com/google-research/self-organising-systems)（texture_nca_pytorch.ipynb，Apache-2.0）。
- Ma et al., "Hidden turbulence in van Gogh's *The Starry Night*," *Physics of Fluids* 36(9), 095140, 2024. [arXiv:2310.03415](https://arxiv.org/abs/2310.03415).
- Gad-el-Hak et al., "Comment on 'Hidden turbulence in van Gogh's *The Starry Night*'," *Physics of Fluids* 37(12), 129101, 2025.
- Aragón et al., "Turbulent Luminance in Impassioned van Gogh Paintings," *JMIV* 30(3), 2008. [arXiv:physics/0606246](https://arxiv.org/abs/physics/0606246).
- Znah, [Hexells](https://znah.net/hexells/)；Sakana AI, [Digital Ecosystem](https://pub.sakana.ai/digital-ecosystem/)；[DyNCA](https://dynca.github.io/)。

## 许可与致谢

- 原创代码 MIT（`LICENSE`）；训练权重 CC BY 4.0。
- Van Gogh, *The Starry Night* (1889)：公有领域二维作品，来源与 SHA256 见 `data/target_metadata.json` 与 `data/exemplar_metadata.json`。
- 上游训练路线 Apache-2.0；ONNX Runtime Web 遵循其自身许可；完整第三方许可见 `LICENSES.md`。

---

作者：Wang Cheng · 2026 · 艺术与科学课程作品
