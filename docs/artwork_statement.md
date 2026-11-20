# 《永不落定的星空》

**2026，交互式 Web 作品，实时神经细胞自动机、WebGPU / WASM，尺寸可变。**

作品让梵高《星空》的整片天空成为一块活着的物质：它会流动、会受伤、会把伤口长回原来的画，也会带着永不愈合的伤口继续漂移。5,856 个参数构成的局部规则只读取每个细胞的近邻，却托住整幅天空的构图与笔触。这里没有播放任何一帧预制画面：观众看到的每一刻，都是局部规则正在协商的结果。拖拽擦除的是 12 个通道的真实状态，而不只是覆盖一层伤口；边界失去记忆后，残余细胞依照同一套规则重新分裂、迁移、编织，把缺口长回记忆中的样子。"破坏—再生"不是附加交互，而是作品的时间结构。

## 三种时间观

- **再生 regenerate**：天空带着完整的记忆醒来——初始状态就是《星空》本身。撕开它，伤口会依照局部规则长回原画。记忆不是贴图表层，而是被训练进规则里的不动点：规则学会"托住这幅画"，也学会"从边界把画补回来"。
- **生长 growing**：把记忆打散成一片稀疏星群。观众看着星点分裂、合并、协商出旋涡与流带，自组织成整幅天空——星空作为发生过程，而非现成图像。
- **永动 persistent**：没有目标图的天空物质。另一套同参数量的纹理规则让画面永远漂移、永不重复；擦除的伤口被永久锁定、不再愈合，天空带着伤痕继续流动——损伤本身成为画面不可抹去的一部分。

三种模式共用同一种科学语法：局部规则 → 全局形态。差别只在于规则是否携带记忆、以及携带多少。艺术观念由此成立：**星空不是状态，而是过程**——可以被记住的过程、正在发生的过程、永不落定的过程。

## 技术说明

两个模型均为 12 通道、96 隐单元、恰好 5,856 个可训练参数的神经细胞自动机，四固定感知核（identity / Sobel-x / Sobel-y / Laplacian）、circular padding、随机异步更新。

- **再生 / 生长模型**：Growing-CA 式（Mordvintsev et al., Distill 2020）目标训练。前 4 通道直接携带 RGBA∈[0,1]，零状态即透明即死亡；每步更新前后以 3×3 max-pool 生命门约束"死细胞不自发复活、活细胞不猝死"。目标为整幅天空 256×144（含月亮、星晕、旋涡与柏树剪影），训练池一半以"活着的画本身"（RGBA=目标、隐通道=0，与网页再生模式启动状态完全一致）初始化并施加随机圆形损伤，学习稳定托住原画与伤口再生；另一半从单细胞种子与随机损伤状态生长，供再生课程使用。网页生长模式与再生模式共用这套权重，但初始条件不同：约 60% 像素随机携带原画色素的"记忆星群"，由再生动力学自组织成整幅画（单细胞 frontier 在 5,856 参数下被诊断证明长不出连续色调原画）。网页再生模式的初始状态即该记忆状态本身，其后的每一帧运动仍只来自 NCA 推理。
- **永动模型**：texture NCA（Self-Organising Textures, Distill 2021），以《星空》天空局部为 exemplar，在 VGG16 特征空间以 sliced optimal transport 训练，无目标图、永不收敛。网页永动模式对笔刷核心施加"损伤锁定"边界条件（每次推理后钉零），实现 Distill persistent 的"损伤不愈"语义；锁定不是学习到的权重，运动仍全部来自 NCA。

渲染层只做材质：厚涂纤维 relief、静态颗粒、稳定高光与伤口夜色合成；不产生任何运动。

## 关于湍流，保持诚实

《星空》的旋涡常被联想为湍流；NCA 的"局部规则生成全局秩序"与湍流的"局部作用形成涡旋级联"形成结构上的同构。需要诚实说明：关于其亮度谱符合 Kolmogorov −5/3 律的 2024 年测量，2025 年已有同行评论，相关解释仍有争议。本作借用的是一种审美同构，而非物理证明。

## 文献

- Mordvintsev, A., Randazzo, E., Niklasson, E., Levin, M., "Growing Neural Cellular Automata," Distill, 2020. DOI: 10.23915/distill.00023.
- Mordvintsev, A. et al., "Self-Organising Textures," Distill, 2021. DOI: 10.23915/distill.00023.003.
- Ma, C. et al., "Hidden turbulence in van Gogh's *The Starry Night*," *Physics of Fluids* 36(9), 095140, 2024. DOI: 10.1063/5.0213627.
- Gad-el-Hak, M. et al., "Comment on 'Hidden turbulence in van Gogh's *The Starry Night*'," *Physics of Fluids* 37(12), 129101, 2025. DOI: 10.1063/5.0306945.
- Aragón, J. L. et al., "Turbulent Luminance in Impassioned van Gogh Paintings," *Journal of Mathematical Imaging and Vision* 30(3), 275–283, 2008.

## 先例

Distill 的 Growing Neural Cellular Automata 提供了"损伤后长回目标形态"的方法框架与 regenerate / growing / persistent 三种时间观的直接先例；Self-Organising Textures 与 Hexells 展示了细胞自动机作为生成艺术媒介的可能；DyNCA 将 NCA 推向持续运动。本作把"记忆—再生—永动"并置为同一作品的三种时间结构。
