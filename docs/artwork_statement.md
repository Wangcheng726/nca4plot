# 《永不落定的星空》

**2026，交互式 Web 作品，实时神经细胞自动机、WebGPU / WASM，尺寸可变。**

作品不是播放一张《星空》，而是让一片星空物质从零开始生长。由 5,856 个参数构成的局部规则只读取每个细胞的近邻，却逐步协商出旋涡、流带和厚涂般的笔触。这里没有固定的目标图，也没有最终帧；随机异步更新让画面持续漂移，永远不抵达平衡。观众拖拽时，擦除的是 12 个通道的真实状态，而不只是覆盖一层伤口。边界失去记忆后，残余细胞依照同一套规则重新分裂、迁移、编织，缺口因此成为下一次生长的起点。“破坏—再生”不是附加交互，而是作品的时间结构。

科学在这里不是装饰。《星空》的旋涡常被联想为湍流；NCA 的“局部规则生成全局秩序”与湍流的“局部作用形成涡旋级联”形成结构上的同构。需要诚实说明，关于其亮度谱符合 Kolmogorov −5/3 律的解释在 2025 年已有同行评论，仍有争议；本作借用的是一种审美同构，而非物理证明。星空因此不是状态，而是一场仍在发生、可以被触碰的过程。

## 技术说明

模型为 12 通道 texture NCA，约 5,856 个可训练参数，使用四固定感知核、circular padding、随机异步更新；以《星空》纯天空旋涡局部为 exemplar，通过 VGG16 特征空间的 sliced optimal transport 训练。网页端从全零种子生长，擦除会清零全部通道，随后由同一局部规则重新组织。作品不含月亮、柏树或村庄构图，关注的是天空物质本身的流动与再生。

## 文献

- Ma, C. et al., “Hidden turbulence in van Gogh’s *The Starry Night*,” *Physics of Fluids* 36(9), 095140, 2024. DOI: 10.1063/5.0213627.
- Gad-el-Hak, M. et al., “Comment on ‘Hidden turbulence in van Gogh’s *The Starry Night*’,” *Physics of Fluids* 37(12), 129101, 2025. DOI: 10.1063/5.0306945.
- Aragón, J. L. et al., “Turbulent Luminance in Impassioned van Gogh Paintings,” *Journal of Mathematical Imaging and Vision* 30(3), 275–283, 2008.
- Mordvintsev, A. et al., “Self-Organising Textures,” Distill, 2021.

## 先例

Distill 的 Growing Neural Cellular Automata 与 Self-Organising Textures 提供了“局部规则生成、损伤后自组织恢复”的方法框架；Hexells 展示了细胞自动机作为生成艺术媒介的可能；DyNCA 将 NCA 推向持续运动。本作不追求复刻完整画面，而是把“永不落定”和“可被触碰的再生”作为作品主体。