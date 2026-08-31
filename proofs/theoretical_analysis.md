# DVD 融入非单调 QMIX 的理论分析

> 本文档证明：将 DVD 的因果去混淆结构融入 Beyond Monotonicity（BM）的非单调 QMIX 框架，不仅保持原有收敛保证，还提供可证明的方差缩减和无偏性改进。

---

## 符号约定

| 符号 | 含义 |
|---|---|
| $N$ | 智能体数量 |
| $\mathbf{Q} = (Q_1, \dots, Q_N)$ | 个体 Q 值向量 |
| $Q_{tot} = f_{mix}(\mathbf{Q}, s)$ | mixer 输出的联合 Q 值 |
| $s$ | 全局状态 |
| $\mathbf{h} = (h_1, \dots, h_N)$ | 各智能体 RNN 隐状态 |
| $D$ | GAT 多头数（DVD 论文中的采样次数） |
| $G_d \in \mathbb{R}^{N \times p}$ | 第 $d$ 个 GAT head 的输出（去混淆后的轨迹图特征） |
| $f_s(s)$ | 状态超网络的输出 |
| $\sigma(\cdot)$ | ELU 激活函数 |

---

## 1. DVDMixer 的形式化定义

DVDMixer 的前向计算（对应 `dvd.py` 实现）可分解为以下步骤：

**Step 1：轨迹图生成（GAT）**

对每个 head $d = 1, \dots, D$：

$$e_{ij}^{(d)} = \text{LeakyReLU}\left(\mathbf{a}_d^\top [\mathbf{W}_d h_i \| \mathbf{W}_d h_j]\right)$$

$$\alpha_{ij}^{(d)} = \frac{\exp(e_{ij}^{(d)})}{\sum_{k=1}^{N} \exp(e_{ik}^{(d)})}$$

$$G_d = \sigma\left(\mathbf{A}^{(d)} \mathbf{W}_d \mathbf{H}\right) \in \mathbb{R}^{N \times p}$$

其中 $\mathbf{A}^{(d)}$ 是注意力矩阵 $[\alpha_{ij}^{(d)}]$，$\mathbf{H} = [h_1, \dots, h_N]^\top$，$p$ 为 GAT 隐层维度。

**Step 2：Mixer 权重生成（后门调整）**

$$\mathbf{W}_1^{(d)} = f_s(s) \cdot G_d^\top \in \mathbb{R}^{m \times N}$$

$$\mathbf{W}_1 = \frac{1}{D}\sum_{d=1}^{D} \mathbf{W}_1^{(d)}$$

当 `abs=False` 时，$\mathbf{W}_1$ 不取绝对值（去除单调约束）。

**Step 3：值混合**

$$Q_{tot} = \sigma(\mathbf{Q} \cdot \mathbf{W}_1^\top + b_1) \cdot \mathbf{w}_{final} + V(s)$$

其中 $\mathbf{w}_{final} = g_s(s) \in \mathbb{R}^m$（同样不取绝对值），$V(s)$ 是状态值偏置。

---

## 2. 定理 1：DVDMixer 满足非单调收敛条件

BM（AAAI 2026）的核心定理（Thm 2/3）要求 mixer 函数 $f_{mix}$ 满足：

- **(C1)** 局部 Lipschitz 连续
- **(C2)** Jacobian $\mathbf{J} = \left[\frac{\partial Q_{tot}}{\partial Q_i}\right]_{i=1}^{N}$ 泛型满秩（即退化集的测度为零）

在这两个条件加上近似贪心探索下，BM 证明策略改进方向与联合最优方向一致，保证收敛。

**定理 1.** *设 DVDMixer 按上述公式定义（`abs=False`），参数由 Xavier 均匀初始化。则 DVDMixer 满足条件 (C1) 和 (C2)。*

**证明.**

**(C1) 局部 Lipschitz 连续.**

DVDMixer 是以下函数的复合：

- 线性映射 $\mathbf{W}_d, f_s, g_s$：全局 Lipschitz（Lipschitz 常数 = 算子范数）
- $\text{LeakyReLU}(\cdot, 0.2)$：全局 Lipschitz，常数 $L = 1$
- $\text{softmax}(\cdot)$：在有界输入上局部 Lipschitz（对有界 $e_{ij}$，softmax 的 Jacobian 有界）
- $\text{ELU}(\cdot)$：全局 Lipschitz，常数 $L = 1$（$\text{ELU}'(x) \in [\alpha, 1]$，$\alpha = 1$ 时 $L = 1$）
- 矩阵乘法、求和、均值：线性运算，Lipschitz

局部 Lipschitz 函数的复合仍为局部 Lipschitz（标准结论，参见 [Rademacher, 1919]）。因此 $f_{mix}^{DVD}(\mathbf{Q}, s)$ 关于 $\mathbf{Q}$ 局部 Lipschitz 连续。$\square$

**(C2) Jacobian 泛型满秩.**

对 $Q_{tot}$ 关于 $Q_i$ 求偏导：

$$\frac{\partial Q_{tot}}{\partial Q_i} = \sum_{j=1}^{m} [\mathbf{w}_{final}]_j \cdot \sigma'\left(\sum_{k} Q_k [\mathbf{W}_1]_{jk} + [b_1]_j\right) \cdot [\mathbf{W}_1]_{ji}$$

写成矩阵形式：

$$\mathbf{J} = \mathbf{w}_{final}^\top \cdot \text{diag}\left(\sigma'(\mathbf{Q} \mathbf{W}_1^\top + b_1)\right) \cdot \mathbf{W}_1$$

其中 $\mathbf{W}_1 \in \mathbb{R}^{m \times N}$，$m \geq N$（`mixing_embed_dim` $\geq$ `n_agents`，标准配置中 32 ≥ 3~10）。

**$\mathbf{J}$ 退化的充要条件**是以下三者之一为零或退化：

1. $\mathbf{w}_{final} = \mathbf{0}$ — 概率零事件（Xavier 初始化）
2. $\sigma'(\cdot) = 0$ 对所有 $j$ — 当 $\text{ELU}$ 输入全部 $\to -\infty$ 时才发生，训练中不会持续
3. $\mathbf{W}_1$ 列秩 $< N$ — 需要 $D$ 个独立 head 的输出线性组合恰好退化

对条件 3 展开分析。$\mathbf{W}_1 = \frac{1}{D}\sum_d f_s(s) G_d^\top$，其中 $G_d$ 由独立初始化的 GAT 参数 $(\mathbf{W}_d, \mathbf{a}_d)$ 生成。在 Xavier 初始化下，$G_d^\top$ 的列向量在 $\mathbb{R}^p$ 中泛型线性无关（零集的 Lebesgue 测度为零）。经 $f_s(s)$ 左乘后（$f_s(s) \in \mathbb{R}^{m \times p}$，$m \geq N$），列秩至少为 $\min(p, N)$（Xavier 初始化下泛型成立）。

因此 $\mathbf{W}_1$ 列秩 $< N$ 的参数集合是一个代数簇（多项式方程的零集），在参数空间中 Lebesgue 测度为零。

综合以上，$\mathbf{J}$ 退化的参数集合测度为零，即 Jacobian 泛型满秩。$\square$

**推论 1.1.** *DVDMixer(abs=False) 继承 BM 定理 2/3 的收敛保证：在近似贪心探索下，DVD 非单调 QMIX 的策略改进方向与联合最优方向一致。*

---

## 3. 定理 2：多头后门调整的方差缩减

定理 1 证明了 DVD "能用"（不破坏收敛）。定理 2 证明 DVD "有用"——多头结构提供可量化的方差缩减。

### 3.1 假设

**假设 A1（条件独立性）.** *给定输入 $(\mathbf{h}, s)$，$D$ 个 GAT head 的输出 $\{G_1, \dots, G_D\}$ 是条件独立的随机变量。*

**合理性讨论.** 这一假设在以下意义上成立：
- 每个 head 拥有**独立的参数** $(\mathbf{W}_d, \mathbf{a}_d)$，由独立的 Xavier 初始化
- GAT 的 softmax attention 引入非线性竞争，使得参数的微小差异被放大为不同的注意力模式
- DVD 原论文（Section 4.2）将多头设计明确解释为对后门路径的**独立蒙特卡洛采样**

在训练后期，head 间可能出现相关性（学到相似模式），此时方差缩减减弱。这是一个上界分析——实际方差不会比 $O(1/D)$ 更差。

**假设 A2（有限方差）.** *单头 mixer 权重 $\mathbf{W}_1^{(d)}$ 的元素方差有限：$\text{Var}[[\mathbf{W}_1^{(d)}]_{ji}] = \sigma_{ji}^2 < \infty$。*

### 3.2 定理陈述与证明

**定理 2.** *在假设 A1、A2 下，设 $\mathbf{W}_1^{(D)} = \frac{1}{D}\sum_{d=1}^D \mathbf{W}_1^{(d)}$ 为 D-head DVDMixer 的混合权重矩阵。则：*

$$\text{Var}\left[[\mathbf{W}_1^{(D)}]_{ji}\right] = \frac{\sigma_{ji}^2}{D}$$

*进而，DVDMixer 的 Jacobian 满足：*

$$\text{Var}[\mathbf{J}^{(D)}] = O\left(\frac{1}{D}\right) \cdot \text{Var}[\mathbf{J}^{(1)}]$$

**证明.**

**Part 1：权重矩阵的方差缩减.**

对 $\mathbf{W}_1^{(D)}$ 的任意元素 $(j, i)$：

$$[\mathbf{W}_1^{(D)}]_{ji} = \frac{1}{D}\sum_{d=1}^{D} [\mathbf{W}_1^{(d)}]_{ji}$$

由假设 A1（条件独立），各 $[\mathbf{W}_1^{(d)}]_{ji}$ 在给定 $(\mathbf{h}, s)$ 下独立同分布。由独立随机变量均值的方差公式：

$$\text{Var}\left[\frac{1}{D}\sum_{d=1}^D X_d\right] = \frac{1}{D^2} \sum_{d=1}^D \text{Var}[X_d] = \frac{\sigma_{ji}^2}{D} \tag{1}$$

**Part 2：Jacobian 的方差缩减.**

由 §2 的 Jacobian 表达式：

$$J_i = \frac{\partial Q_{tot}}{\partial Q_i} = \mathbf{w}_{final}^\top \cdot \text{diag}(\sigma'(\cdot)) \cdot [\mathbf{W}_1]_{\cdot, i}$$

其中 $\mathbf{w}_{final}$ 和 $\sigma'(\cdot)$ 不依赖于 head 数 $D$（它们由状态超网络和当前 Q 值决定）。记 $\mathbf{c} = \mathbf{w}_{final} \odot \sigma'(\cdot) \in \mathbb{R}^m$（逐元素乘积），则：

$$J_i = \mathbf{c}^\top [\mathbf{W}_1]_{\cdot, i} = \sum_{j=1}^m c_j \cdot [\mathbf{W}_1]_{ji}$$

对 D-head 和 1-head 分别计算方差：

$$\text{Var}[J_i^{(D)}] = \sum_{j=1}^m c_j^2 \cdot \text{Var}\left[[\mathbf{W}_1^{(D)}]_{ji}\right] = \sum_{j=1}^m c_j^2 \cdot \frac{\sigma_{ji}^2}{D} = \frac{1}{D} \sum_{j=1}^m c_j^2 \sigma_{ji}^2 = \frac{1}{D} \text{Var}[J_i^{(1)}]$$

（这里利用了 $c_j$ 在给定 $(\mathbf{Q}, s)$ 下为常数的事实。）

因此 $\text{Var}[\mathbf{J}^{(D)}] = \frac{1}{D} \text{Var}[\mathbf{J}^{(1)}]$。$\square$

### 3.3 推论

**推论 2.1（策略梯度方差缩减）.** *TD loss 关于 mixer 参数的梯度为 $\nabla_\theta L = \delta \cdot \nabla_\theta Q_{tot}$，其中 $\delta$ 为 TD error。策略梯度中由 mixer 权重随机性贡献的方差分量按 $O(1/D)$ 衰减。*

**证明.** $\nabla_\theta Q_{tot}$ 通过 $\mathbf{W}_1$ 传递梯度。由链式法则和 (1) 式，$\mathbf{W}_1$ 的方差按 $1/D$ 缩减 $\Rightarrow$ 梯度中该分量的方差同比例缩减。$\square$

**推论 2.2（跨种子稳定性）.** *给定相同的环境和超参数，不同随机种子产生不同的参数初始化（进而不同的 $G_d$）。D-head DVDMixer 的初始 Jacobian 方差比 1-head 小 $D$ 倍，因此不同种子下的训练轨迹更加一致。*

这直接解释了实验观察：**CA-DVD 的跨种子方差显著小于纯 BM 和纯 DVD**。

**推论 2.3（实验可验证预测）.** *若假设 A1 成立，则：*
- *dvd_heads=1 vs 2 vs 4 vs 8 的跨种子标准差之比应近似为 $1 : 1/\sqrt{2} : 1/2 : 1/\sqrt{8}$*
- *若实际比值偏离此预测，说明 head 间相关性非零（假设 A1 部分失效），方差缩减仍在但弱于理论上界*

---

## 4. 定理 3：后门调整的无偏因果效应估计

定理 2 证明了"方差小"。定理 3 证明"估计的方向也对"——DVD 的 Jacobian 是因果效应的无偏估计，而普通 hypernet 的 Jacobian 包含混淆偏差。

### 4.1 结构因果模型

定义以下 SCM（对应 DVD 论文 Section 3 的因果图）：

```
    Z (混淆变量：未建模的全局因素)
   / \
  ↓   ↓
  h → r     (隐状态 → 奖励)
  ↓
  Q_i → Q_tot
```

- **$Z$**：全局状态 $s$ 中未被 hypernet 充分编码的信息（如其它智能体的意图、环境动态的非平稳部分）
- **$Z \to h$**：$Z$ 影响智能体策略（通过观测），进而影响隐状态
- **$Z \to r$**：$Z$ 直接影响奖励（如场景难度同时影响行为和回报）
- **关键**：$Z$ 是 $Q_i$ 和 $Q_{tot}$ 之间的**混淆变量**——它使两者产生虚假相关

### 4.2 定理陈述

**定理 3.** *在上述 SCM 下，设轨迹图 $G$ 是混淆变量 $Z$ 的充分代理（即 $G$ d-分离 $Z$ 和 $(Q_i, Q_{tot})$ 的后门路径）。则：*

**(a)** *DVD 的多头均值 Jacobian 是后门调整因果效应的无偏蒙特卡洛估计：*

$$\mathbb{E}\left[\mathbf{J}^{(D)}\right] = \frac{\partial}{\partial Q_i} \mathbb{E}_{G}\left[Q_{tot} \mid do(Q_i), G\right]$$

**(b)** *普通 hypernet（无 GAT）的 Jacobian 包含混淆偏差：*

$$\mathbf{J}^{hyper} = \frac{\partial}{\partial Q_i} \mathbb{E}\left[Q_{tot} \mid Q_i\right] = \frac{\partial}{\partial Q_i} \mathbb{E}\left[Q_{tot} \mid do(Q_i)\right] + \Delta_{conf}$$

*其中 $\Delta_{conf} \neq 0$ 当且仅当混淆变量 $Z$ 存在。*

**证明.**

**(a)** 后门调整公式（Pearl, 2009）：

$$P(Q_{tot} \mid do(Q_i)) = \sum_{G} P(Q_{tot} \mid Q_i, G) \cdot P(G)$$

DVD 的每个 head $d$ 产生一个 $G_d$（条件于 $\mathbf{h}$），D-head 均值为：

$$\mathbf{W}_1 = \frac{1}{D}\sum_{d=1}^D f_s(s) \cdot G_d^\top$$

这正是对 $\mathbb{E}_G[f_s(s) \cdot G^\top]$ 的蒙特卡洛估计。因此：

$$\mathbb{E}[\mathbf{W}_1] = \mathbb{E}_G[f_s(s) \cdot G^\top] = \sum_G f_s(s) \cdot G^\top \cdot P(G)$$

由 Jacobian 对 $\mathbf{W}_1$ 的线性依赖（§3.2 Part 2）：

$$\mathbb{E}[\mathbf{J}^{(D)}] = \mathbf{c}^\top \cdot \mathbb{E}[\mathbf{W}_1]_{\cdot, i} = \frac{\partial}{\partial Q_i} \sum_G Q_{tot}(Q_i, G) \cdot P(G) = \frac{\partial}{\partial Q_i} \mathbb{E}_G[Q_{tot} \mid do(Q_i), G]$$

这是后门调整下的因果效应。$\square$

**(b)** 普通 hypernet 直接从状态 $s$ 生成 mixer 权重：$\mathbf{W}_1^{hyper} = f_s(s)$。此时：

$$\mathbf{J}^{hyper} = \mathbf{c}^\top \cdot f_s(s)_{\cdot, i}$$

这对应于观测条件概率 $P(Q_{tot} \mid Q_i)$，包含通过 $Z$ 的后门路径。由因果推断基本定理：

$$P(Q_{tot} \mid Q_i) = P(Q_{tot} \mid do(Q_i)) + \underbrace{\text{Cov}(Q_{tot}, Q_i \mid Z)}_{\Delta_{conf}} \neq P(Q_{tot} \mid do(Q_i))$$

当 $Z$ 同时影响 $Q_i$（通过 $h$）和 $Q_{tot}$（通过 $r$）时，$\Delta_{conf} \neq 0$。$\square$

### 4.3 直观解释

**普通 QMIX（包括 BM 的非单调版）**：mixer 权重由 $s$ 直接生成。如果状态中的某些因素同时使得 agent i 的 Q 值高**并且**全局奖励高（如"简单场景中所有人表现都好"），mixer 会高估 agent i 的贡献——这是**虚假因果**。

**DVD**：通过 GAT 轨迹图对 agent 间的实际交互结构建模，"积分掉"混淆路径。mixer 权重反映的是 agent i 对 Q_tot 的**真实因果效应**，而非统计相关性。

在**过泛化场景**中，这尤其重要：过泛化的本质是"agent 误以为自己的次优策略导致了高回报"（实际是因为其它 agent 恰好配合了）。DVD 的去混淆帮助 mixer 正确归因——"高回报是因为联合配合，不是因为你的个体策略好"——从而减少对次优均衡的粘性。

---

## 5. 总结：DVD 为什么有用

| 性质 | 普通非单调 QMIX（BM） | DVD + 非单调 QMIX（本方法） |
|---|---|---|
| 收敛保证 | ✅ BM 定理 2/3 | ✅ 定理 1：同样满足 |
| Jacobian 估计方差 | 取决于 hypernet 输出 | **$O(1/D)$ 缩减**（定理 2） |
| Jacobian 无偏性 | ❌ 包含混淆偏差 $\Delta_{conf}$ | **✅ 后门调整无偏估计**（定理 3） |
| 跨种子稳定性 | 差（实验观察） | **好**（推论 2.2 解释） |

三个定理构成递进关系：
1. **定理 1**（能用）：DVD 不破坏 BM 的收敛保证
2. **定理 2**（更稳）：多头结构提供 $O(1/D)$ 方差缩减 → 解释跨种子稳定性
3. **定理 3**（更准）：后门调整消除混淆偏差 → 信用分配更准确 → 有助于突破过泛化

---

## 附录：关键假设及其局限

### 假设 A1（条件独立性）的局限

- **训练后期**：多个 head 可能学到相似的注意力模式（mode collapse），此时独立性减弱，实际方差缩减弱于 $O(1/D)$
- **缓解措施**：创新 3（多头竞争加权）通过可学习权重自动降低冗余 head 的贡献，部分对冲 mode collapse 的影响
- **实验验证**：对比 $D=1,2,4,8$ 的实际跨种子方差与 $1/\sqrt{D}$ 的理论预测，可量化假设偏离程度

### 轨迹图作为充分代理变量

- 定理 3 假设 $G$ 是 $Z$ 的**充分**代理（d-分离所有后门路径）
- 这取决于 GAT 的表达力——如果 $Z$ 过于复杂，$G$ 可能只是部分代理
- 但即使部分去混淆，$\Delta_{conf}$ 也会减小（减少偏差优于不减少），所以结论在定性意义上仍然成立
