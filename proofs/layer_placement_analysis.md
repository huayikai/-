# DVD 去混淆的层次放置分析：为什么第一层优于第二层

> 本文档从**表达力/信息瓶颈**的角度，论证 CA-DVD 将 GAT 去混淆放在**第一层**（作用于原始 agent Q）相对于 DVD 原文放在**第二层**（作用于混合后的 credits）的优势。结论与 `theoretical_analysis.md` 定理 3（后门调整的无偏性）互补：定理 3 说明"去混淆消除偏差"，本文说明"去混淆必须放在第一层才能保留 per-agent 的针对性"。
>
> 代码对应：第一层版 `dvd.py: _forward_first`，第二层版 `dvd.py: _forward_second`（`dvd_layer` 开关）。

---

## 1. 问题设定

DVD 的核心是用轨迹图 $G$ 作为代理混淆变量，对信用分配做后门调整。一个被原文忽略的实现自由度是：**GAT 特征注入到 mixer 的哪一层**。

- **原文（第二层）**：第一层权重 $W_{\text{plain}}=|f_w(s)|$ 是纯状态的；GAT 用于生成第二层 credits $K=|f_s(s)G|$（附录 A 公式 16–19）。
- **CA-DVD（第一层）**：GAT 用于生成第一层权重 $W_1=f_s(s)G^\top$，直接作用在原始 agent Q 上；第二层 $w_{\text{final}}$ 是纯状态的。

用户的直觉是："第一层混合之后，各 agent 的原始信息已经融合，此时再去混淆效果不好。" 下面把这个直觉严格化。

### 符号

| 符号 | 含义 |
|---|---|
| $q=(Q_1,\dots,Q_N)^\top$ | 各 agent 选定动作的局部 Q 值 |
| $G=[g_1;\dots;g_N]\in\mathbb{R}^{N\times p}$ | GAT 图特征，$g_i\in\mathbb{R}^p$ 是 agent $i$ 的节点特征，$p=$ `gat_dim` |
| $m$ | `mixing_embed_dim` |
| $f_s(s),f_w(s),f_{\text{fin}}(s)$ | 状态超网络 |
| $\sigma$ | ELU 激活，$\sigma'(\cdot)\in(0,1]$ |
| $\bar g=\text{pool}(G)$ | 图级池化特征（第二层版必需，见 §2.2） |

为聚焦核心，下面以单 head（$D=1$）、省略 $|\cdot|$ 退火写出；多头均值与 abs 不改变 per-agent vs 全局的结论。

---

## 2. 两种结构的前向与 Jacobian

信用分配的本质由 Jacobian $J_i=\partial Q_{tot}/\partial Q_i$ 刻画——它是 mixer 分给 agent $i$ 的**有效信用权重**。去混淆是否"针对 agent $i$"，就看 $J_i$ 是否由 agent $i$ 专属的图特征 $g_i$ 调制。

### 2.1 第一层版（CA-DVD，`_forward_first`）

$$W_1 = f_s(s)\,G^\top\in\mathbb{R}^{m\times N},\qquad [W_1]_{:,i}=f_s(s)\,g_i$$

$$\text{hidden}=\sigma(W_1 q + b_1),\qquad Q_{tot}=w_{\text{final}}^\top\,\text{hidden}+V(s),\quad w_{\text{final}}=f_{\text{fin}}(s)$$

Jacobian：

$$\boxed{\,J_i^{\text{(1)}}=w_{\text{final}}^\top\,\mathrm{diag}(\sigma')\,[W_1]_{:,i}=w_{\text{final}}^\top\,\mathrm{diag}(\sigma')\,f_s(s)\,g_i\,}$$

**关键**：$J_i^{\text{(1)}}$ 显式依赖 agent $i$ 专属的 $g_i$。不同 agent 的信用权重被各自的图特征独立调制。

### 2.2 第二层版（原文式，`_forward_second`）

第一层纯状态混合（不含 $G$）：

$$\text{hidden}=\sigma(W_{\text{plain}}q+b_1),\qquad W_{\text{plain}}=f_w(s)\in\mathbb{R}^{m\times N}$$

第二层 credits 由 $G$ 生成。**注意维度**：credits 是 $m$ 维（作用在 $m$ 个 inter values 上），**没有 agent 维**，而 $G\in\mathbb{R}^{N\times p}$ 有 agent 维。因此必须先把 $N$ 个节点聚合成图级表示 $\bar g=\text{pool}(G)\in\mathbb{R}^p$：

$$w_{\text{final}}=f_s(s)\,\bar g\in\mathbb{R}^{m}$$

$$Q_{tot}=w_{\text{final}}^\top\,\text{hidden}+V(s)$$

Jacobian：

$$\boxed{\,J_i^{\text{(2)}}=w_{\text{final}}^\top\,\mathrm{diag}(\sigma')\,[W_{\text{plain}}]_{:,i}=\underbrace{r(s,G)^\top}_{\text{对所有 }i\text{ 相同}}\,[W_{\text{plain}}]_{:,i}\,}$$

其中 $r(s,G)^\top:=w_{\text{final}}^\top\mathrm{diag}(\sigma')$ 是一个**与 $i$ 无关**的公共行向量，而 $[W_{\text{plain}}]_{:,i}$ **不含 $G$**。

图信息 $G$ 只通过公共的 $r(s,G)$ 进入所有 $J_i^{\text{(2)}}$。

---

## 3. 定理 L1：第一层可实现 per-agent 异质去混淆

**定理 L1.** *设 GAT 满秩（$p\ge N$ 且注意力使各节点输出泛型不同）。则第一层版的信用权重族 $\{J_i^{\text{(1)}}\}_{i=1}^N$ 的 $G$-依赖方向泛型线性无关：可以独立地调制每个 agent 的信用。*

**证明.** $J_i^{\text{(1)}}=c(s)^\top f_s(s)\,g_i$，其中 $c(s)^\top=w_{\text{final}}^\top\mathrm{diag}(\sigma')$。$G$ 对 $J_i^{\text{(1)}}$ 的影响完全由 $g_i$（$G$ 的第 $i$ 行）承载。GAT 的 softmax 注意力对不同节点产生不同聚合，故 $\{g_i\}_{i=1}^N$ 泛型线性无关；经满秩线性映射 $f_s(s)$ 后仍然如此。因此改变 $g_i$ 只影响 $J_i^{\text{(1)}}$ 而不牵连 $J_{j\ne i}$，即各 agent 的去混淆修正可独立施加。$\square$

---

## 4. 定理 L2：第二层的去混淆瓶颈

**定理 L2.** *第二层版的信用权重满足*

$$J_i^{\text{(2)}}=r(s,G)^\top[W_{\text{plain}}]_{:,i},$$

*其中 $r(s,G)$ 与 $i$ 无关、$[W_{\text{plain}}]_{:,i}$ 不含 $G$。因此 $G$ 对所有 agent 的信用影响被同一个因子 $r(s,G)$ 捆绑——第二层无法表达随 agent 变化的去混淆修正 $c_i(G)$。*

**证明.** 由 §2.2，$w_{\text{final}}=f_s(s)\bar g$ 是**单个**向量，故 $r(s,G)^\top=w_{\text{final}}^\top\mathrm{diag}(\sigma')$ 对所有 $i$ 是同一行向量。对任意两 agent $i\ne j$：

$$\frac{\partial J_i^{\text{(2)}}}{\partial G}=\Big(\frac{\partial r}{\partial G}\Big)^\top[W_{\text{plain}}]_{:,i},\qquad \frac{\partial J_j^{\text{(2)}}}{\partial G}=\Big(\frac{\partial r}{\partial G}\Big)^\top[W_{\text{plain}}]_{:,j}.$$

二者共享同一个 $\partial r/\partial G$。因此 $G$ 无法对 $i$ 和 $j$ 施加**方向不同**的调制——它只能通过公共标量场对所有 agent 做同向缩放。要实现 per-agent 异质修正 $c_i(G)\ne c_j(G)$，唯一的 $i$-依赖来自 $[W_{\text{plain}}]_{:,i}$，而它不含 $G$，无法承载去混淆信息。$\square$

**注（池化并非问题的根源，而是症状）.** 即使不池化、用 flatten 保留全部 $G$ 来生成 $w_{\text{final}}$，$w_{\text{final}}$ 仍是作用在 $\text{hidden}$ 上的**单一** $m$ 维向量；而 $\text{hidden}=\sigma(W_{\text{plain}}q)$ 已把各 agent 线性混合。去混淆作用的对象（inter values）本身已无 per-agent 结构，故无法解耦回单个 $Q_i$。**这正是"第一层混合后信息已融合、再去混淆无从针对"的严格表述。**

---

## 5. 推论：与相对过泛化的对症性

**相对过泛化的机制**：agent $i$ 因其他 agent 恰好配合而误将高回报归因于自己的次优动作，需要一个**针对 agent $i$** 的信用下调 $c_i(G)<0$ 来纠正虚假信用。

| | 能否表达 $c_i(G)$（per-agent 去混淆） | 对过泛化 |
|---|---|---|
| 第一层版 | ✅ 由 $g_i$ 独立调制 $J_i$（定理 L1） | **对症**：可单独下调 agent $i$ 的虚假信用 |
| 第二层版 | ❌ 只能全局同向调制（定理 L2） | 无法定位到 agent $i$，去混淆退化为全局缩放 |

因此在 DVD 的核心目标（per-agent 去混淆信用分配）上，第一层结构**严格更具表达力**，且这一优势恰好落在过泛化最需要的地方。

---

## 6. 诚实边界

1. **表达力 ≠ 性能**。定理 L1/L2 证明的是第一层在 per-agent 去混淆上的表达力严格更强；更强表达力也可能带来更难优化或过拟合。最终性能由 layer-placement 消融实验判定。
2. **第二层版并非"无去混淆"**。它 $=$ 标准 QMIX 第一层 $+$ 全局去混淆第二层，表达力介于纯 BM 与第一层 DVD 之间——保留了 per-agent 的"混合"能力，只是丧失 per-agent 的"去混淆"能力。
3. **前提依赖**。结论建立在"去混淆需要 per-agent"之上。若某任务的混淆是全局同质的（所有 agent 混淆强度相同），第一层优势消失、两者等价。过泛化任务通常 per-agent 异质，故第一层占优。

---

## 7. 可证伪预测（接实验）

- **P1**：layer-placement 消融中，第一层版在 per-agent 信用异质性强的任务（Stag Hunt 惩罚版、SMAC `3s5z_vs_3s6z` 的"牺牲一体、诱敌"策略）上应显著优于第二层版；在混淆同质任务上两者接近。
- **P2**：若实测第二层版性能 $\approx$ 纯 BM，说明其第二层去混淆退化为"无效全局调制"（印证定理 L2 的瓶颈）。
- **P3**：可视化 Jacobian $\{J_i\}$ 的跨 agent 方差——第一层版应显著大于第二层版（后者被公共 $r(s,G)$ 压平）。

---

## 8. 小结

| | 第一层版（CA-DVD） | 第二层版（原文式） |
|---|---|---|
| GAT 注入 | 第一层 $W_1$（原始 agent Q） | 第二层 credits（混合后 inter values） |
| Jacobian $J_i$ 的 $G$ 依赖 | agent $i$ 专属 $g_i$ | 公共 $r(s,G)$，$i$-无关 |
| per-agent 去混淆 | ✅（定理 L1） | ❌ 瓶颈（定理 L2） |
| 对相对过泛化 | 对症 | 全局调制、难定位 |

**一句话**：去混淆要对单个 agent 生效，就必须在 agent 信息被混合之前（第一层）注入；过了第一层，agent 信息已纠缠，第二层的去混淆只能全局施力。这把用户"第一层更好"的直觉，落成了 Jacobian 层面的表达力定理。
