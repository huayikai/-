# CA-DVD：因果自适应值分解（Causal-Adaptive DVD）

基于 PyMARL2 框架，在 DVD（ICML 2022）和 Beyond Monotonicity（AAAI 2026）的基础上，提出三个创新机制，使因果去混淆与非单调值分解产生深度耦合，解决多智能体相对过泛化问题。

## 创新点

### 创新 1：渐进式单调约束退火（Progressive Monotonicity Annealing）

**问题**：直接去掉单调约束（abs=False）导致优化自由度过大，收敛缓慢；但全程保持单调约束又会被锁死在次优解。

**方案**：训练初期保持单调约束（利用 IGM 的归纳偏置快速建立粗略 Q 值景观），随训练逐步退火到完全非单调，让 mixer 突破局部最优。

```
abs_weight = max(0, 1 - t_env / anneal_steps)
w1 = abs_weight * |w1| + (1 - abs_weight) * w1
```

**配置**：
```yaml
use_abs_anneal: True          # 开关
abs_anneal_steps: 2000000     # 退火步数（在此步数时完全非单调）
```

### 创新 2：因果感知探索（Causal-Aware Exploration）

**问题**：RND 探索的是"状态新颖性"，与因果结构无关——它不知道哪些区域的 agent 间因果关系尚未被 GAT 建模清楚。

**方案**：利用 GAT 的 attention 分布熵作为额外探索信号。Attention 均匀（熵高）说明模型尚未搞清 agent 间的因果依赖，需要更多探索；attention 集中（熵低）说明因果结构已清晰。

```
H(attention) = -Σ α_ij * log(α_ij)
r_explore = β_rnd * r_rnd + β_causal * H(attention)
```

形成**闭环**：GAT 学习因果结构 → attention 熵指导探索 → 探索反馈改善 GAT 学习。

**配置**：
```yaml
use_causal_explore: True      # 开关
causal_explore_beta: 0.01     # 因果探索奖励权重
```

### 创新 3：多头竞争信用分配（Competitive Multi-Head Credit Assignment）

**问题**：DVD 原论文中 D 个 head 取简单均值，假设所有因果假说同等重要。但在过泛化场景中，某些 head 可能捕捉到了关键的协调结构，另一些则是噪声。

**方案**：引入可学习的 head 权重（softmax 归一化），让模型自动发现哪个因果假说（哪张因果图）对值分解最有用。

```python
head_weights = softmax(head_logits)   # 可学习参数
w1 = Σ_d head_weights[d] * w1_heads[d]   # 加权求和替代均值
```

**配置**：
```yaml
use_head_competition: True    # 开关
```

## 项目结构

```
├── main.py                              # 入口（Sacred 实验框架）
├── config/algs/
│   ├── dvd+qmix_without_abs_v2.yaml     # ★ 新方法配置（三个创新全开）
│   ├── dvd+qmix_without_abs.yaml        # 基础 DVD+BM（无创新机制）
│   ├── qmix_without_abs.yaml            # 对照：纯 BM
│   ├── dvd.yaml                         # 对照：原始 DVD
│   └── qmix.yaml                        # 基线：标准 QMIX
├── modules/mixers/
│   └── dvd.py                           # ★ DVDMixer（含三个创新机制实现）
├── learners/
│   └── dvd_nq_learner_with_sarsa.py     # ★ 主 Learner（对接创新机制）
├── modules/exploration/rnd.py           # RND 内在奖励
├── config/envs/                         # 环境配置（SMAC、Stag Hunt 等）
├── controllers/                         # Multi-Agent Controller
├── runners/                             # 并行/单线程采样器
├── components/                          # Buffer、动作选择器等
├── envs/                                # 环境封装
└── utils/                               # 工具函数
```

## 快速启动

### 环境依赖

```bash
pip install torch sacred numpy pyyaml tensorboard
pip install git+https://github.com/oxwhirl/smac.git
```

### 运行新方法（三个创新全开）

```bash
# SMAC 地图
python main.py --config=dvd+qmix_without_abs_v2 --env-config=sc2 with env_args.map_name=3s5z_vs_3s6z

# Stag Hunt（过泛化测试）
python main.py --config=dvd+qmix_without_abs_v2 --env-config=stag_hunt

# Matrix Game
python main.py --config=dvd+qmix_without_abs_v2 --env-config=one_step_matrix_game
```

### 消融实验（单独开关每个创新）

```bash
# 只开渐进式退火
python main.py --config=dvd+qmix_without_abs_v2 --env-config=sc2 with \
    env_args.map_name=3s5z_vs_3s6z use_causal_explore=False use_head_competition=False

# 只开因果感知探索
python main.py --config=dvd+qmix_without_abs_v2 --env-config=sc2 with \
    env_args.map_name=3s5z_vs_3s6z use_abs_anneal=False use_head_competition=False

# 只开多头竞争
python main.py --config=dvd+qmix_without_abs_v2 --env-config=sc2 with \
    env_args.map_name=3s5z_vs_3s6z use_abs_anneal=False use_causal_explore=False

# dvd_heads 消融（验证方差缩减定理）
python main.py --config=dvd+qmix_without_abs_v2 --env-config=sc2 with \
    env_args.map_name=3s5z_vs_3s6z dvd_heads=1
python main.py --config=dvd+qmix_without_abs_v2 --env-config=sc2 with \
    env_args.map_name=3s5z_vs_3s6z dvd_heads=4
```

### 对照实验

```bash
# 基础 DVD+BM（无创新机制）
python main.py --config=dvd+qmix_without_abs --env-config=sc2 with env_args.map_name=3s5z_vs_3s6z

# 纯 BM
python main.py --config=qmix_without_abs --env-config=sc2 with env_args.map_name=3s5z_vs_3s6z

# 原始 DVD
python main.py --config=dvd --env-config=sc2 with env_args.map_name=3s5z_vs_3s6z

# 标准 QMIX
python main.py --config=qmix --env-config=sc2 with env_args.map_name=3s5z_vs_3s6z
```

### 常用参数

```bash
seed=42                    # 随机种子
use_cuda=True              # GPU 训练
abs_anneal_steps=3000000   # 调整退火速度
causal_explore_beta=0.05   # 调整因果探索强度
rnd_beta=0.05              # 调整 RND 强度
```

### 查看训练结果

```bash
tensorboard --logdir=results/tb_logs
```

新增 TensorBoard 监控指标：
- `attention_entropy`：GAT attention 熵（反映因果结构学习进度）
- `abs_weight`：当前单调约束权重（从 1 退火到 0）
- `head_weight_0` ~ `head_weight_7`：各 head 的竞争权重

## 配置开关速查

| 配置项 | 默认值 | 说明 |
|---|---|---|
| `use_abs_anneal` | `False` | 渐进式 abs 退火开关 |
| `abs_anneal_steps` | `2000000` | 退火完成步数 |
| `use_causal_explore` | `False` | 因果感知探索开关 |
| `causal_explore_beta` | `0.01` | 因果探索奖励权重 |
| `use_head_competition` | `False` | 多头竞争开关 |
| `use_rnd` | `True` | RND 探索开关 |
| `dvd_heads` | `8` | GAT 多头数 |
| `abs` | `False` | 全局单调约束（退火模式下被覆盖） |

## 参考文献

- **DVD**: Wen et al., *Deconfounded Value Decomposition for Multi-Agent Reinforcement Learning*, ICML 2022
- **Beyond Monotonicity**: *Revisiting Factorization Principles in Multi-Agent Q-Learning*, AAAI 2026 (Oral)
