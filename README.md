# DVD + 非单调 QMIX（去 IGM 约束的因果值分解）

基于 PyMARL2 框架，实现 **DVD（ICML 2022）因果去混淆值分解** 与 **Beyond Monotonicity（AAAI 2026）非单调 QMIX** 的组合方法，研究去除 IGM 约束后因果后门调整对缓解多智能体相对过泛化问题的效果。

## 核心思路

| 组件 | 来源 | 作用 |
|---|---|---|
| 多头 GAT 轨迹图 + 后门调整 | DVD | 去除值分解中的混淆偏差，实现因果信用分配 |
| `abs=False` 去单调约束 | Beyond Monotonicity | 允许 mixer 表达非单调 Q 值景观 |
| SARSA 目标（非 max） | Beyond Monotonicity | 避免非单调 mixer 下的 Q 值高估 |
| TD(λ) | Beyond Monotonicity | 平滑多步回报，稳定非单调学习 |
| RND 内在奖励（归一化 + 衰减） | Beyond Monotonicity | 驱动探索未知状态，打破次优均衡 |

## 项目结构

```
├── main.py                          # 入口（Sacred 实验框架）
├── config/
│   ├── default.yaml                 # 全局默认配置
│   ├── algs/
│   │   ├── dvd+qmix_without_abs.yaml   # ★ 主实验：DVD × 非单调 QMIX
│   │   ├── qmix_without_abs.yaml        # 对照：纯非单调 QMIX（BM 复现）
│   │   ├── dvd.yaml                     # 对照：原始 DVD（单调）
│   │   ├── qmix.yaml                    # 基线：标准 QMIX
│   │   └── ...                          # 其它算法配置
│   └── envs/
│       ├── sc2.yaml                 # StarCraft II (SMAC)
│       ├── stag_hunt.yaml           # Stag Hunt（过泛化测试环境）
│       └── ...
├── learners/
│   ├── dvd_nq_learner_with_sarsa.py # ★ 主实验 Learner（DVD + SARSA + TD(λ) + RND）
│   ├── nq_learner_with_sarsa.py     # 对照 Learner（纯 BM：SARSA + RND）
│   ├── dvd_learner.py               # 原始 DVD Learner
│   └── ...
├── modules/
│   ├── mixers/
│   │   ├── dvd.py                   # ★ DVDMixer（多头 GAT + 超网络权重生成，支持 abs=True/False）
│   │   ├── nmix.py                  # 非单调 Mixer（BM 对照用）
│   │   └── ...
│   ├── exploration/
│   │   └── rnd.py                   # RND 内在奖励模块
│   └── agents/                      # 各类 Agent 网络（RNN / MLP 等）
├── controllers/                     # MAC（Multi-Agent Controller）
├── runners/
│   ├── parallel_runner.py           # 并行采样器
│   └── episode_runner.py            # 单线程采样器
├── components/                      # Episode Buffer、动作选择器等基础组件
├── envs/                            # 环境封装（SMAC、Stag Hunt、Matrix Game）
├── run/                             # 训练主循环
└── utils/                           # 工具函数（TD(λ)、日志等）
```

## 算法配置说明

### 主实验：`dvd+qmix_without_abs`

```yaml
learner: dvd_nq_learner_with_sarsa   # DVD 结构 + SARSA 目标
mixer: dvd                            # DVDMixer（GAT 轨迹图）
abs: False                            # 去除单调约束
td_lambda: 0.3                        # TD(λ) 平滑
use_rnd: True                         # RND 探索
rnd_beta: 0.01                        # 内在奖励权重（线性衰减到 0）
dvd_heads: 8                          # GAT 多头数（后门调整采样次数）
```

### 对照组

| 配置文件 | Learner | Mixer | 单调 | RND | 定位 |
|---|---|---|---|---|---|
| `dvd+qmix_without_abs.yaml` | `dvd_nq_learner_with_sarsa` | DVDMixer | 否 | 是 | **主实验** |
| `qmix_without_abs.yaml` | `nq_learner_with_sarsa` | Mixer | 否 | 是 | BM 复现 |
| `dvd.yaml` | `dvd_learner` | DVDMixer | 是 | 否 | DVD 复现 |
| `qmix.yaml` | `q_learner` | QMIX | 是 | 否 | 标准基线 |

## 快速启动

### 环境依赖

```bash
# Python 3.8+, PyTorch, Sacred, SMAC
pip install torch sacred numpy pyyaml tensorboard
pip install git+https://github.com/oxwhirl/smac.git
```

### 运行主实验（DVD + 非单调 QMIX）

```bash
# SMAC 地图
python main.py --config=dvd+qmix_without_abs --env-config=sc2 with env_args.map_name=3s5z_vs_3s6z

# Stag Hunt（过泛化测试）
python main.py --config=dvd+qmix_without_abs --env-config=stag_hunt

# Matrix Game
python main.py --config=dvd+qmix_without_abs --env-config=one_step_matrix_game
```

### 运行对照实验

```bash
# 纯非单调 QMIX（BM 复现）
python main.py --config=qmix_without_abs --env-config=sc2 with env_args.map_name=3s5z_vs_3s6z

# 原始 DVD
python main.py --config=dvd --env-config=sc2 with env_args.map_name=3s5z_vs_3s6z

# 标准 QMIX 基线
python main.py --config=qmix --env-config=sc2 with env_args.map_name=3s5z_vs_3s6z
```

### 常用参数覆盖

```bash
# 指定种子
python main.py --config=dvd+qmix_without_abs --env-config=sc2 with env_args.map_name=3s5z_vs_3s6z seed=42

# 调整 DVD 多头数（消融实验）
python main.py --config=dvd+qmix_without_abs --env-config=sc2 with env_args.map_name=3s5z_vs_3s6z dvd_heads=1

# 关闭 RND（消融实验）
python main.py --config=dvd+qmix_without_abs --env-config=sc2 with env_args.map_name=3s5z_vs_3s6z use_rnd=False

# 使用 GPU
python main.py --config=dvd+qmix_without_abs --env-config=sc2 with env_args.map_name=3s5z_vs_3s6z use_cuda=True
```

### 查看训练结果

训练日志保存在 `results/` 目录，可用 TensorBoard 查看：

```bash
tensorboard --logdir=results/tb_logs
```

## 参考文献

- **DVD**: Wen et al., *Deconfounded Value Decomposition for Multi-Agent Reinforcement Learning*, ICML 2022
- **Beyond Monotonicity**: *Revisiting Factorization Principles in Multi-Agent Q-Learning*, AAAI 2026 (Oral)
