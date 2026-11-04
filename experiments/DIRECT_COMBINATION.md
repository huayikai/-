# Direct BM + trajectory control (2026-11-02)

## 最新状态（2026-11-03）：既定集合结果已齐

3s79/80/81、6h80/84/87/88/90 共八条均覆盖 5M。直接组合在 3s 三颗全部已记录测试为零，6h 四颗有非零胜率、87 全零。最近 5M 单点的种子等权均值：3s BACM/直接为 70.833%/0%，6h 为 77.5%/65%；6h80/84/90 的单点仍为直接组合更高。结果对照整套融合及训练安排，不单独证明门控必要性；6h 为此前筛选的开发集合。详细配对、配置核验及未平滑曲线见 [完整补齐报告](D:/论文/代码/experiments/2026-11-03_DIRECT_COMBINATION_COMPLETE/report.md)。当前下一步按用户决定转向 MMM2 新种子，下方首批表格和命令作为历史追溯。

## 固定的实验定义

这是同 BM 骨架的直接接入对照，配置为 `dvd_audit_direct`，mixer 为 `dvd_direct`。它与历史 `dvd_bm_audit` 分开记录，也不作为原论文 DVD 的严格复现。

令 H 为智能体 GRU 隐藏状态，G_d(H) 为第 d 个图注意力头的输出，M_d(s) 为状态条件超网络输出。直接组合使用有符号原始候选：

```
W_direct(s,H) = mean_d[ G_d(H) M_d(s)^T ]
Q_direct = ELU(q W_direct + b1(s)) W2(s) + b2(s)
```

- b1、W2、b2 的网络结构和初始数值与相同 seed 的 BACM/BM 对齐；GAT 和候选超网络的初始数值与 BACM 对齐。
- 不保留动态门控或 BM 的独立 W1；不做候选 RMS 重缩放、不做零候选回退。零候选时 W1 就是零矩阵。
- 使用单一 masked TD 平方损失。目标为该直接组合网络自身的 target mixer，采用相同的 replay SARSA、修正 TD(lambda)、每次更新 tau=0.01 的软更新及 RND 流程。
- 没有额外 BM 拟合、候选辅助损失、路由监督或稀疏损失。不能通过关闭路由损失、冻结一个旧 gate 来冒充该直接接入结构。
- hidden 路径与各地图的主配置固定一致：3s attached，6h detached；detach 仍保留 Q 输入、GAT 和候选参数的梯度。本轮不单独检验 detach 的作用。
- 训练预算 5.05M；优先查看相同 seed 的原始曲线、最近 5M 评估和末段评估，不采用时间加权胜率。

初始化时临时构造原有 BACM 模块以复现其随机数消耗，然后仅保留上述活动模块。临时门控与未使用 BM W1 不进入最终模型、前向计算或优化器；共同部分的初始化和后续 RND 随机流保持对齐。它不是加载预训练权重。

这个比较检验融合及其损失/目标安排的整体价值，不能单独把差异归因于动态门控。RMS 现有结果保留，本轮不追加 RMS 消融。

## 首批运行（结果已于 2026-11-03 回传）

| GPU | 地图 | seed | hidden 路径 | 运行名称 |
| --- | --- | ---: | --- | --- |
| 0 | 3s5z_vs_3s6z | 80 | attached | baseline_bm_dvd_direct_raw_attached_5m_3s5z_vs_3s6z_s80 |
| 1 | 3s5z_vs_3s6z | 81 | attached | baseline_bm_dvd_direct_raw_attached_5m_3s5z_vs_3s6z_s81 |
| 2 | 6h_vs_8z | 80 | detached | baseline_bm_dvd_direct_raw_detached_5m_6h_vs_8z_s80 |
| 3 | 6h_vs_8z | 84 | detached | baseline_bm_dvd_direct_raw_detached_5m_6h_vs_8z_s84 |

当前代码已直接修改在 `D:/论文/代码`。本目录是后续维护和同步的主版本。使用独立名称，输出仍为 `ablation_results_dvd`。主要源码、配置及指纹由 audit 入口自动保存。首批四条日志均已覆盖 5M；本地 CPU 检查与服务器学习效果分开记录，Sacred 的 RUNNING 状态不用于判断当前进程状态。

最近 5M 原始胜率如下，不插值、不时间加权：

| 地图/seed | 纯 BM | 直接 A+B | 完整 BACM |
| --- | ---: | ---: | ---: |
| 3s/80 | 59.375% | 0% | 68.75% |
| 3s/81 | 75% | 0% | 65.625% |
| 6h/80 | 81.25% | 75% | 62.5% |
| 6h/84 | 65.625% | 96.875% | 90.625% |

首批 3s 两条全部已记录测试为零，6h 两条最近 5M 与末次均高于对应 BACM。这是整套结构及训练安排的对照，不支持完整融合普遍更好，也不能独立证明门控的作用。首批完整末段、真实步数及源码核验见 [首批结果报告](D:/论文/代码/experiments/2026-11-03_DIRECT_COMBINATION_RESULTS/report.md)。后续 3s79、6h87/88/90 现已回传，当前以顶部完整补齐报告为准。6h 既定集合来自此前 BACM 表现筛选，不能当作未筛选种子的总体稳定性估计。RMS 扩展继续暂停。

## 补齐批次（2026-11-03，结果已回传）

保持首批定义及训练参数，GPU 0 对应 3s79 attached，GPU 1/2/3 分别对应 6h87/88/90 detached，预算均为 5.05M。四条均已收到覆盖 5M 的日志，24 项共同训练设置及源码指纹核验通过。[本批四条命令](D:/论文/代码/experiments/2026-11-03_DIRECT_COMBINATION_NEXT/commands.md) 保留作存档，不表示需要重跑。助手未启动训练；本次没有修改训练代码或添加参数变体。

## 本地维护与服务器同步

本地 `D:/论文/代码` 对应服务器 `/home/zhangbei/pymarl2/src`。运行前将本目录的最新源码同步到服务器，重点包括：

| 本地文件 | 作用 |
| --- | --- |
| [main_audit.py](D:/论文/代码/main_audit.py) | 接受新配置，记录新 mixer 源码 |
| [dvd_audit_learner.py](D:/论文/代码/learners/dvd_audit_learner.py) | 接入 direct 的在线/目标前向及标记 |
| [dvd_direct.py](D:/论文/代码/modules/mixers/dvd_direct.py) | 同骨架直接轨迹权重 |
| [dvd_audit_direct.yaml](D:/论文/代码/config/algs/dvd_audit_direct.yaml) | 固定共同训练参数 |
| [_test_dvd_direct.py](D:/论文/代码/_test_dvd_direct.py) | 新对照的 CPU 验证 |

原始训练结果仍保留在现有三个结果根目录，本目录只保存整理后的实验日志与引用。10 月 26 日以来的结果详见 [阶段实验日志](D:/论文/代码/experiments/2026-10-26_to_2026-11-02/report.md)。旧补丁包和安装器只是此前交付的可选副本，后续以本目录源码为准。

## 首批启动命令存档（已有回传结果）

以下为已回传首批结果对应的命令，保留以便追溯，不表示需要重复运行。每个服务器终端先进入环境，之后在四个独立终端分别执行对应的一条命令：

```bash
cd /home/zhangbei/pymarl2
conda activate pymarl
```

GPU 0，3s seed80：

```bash
CUDA_VISIBLE_DEVICES=0 python3 src/main_audit.py --config=dvd_audit_direct --env-config=sc2 with env_args.map_name=3s5z_vs_3s6z seed=80 name=baseline_bm_dvd_direct_raw_attached_5m_3s5z_vs_3s6z_s80 audit_detach_mixed_hidden=False t_max=5050000 local_results_path=ablation_results_dvd
```

GPU 1，3s seed81：

```bash
CUDA_VISIBLE_DEVICES=1 python3 src/main_audit.py --config=dvd_audit_direct --env-config=sc2 with env_args.map_name=3s5z_vs_3s6z seed=81 name=baseline_bm_dvd_direct_raw_attached_5m_3s5z_vs_3s6z_s81 audit_detach_mixed_hidden=False t_max=5050000 local_results_path=ablation_results_dvd
```

GPU 2，6h seed80：

```bash
CUDA_VISIBLE_DEVICES=2 python3 src/main_audit.py --config=dvd_audit_direct --env-config=sc2 with env_args.map_name=6h_vs_8z seed=80 name=baseline_bm_dvd_direct_raw_detached_5m_6h_vs_8z_s80 audit_detach_mixed_hidden=True t_max=5050000 local_results_path=ablation_results_dvd
```

GPU 3，6h seed84：

```bash
CUDA_VISIBLE_DEVICES=3 python3 src/main_audit.py --config=dvd_audit_direct --env-config=sc2 with env_args.map_name=6h_vs_8z seed=84 name=baseline_bm_dvd_direct_raw_detached_5m_6h_vs_8z_s84 audit_detach_mixed_hidden=True t_max=5050000 local_results_path=ablation_results_dvd
```

实现阶段已通过 31 项 CPU 检查，1 项因本地缺少 Sacred 跳过。首批 SC2 训练日志已回传并单独核验；本次结果整理没有修改训练源码、原始实验日志或论文正文。完整配置可用 `--audit-print-config` 检查。
