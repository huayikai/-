# Direct BM + trajectory control (2026-11-02)

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

## 首批预定运行

| GPU | 地图 | seed | hidden 路径 | 运行名称 |
| --- | --- | ---: | --- | --- |
| 0 | 3s5z_vs_3s6z | 80 | attached | baseline_bm_dvd_direct_raw_attached_5m_3s5z_vs_3s6z_s80 |
| 1 | 3s5z_vs_3s6z | 81 | attached | baseline_bm_dvd_direct_raw_attached_5m_3s5z_vs_3s6z_s81 |
| 2 | 6h_vs_8z | 80 | detached | baseline_bm_dvd_direct_raw_detached_5m_6h_vs_8z_s80 |
| 3 | 6h_vs_8z | 84 | detached | baseline_bm_dvd_direct_raw_detached_5m_6h_vs_8z_s84 |

当前代码已直接修改在 `D:/论文/代码`。本目录是后续维护和同步的主版本。使用独立名称，输出仍为 `ablation_results_dvd`。主要源码、配置及指纹由 audit 入口自动保存。实验结果仍待服务器训练后回传，本地 CPU 检查不是 SC2 学习效果验证。

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

## 启动命令

每个服务器终端先进入环境，之后在四个独立终端分别执行对应的一条命令：

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

当前已完成 31 项 CPU 检查，1 项因本地缺少 Sacred 跳过；实际 SC2 训练待用户在服务器启动。没有修改旧 mixer 的计算、原始实验日志或论文正文。完整配置可用 `--audit-print-config` 检查。
