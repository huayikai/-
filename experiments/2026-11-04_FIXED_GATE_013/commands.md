# 2026-11-04：固定门控 0.13 首批四条

状态更新（2026-11-05）：首批四条均已回传覆盖5M的结果，实际门控固定行为已核验，详见 [结果报告](D:/论文/代码/experiments/2026-11-05_FIXED_GATE_013_RESULTS/report.md)。以下首批命令保留追溯，不表示需要重跑；助手未启动 SC2 训练，本轮沿用现有源码。

用户最新要求：不同种子方向不一致时先不急着下结论，后续需要更多种子验证。当前只记录逐seed表现，先补齐本文件原先预定的3s79、6h87/88/90；再按预先固定的新集合做独立配对，所有方向均保留。

2026-11-05 后续补齐更新：3s79、6h87/88/90的四条现已回传，既定八条已齐，见 [完整补齐报告](D:/论文/代码/experiments/2026-11-05_FIXED_GATE_013_COMPLETE/report.md)。下一阶段扩大独立种子验证，本文件命令与首批状态作为历史追溯。

2026-11-04 范围确认：用户决定先完成现有两图，MMM2 留待后续具体修改意见再评估。本批四条及随后预定的剩余四条继续按本文件定义安排，无需因该范围决定修改命令或重启已有训练。

## 目的与固定值依据

检验固定近似后段融合比例的简单混合能否达到动态门控的表现。取已有既定开发集合的 masked counterfactual_gate_mean，在 [4M,5M) 内先对每条运行的已记录 batch 均值作算术平均，再对种子等权平均，统一四舍五入到两位小数：

| 地图 | 动态 4–5M 门控均值 | 动态 0–5M 门控均值 | 本轮固定值 |
| --- | ---: | ---: | ---: |
| 3s5z_vs_3s6z | 0.1316316 | 0.1226982 | 0.13 |
| 6h_vs_8z | 0.1321846 | 0.1250958 | 0.13 |

这只是近似后段融合量控制，不是整个训练期或逐阶段精确均值匹配。平均的是日志中的 masked batch gate 均值，不是所有 replay transition 按数量汇总的总体均值，也不是胜率加权。统计来源及逐 seed、逐阶段记录见 [gate_calibration.json](D:/论文/代码/experiments/2026-11-04_FIXED_GATE_013/gate_calibration.json)，共有 Sacred/TensorBoard gate 记录一致。

## 控制设置

- adaptive_gate_init=0.13、counterfactual_route_loss_weight=0、counterfactual_sparse_weight=0。原混合值对 gate 已停梯度，门控保持初始常数，不是只设初值后继续学习。
- 保留原有零候选信号的 BM 回退：无有效轨迹信号时实际 gate 为 0；其他样本的名义固定 gate 为 0.13。
- 保留 learned 图聚合、RMS、BM 自举目标、BM/混合损失比例 1:0.25、候选辅助损失 0.1、辅助 hidden 隔离及 RND。
- 3s attached、6h detached；训练 5.05M，输出 ablation_results_dvd。共同训练配置与各自配对动态运行一致，训练处理差异为 gate 初值 0.05→0.13 与路由监督系数 0.1→0。
- 固定从 0.13 开始也改变了动态门控原本 0.05 的初始先验和训练时间轨迹；本轮比较固定比例与动态机制的整套配置，不能单独把差异归因于逐样本自适应或路由损失。

## 本地核验

四个对应 seed/hidden/梯度日志设置在真实 learner 的小型 CPU 合成 batch 上各更新三次：门控参数逐位不变、门控梯度为零，agent 与候选参数更新且有限，非零轨迹信号 gate 为 0.13，零信号仍回退到 0。除门控末层 bias 外，共同网络初始化与动态版本相同，后续 Torch RNG 状态一致，RND 开启。该检查验证控制行为，不验证 SC2 学习效果。见 [cpu_check.json](D:/论文/代码/experiments/2026-11-04_FIXED_GATE_013/cpu_check.json) 和 [有效配置](D:/论文/代码/experiments/2026-11-04_FIXED_GATE_013/config_check.json)。

## 首批运行

| GPU | 地图 | seed | hidden | 梯度日志 |
| --- | --- | ---: | --- | --- |
| 0 | 3s5z_vs_3s6z | 80 | attached | False |
| 1 | 3s5z_vs_3s6z | 81 | attached | True |
| 2 | 6h_vs_8z | 80 | detached | True |
| 3 | 6h_vs_8z | 84 | detached | True |

梯度日志开关沿用各配对动态运行，避免把观察设置变化混入本批配置。本批没有新增训练源码，沿用上一批服务器代码。四张卡空闲后，在四个独立终端执行；每个终端先运行：

```bash
cd /home/zhangbei/pymarl2
conda activate pymarl
```

GPU 0，3s5z_vs_3s6z seed80：

```bash
CUDA_VISIBLE_DEVICES=0 python3 src/main_audit.py --config=dvd_audit_router --env-config=sc2 with env_args.map_name=3s5z_vs_3s6z seed=80 name=ablation_bacm_fixedgate013_attached_5m_3s5z_vs_3s6z_s80 adaptive_gate_init=0.13 counterfactual_route_loss_weight=0.0 counterfactual_sparse_weight=0.0 audit_detach_mixed_hidden=False audit_log_agent_gradients=False use_tensorboard=True t_max=5050000 local_results_path=ablation_results_dvd
```

GPU 1，3s5z_vs_3s6z seed81：

```bash
CUDA_VISIBLE_DEVICES=1 python3 src/main_audit.py --config=dvd_audit_router --env-config=sc2 with env_args.map_name=3s5z_vs_3s6z seed=81 name=ablation_bacm_fixedgate013_attached_5m_3s5z_vs_3s6z_s81 adaptive_gate_init=0.13 counterfactual_route_loss_weight=0.0 counterfactual_sparse_weight=0.0 audit_detach_mixed_hidden=False audit_log_agent_gradients=True use_tensorboard=True t_max=5050000 local_results_path=ablation_results_dvd
```

GPU 2，6h_vs_8z seed80：

```bash
CUDA_VISIBLE_DEVICES=2 python3 src/main_audit.py --config=dvd_audit_router --env-config=sc2 with env_args.map_name=6h_vs_8z seed=80 name=ablation_bacm_fixedgate013_detached_5m_6h_vs_8z_s80 adaptive_gate_init=0.13 counterfactual_route_loss_weight=0.0 counterfactual_sparse_weight=0.0 audit_detach_mixed_hidden=True audit_log_agent_gradients=True use_tensorboard=True t_max=5050000 local_results_path=ablation_results_dvd
```

GPU 3，6h_vs_8z seed84：

```bash
CUDA_VISIBLE_DEVICES=3 python3 src/main_audit.py --config=dvd_audit_router --env-config=sc2 with env_args.map_name=6h_vs_8z seed=84 name=ablation_bacm_fixedgate013_detached_5m_6h_vs_8z_s84 adaptive_gate_init=0.13 counterfactual_route_loss_weight=0.0 counterfactual_sparse_weight=0.0 audit_detach_mixed_hidden=True audit_log_agent_gradients=True use_tensorboard=True t_max=5050000 local_results_path=ablation_results_dvd
```

## 已预定补齐及判读

2026-11-05：剩余四条的 [正式命令与核验](D:/论文/代码/experiments/2026-11-05_FIXED_GATE_013_NEXT/commands.md) 对应结果现已收到，保持本文件定义并通过核验；既定八条已齐，助手未启动训练，命令不需要原样重跑。

首批后按同一定义补 3s79、6h87/88/90，总计既定八条固定门控运行；不根据首批方向决定是否补齐。动态性能复用既有配对结果，不重跑挑优。6h 集合仍为此前筛选的开发集合，新独立验证另列。

查看距离 5M 最近的一次未平滑评估、末段若干评估和原始曲线，不插值、不时间加权。先比较固定 0.13 与动态，再结合已有固定 0.05/0.10，所有方向均记录。之后如需进一步区分阶段调节与样本调节，再做阶段匹配/打乱/拟合网格诊断。

创新点和默认主方法保持不变，RMS 不追加消融，MMM2 当前原配置暂停批量扩种子。本次未启动训练、修改原始结果或论文正文、提交或推送。
