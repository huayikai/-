# 2026-11-05：固定门控0.13补齐批次

状态更新：本批四条均已回传覆盖5M的结果，固定0.13既定八条已齐，详见 [补齐结果报告](D:/论文/代码/experiments/2026-11-05_FIXED_GATE_013_COMPLETE/report.md)。以下命令保留追溯，不表示需要重跑；助手未启动训练。

本批是事前已预定的剩余四条，保持首批定义与配置，仅补齐种子。首批结果和不同方向均保留，见 [首批结果报告](D:/论文/代码/experiments/2026-11-05_FIXED_GATE_013_RESULTS/report.md)。用户要求不同seed方向不一时先不急着下结论，补更多种子实验。

## 本批安排

| GPU | 地图 | seed | hidden |
| --- | --- | ---: | --- |
| 0 | 3s5z_vs_3s6z | 79 | attached |
| 1 | 6h_vs_8z | 87 | detached |
| 2 | 6h_vs_8z | 88 | detached |
| 3 | 6h_vs_8z | 90 | detached |

固定gate初值0.13，路由监督和稀疏系数均为0。保留原RMS、learned图聚合、BM自举目标、BM/混合/候选损失、辅助hidden隔离、RND和零候选信号回退，预算5.05M，显式开启TensorBoard和agent梯度日志，输出ablation_results_dvd。

当前运行代码沿用首批，无需修改训练源码。四张卡空闲后，在四个独立终端运行；每个终端先执行：

```bash
cd /home/zhangbei/pymarl2
conda activate pymarl
```

GPU 0，3s5z_vs_3s6z seed79：

```bash
CUDA_VISIBLE_DEVICES=0 python3 src/main_audit.py --config=dvd_audit_router --env-config=sc2 with env_args.map_name=3s5z_vs_3s6z seed=79 name=ablation_bacm_fixedgate013_attached_5m_3s5z_vs_3s6z_s79 adaptive_gate_init=0.13 counterfactual_route_loss_weight=0.0 counterfactual_sparse_weight=0.0 audit_detach_mixed_hidden=False audit_log_agent_gradients=True use_tensorboard=True t_max=5050000 local_results_path=ablation_results_dvd
```

GPU 1，6h_vs_8z seed87：

```bash
CUDA_VISIBLE_DEVICES=1 python3 src/main_audit.py --config=dvd_audit_router --env-config=sc2 with env_args.map_name=6h_vs_8z seed=87 name=ablation_bacm_fixedgate013_detached_5m_6h_vs_8z_s87 adaptive_gate_init=0.13 counterfactual_route_loss_weight=0.0 counterfactual_sparse_weight=0.0 audit_detach_mixed_hidden=True audit_log_agent_gradients=True use_tensorboard=True t_max=5050000 local_results_path=ablation_results_dvd
```

GPU 2，6h_vs_8z seed88：

```bash
CUDA_VISIBLE_DEVICES=2 python3 src/main_audit.py --config=dvd_audit_router --env-config=sc2 with env_args.map_name=6h_vs_8z seed=88 name=ablation_bacm_fixedgate013_detached_5m_6h_vs_8z_s88 adaptive_gate_init=0.13 counterfactual_route_loss_weight=0.0 counterfactual_sparse_weight=0.0 audit_detach_mixed_hidden=True audit_log_agent_gradients=True use_tensorboard=True t_max=5050000 local_results_path=ablation_results_dvd
```

GPU 3，6h_vs_8z seed90：

```bash
CUDA_VISIBLE_DEVICES=3 python3 src/main_audit.py --config=dvd_audit_router --env-config=sc2 with env_args.map_name=6h_vs_8z seed=90 name=ablation_bacm_fixedgate013_detached_5m_6h_vs_8z_s90 adaptive_gate_init=0.13 counterfactual_route_loss_weight=0.0 counterfactual_sparse_weight=0.0 audit_detach_mixed_hidden=True audit_log_agent_gradients=True use_tensorboard=True t_max=5050000 local_results_path=ablation_results_dvd
```

## 已完成的核验与后续

- 四条通过main_audit的本地配置加载/覆盖检查，与同地图首批已验证配置相比仅seed、名称及其记录指纹等身份字段变化；与对应动态配置的训练处理差异为gate初值0.05→0.13、路由监督系数0.1→0。
- 门控固定原理由既有计算图和首批CPU/实际日志核验支持；本批不重复运行相同检查，也不把配置核验称为SC2学习效果验证。零轨迹信号仍使用原BM回退，此时实际gate为0。
- 运行后查看最近5M的未平滑评估、末段和整段曲线，不插值、不时间加权；不同seed方向不一时只记录配对观察。
- 本批回传后，既定3s三颗、6h五颗的固定0.13集合补齐，再按预先固定的新集合扩大独立验证，候选为3s84/85、6h92/93的动态/固定0.13/BM配对。
- 创新点、默认主方法不改；MMM2/新地图暂缓，外部算法对照低优先级，RMS扩展暂停。

[配置核验记录](D:/论文/代码/experiments/2026-11-05_FIXED_GATE_013_NEXT/config_check.json)；[原固定对照定义与统计依据](D:/论文/代码/experiments/2026-11-04_FIXED_GATE_013/commands.md)。

本次只准备命令与记录，未改训练源码、原始日志或论文正文，未启动训练、提交或推送。
