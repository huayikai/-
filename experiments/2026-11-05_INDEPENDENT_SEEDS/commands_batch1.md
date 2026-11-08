# 2026-11-05：新独立种子配对验证，预定12条

11 月 8 日更新。集合已扩为 3s84–87、6h92–95，24 个三方法条件均有覆盖 5M 的记录，见 [完整结果](../2026-11-08_INDEPENDENT_RESULTS/report.md)。本文保留最初 12 条的命令与预定顺序。

11 月 5 日快照：首批已回传3个不同的预定条件；6h92动态有两份逐标量一致的运行记录，3s84固定0.13本地未找到。已回传日志覆盖5M，完整收尾尚未在本地确认；详见 [首批结果](D:/论文/代码/experiments/2026-11-05_INDEPENDENT_SEEDS/batch1_report.md)。第二、三批当时尚未见本地记录。

发命令时状态：12条配置及三批顺序已预先固定并核验；当时提供首批四条，本地尚无这些运行，助手未启动训练。

## 验证目标与集合

既定开发集合的固定0.13已补齐，用户要求结果方向不一致时先增加更多种子。新集合固定为3s84/85、6h92/93，每颗分别比较动态BACM、固定0.13、纯BM，共12条。所有结果保留，新验证与原6h开发筛选集合分开报告，不根据首批表现更换种子、系数、方法版本或取消后续批次。

固定比例继续采用已有开发日志确定的0.13，不利用新验证结果重新选择常数。3s attached、6h detached，RND开启，5.05M，32局测试，每10K评估，输出ablation_results_dvd。创新点和默认主方法保持现状。

## 已固定的三批顺序

| 批次 | GPU0 | GPU1 | GPU2 | GPU3 |
| --- | --- | --- | --- | --- |
| 1（本次） | 3s84 动态 | 3s84 固定0.13 | 6h92 动态 | 6h92 固定0.13 |
| 2 | 3s85 动态 | 3s85 固定0.13 | 6h93 动态 | 6h93 固定0.13 |
| 3 | 3s84 BM | 3s85 BM | 6h92 BM | 6h93 BM |

前两批先取得动态与固定的同seed配对，第三批补齐四颗纯BM参照。三批是四卡算力安排，不把不同方法样本量暂不一致的中间结果当成完整三方法结论。动态/固定均开启相同的agent梯度日志和TensorBoard；BM按原纯BM配置保留，不在预算内启用轨迹接管。

## 首批四条命令

沿用上一批服务器训练代码，无需修改训练源码。四张卡空闲后，在四个独立终端运行；每个终端先执行：

```bash
cd /home/zhangbei/pymarl2
conda activate pymarl
```

GPU 0，3s5z_vs_3s6z seed84，动态 BACM：

```bash
CUDA_VISIBLE_DEVICES=0 python3 src/main_audit.py --config=dvd_audit_router --env-config=sc2 with env_args.map_name=3s5z_vs_3s6z seed=84 name=diag_bacm_full_attached_5m_3s5z_vs_3s6z_s84 audit_detach_mixed_hidden=False audit_log_agent_gradients=True use_tensorboard=True t_max=5050000 local_results_path=ablation_results_dvd
```

GPU 1，3s5z_vs_3s6z seed84，固定门控0.13：

```bash
CUDA_VISIBLE_DEVICES=1 python3 src/main_audit.py --config=dvd_audit_router --env-config=sc2 with env_args.map_name=3s5z_vs_3s6z seed=84 name=ablation_bacm_fixedgate013_attached_5m_3s5z_vs_3s6z_s84 adaptive_gate_init=0.13 counterfactual_route_loss_weight=0.0 counterfactual_sparse_weight=0.0 audit_detach_mixed_hidden=False audit_log_agent_gradients=True use_tensorboard=True t_max=5050000 local_results_path=ablation_results_dvd
```

GPU 2，6h_vs_8z seed92，动态 BACM：

```bash
CUDA_VISIBLE_DEVICES=2 python3 src/main_audit.py --config=dvd_audit_router --env-config=sc2 with env_args.map_name=6h_vs_8z seed=92 name=diag_bacm_full_detached_5m_6h_vs_8z_s92 audit_detach_mixed_hidden=True audit_log_agent_gradients=True use_tensorboard=True t_max=5050000 local_results_path=ablation_results_dvd
```

GPU 3，6h_vs_8z seed92，固定门控0.13：

```bash
CUDA_VISIBLE_DEVICES=3 python3 src/main_audit.py --config=dvd_audit_router --env-config=sc2 with env_args.map_name=6h_vs_8z seed=92 name=ablation_bacm_fixedgate013_detached_5m_6h_vs_8z_s92 adaptive_gate_init=0.13 counterfactual_route_loss_weight=0.0 counterfactual_sparse_weight=0.0 audit_detach_mixed_hidden=True audit_log_agent_gradients=True use_tensorboard=True t_max=5050000 local_results_path=ablation_results_dvd
```

## 配置核验与判读边界

- 本地main_audit配置加载/覆盖核验通过，12条共有训练设置一致。各方法沿用已验证参考设置；历史BM的10.05M总预算在本验证统一覆盖为5.05M，不改变目标、探索或优化设置。
- 每个seed的动态/固定配对仅有两项训练处理差异：gate初值0.05→0.13、路由监督系数0.1→0。稀疏项均0，RMS、BM目标、其他损失、图聚合、hidden路径及梯度日志设置一致。
- 固定0.13的实际常数行为已经先前CPU检查及八条开发运行日志验证，本批只换新种子，不重复同一行为测试。原零轨迹信号回退仍保留。
- 固定从0.13开始同时改变初始先验和训练时间轨迹；新结果仍不能单独解释为逐样本路由的因果收益。
- 使用距5M最近的一次原始未平滑评估、末段和曲线，不插值、不时间加权。逐seed记录差异及不确定性，不把相邻测试点当独立训练重复。
- 后续结果无论方向均保留；MMM2/新地图暂缓，外部算法对照低优先级，RMS扩展暂停。

[12条配置及预定顺序](D:/论文/代码/experiments/2026-11-05_INDEPENDENT_SEEDS/config_check.json)；[已齐开发集合报告](D:/论文/代码/experiments/2026-11-05_FIXED_GATE_013_COMPLETE/report.md)。

本次只准备配置、命令与记录，未修改训练源码、原始日志或论文正文，未启动训练、提交或推送。
