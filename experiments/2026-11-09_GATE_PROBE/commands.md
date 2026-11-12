# 同模型、同样本门控拟合诊断

2026-11-12 首批用 GPU 0、1、3 跑 `3s85`、`6h92`、`3s84`。GPU 2 留给正在运行的 Qatten `6h94`。原计划的第四个诊断案例 `6h94` 等空卡后再排。

使用原动态设置，关闭训练期打乱，开启只读诊断记录。首批沿用已定的三个机制案例。每条训练至 5.05M，在 1M、2M、3M、4M、5M 附近各保存连续三批更新前的全部有效 transition，每条 15 份。文件记录实际保存步数。

记录BM/DVD激活前特征、输出权重/偏置、共同BM TD目标、动态门控、路由标签及原始三分支输出。保留零候选回退和valid mask；无额外回放采样或随机抽取，记录期间不执行额外agent前向或参数更新。每个快照的模型参数固定，阶段间及连续批次间模型仍照常训练。

离线比较：g=0、动态g、固定0.13、该batch有效候选的平均g、20次门控置换，以及[0,g_max]上101点网格并纳入动态/常数/路由标签/各次置换候选的逐样本拟合oracle。原始端点/混合输出用于重建检查。记录MSE、MAE、改善/变差比例、oracle差距及正标签插值的变差比例。oracle只衡量保存的TD目标拟合；连续batch和20次置换不算独立训练重复。

## 同步与预检

从本次提交同步以下文件到服务器 `/home/zhangbei/pymarl2/src` 下的对应位置。

- main_audit.py
- learners/dvd_audit_learner.py
- utils/gate_probe.py
- config/algs/dvd_audit_router.yaml
- experiments/2026-11-09_GATE_PROBE/analyze.py
- _test_gate_probe.py、_test_dvd_audit.py（本批预检）

本地Python 3.8 / PyTorch 2.4.1+cpu：45项通过、1项因缺少Sacred跳过；含observer开关两种hidden路径下参数/RNG/target/日志一致、离线重建、已知内点最优解和非网格置换最优解。尚未在服务器PyTorch 1.12.1或SC2上运行。

先在项目根目录执行一次预检：

```bash
cd /home/zhangbei/pymarl2
conda activate pymarl
python3 src/_test_gate_probe.py
```

预检通过后，在三个终端分别进入同一目录、激活环境并各运行一条。

## GPU 0：3s5z_vs_3s6z seed85

```bash
CUDA_VISIBLE_DEVICES=0 python3 src/main_audit.py --debug --capture=no --config=dvd_audit_router --env-config=sc2 with env_args.map_name=3s5z_vs_3s6z seed=85 audit_detach_mixed_hidden=False audit_log_agent_gradients=True use_tensorboard=True t_max=5050000 local_results_path=ablation_results_11_6 name=diag_bacm_gateprobe_attached_5m_3s5z_vs_3s6z_s85 audit_shuffle_gate=False audit_save_gate_probe=True
```

## GPU 1：6h_vs_8z seed92

```bash
CUDA_VISIBLE_DEVICES=1 python3 src/main_audit.py --debug --capture=no --config=dvd_audit_router --env-config=sc2 with env_args.map_name=6h_vs_8z seed=92 audit_detach_mixed_hidden=True audit_log_agent_gradients=True use_tensorboard=True t_max=5050000 local_results_path=ablation_results_11_6 name=diag_bacm_gateprobe_detached_5m_6h_vs_8z_s92 audit_shuffle_gate=False audit_save_gate_probe=True
```

## GPU 3：3s5z_vs_3s6z seed84

```bash
CUDA_VISIBLE_DEVICES=3 python3 src/main_audit.py --debug --capture=no --config=dvd_audit_router --env-config=sc2 with env_args.map_name=3s5z_vs_3s6z seed=84 audit_detach_mixed_hidden=False audit_log_agent_gradients=True use_tensorboard=True t_max=5050000 local_results_path=ablation_results_11_6 name=diag_bacm_gateprobe_attached_5m_3s5z_vs_3s6z_s84 audit_shuffle_gate=False audit_save_gate_probe=True
```

## 检查与回传

1M后应出现ablation_results_11_6/gate_probes/diag_bacm_gateprobe_*/*.npz；每阶段三份。保存器保留已有文件，如重启撞到相同文件名将报错，不覆盖旧轨迹。

训练后可在服务器直接运行：

```bash
python3 src/experiments/2026-11-09_GATE_PROBE/analyze.py --input ablation_results_11_6/gate_probes --output ablation_results_11_6/gate_probe_analysis
```

回传本批 `sacred`、`tb_logs`、`gate_probes` 和 `gate_probe_analysis`（若已执行离线分析）。所有门控方案在同一份记录上离线比较。

先核对重建误差与保存数量，再逐seed逐阶段比较动态、常数、打乱和oracle，不把所有transition视作独立训练重复。若动态在同模型同数据上持续劣于常数或打乱，再据机制证据决定是否修改监督。
