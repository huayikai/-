# 训练期门控打乱的两条机制预实验

本轮使用 3s5z_vs_3s6z seed85 和 6h_vs_8z seed92。这两颗依据已有动态运行能够学习的曲线选为机制案例，原动态日志作为配对参照。新增实验观察改变门控与样本对应关系后的训练表现。

## 实现与核验口径

`audit_shuffle_gate` 默认 False，以下命令显式设为 True。每次在线训练前向，在 learner 提供的 filled、非终止后位置掩码内，对具有有效 DVD 候选信号的 transition 做一次均匀随机排列。排列跨 batch 中的有效 transition，允许部分位置保持原位。零候选信号的 BM 回退行与掩码外的位置保留原门控。

混合价值使用打乱后的门控。路由损失使用原网络输出与原样本标签。目标网络和 no-grad/eval 前向使用原门控。初始门控全部相等时，重排后的值也相等；模型学出样本差异后可以由日志观察到实际数值变化。

重排使用独立 CPU torch.Generator，种子为实验 seed 与 0x5A17 的异或。GPU 前向仅搬运排列索引，全局采样、探索及 RND 随机流没有额外的随机抽取。保存模型时另存 gate_shuffle_rng.th，加载含该文件的模型时恢复此私有流。

TensorBoard 新增以下记录。

- audit_shuffle_gate 为开关状态。
- audit_gate_shuffle_valid_count、eligible_count、fallback_count 为有效、参与排列和回退行数。
- audit_gate_shuffle_changed_fraction、mean_abs_change 描述门控值实际改变情况。
- audit_gate_shuffle_mean_delta、std_delta、distribution_error 核对打乱前后均值、标准差和排序后逐值差异。distribution_error 应为零，前两项可有浮点求和量级误差。

原 counterfactual_gate_* 继续描述用于路由监督的原输出。counterfactual_diag_* 的拟合收益分组使用实际进入混合价值的门控。

同次前向打乱前后的门控边际分布保持相同。两种训练条件的数据分布和门控分布会随各自训练过程演化，结果按该训练干预解释。

## 同步文件

把下列本地文件同步到服务器 `/home/zhangbei/pymarl2/src/` 下的对应位置，也可使用同步补丁包中的 src 目录。

- config/algs/dvd_audit_router.yaml
- learners/dvd_audit_learner.py
- modules/mixers/dvd_adaptive_takeover.py
- modules/mixers/dvd_counterfactual_router.py
- modules/mixers/dvd_counterfactual_router_audit.py

有效参数与原动态配置逐项核对见 config_check.json。测试使用本地 CPU 合成 batch，见 cpu_validation.json。GPU 与 SC2 完整训练由本批运行验证。

## 运行

两个终端分别准备环境。

```bash
cd /home/zhangbei/pymarl2
conda activate pymarl
```

GPU 0，3s seed85，attached。

```bash
CUDA_VISIBLE_DEVICES=0 python3 src/main_audit.py \
  --config=dvd_audit_router --env-config=sc2 with \
  env_args.map_name=3s5z_vs_3s6z seed=85 \
  name=ablation_bacm_shufflegate_attached_5m_3s5z_vs_3s6z_s85 \
  audit_shuffle_gate=True audit_detach_mixed_hidden=False \
  audit_log_agent_gradients=True use_tensorboard=True \
  t_max=5050000 local_results_path=ablation_results_11_6
```

GPU 1，6h seed92，detached。

```bash
CUDA_VISIBLE_DEVICES=1 python3 src/main_audit.py \
  --config=dvd_audit_router --env-config=sc2 with \
  env_args.map_name=6h_vs_8z seed=92 \
  name=ablation_bacm_shufflegate_detached_5m_6h_vs_8z_s92 \
  audit_shuffle_gate=True audit_detach_mixed_hidden=True \
  audit_log_agent_gradients=True use_tensorboard=True \
  t_max=5050000 local_results_path=ablation_results_11_6
```

回传以上两条的 Sacred 与 TensorBoard 日志。先核对排列分布及原始评估，再按各自真实步数比较最近 5M、末段和完整学习曲线。
