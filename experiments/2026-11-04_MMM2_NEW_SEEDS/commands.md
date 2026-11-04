# 2026-11-04：MMM2 新种子首批命令

状态更新：本批四条均已回传覆盖 5M 的结果，全部已记录测试为零胜率，详见 [结果报告](D:/论文/代码/experiments/2026-11-04_MMM2_RESULTS/report.md)。以下命令保留追溯，不表示需要重复训练；助手未启动训练。

## 本批安排

按用户决定先补 MMM2 其他种子，沿用现有 BACM/BM 配置，预算 5.05M、RND 开启。BACM 保持 detached、候选辅助隐藏隔离、BM/混合损失比例 1:0.25、辅助/路由损失各 0.1。纯 BM 在整个预算内不启动轨迹接管。本轮不改创新点或默认主方法。

| GPU | seed | 方法 |
| --- | ---: | --- |
| 0 | 80 | BACM detached |
| 1 | 80 | 纯 BM |
| 2 | 81 | BACM detached |
| 3 | 81 | 纯 BM |

沿用上一批服务器训练代码，本次未改训练源码。四张卡空闲后，在四个独立终端分别执行；每个终端先运行：

```bash
cd /home/zhangbei/pymarl2
conda activate pymarl
```

GPU 0，MMM2 seed80，BACM detached：

```bash
CUDA_VISIBLE_DEVICES=0 python3 src/main_audit.py --config=dvd_audit_router --env-config=sc2 with env_args.map_name=MMM2 seed=80 name=audit_router_mix025_detachhidden_5m_MMM2_s80 audit_detach_mixed_hidden=True audit_log_agent_gradients=True use_tensorboard=True t_max=5050000 local_results_path=ablation_results_dvd
```

GPU 1，MMM2 seed80，纯 BM：

```bash
CUDA_VISIBLE_DEVICES=1 python3 src/main_audit.py --config=dvd_audit_bm --env-config=sc2 with env_args.map_name=MMM2 seed=80 name=audit_bm_td_5m_MMM2_s80 use_tensorboard=True t_max=5050000 local_results_path=ablation_results_dvd
```

GPU 2，MMM2 seed81，BACM detached：

```bash
CUDA_VISIBLE_DEVICES=2 python3 src/main_audit.py --config=dvd_audit_router --env-config=sc2 with env_args.map_name=MMM2 seed=81 name=audit_router_mix025_detachhidden_5m_MMM2_s81 audit_detach_mixed_hidden=True audit_log_agent_gradients=True use_tensorboard=True t_max=5050000 local_results_path=ablation_results_dvd
```

GPU 3，MMM2 seed81，纯 BM：

```bash
CUDA_VISIBLE_DEVICES=3 python3 src/main_audit.py --config=dvd_audit_bm --env-config=sc2 with env_args.map_name=MMM2 seed=81 name=audit_bm_td_5m_MMM2_s81 use_tensorboard=True t_max=5050000 local_results_path=ablation_results_dvd
```

## 核验与结果口径

- 四条命令通过本地 main_audit 配置加载与覆盖检查，无需启动 SC2。除了种子、运行名称及输出目录，训练配置与各自历史 MMM2 seed79 对照一致；旧配置未显式记录的 learned 图聚合和 RMS 开关按其默认值处理。当前源码为已审计版本，不宣称与历史 seed79 源码逐字一致或逐位复现。
- 显式 use_tensorboard=True；BACM 显式 audit_log_agent_gradients=True，与历史 MMM2 主配置一致。main_audit 自动记录当前配置和源码指纹。
- 输出根目录预计为 /home/zhangbei/pymarl2/ablation_results_dvd；新增运行名独立于历史 seed79，保留其既有失败及其他对照。
- 训练至 5.05M，比较距 5M 最近的一次未平滑评估及末段原始曲线，不插值、不时间加权。
- 完成后先分析本批四条；下一阶段优先固定近似同融合量门控，具体固定值先核对地图与阶段。RMS 扩展及 seed79 诊断不列入本批。

[配置核验记录](D:/论文/代码/experiments/2026-11-04_MMM2_NEW_SEEDS/config_check.json)；[整体实验补充计划](D:/论文/代码/experiments/2026-11-03_EXPERIMENT_GAPS/plan.md)。
