# 2026-11-03：直接 A+B 补齐命令

状态：命令已准备并核验；助手未启动训练，本地尚无本批直接对照结果。

沿用上一批 `dvd_audit_direct` 的定义和服务器代码，只补齐 3s79、6h87/88/90。3s attached、6h detached，5.05M，输出 `ablation_results_dvd`；不追加 RMS、门控或其他参数变体。完成后与同 seed 的完整 BACM、纯 BM 比较原始最近 5M 和末段评估，不时间加权。

首批结果见 [结果报告](D:/论文/代码/experiments/2026-11-03_DIRECT_COMBINATION_RESULTS/report.md)。本批四条与首批配对的 24 项共同训练参数一致，地图/seed/name/hidden 路径按下表固定；本轮没有修改训练源码。

| GPU | 地图 | seed | hidden 路径 |
| --- | --- | ---: | --- |
| 0 | 3s5z_vs_3s6z | 79 | attached |
| 1 | 6h_vs_8z | 87 | detached |
| 2 | 6h_vs_8z | 88 | detached |
| 3 | 6h_vs_8z | 90 | detached |

四张卡空闲后，在四个独立服务器终端执行；每个终端先运行：

```bash
cd /home/zhangbei/pymarl2
conda activate pymarl
```

GPU 0，3s5z_vs_3s6z seed79（attached）：

```bash
CUDA_VISIBLE_DEVICES=0 python3 src/main_audit.py --config=dvd_audit_direct --env-config=sc2 with env_args.map_name=3s5z_vs_3s6z seed=79 name=baseline_bm_dvd_direct_raw_attached_5m_3s5z_vs_3s6z_s79 audit_detach_mixed_hidden=False t_max=5050000 local_results_path=ablation_results_dvd
```

GPU 1，6h_vs_8z seed87（detached）：

```bash
CUDA_VISIBLE_DEVICES=1 python3 src/main_audit.py --config=dvd_audit_direct --env-config=sc2 with env_args.map_name=6h_vs_8z seed=87 name=baseline_bm_dvd_direct_raw_detached_5m_6h_vs_8z_s87 audit_detach_mixed_hidden=True t_max=5050000 local_results_path=ablation_results_dvd
```

GPU 2，6h_vs_8z seed88（detached）：

```bash
CUDA_VISIBLE_DEVICES=2 python3 src/main_audit.py --config=dvd_audit_direct --env-config=sc2 with env_args.map_name=6h_vs_8z seed=88 name=baseline_bm_dvd_direct_raw_detached_5m_6h_vs_8z_s88 audit_detach_mixed_hidden=True t_max=5050000 local_results_path=ablation_results_dvd
```

GPU 3，6h_vs_8z seed90（detached）：

```bash
CUDA_VISIBLE_DEVICES=3 python3 src/main_audit.py --config=dvd_audit_direct --env-config=sc2 with env_args.map_name=6h_vs_8z seed=90 name=baseline_bm_dvd_direct_raw_detached_5m_6h_vs_8z_s90 audit_detach_mixed_hidden=True t_max=5050000 local_results_path=ablation_results_dvd
```

本批采用既有开发集合，不依据直接组合的表现更换种子或修改方法。6h 集合此前依据 BACM 表现筛选，补齐后仍是该集合内的条件性比较。既有零胜率与反向结果均保留。
