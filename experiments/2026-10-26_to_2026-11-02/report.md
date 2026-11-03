# 2026-10-26 至 2026-11-02 实验日志

更新日期：2026-11-02，时区 Asia/Shanghai。仅整理已存在的原始结果；本轮没有启动训练。

> 快照说明（2026-11-03 补注）：本文的 47 条及 252 条 config 总数对应 11 月 2 日当时已回传数据。直接 A+B 首批四条虽于 11 月 2 日晚启动，但在 11 月 3 日回传，另见 [新结果报告](D:/论文/代码/experiments/2026-11-03_DIRECT_COMBINATION_RESULTS/report.md)。原快照条数和当日待办保留；当前 config 总数为 256（104 + 77 + 75），最新安排以新报告为准。

## 当前决定

- 直接在 `D:/论文/代码` 维护代码、配置、命令与实验记录，独立补丁包仅作可选副本。
- 下一轮优先同 BM 骨架的直接 A+B 对照。新 `dvd_audit_direct` 代码及命令已就绪，尚无服务器学习结果；定义见 [DIRECT_COMBINATION.md](D:/论文/代码/experiments/DIRECT_COMBINATION.md)。
- RMS 作为尺度处理/实现细节，现有结果保留，暂不继续追加其消融或参数扫描。
- 既有成功、失败及反向结果全部保留。6h 既定开发集合按 BACM 表现筛选，其比较不能当作随机种子总体稳定性估计。

## 数据范围和口径

本日志按北京时间启动日期收录 47 条新增运行，来自 `D:/论文/ablation_results_dvd`。三个数据根目录共包含 252 条 config 记录（104 + 77 + 71）；下表不把旧对照或同 seed 诊断重跑当成新增独立种子。

5M 指距离 5,000,000 最近的一次未平滑原始测试，不插值；末次通常在 5.04M 左右，保留真实步数。最近 5M 的单点、最后单点与曲线共同查看，不使用历史时间加权胜率。论文如使用边界插值，需单独注明并统一复算，不能把本表数值无标记混用。测试局数通常为 32，单点有波动。

47 条均记录到 5M 以上；Sacred/TensorBoard 共有胜率不一致数为 0，发现的非有限 simple scalar 数为 0。Sacred 有 2 条 info.json 空缺/截断，采用对应 TensorBoard 原始评估。RUNNING 元数据不代表服务器进程仍在运行。

完整运行身份、参数、真实步数、末五次值和事件文件 SHA-256 见 [run_summary.json](D:/论文/代码/experiments/2026-10-26_to_2026-11-02/run_summary.json)。当前读取器支持本批 simple_value 标量，检查记录长度，未验证 TFRecord CRC。

## 分阶段记录

### 01 门控初始化对照

本阶段 4 条；启动时间 2026-10-26T10:16 至 2026-10-26T10:18。

完成 79/80 的动态初始化 0.10 与固定 0.05。默认动态 0.05 与历史固定 0.10 保留。gate.detach() 且稀疏系数为零时，关闭路由损失会冻结 gate，不是保留门控学习的独立监督消融。

| 运行身份 | 地图/seed | 最近 5M：步数 / 胜率 | 末次：步数 / 胜率 |
| --- | --- | --- | --- |
| `diag_bacm_dynamicinit010_detached_5m_6h_vs_8z_s80` | 6h_vs_8z/80 | 5,000,807 / 81.25% | 5,041,276 / 84.375% |
| `diag_bacm_fixedgate005_detached_5m_6h_vs_8z_s80` | 6h_vs_8z/80 | 4,998,070 / 53.125% | 5,048,233 / 59.375% |
| `diag_bacm_dynamicinit010_detached_5m_6h_vs_8z_s79` | 6h_vs_8z/79 | 4,999,024 / 59.375% | 5,049,421 / 75% |
| `diag_bacm_fixedgate005_detached_5m_6h_vs_8z_s79` | 6h_vs_8z/79 | 4,999,741 / 90.625% | 5,050,044 / 81.25% |

### 02 6h 组件补齐

本阶段 8 条；启动时间 2026-10-26T22:27 至 2026-10-27T10:14。

补齐 84/87/88/90 的去辅助拟合、去跨智能体聚合，与原有 seed80 构成既定五颗组件对照。identity 仍保留逐智能体投影及后续超网络，不能改称删掉全部 GAT。旧 79/83 结果保留。

| 运行身份 | 地图/seed | 最近 5M：步数 / 胜率 | 末次：步数 / 胜率 |
| --- | --- | --- | --- |
| `ablation_bacm_noaux_detached_5m_6h_vs_8z_s84` | 6h_vs_8z/84 | 4,998,877 / 43.75% | 5,049,178 / 28.125% |
| `ablation_bacm_noaggregate_detached_5m_6h_vs_8z_s84` | 6h_vs_8z/84 | 5,001,910 / 50% | 5,042,152 / 50% |
| `ablation_bacm_noaux_detached_5m_6h_vs_8z_s87` | 6h_vs_8z/87 | 4,998,765 / 75% | 5,049,154 / 78.125% |
| `ablation_bacm_noaggregate_detached_5m_6h_vs_8z_s87` | 6h_vs_8z/87 | 5,001,396 / 75% | 5,041,732 / 71.875% |
| `ablation_bacm_noaux_detached_5m_6h_vs_8z_s88` | 6h_vs_8z/88 | 4,996,381 / 68.75% | 5,046,974 / 65.625% |
| `ablation_bacm_noaggregate_detached_5m_6h_vs_8z_s88` | 6h_vs_8z/88 | 4,996,813 / 62.5% | 5,047,295 / 65.625% |
| `ablation_bacm_noaux_detached_5m_6h_vs_8z_s90` | 6h_vs_8z/90 | 5,000,245 / 71.875% | 5,040,463 / 62.5% |
| `ablation_bacm_noaggregate_detached_5m_6h_vs_8z_s90` | 6h_vs_8z/90 | 4,998,388 / 81.25% | 5,048,632 / 75% |

### 03 第二层 DVD 共同参数

本阶段 8 条；启动时间 2026-10-27T23:06 至 2026-10-28T10:06。

3s79/80/81、6h80/84/87/88/90 已齐。属于本地第二层 DVD 实现与共同数值参数的工程比较；DVD learner 的 Double Q、硬更新、无 RND 与 BACM 的训练流程仍不同，不是严格原论文复现。3s81 的强 DVD 结果继续保留。

| 运行身份 | 地图/seed | 最近 5M：步数 / 胜率 | 末次：步数 / 胜率 |
| --- | --- | --- | --- |
| `baseline_dvd_second_matched_5m_3s5z_vs_3s6z_s79` | 3s5z_vs_3s6z/79 | 4,997,796 / 56.25% | 5,049,546 / 78.125% |
| `baseline_dvd_second_matched_5m_3s5z_vs_3s6z_s80` | 3s5z_vs_3s6z/80 | 4,997,212 / 0% | 5,048,576 / 0% |
| `baseline_dvd_second_matched_5m_6h_vs_8z_s80` | 6h_vs_8z/80 | 4,995,419 / 0% | 5,045,868 / 0% |
| `baseline_dvd_second_matched_5m_6h_vs_8z_s84` | 6h_vs_8z/84 | 4,997,780 / 0% | 5,048,431 / 0% |
| `baseline_dvd_second_matched_5m_3s5z_vs_3s6z_s81` | 3s5z_vs_3s6z/81 | 5,002,614 / 87.5% | 5,043,090 / 90.625% |
| `baseline_dvd_second_matched_5m_6h_vs_8z_s87` | 6h_vs_8z/87 | 5,001,196 / 0% | 5,042,110 / 0% |
| `baseline_dvd_second_matched_5m_6h_vs_8z_s88` | 6h_vs_8z/88 | 5,004,058 / 0% | 5,045,102 / 0% |
| `baseline_dvd_second_matched_5m_6h_vs_8z_s90` | 6h_vs_8z/90 | 5,004,056 / 0% | 5,044,806 / 0% |

### 04 第一层 DVD 共同参数

本阶段 4 条；启动时间 2026-10-28T23:45 至 2026-10-28T23:46。

只改变注入版本的工程诊断。3s81 学起，3s80 与两颗 6h 未恢复。不能单凭这些运行把失败归因于第二层池化。

| 运行身份 | 地图/seed | 最近 5M：步数 / 胜率 | 末次：步数 / 胜率 |
| --- | --- | --- | --- |
| `baseline_dvd_first_matched_5m_3s5z_vs_3s6z_s80` | 3s5z_vs_3s6z/80 | 4,999,006 / 0% | 5,050,541 / 0% |
| `baseline_dvd_first_matched_5m_3s5z_vs_3s6z_s81` | 3s5z_vs_3s6z/81 | 5,003,203 / 84.375% | 5,043,892 / 93.75% |
| `baseline_dvd_first_matched_5m_6h_vs_8z_s80` | 6h_vs_8z/80 | 4,996,249 / 0% | 5,047,543 / 0% |
| `baseline_dvd_first_matched_5m_6h_vs_8z_s84` | 6h_vs_8z/84 | 5,001,013 / 0% | 5,041,873 / 0% |

### 05 旧 DVD 配置复验及补种子

本阶段 11 条；启动时间 2026-10-29T20:14 至 2026-10-30T20:17。

恢复旧探索、batch/buffer、TD lambda、头数等配置。seed79 的 6h 与 MMM2 两组在覆盖范围内原始记录逐项复现；3s79 前约 4.14M 一致后分叉，但仍学起。补种子后，同配置在其他 seed 上表现不同。不同结构/协议分开列，旧 10M 末点不能与新 5M 末点混比。

| 运行身份 | 地图/seed | 最近 5M：步数 / 胜率 | 末次：步数 / 胜率 |
| --- | --- | --- | --- |
| `repro_dvd_legacycfg_5m_6h_vs_8z_s79_h4_first` | 6h_vs_8z/79 | 4,996,990 / 21.875% | 5,047,844 / 31.25% |
| `repro_dvd_legacycfg_5m_3s5z_vs_3s6z_s79_h8_second` | 3s5z_vs_3s6z/79 | 4,998,000 / 43.75% | 5,050,051 / 62.5% |
| `repro_dvd_legacycfg_5m_MMM2_s79_h8_first` | MMM2/79 | 4,998,558 / 90.625% | 5,049,383 / 84.375% |
| `repro_dvd_legacycfg_5m_MMM2_s79_h8_second` | MMM2/79 | 4,998,314 / 96.875% | 5,049,824 / 90.625% |
| `repro_dvd_legacycfg_5m_6h_vs_8z_s80_h4_first` | 6h_vs_8z/80 | 4,995,507 / 0% | 5,047,315 / 0% |
| `repro_dvd_legacycfg_5m_6h_vs_8z_s84_h4_first` | 6h_vs_8z/84 | 5,003,169 / 0% | 5,045,696 / 0% |
| `repro_dvd_legacycfg_5m_3s5z_vs_3s6z_s80_h8_second` | 3s5z_vs_3s6z/80 | 4,997,004 / 37.5% | 5,049,553 / 34.375% |
| `repro_dvd_legacycfg_5m_3s5z_vs_3s6z_s81_h8_second` | 3s5z_vs_3s6z/81 | 5,004,670 / 12.5% | 5,046,636 / 15.625% |
| `repro_dvd_legacycfg_5m_6h_vs_8z_s87_h4_first` | 6h_vs_8z/87 | 4,995,305 / 12.5% | 5,046,126 / 12.5% |
| `repro_dvd_legacycfg_5m_6h_vs_8z_s88_h4_first` | 6h_vs_8z/88 | 5,000,484 / 0% | 5,042,160 / 0% |
| `repro_dvd_legacycfg_5m_6h_vs_8z_s90_h4_first` | 6h_vs_8z/90 | 4,994,930 / 3.125% | 5,047,817 / 0% |

### 06 TD 修正后的 DVD+BM base

本阶段 8 条；启动时间 2026-11-01T10:10 至 2026-11-01T21:11。

3s79/80/81 与 6h80/84/87/88/90 已齐。3s79、6h87 学起，其余六条在已记录测试中均零胜率。它是一种直接组合实现，但默认第二层超网络和 0.01 初始化缩放与 BACM 骨架不同；不能将所有差距单独归因于门控。与纯 BM 比较仍有反向 seed。

| 运行身份 | 地图/seed | 最近 5M：步数 / 胜率 | 末次：步数 / 胜率 |
| --- | --- | --- | --- |
| `baseline_dvd_bm_audit_5m_3s5z_vs_3s6z_s80` | 3s5z_vs_3s6z/80 | 4,995,313 / 0% | 5,047,195 / 0% |
| `baseline_dvd_bm_audit_5m_3s5z_vs_3s6z_s81` | 3s5z_vs_3s6z/81 | 5,001,032 / 0% | 5,042,223 / 0% |
| `baseline_dvd_bm_audit_5m_6h_vs_8z_s80` | 6h_vs_8z/80 | 5,002,944 / 0% | 5,043,105 / 0% |
| `baseline_dvd_bm_audit_5m_6h_vs_8z_s84` | 6h_vs_8z/84 | 5,003,851 / 0% | 5,044,004 / 0% |
| `baseline_dvd_bm_audit_5m_3s5z_vs_3s6z_s79` | 3s5z_vs_3s6z/79 | 5,004,788 / 50% | 5,045,626 / 59.375% |
| `baseline_dvd_bm_audit_5m_6h_vs_8z_s87` | 6h_vs_8z/87 | 5,001,754 / 62.5% | 5,042,418 / 56.25% |
| `baseline_dvd_bm_audit_5m_6h_vs_8z_s88` | 6h_vs_8z/88 | 5,003,275 / 0% | 5,043,482 / 0% |
| `baseline_dvd_bm_audit_5m_6h_vs_8z_s90` | 6h_vs_8z/90 | 5,002,935 / 0% | 5,043,121 / 0% |

### 07 去 RMS 归一化

本阶段 4 条；启动时间 2026-11-02T10:06 至 2026-11-02T10:07。

新四条与旧 3s79 合看，结果有正有反；6h84 在最近 5M 与末次的排序不同。用户决定暂不扩展该消融。它仍保留门控及配套损失，不是直接 A+B 对照。

| 运行身份 | 地图/seed | 最近 5M：步数 / 胜率 | 末次：步数 / 胜率 |
| --- | --- | --- | --- |
| `ablation_bacm_nornorm_attached_5m_3s5z_vs_3s6z_s80` | 3s5z_vs_3s6z/80 | 4,997,945 / 59.375% | 5,048,597 / 46.875% |
| `ablation_bacm_nornorm_attached_5m_3s5z_vs_3s6z_s81` | 3s5z_vs_3s6z/81 | 5,003,720 / 68.75% | 5,044,190 / 81.25% |
| `ablation_bacm_nornorm_detached_5m_6h_vs_8z_s80` | 6h_vs_8z/80 | 5,000,797 / 84.375% | 5,041,010 / 78.125% |
| `ablation_bacm_nornorm_detached_5m_6h_vs_8z_s84` | 6h_vs_8z/84 | 5,000,717 / 56.25% | 5,041,077 / 78.125% |

## 修正 base 与完整 BACM 的配对摘要

| 地图/seed | base @5M | BACM @5M | base 末次 | BACM 末次 |
| --- | ---: | ---: | ---: | ---: |
| 3s5z_vs_3s6z/80 | 0% | 68.75% | 0% | 68.75% |
| 3s5z_vs_3s6z/81 | 0% | 65.625% | 0% | 65.625% |
| 6h_vs_8z/80 | 0% | 62.5% | 0% | 68.75% |
| 6h_vs_8z/84 | 0% | 90.625% | 0% | 59.375% |
| 3s5z_vs_3s6z/79 | 50% | 78.125% | 59.375% | 68.75% |
| 6h_vs_8z/87 | 62.5% | 81.25% | 56.25% | 65.625% |
| 6h_vs_8z/88 | 0% | 75% | 0% | 75% |
| 6h_vs_8z/90 | 0% | 78.125% | 0% | 87.5% |

## RMS 数据暂存

| 地图/seed | 完整 @5M | 去 RMS @5M | 完整末次 | 去 RMS 末次 |
| --- | ---: | ---: | ---: | ---: |
| 3s5z_vs_3s6z/79 | 78.125% | 0% | 68.75% | 0% |
| 3s5z_vs_3s6z/80 | 68.75% | 59.375% | 68.75% | 46.875% |
| 3s5z_vs_3s6z/81 | 65.625% | 68.75% | 65.625% | 81.25% |
| 6h_vs_8z/80 | 62.5% | 84.375% | 68.75% | 78.125% |
| 6h_vs_8z/84 | 90.625% | 56.25% | 59.375% | 78.125% |

RMS 不作为已证实的普遍性能收益或单独核心创新；这里保留完整记录供后续决定用途。

## 尚未产生结果的新实验

- `dvd_audit_direct`：同 BM 骨架，raw 轨迹候选直接替换 W1，单一 TD 使用自身 target mixer，无动态门控、额外 BM/候选/路由损失、RMS 或 BM fallback。
- 首批预定：3s80/81 attached；6h80/84 detached；预算 5.05M。CPU 检查 31 项通过，1 项因本地缺少 Sacred 跳过；未进行本地 SC2 训练。
- 以本目录代码为主，服务器同步后直接运行。四条命令见 [DIRECT_COMBINATION.md](D:/论文/代码/experiments/DIRECT_COMBINATION.md)，不将已准备命令当作运行完成。
