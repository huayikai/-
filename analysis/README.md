# 实验分析记录

本目录保存截至2026-10-04的报告、结果表、源码核验、曲线及重算脚本。原始日志位于本地`D:\科研\ablation_results`，服务器训练目录通常为`~/pymarl2/src`。

## 先看这些文件

- [最新结果与5M预算决策](../experiments/2026-10-04_RESULTS_AND_BUDGET_LOG.md)
- [10月4日报告](audit_update_2026-10-04/report.md)：3s seed81回退定位、6h首组配对。
- [5M逐运行指标](audit_update_2026-10-04/five_m_metrics.json)：统一截取已有运行的0–5M与4–5M窗口。
- [10月3日报告](audit_update_2026-10-03/report.md)：修正版BM79/80与Router80/81。
- [10月2日报告](audit_update_2026-10-02/report.md)：seed79的legacy/td/aux/both四组。
- [10月1日历史核查](experiment_audit/report.md)：84组原始实验与公平性问题。

日期目录是当时复制日志的分析快照。旧报告中的“下一轮”按当时状态记录；最新研究安排以10月4日更新日志为准。10月4日`next_round_commands.sh`对应用户已启动、仍在运行的10M诊断轮次，不能当作已经得到结果的实验。

## 输出与统计口径

`runs.csv`、`summary.json`保存覆盖范围和分组结果；`paired_metrics.json`只汇总相同种子、相同地图的配对；`source_verification.json`核验Sacred保存快照及manifest；`config_comparison.json`保存配置差异。PNG/SVG中的平滑仅用于展示，统计使用未平滑评估记录。AUC是区间内胜率时间积分除以区间长度，窗口边界线性插值，同step重复记录保留wall_time最新者。

完整标量缓存`details.json`/`details.json.gz`以及`historical_controls.json.gz`留在本地并由Git忽略。仓库不含原始TensorBoard/Sacred日志；这些缓存可用原始日志和解析脚本重建，曲线及指标表已入库。Sacred的RUNNING状态不等于服务器进程当前正在运行，需区分实际覆盖与用户反馈。

## 重算

依赖Python、NumPy、TensorBoard；绘图另需Matplotlib。本机已验证解释器为`D:\python\python.exe`。在仓库根目录用新的输出目录解析最新日志，避免覆盖历史快照：

```powershell
& 'D:\python\python.exe' -B analysis/analyze_experiments.py --root 'D:\科研\ablation_results' --out analysis/local_reparse --name-regex '^audit_'
& 'D:\python\python.exe' -B analysis/summarize_5m.py --input analysis/local_reparse/details.json.gz
```

`analyze_experiments.py`直接解析TFRecord/protobuf标量，无需启动TensorBoard服务或安装TensorFlow，支持中文路径。`summarize_5m.py`选择正式BM/Router both名称，将独立nomix等消融排除；覆盖不足5M时指标为null，不能按完整运行汇总。它输出逐运行指标，配对均值仍须只使用共同种子。

三个日期专用`report_audit_*.py`依赖各自日期目录的完整标量缓存和历史参照，部分还依赖上轮缓存做逐点核对。它们用于复现对应快照，不是任意新数据的通用报告入口。尤其`report_experiments.py`含10月1日84组、`report_audit_update.py`含10月2日88组的固定叙述，不能直接用于更新后的全目录。重新解析只能重建当时选中的运行内容，不能替代保存当时的目录范围；后续新增运行请输出到新的日期目录。
