"""Summarize the four seed-79 audit runs without changing raw logs."""
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent / 'audit_update_2026-10-02'
details = json.loads(gzip.decompress((OUT / 'details.json.gz').read_bytes()))
historical = json.loads(gzip.decompress((OUT / 'historical_controls.json.gz').read_bytes()))
verification = json.loads((OUT / 'source_verification.json').read_text(encoding='utf-8'))
variants = ['legacy', 'td', 'aux', 'both']
labels = ['Legacy', 'TD only', 'Aux only', 'TD + Aux']
chinese = ['原行为对照', '仅修正 TD(λ)', '仅隔离辅助 hidden 梯度', '两项同时修正']
colors = ['#808080', '#2878b5', '#d88923', '#28845b']
runs = [next(d for d in details if '_{}_'.format(v) in d['row']['name']) for v in variants]
bm = next(d for d in historical if d['row']['name'] == 'takeoveroff_3s5z_s79')


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join('---' for _ in headers) + ' |'] +
                     ['| ' + ' | '.join(str(x) for x in row) + ' |' for row in rows])


plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
grid = np.arange(0, 10_000_001, 20_000)
for d, label, color in zip(runs + [bm], labels + ['Historical BM (old TD)'], colors + ['#2b3340']):
    x, y = np.asarray(d['series']['test_battle_won_mean']).T
    curve = np.interp(grid, x, y)
    smooth = np.convolve(np.pad(curve, (5, 5), mode='edge'), np.ones(11) / 11, mode='valid')
    axes[0].plot(grid / 1e6, smooth, label=label, color=color,
                 linestyle='--' if d is bm else '-', linewidth=1.8)
axes[0].set(xlabel='Environment steps (millions)', ylabel='Test win rate',
            title='Learning curves (0.20M display smoothing)', ylim=(-.02, 1.02))
axes[0].legend(fontsize=9, loc='upper left')
axes[0].grid(alpha=.15)
xs = np.arange(5)
axes[1].bar(xs - .18, [d['row']['auc_0_10m'] for d in runs + [bm]], .36, label='0-10M AUC', color='#2878b5')
axes[1].bar(xs + .18, [d['row']['win_9_10m'] for d in runs + [bm]], .36, label='9-10M mean win rate', color='#28845b')
axes[1].set_xticks(xs, labels + ['Hist. BM\nold TD'])
axes[1].set(title='Unsmoothed, time-weighted metrics', ylim=(0, 1.02))
axes[1].legend(fontsize=9)
axes[1].grid(axis='y', alpha=.15)
fig.suptitle('3s5z_vs_3s6z | seed 79 only | TD target and auxiliary-gradient audit', fontsize=13)
fig.savefig(OUT / 'audit_comparison.png', dpi=180)
fig.savefig(OUT / 'audit_comparison.svg')
plt.close(fig)

metrics = []
for label, d in zip(chinese + ['历史 BM（旧 TD，仅参考）'], runs + [bm]):
    r = d['row']
    metrics.append([label, f"{r['auc_0_10m']:.4f}",
                    f"{100*r['win_3_5m']:.2f}%", f"{100*r['win_7_9m']:.2f}%",
                    f"{100*r['win_9_10m']:.2f}%", f"{100*r['final']:.2f}%"])
diagnostics = []
for label, d in zip(chinese, runs):
    z = d['diagnostics_9_10m']
    diagnostics.append([label, f"{z['counterfactual_gate_mean']:.4f}",
                        f"{z['counterfactual_gate_std']:.4f}",
                        f"{z['counterfactual_gate_target_correlation']:.4f}",
                        f"{z['counterfactual_absolute_advantage_mean']:.5f}"])
rows = [
    '# 2026-10-02：TD(λ) / 辅助 hidden 梯度四组实验核查',
    '数据目录：`D:\\科研\\ablation_results`。相较 2026-10-01 的 84 组记录，新增 4 组，总计 88 组；未删除旧记录。新增运行全部为 `3s5z_vs_3s6z`、seed79。',
    '## 结论',
    '这轮结果值得继续做有限配对复验。旧行为对照仍陷入低胜率，三组修正均恢复了学习；两项同时修正的 0–10M AUC 最高，仅辅助隔离的 9–10M 平均胜率最高。当前证据支持“实现路径影响了 seed79 的失败”，尚不支持跨种子稳定增益、两项修正普遍最优或 DVD 路由的独立贡献。',
    table(['方案', '0–10M AUC', '3–5M 胜率', '7–9M 胜率', '9–10M 胜率', '最后一次评估'], metrics),
    'AUC 为胜率曲线的时间积分除以 10M；各区间胜率使用时间加权平均，窗口边界线性插值。同 step 重复记录保留 wall_time 最新者。最后单点评估仅用于展示，选择方案以 AUC 和完整尾段为主。',
    '![四组曲线与指标](audit_comparison.png)',
    '## 覆盖与可比性',
    table(['方案', '最后评估步', '配置开关 TD / Aux', 'Sacred 状态'],
          [[label, d['row']['last_step'],
            f"{d['config']['audit_fix_td_lambda']} / {d['config']['audit_isolate_dvd_aux_hidden']}",
            d['row']['status']] for label, d in zip(chinese, runs)]),
    '四组评估均超过 10M，覆盖所比较的全部区间；aux/both/td 的最后评估略低于配置总预算 10.05M，不能据此声称完整训练进程已结束。所有元数据仍为 RUNNING，且没有 stop_time。此处只判断已复制日志的评估覆盖，不推断服务器进程状态。四组全部标量均有限，没有事件解析异常。',
    '逐组 Sacred 配置除 name 和两个实验开关外完全一致；TensorBoard 记录的开关也分别为 0/0、1/0、0/1、1/1。四组的 20 个 manifest 文件逐一与实际保存的源码/资源 SHA-256 一致；源码及 YAML 在忽略 Windows/Linux 换行差异后与本地 b7d1d30 相同。运行元数据没有 Git repository 信息，所以版本核对以文件快照与哈希为依据。',
    f"legacy 与旧 safe-v2 seed79 的胜率曲线逐点一致，全部 {verification['legacy_reproduction']['common_tags']} 个共有标量标签也逐点一致。这加强了关闭开关的复现证据，而不仅是胜率区间近似相同。",
    '## 两项修正分别带来了什么',
    '仅 TD：AUC 从 0.0169 升至 0.4479，尾段从 1.62% 升至 73.91%；前期启动最快，1–3M 时间均值为 24.50%。这支持末尾/边界目标处理对本 seed 的学习有实质影响。修正同时处理最后有效步、终止、截断和 padding，本轮不能进一步将收益只归因于其中一项。',
    '仅辅助隔离：AUC 达 0.4966，尾段 83.80%；在旧 TD 下也能解除 seed79 的失败，说明原辅助损失到 agent hidden 的梯度路径值得认真处理。',
    '两项同时修正：AUC 为 0.5209，比仅辅助隔离高约 0.0243；尾段 82.41%，比仅辅助隔离低约 1.38 个百分点。3–5M / 5–7M 更强，因此累计学习效率最好。单 seed 的差异不足以证明两个开关收益可相加或某个方案普遍最优。',
    '## 机制诊断',
    table(['方案', 'Gate 均值', 'Gate 标准差', 'Gate/标签相关性', 'DVD 绝对 TD 优势'], diagnostics),
    '以上为 9–10M 时间均值。修正后三组 gate/标签相关性约 0.056–0.081，仍较弱；DVD 平均绝对 TD 优势仍为负。胜率恢复并没有同时提供路由监督已经有效跟随、DVD 专家平均更优的证据。不同方案访问的轨迹分布已经改变，跨运行 TD loss 的绝对大小不能直接当作策略质量排名。',
    '## 历史 BM 与下一轮',
    '历史同 seed BM 的 AUC 为 0.3747，尾段 77.55%。两项同时修正分别高约 0.1462 和 4.86 个百分点，但历史 BM 使用旧 TD 目标。正式比较必须补同样启用 TD 修正的 BM；若 BM 也因目标修正提高，就需要重新判断 DVD 的独立增量。',
    '当前优先补 `dvd_audit_bm` + `audit_fix_td_lambda=True` 的 seed79 对照，并将两项同时修正扩展到 seed80/81/82；之后补齐相同 TD 修正的 BM 80/81/82，形成四种子的严格配对。若资源有限，先做 seed79/80 的配对。保留 seed82 和失败结果。基础 DVD 的 TD 开/关复验属于下一优先级，暂不扩展 gate 超参。',
    '## 文件',
    '`runs.csv`：新增四组的覆盖和胜率窗口；`details.json.gz`：原始标量与配置；`source_verification.json`：源码核查和旧行为逐点复现；`config_and_diagnostics.json`：配置与后期诊断；`historical_controls.json.gz`：旧 safe-v2 和 BM 的参照曲线。所有输出仅写入本地分析目录，原始实验记录未改动。',
]
(OUT / 'report.md').write_text('\n\n'.join(rows) + '\n', encoding='utf-8')
print('Saved report, PNG and SVG:', OUT)
