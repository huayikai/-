"""Render the experiment audit without altering input logs or research code."""
import json
import gzip
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent / "experiment_audit"
with gzip.open(OUT / "details.json.gz", "rt", encoding="utf-8") as stream:
    details = json.load(stream)
summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))


def members(method):
    return sorted([d for d in details if d['row']['method'] == method and d['row']['full_10m'] and not d['row']['nonfinite_tags']], key=lambda d: d['row']['seed'])


def fmt(value):
    return "—" if value is None else f"{value:.4f}"


def table(rows, columns):
    lines = ["| " + " | ".join(label for key, label in columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row[key]) if isinstance(row[key], float) else str(row[key]) for key, label in columns) + " |")
    return "\n".join(lines)


def paired(first, second):
    a = {d['row']['seed']: d['row'] for d in members(first)}
    b = {d['row']['seed']: d['row'] for d in members(second)}
    return [{"seed": seed, "a_auc": a[seed]['auc_0_10m'], "b_auc": b[seed]['auc_0_10m'], "delta_auc": a[seed]['auc_0_10m'] - b[seed]['auc_0_10m'], "a_tail": a[seed]['win_9_10m'], "b_tail": b[seed]['win_9_10m'], "delta_tail": a[seed]['win_9_10m'] - b[seed]['win_9_10m']} for seed in sorted(a.keys() & b.keys())]


grid = np.arange(0, 10_000_001, 20_000)


def smooth_curve(d):
    x, y = np.asarray(d['series']['test_battle_won_mean']).T
    values = np.interp(grid, x, y)
    # Centered 0.20M display smoothing; numerical metrics use unsmoothed logs.
    return np.convolve(np.pad(values, (5, 5), mode='edge'), np.ones(11) / 11, mode='valid')


plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True, constrained_layout=True)
for seed, ax in zip(range(79, 83), axes.flat):
    for method, label, color in [('takeoveroff_3s5z', 'BM control', '#27364a'), ('counterfactual_router_safev2_3s5z', 'Router safe-v2', '#b95320')]:
        d = next(d for d in members(method) if d['row']['seed'] == seed)
        ax.plot(grid / 1e6, smooth_curve(d), label=label, color=color, linewidth=1.8)
    ax.set_title(f'Seed {seed}')
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.15)
    ax.set_xlabel('Environment steps (millions)')
    ax.set_ylabel('Test win rate')
axes[0, 0].legend(loc='upper left')
fig.suptitle('3s5z_vs_3s6z: safe-v2 versus BM, paired seeds', fontsize=14)
fig.savefig(OUT / 'safev2_paired.png', dpi=160)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
for ax, specs, title in [
    (axes[0], [('takeoveroff_3s5z', 'BM control', '#27364a'), ('takeover_floor0_3s5z', 'Floor0', '#2d7d79'), ('counterfactual_router_safev2_3s5z', 'Router safe-v2', '#b95320')], '3s5z_vs_3s6z'),
    (axes[1], [('creditoff_6h', 'Credit structure, DVD off', '#27364a'), ('creditdvd_6h', 'Credit DVD', '#b95320')], '6h_vs_8z'),
]:
    for method, label, color in specs:
        values = np.stack([smooth_curve(d) for d in members(method)])
        ax.plot(grid / 1e6, values.mean(axis=0), label=label, color=color, linewidth=2)
    ax.set_title(title + ' | mean of seeds 79-82')
    ax.set_xlabel('Environment steps (millions)')
    ax.set_ylabel('Test win rate')
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.15)
    ax.legend(loc='lower right', fontsize=9)
fig.suptitle('Raw-log audit: learning curves (0.20M display smoothing)', fontsize=13)
fig.savefig(OUT / 'comparison.png', dpi=160)
plt.close(fig)

safe_pairs = paired('counterfactual_router_safev2_3s5z', 'takeoveroff_3s5z')
credit_pairs = paired('creditdvd_6h', 'creditoff_6h')
diag_rows = []
for d in members('counterfactual_router_safev2_3s5z'):
    z = d['diagnostics_9_10m']
    diag_rows.append({'seed': d['row']['seed'], **{k: z['counterfactual_' + k] for k in ['gate_mean','gate_std','gate_target_correlation','route_positive_rate','route_reliable_rate','absolute_advantage_mean']}})
config_diffs = []
for first, second in [('creditdvd_6h', 'creditoff_6h'), ('counterfactual_router_safev2_3s5z','takeoveroff_3s5z')]:
    a = {d['row']['seed']: d for d in members(first)}
    b = {d['row']['seed']: d for d in members(second)}
    for seed in sorted(a.keys() & b.keys()):
        ca, cb = a[seed]['config'], b[seed]['config']
        diff = {key: [ca.get(key), cb.get(key)] for key in ca.keys() | cb.keys() if key != 'name' and ca.get(key) != cb.get(key)}
        config_diffs.append({'first': first, 'second': second, 'seed': seed, 'differences': diff})
(OUT / 'paired_config_differences.json').write_text(json.dumps(config_diffs, ensure_ascii=False, indent=2), encoding='utf-8')

report = [
    '# 原始实验记录核查（2026-10-01）',
    '数据目录：`D:\\科研\\ablation_results`。全部 84 个 TensorBoard 目录与 84 份 Sacred 配置一一匹配；两次日志创建时间比 Sacred start_time 晚 1 秒，使用唯一同名且相差不超过 2 秒的匹配。',
    '80 个运行的胜率记录覆盖到 10M；其中 4 个修复前 Takeover 运行含 NaN，单独保留为数值失败记录。其余 4 个运行在约 5.1–5.4M 中断，不能推断它们的 10M 表现。正式 10M 分组表包含 76 个数值有限的运行。',
    'Sacred 的 RUNNING 状态与记录已覆盖训练预算是两件事；这里以日志实际步数判断覆盖范围，不据此声称运行进程已退出。',
    '统计口径：去重同 step 记录时保留 wall_time 最晚值；AUC 为 0–10M 胜率曲线梯形积分除以 10M；后期指标为 9–10M 时间加权平均。窗口边界线性插值，首条记录前使用首条胜率；本批运行首次评估均接近零步。所有主表保留 seed82，不按是否学起来删 seed。重复评估点不是独立种子。曲线平滑只用于绘图。',
    '本文与旧文档少数数字不同，主要因为统一窗口边界、时间加权及使用全四种子。当前配置不能单独证明训练时的 learner/mixer 内部实现。',
    '## 主要判断',
    '- safe-v2 已运行到 10M，明显优于 aggressive-v1，但四种子 AUC 和后期胜率均低于 BM 对照。seed79 低胜率、seed80 接近但后期落后、seed81 后期有收益、seed82 两者都失败；不能把 seed81 单独作为稳定优势。',
    '- Credit DVD 在 6h 的四种子结果较好，但同结构 DVD-off 对照几乎同样好。AUC 平均增量约 0.0092，9–10M 胜率增量约 0.0060，逐种子方向不一致；尚无可靠的 DVD 独立贡献证据。',
    '- 旧基础组合在 3s5z 的正结果仅有 seed79；其 epsilon_start=1.0，而旧 BM 为 0.5，且 learner/RND/target 更新不一致。可保留为待复验候选，不能据此声称稳定优于 BM。',
    '- 6h 基础组合 eps075 的 seed80/81 在约 5M 停止时仍零胜率；它们不是完整 10M 结果，但必须在稳健性讨论中保留。',
    '- 优先修正当前实现已验证的 TD(lambda) 序列末尾奖励遗漏、辅助 DVD loss 的 hidden-state 梯度路径、直接 qmix_without_abs 分支的默认 abs 参数，再做必要的有限配对复验。当前记录没有保存 learner/mixer 源码，不能确认上述问题在历史服务器运行中是否完全相同。',
    '## 3s5z 四种子完整结果',
]
selected = [s for s in summary if s['map']=='3s5z_vs_3s6z' and s['n']==4]
report.append(table(selected, [('method','方法'),('n','种子数'),('auc_0_10m','AUC'),('auc_0_10m_sd','跨种子样本SD'),('win_9_10m','9–10M胜率'),('win_9_10m_sd','跨种子样本SD')]))
report += ['## safe-v2 与 BM 的逐种子配对', table(safe_pairs, [('seed','Seed'),('a_auc','safe-v2 AUC'),('b_auc','BM AUC'),('delta_auc','AUC差'),('a_tail','safe-v2后期'),('b_tail','BM后期'),('delta_tail','后期差')]), '![逐种子曲线](safev2_paired.png)', '## safe-v2 后期机制指标', table(diag_rows, [('seed','Seed'),('gate_mean','Gate均值'),('gate_std','Gate标准差'),('gate_target_correlation','Gate/标签相关性'),('route_positive_rate','正标签比例'),('route_reliable_rate','绝对门槛通过比例'),('absolute_advantage_mean','平均绝对优势')]), '这些是有效样本内统计量的 9–10M 时间均值。seed80/81 的 gate/标签相关性约 0.075/0.069，状态依赖监督跟随较弱；这不是严格常数 gate，但也不能仅凭 gate 均值接近标签均值证明路由有效。所有 seed 的平均绝对优势为负，不排除局部正优势，但需要区分局部拟合优势与策略改善。', '## 6h Credit DVD 与关闭修正的对照', table(credit_pairs, [('seed','Seed'),('a_auc','DVD-on AUC'),('b_auc','DVD-off AUC'),('delta_auc','AUC差'),('a_tail','DVD-on后期'),('b_tail','DVD-off后期'),('delta_tail','后期差')]), '四个 seed 的 Sacred 配置均只在 name 和 credit_gate_warmup_steps 上不同：DVD-on 为 1M，DVD-off 为 20M。在当前代码下，后者在 10.05M 训练中不启用 DVD 修正。Credit 构造会消耗额外初始化随机数，因此与更早的直接 matched_bm 运行不是同一初始化对照；DVD-on/off 配对更有解释力。', '![平均曲线](comparison.png)', '## 旧基础版和历史 BM（仅 seed79，配置未匹配）']
old_rows = [s for s in summary if s['method'] in ['dvd_nqmix_on_3s5z_vs_3s6z','nqmix_on_3s5z_vs_3s6z','dvd_nqmix_on_6h_vs_8z','nqmix_on_6h_vs_8z','dvd_nqmix_on_MMM2','nqmix_on_MMM2']]
report.append(table(old_rows, [('map','地图'),('method','方法'),('auc_0_10m','AUC'),('win_9_10m','9–10M胜率'),('final','最后单点评估')]))
report += ['## 不完整与数值失败记录', table([d['row'] for d in details if not d['row']['full_10m']], [('name','运行'),('last_step','最后评估步'),('final','最后胜率'),('never_won','记录中始终零胜率')]), '\n'.join('- ' + d['row']['name'] + '__' + d['row']['timestamp'] for d in details if d['row']['nonfinite_tags']), '## 全部完整、数值有限运行的分组汇总', table(summary, [('map','地图'),('method','方法'),('n','种子数'),('auc_0_10m','AUC'),('win_9_10m','9–10M胜率')]), '## 可复核文件', '- runs.csv：全部 84 个运行，含配置路径、事件目录、覆盖范围及异常标记。\n- details.json.gz：全部标量序列、配置和元数据。\n- paired_config_differences.json：逐种子配置差异。\n- issues.json：两次 1 秒时间差匹配记录。\n- analyze_experiments.py / report_experiments.py：重算脚本位于上级 analysis 目录。']
(OUT / 'report.md').write_text('\n\n'.join(report) + '\n', encoding='utf-8')
print('Report and two figures saved:', str(OUT))
print('SAFE_PAIRS', json.dumps(safe_pairs))
print('CREDIT_PAIRS', json.dumps(credit_pairs))
