"""Read-only source verification and paired analysis for the 2026-10-03 update."""
import gzip
import hashlib
import json
import subprocess
from pathlib import Path, PurePosixPath

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent / 'audit_update_2026-10-03'
details = json.loads(gzip.decompress((OUT / 'details.json.gz').read_bytes()))
history = json.loads(gzip.decompress((OUT / 'historical_reference/details.json.gz').read_bytes()))
routers = {d['row']['seed']: d for d in details if '_both_' in d['row']['name']}
bms = {d['row']['seed']: d for d in details if d['row']['name'].startswith('audit_bm_td_')}
blob_cache = {}
verification = []
for d in details:
    parent = Path(d['row']['config_path']).parent.parent
    saved = {name: parent / store for name, store in d['run']['experiment']['sources']}
    for original, store in d['run']['resources']:
        saved[original.split('/src/', 1)[1]] = parent / '_resources' / PurePosixPath(store).name
    errors = []
    for item in d['config']['audit_source_manifest']:
        name = item['path']
        if name not in saved or not saved[name].is_file():
            errors.append({'path': name, 'error': 'snapshot missing'})
            continue
        blob = saved[name].read_bytes()
        if hashlib.sha256(blob).hexdigest() != item['sha256']:
            errors.append({'path': name, 'error': 'snapshot does not match manifest'})
        if name not in blob_cache:
            blob_cache[name] = subprocess.run(['git', 'show', 'HEAD:' + name], check=True,
                                              stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
        if blob.replace(b'\r\n', b'\n') != blob_cache[name].replace(b'\r\n', b'\n'):
            errors.append({'path': name, 'error': 'snapshot differs from current Git source'})
    verification.append({'name': d['row']['name'], 'manifest_files': len(d['config']['audit_source_manifest']), 'errors': errors})
if any(v['errors'] for v in verification):
    raise RuntimeError(json.dumps(verification, ensure_ascii=False))
(OUT / 'source_verification.json').write_text(json.dumps(verification, ensure_ascii=False, indent=2), encoding='utf-8')

paired = []
for seed in sorted(routers.keys() & bms.keys()):
    a, b = routers[seed]['row'], bms[seed]['row']
    paired.append({'seed': seed, 'router_auc': a['auc_0_10m'], 'bm_auc': b['auc_0_10m'],
                   'auc_delta': a['auc_0_10m'] - b['auc_0_10m'],
                   'router_tail': a['win_9_10m'], 'bm_tail': b['win_9_10m'],
                   'tail_delta': a['win_9_10m'] - b['win_9_10m']})
means = {key: float(np.mean([p[key] for p in paired])) for key in paired[0] if key != 'seed'}
(OUT / 'paired_metrics.json').write_text(json.dumps({'pairs': paired, 'paired_mean': means}, indent=2), encoding='utf-8')

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False})
grid = np.arange(0, 10_000_001, 20_000)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), sharex=True, sharey=True, constrained_layout=True)
for seed, ax in zip([79, 80, 81], axes):
    specs = [(routers[seed], 'Router: fixed TD + isolated aux', '#28845b', '-')]
    if seed in bms:
        specs.append((bms[seed], 'BM: fixed TD', '#2878b5', '-'))
    else:
        old = next(d for d in history if d['row']['name'] == 'counterfactual_router_safev2_3s5z_s81')
        specs.append((old, 'Historical router: old targets/aux', '#a7773b', '--'))
        ax.text(.03, .52, 'Matched BM seed 81 pending', transform=ax.transAxes, fontsize=9)
    for d, label, color, style in specs:
        x, y = np.asarray(d['series']['test_battle_won_mean']).T
        curve = np.interp(grid, x, y)
        curve = np.convolve(np.pad(curve, (5, 5), mode='edge'), np.ones(11) / 11, mode='valid')
        ax.plot(grid / 1e6, curve, label=label, color=color, linestyle=style, linewidth=1.8)
    ax.axvline(6, color='#999999', linestyle=':', alpha=.5)
    ax.set(title='Seed {}'.format(seed), xlabel='Environment steps (millions)', ylim=(-.02, 1.02))
    ax.grid(alpha=.15)
    ax.legend(loc='lower left', fontsize=8)
axes[0].set_ylabel('Test win rate')
fig.suptitle('3s5z_vs_3s6z | matched targets for current BM / Router | 0.20M display smoothing', fontsize=12)
fig.savefig(OUT / 'paired_curves.png', dpi=180)
fig.savefig(OUT / 'paired_curves.svg')
plt.close(fig)


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join('---' for _ in headers) + ' |'] +
                     ['| ' + ' | '.join(str(v) for v in row) + ' |' for row in rows])


commands = [
    ('0', 'dvd_audit_bm', '3s5z_vs_3s6z', 81, 'audit_bm_td_3s5z_vs_3s6z_s81', True, False),
    ('1', 'dvd_audit_router', '3s5z_vs_3s6z', 81, 'audit_router_aux_3s5z_vs_3s6z_s81', False, True),
    ('2', 'dvd_audit_bm', '6h_vs_8z', 79, 'audit_bm_td_6h_vs_8z_s79', True, False),
    ('3', 'dvd_audit_router', '6h_vs_8z', 79, 'audit_router_both_6h_vs_8z_s79', True, True),
]
launch = '\n\n'.join(
    'CUDA_VISIBLE_DEVICES={} python3 src/main_audit.py --config={} --env-config=sc2 with '
    'env_args.map_name={} use_tensorboard=True seed={} name={} audit_fix_td_lambda={} '
    'audit_isolate_dvd_aux_hidden={}'.format(*c) for c in commands)
(OUT / 'next_round_commands.sh').write_text('# Run each command in its own terminal from the server repository root.\n\n' + launch + '\n', encoding='utf-8')
main_rows = []
for seed in [79, 80, 81]:
    a = routers[seed]['row']
    b = bms[seed]['row'] if seed in bms else None
    main_rows.append([seed, f"{a['auc_0_10m']:.4f}", f"{b['auc_0_10m']:.4f}" if b else '待跑',
                      f"{100*a['win_9_10m']:.2f}%", f"{100*b['win_9_10m']:.2f}%" if b else '待跑'])
z_rows = []
for seed in [79, 80, 81]:
    z = routers[seed]['diagnostics_9_10m']
    z_rows.append([seed, f"{z['counterfactual_gate_mean']:.4f}",
                   f"{z['counterfactual_gate_target_correlation']:.4f}",
                   f"{z['counterfactual_absolute_advantage_mean']:.5f}"])
report = [
    '# 2026-10-03：修正版 BM 配对与 Router seed81 回退分析',
    '目录为 `D:\\科研\\ablation_results`。本轮总计92份 Sacred 配置，相较上次新增4个运行：修正版 BM79/80 与双修正版 Router80/81。全部8个 audit 运行的评估已覆盖10M，全部标量有限，无事件解析异常。Sacred 仍为 RUNNING，不能据此判断实际进程是否已经退出；比较使用已有的0–10M完整区间。',
    '## 主要结果',
    table(['Seed', 'Router AUC', '修正版 BM AUC', 'Router 9–10M', '修正版 BM 9–10M'], main_rows),
    'seed79：Router显著改善该次运行，修正版 BM 却落入低胜率。seed80：Router AUC 增加0.0373，后期低3.26个百分点，表现为累计学习效率改善而非后期全面优于 BM。seed81：Router 3–5M胜率64.19%，5–7M为53.50%，7–9M为34.56%，9–10M为22.38%，存在持续回退。该 seed 的同目标 BM 尚缺，不能断言回退是 Router 特有。',
    '79/80的严格配对均值：Router AUC {:.4f}，BM {:.4f}；尾段 {:.2f}% / {:.2f}%。均值优势主要来自 BM79 失败，不足以认定普遍优势。Router三个种子均值不可与仅两个种子的 BM 均值直接作公平比较。'.format(means['router_auc'], means['bm_auc'], 100*means['router_tail'], 100*means['bm_tail']),
    '![配对曲线](paired_curves.png)',
    '## 可比性与源码检查',
    '逐个核对8组各20份manifest文件：SHA-256均与实际源码/资源快照一致，忽略Windows/Linux换行格式后均与本地b7d1d30相同。BM和Router使用同一个新learner、相同TD修正、epsilon、RND、target软更新、优化器和训练预算；方法专属参数不同，算法配置外不存在共享训练参数差异。BM的takeover_bm_abs=False，warm-up=20M，事件takeover_alpha_mean始终为0，确认为训练期间纯BM。开关记录与配置一致。',
    '历史 BM79（旧目标）AUC0.3747、后期77.55%，新BM79 AUC0.0177、后期0.03%；新BM80仍能达到85.87%。目标计算正确不保证每个有限预算运行都更好，目标修正也会改变学习轨迹。现有数据不支持把旧目标恢复为正式主结果；旧代码没有完整快照，历史比较是排查线索。',
    '旧safe-v2 seed81后期87.07%，当前双修正版22.38%。两个开关一起改变，现阶段不能确定由TD边界处理、辅助hidden梯度隔离、mixed分支或其交互造成回退。',
    '## 机制诊断',
    table(['Seed', 'Gate均值', 'Gate/标签相关性', 'DVD平均绝对TD优势'], z_rows),
    '三个种子的gate均值均约0.13，gate/标签相关性约0.064–0.073；seed81的DVD平均绝对优势更负。这提供排查线索，不能证明是回退原因。不同训练访问分布上的TD误差也不能直接比较为因果证据。6M处RND衰减归零，图中竖线只标明该已知时间点，不表示已识别回退由RND导致。',
    '## 下一轮四卡',
    table(['GPU', '任务', '目的'], [
        [0, '3s 修正版 BM seed81', '区分两种方法共同失败与Router特有回退'],
        [1, '3s Router aux-only seed81', '与已有both81只差TD开关，定位目标处理变化的影响'],
        [2, '6h 修正版 BM seed79', '跨地图配对基线'],
        [3, '6h Router both seed79', '冻结主配置的跨地图验证'],
    ]),
    'aux-only故意保留旧TD计算，仅作诊断，不能用它代替修正版BM的正式主比较，也不能将仅seed79的好结果作为选用旧目标的依据。所有命令使用已有配置和开关，无须修改训练代码。',
    '```bash\n' + launch + '\n```',
    '## 结果出来后的分支',
    '若BM81同样回退，优先排查共同训练动力学；若BM81稳定而Router81回退，优先做counterfactual_mix_loss_weight=0的mixed更新消融。若aux-only81恢复，说明在该条件下TD处理变化是关键影响因素之一；若它也失败，需要再补TD-only81来检验辅助梯度隔离及两项交互，不能直接归因mixed。此后补3s BM/Router82与6h BM/Router80，再按跨地图结果决定补81/82与定向改动。保留全部失败结果，按配对AUC、尾段和回退幅度判断，暂不扫gate超参。',
    '## 文件',
    'runs.csv与details.json.gz保存全部audit运行；paired_metrics.json仅使用已完成配对79/80；source_verification.json保存逐文件核对；historical_reference/为从当前原始记录重新解析的历史对照；next_round_commands.sh保存本轮建议。原始日志与训练代码未改动。',
]
(OUT / 'report.md').write_text('\n\n'.join(report) + '\n', encoding='utf-8')
print(json.dumps({'paired_means': means, 'verified_runs': len(verification), 'report': str(OUT / 'report.md')}, ensure_ascii=False))
