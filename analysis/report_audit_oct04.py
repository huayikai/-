"""Read-only analysis of the October 4 audit experiment update."""
import gzip
import hashlib
import json
import subprocess
from pathlib import Path, PurePosixPath

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent / 'audit_update_2026-10-04'
details = json.loads(gzip.decompress((OUT / 'details.json.gz').read_bytes()))
by_name = {d['row']['name']: d for d in details}
previous = json.loads(gzip.decompress((OUT.parent / 'audit_update_2026-10-03/details.json.gz').read_bytes()))
assert all(d['row']['full_10m'] for d in details)
assert all(by_name[d['row']['name']]['series'] == d['series'] for d in previous)

verification, blob_cache = [], {}
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
            errors.append({'path': name, 'error': 'snapshot differs from Git HEAD'})
    for tag in ('audit_fix_td_lambda', 'audit_isolate_dvd_aux_hidden'):
        expected = int(d['config'][tag]) if d['config']['mixer'] == 'dvd_counterfactual_router' or tag == 'audit_fix_td_lambda' else 0
        if any(value != expected for _, value in d['series'][tag]):
            errors.append({'tag': tag, 'error': 'event flag does not match configuration'})
    if d['row']['name'].startswith('audit_bm_'):
        assert not d['config']['takeover_bm_abs']
        assert all(value == 0 for _, value in d['series']['takeover_alpha_mean'])
    verification.append({'name': d['row']['name'], 'manifest_files': len(d['config']['audit_source_manifest']), 'errors': errors})
assert not any(v['errors'] for v in verification), verification
(OUT / 'source_verification.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')

def get(method, map_name, seed):
    return by_name['audit_{}_{}_s{}'.format(method, map_name, seed)]

def differences(a, b):
    return {k: [a.get(k), b.get(k)] for k in sorted(a.keys() | b.keys()) if a.get(k) != b.get(k)}

pairs, config_checks = [], []
for map_name, seeds in [('3s5z_vs_3s6z', [79, 80, 81]), ('6h_vs_8z', [79])]:
    for seed in seeds:
        a, b = get('router_both', map_name, seed), get('bm_td', map_name, seed)
        diff = differences(a['config'], b['config'])
        allowed = {'mixer', 'name', 'audit_source_manifest', 'audit_isolate_dvd_aux_hidden'}
        unexpected = [k for k in diff if k not in allowed and not k.startswith(('counterfactual_', 'adaptive_', 'takeover_'))]
        assert not unexpected, unexpected
        config_checks.append({'map': map_name, 'seed': seed, 'differences': diff, 'unexpected': unexpected})
        x, y = a['row'], b['row']
        pairs.append({'map': map_name, 'seed': seed, 'router_auc': x['auc_0_10m'], 'bm_auc': y['auc_0_10m'],
                      'auc_delta': x['auc_0_10m'] - y['auc_0_10m'], 'router_tail': x['win_9_10m'],
                      'bm_tail': y['win_9_10m'], 'tail_delta': x['win_9_10m'] - y['win_9_10m']})
aux = get('router_aux', '3s5z_vs_3s6z', 81)
both = get('router_both', '3s5z_vs_3s6z', 81)
switch_diff = differences(aux['config'], both['config'])
assert set(switch_diff) == {'name', 'audit_fix_td_lambda'}
(OUT / 'config_comparison.json').write_text(json.dumps({'pairs': config_checks, 'aux_both_seed81': switch_diff}, indent=2), encoding='utf-8')
three = [p for p in pairs if p['map'] == '3s5z_vs_3s6z']
means = {key: float(np.mean([p[key] for p in three])) for key in ('router_auc', 'bm_auc', 'auc_delta', 'router_tail', 'bm_tail', 'tail_delta')}
(OUT / 'paired_metrics.json').write_text(json.dumps({'pairs': pairs, '3s_paired_mean': means,
    '3s_seed81_aux_minus_both_tail': aux['row']['win_9_10m'] - both['row']['win_9_10m']}, indent=2), encoding='utf-8')

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
grid = np.arange(0, 10_000_001, 20_000)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True, constrained_layout=True)
specs = [
    [('bm_td', 'BM: fixed TD', '#2878b5'), ('router_both', 'Router: fixed TD + isolated aux', '#bf4c45'),
     ('router_aux', 'Router: old TD + isolated aux (diagnostic)', '#28845b')],
    [('bm_td', 'BM: fixed TD', '#2878b5'), ('router_both', 'Router: fixed TD + isolated aux', '#28845b')],
]
for ax, map_name, seed, lines in zip(axes, ['3s5z_vs_3s6z', '6h_vs_8z'], [81, 79], specs):
    for method, label, color in lines:
        x, y = np.asarray(get(method, map_name, seed)['series']['test_battle_won_mean']).T
        curve = np.interp(grid, x, y)
        curve = np.convolve(np.pad(curve, (5, 5), mode='edge'), np.ones(11) / 11, mode='valid')
        ax.plot(grid / 1e6, curve, label=label, color=color, linewidth=1.8)
    ax.set(title='{} | seed {}'.format(map_name, seed), xlabel='Environment steps (millions)', ylim=(-.02, 1.02))
    ax.grid(alpha=.15)
    ax.legend(loc='lower right', fontsize=8)
axes[0].set_ylabel('Test win rate')
fig.suptitle('October 4 update | 0.20M display smoothing; metrics use unsmoothed records', fontsize=11)
fig.savefig(OUT / 'update_curves.png', dpi=180)
fig.savefig(OUT / 'update_curves.svg')
plt.close(fig)

commands = [
    ('0', 'dvd_audit_router', '3s5z_vs_3s6z', 81, 'audit_router_td_3s5z_vs_3s6z_s81', True, False, ''),
    ('1', 'dvd_audit_router', '3s5z_vs_3s6z', 81, 'audit_router_both_nomix_3s5z_vs_3s6z_s81', True, True, ' counterfactual_mix_loss_weight=0.0'),
    ('2', 'dvd_audit_bm', '6h_vs_8z', 80, 'audit_bm_td_6h_vs_8z_s80', True, False, ''),
    ('3', 'dvd_audit_router', '6h_vs_8z', 80, 'audit_router_both_6h_vs_8z_s80', True, True, ''),
]
launch = '\n\n'.join('CUDA_VISIBLE_DEVICES={} python3 src/main_audit.py --config={} --env-config=sc2 with '
    'env_args.map_name={} use_tensorboard=True seed={} name={} audit_fix_td_lambda={} '
    'audit_isolate_dvd_aux_hidden={}{}'.format(*c) for c in commands)
(OUT / 'next_round_commands.sh').write_text('# Run each command in its own terminal from ~/pymarl2.\n\n' + launch + '\n', encoding='utf-8')

def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join('---' for _ in headers) + ' |'] +
                     ['| ' + ' | '.join(str(v) for v in row) + ' |' for row in rows])

rows = [[p['map'], p['seed'], f"{p['router_auc']:.4f}", f"{p['bm_auc']:.4f}",
         f"{100*p['router_tail']:.2f}%", f"{100*p['bm_tail']:.2f}%", f"{100*p['tail_delta']:+.2f}pp"] for p in pairs]
report = [
    '# 2026-10-04：seed81 定位与 6h 首组配对',
    '原始目录 D:\\科研\\ablation_results：96份Sacred配置，较上轮新增4组。12组audit评估均覆盖10M，解析无异常、无NaN/Inf。旧8组事件系列与上轮逐点一致；12组共240份manifest文件均与实际快照及本地Git HEAD一致（仅忽略换行差异）。事件开关与配置一致。Sacred仍标记RUNNING，不能据此认定服务器进程已经退出。',
    '## 正式修正版配对',
    table(['地图', 'Seed', 'Router AUC', 'BM AUC', 'Router 9–10M', 'BM 9–10M', '尾段差值'], rows),
    '3s三种子严格配对均值：Router AUC {:.4f}、BM {:.4f}；尾段 {:.2f}%、{:.2f}%，平均差值{:+.2f}pp。但79/80/81的尾段差值分别为+82.38、-3.26、-64.17pp，两次落后、一次大幅领先。均值优势受BM79失败强烈影响，当前不能描述为稳定提升。'.format(means['router_auc'], means['bm_auc'], 100*means['router_tail'], 100*means['bm_tail'], 100*means['tail_delta']),
    '## 本轮确认和未确认的内容',
    '3s seed81：修正版BM后期86.55%，双修正版Router22.38%；aux-only Router84.50%。aux-only与both的配置仅name和TD开关不同，因此在该种子、该辅助隔离条件下，改变TD目标处理伴随62.12pp的后期差异。它支持优先检查目标处理与Router更新的组合，而非判定修正公式数学错误。BM81正确目标仍能学好，也不能判定TD修正普遍有害。改变目标会改变后续采样轨迹，一次同种子配置比较没有识别唯一内部机制。',
    '6h seed79：Router AUC0.6086、BM0.6010；尾段76.90%、75.58%，差值+1.32pp。最后一次评估84.38%对68.75%放大了差异，不能用单点评估代替尾段均值。当前可说该次跨地图配对未出现明显劣化，不能说已获得可靠提升。',
    '![更新曲线](update_curves.png)',
    '## 下一轮四卡',
    table(['GPU', '实验', '目的'], [
        [0, '3s Router TD-only seed81', '在正确TD下关闭辅助隔离，与已有both81比较；联合aux81定位开关组合'],
        [1, '3s Router both，mix损失权重=0，seed81', '保留正确TD和辅助隔离，测试mixed直接TD更新是否是必要影响因素'],
        [2, '6h BM TD seed80', '第二组跨地图配对基线'],
        [3, '6h Router both seed80', '检验首组1.32pp差异是否重复出现'],
    ]),
    '无须修改或重新提交训练代码。GPU0、1均使用正确TD目标。GPU1只去掉mixed直接TD损失，BM价值损失从归一化系数0.8变为1.0，DVD辅助、route损失和推理时mixed组合仍保留；它不是纯BM，也不等同于推理阶段完全关闭DVD。目标仍由target BM锚定。辅助/route参数的全局梯度裁剪仍可能影响BM更新，因此不能宣称该实验与纯BM逐步完全相同。',
    '```bash\n' + launch + '\n```',
    '## 下一轮之后如何决定',
    '若TD-only81恢复，优先检查正确TD与辅助隔离组合；若nomix81恢复，则mixed直接价值更新是该条件下的重要影响因素，可在79或82复验，并再测试较小正权重（如0.05）而非直接把0权重作为最终方法。若两者都失败，先复验seed81与检查数据边界处理/学习率，不扩大gate网格搜索。所有结果只能定位当前配置条件下的因素，不能作为唯一因果机制证明。',
    '若6h80再次只出现小差异，继续补81/82配对以判断方差；若明显落后，应保留跨地图局限并优先改进3s已定位的稳定性问题。后续补3s BM/Router82完成四种子；若方法调整，必须用同一最终配置重新组成完整配对，不能拼接各地图/种子最好的不同版本。',
    '论文目前能支持的结论是结构在部分运行中有效、存在稳定性和目标处理敏感性；尚不能宣称两地图一致优越或两个实现修正都提升胜率。实现正确性、算法贡献和经验性能应分别陈述。少量运行应报告不确定性，参见[NeurIPS 2021评估研究](https://proceedings.neurips.cc/paper_files/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html)。',
    '## 保存内容',
    'runs.csv/details.json.gz保存12组audit原始解析，paired_metrics.json保存配对结果，source_verification.json保存快照核验，config_comparison.json保存配对差异，next_round_commands.sh保存四卡命令。原始日志与训练代码未修改。',
]
(OUT / 'report.md').write_text('\n\n'.join(report) + '\n', encoding='utf-8')
print(json.dumps({'verified_runs': len(verification), 'pairs': pairs, '3s_mean': means, 'report': str(OUT / 'report.md')}, ensure_ascii=True))
