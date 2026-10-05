"""Verify and summarize the October 5 audit update without changing training."""
import gzip
import hashlib
import json
import subprocess
from pathlib import Path, PurePosixPath

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from summarize_5m import window_mean

OUT = Path(__file__).parent / 'audit_update_2026-10-05'
details = json.loads(gzip.decompress((OUT / 'details.json.gz').read_bytes()))
previous = json.loads(gzip.decompress((OUT.parent / 'audit_update_2026-10-04/details.json.gz').read_bytes()))
by_name = {d['row']['name']: d for d in details}
assert len(by_name) == len(details) == 16
assert all(d['row']['full_10m'] and not d['row']['nonfinite_tags'] for d in details)
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

def diff(a, b):
    return {k: [a.get(k), b.get(k)] for k in sorted(a.keys() | b.keys()) if a.get(k) != b.get(k)}

bm81 = get('bm_td', '3s5z_vs_3s6z', 81)
both81 = get('router_both', '3s5z_vs_3s6z', 81)
td81 = get('router_td', '3s5z_vs_3s6z', 81)
nomix81 = get('router_both_nomix', '3s5z_vs_3s6z', 81)
config_checks = {'td81_vs_both81': diff(td81['config'], both81['config']),
                 'nomix81_vs_both81': diff(nomix81['config'], both81['config'])}
assert set(config_checks['td81_vs_both81']) == {'name', 'audit_isolate_dvd_aux_hidden'}
assert set(config_checks['nomix81_vs_both81']) == {'name', 'counterfactual_mix_loss_weight'}
shared = sorted(nomix81['series'].keys() & bm81['series'].keys())
exact = [k for k in shared if nomix81['series'][k] == bm81['series'][k]]
assert 'test_battle_won_mean' in exact
equivalence = {'shared_tags': len(shared), 'exact_tags': exact,
               'different_tags': [k for k in shared if k not in exact],
               'win_points': len(bm81['series']['test_battle_won_mean']),
               'scope': 'Observed scalar series only; no saved parameter trajectory comparison.'}
(OUT / 'nomix_bm_equivalence.json').write_text(json.dumps(equivalence, indent=2), encoding='utf-8')

metrics = []
for d in details:
    row, wins = d['row'], d['series']['test_battle_won_mean']
    metrics.append({'name': row['name'], 'map': row['map'], 'seed': row['seed'],
                    'auc_0_5m': window_mean(wins, 0, 5_000_000),
                    'win_4_5m': window_mean(wins, 4_000_000, 5_000_000),
                    'auc_0_10m': row['auc_0_10m'], 'win_9_10m': row['win_9_10m']})
(OUT / 'all_budget_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
metric_by = {r['name']: r for r in metrics}
pairs = []
for seed in (79, 80):
    a, b = get('router_both', '6h_vs_8z', seed), get('bm_td', '6h_vs_8z', seed)
    changes = diff(a['config'], b['config'])
    allowed = {'mixer', 'name', 'audit_source_manifest', 'audit_isolate_dvd_aux_hidden'}
    assert not [k for k in changes if k not in allowed and not k.startswith(('counterfactual_', 'adaptive_', 'takeover_'))]
    config_checks['6h_pair{}'.format(seed)] = changes
    x, y = metric_by[a['row']['name']], metric_by[b['row']['name']]
    pairs.append({'seed': seed, **{k: {'router': x[k], 'bm': y[k], 'delta': x[k] - y[k]}
                                 for k in ('auc_0_5m', 'win_4_5m', 'auc_0_10m', 'win_9_10m')}})
(OUT / 'config_comparison.json').write_text(json.dumps(config_checks, indent=2), encoding='utf-8')
means = {k: {method: float(np.mean([p[k][method] for p in pairs])) for method in ('router', 'bm', 'delta')}
         for k in ('auc_0_5m', 'win_4_5m', 'auc_0_10m', 'win_9_10m')}
(OUT / '6h_paired_metrics.json').write_text(json.dumps({'pairs': pairs, 'paired_mean': means}, indent=2), encoding='utf-8')

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True, constrained_layout=True)
grid = np.arange(0, 10_000_001, 20_000)
specs = [
    [('bm_td', 'BM: fixed TD', '#27364a', '-'), ('router_both', 'Router: both, mix=0.25', '#bf4c45', '-'),
     ('router_td', 'Router: TD only, mix=0.25', '#b18535', '-'),
     ('router_both_nomix', 'Router: both, mix=0 (overlaps BM)', '#28845b', '--')],
    [('bm_td', 'BM: fixed TD', '#27364a', '-'), ('router_both', 'Router: both, mix=0.25', '#bf4c45', '-')],
]
for ax, map_name, seed, lines in zip(axes, ['3s5z_vs_3s6z', '6h_vs_8z'], [81, 80], specs):
    for method, label, color, style in lines:
        x, y = np.asarray(get(method, map_name, seed)['series']['test_battle_won_mean']).T
        curve = np.interp(grid, x, y)
        curve = np.convolve(np.pad(curve, (5, 5), mode='edge'), np.ones(11) / 11, mode='valid')
        ax.plot(grid / 1e6, curve, label=label, color=color, linestyle=style, linewidth=1.8)
    ax.axvline(5, color='#777777', linestyle=':', alpha=.6)
    ax.set(title='{} | seed {}'.format(map_name, seed), xlabel='Environment steps (millions)', ylim=(-.02, 1.02))
    ax.grid(alpha=.15)
    ax.legend(loc='lower right', fontsize=8)
axes[0].set_ylabel('Test win rate')
fig.suptitle('October 5 update | 0.20M display smoothing; dotted line marks the 5M evaluation budget', fontsize=11)
fig.savefig(OUT / 'update_curves.png', dpi=180)
fig.savefig(OUT / 'update_curves.svg')
plt.close(fig)

commands = [(0, '3s5z_vs_3s6z', 79), (1, '3s5z_vs_3s6z', 81), (2, '6h_vs_8z', 79), (3, '6h_vs_8z', 80)]
launch = '\n\n'.join('CUDA_VISIBLE_DEVICES={} python3 src/main_audit.py --config=dvd_audit_router --env-config=sc2 with '
    'env_args.map_name={} use_tensorboard=True seed={} name=audit_router_both_mix005_5m_{}_s{} '
    'audit_fix_td_lambda=True audit_isolate_dvd_aux_hidden=True counterfactual_mix_loss_weight=0.05 '
    't_max=5050000'.format(gpu, map_name, seed, map_name, seed) for gpu, map_name, seed in commands)
(OUT / 'next_round_commands.sh').write_text('# Run each command in its own terminal from ~/pymarl2.\n\n' + launch + '\n', encoding='utf-8')

def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join('---' for _ in headers) + ' |'] +
                     ['| ' + ' | '.join(str(v) for v in row) + ' |' for row in rows])

chosen = [bm81, both81, td81, nomix81, get('router_aux', '3s5z_vs_3s6z', 81),
          get('bm_td', '6h_vs_8z', 80), get('router_both', '6h_vs_8z', 80)]
labels = ['3s81 BM', '3s81 Router both', '3s81 Router TD-only', '3s81 Router both / mix=0',
          '3s81 Router aux-only（旧TD诊断）', '6h80 BM', '6h80 Router both']
rows = []
for label, d in zip(labels, chosen):
    m = metric_by[d['row']['name']]
    rows.append([label, f"{m['auc_0_5m']:.4f}", f"{100*m['win_4_5m']:.2f}%",
                 f"{m['auc_0_10m']:.4f}", f"{100*m['win_9_10m']:.2f}%"])
report = [
    '# 2026-10-05：mixed更新定位及5M候选筛选',
    '原始目录D:\\科研\\ablation_results现有100份Sacred配置，较上次新增4组；16组audit评估全部覆盖10M，解析无异常、无NaN/Inf。前12组系列与10月4日逐点一致。16组共320份manifest文件、实际保存快照及本地训练代码核对一致（忽略换行格式），事件开关正确。Sacred仍标记RUNNING，此处仅判断已有评估覆盖，不判断服务器进程是否退出。',
    '## 新结果与关键参照',
    table(['运行', '0–5M AUC', '4–5M胜率', '0–10M AUC', '9–10M胜率'], rows),
    '3s seed81：关闭辅助隔离但保留正确TD时，尾段从22.38%恢复至75.74%，但仍低于BM86.55%；同时它的4–5M只有41.36%，低于both70.27%和BM74.00%，不适合作为5M主比较下的直接替代。',
    '保留两项修正、将mixed直接TD损失权重设0时，尾段恢复至86.55%。不仅汇总相同，test_battle_won_mean全系列（步数和值）与BM逐点相同；19个共有标量中15个逐点相同，包括训练/评估回报、长度、q_taken_mean和target_mean，loss_td/td_error_abs/grad_norm及aux开关不同。这排除了只看终点巧合，但没有比较保存参数轨迹，不能把所有内部更新都称为已证明逐参数相同。',
    '这一结果将排查重点指向mixed直接价值更新与其余更新的组合。权重从0.25变0时，BM价值损失归一化系数也从0.8变1.0，所以不能唯一归因某个具体梯度分量。分散执行用agent Q选动作，mixer用于训练；aux隔离后、mixed直接损失为0时DVD不直接训练agent/BM，因而这次得到BM一致轨迹是结构上可解释的结果。零权重是诊断对照，不能包装为DVD已产生性能增益。',
    '6h seed80：Router学习明显变慢，4–5M40.23%对BM70.07%，差-29.84pp；0–5M AUC0.1157对0.4781。继续训练后追到9–10M73.86%，仍低于BM79.61%。改为5M比较不会消除该地图的问题。',
    '6h79/80严格配对均值：4–5M Router {:.2f}%、BM {:.2f}%；0–5M AUC {:.4f}/{:.4f}。9–10M均值 {:.2f}%/{:.2f}%，0–10M AUC {:.4f}/{:.4f}。目前不能说当前Router在6h获得一致提升。'.format(
        100*means['win_4_5m']['router'], 100*means['win_4_5m']['bm'], means['auc_0_5m']['router'], means['auc_0_5m']['bm'],
        100*means['win_9_10m']['router'], 100*means['win_9_10m']['bm'], means['auc_0_10m']['router'], means['auc_0_10m']['bm']),
    '![更新曲线](update_curves.png)',
    '## 下一轮：四组固定小mixed权重，全部5M',
    '仅将counterfactual_mix_loss_weight从0.25降到0.05，保留正确TD、辅助隔离、路由、学习率、RND及其6M衰减时钟。归一化后的value loss从0.2 L_mix+0.8 L_BM变为约0.047619 L_mix+0.952381 L_BM；这是损失系数，不是实际梯度幅度比例，也不是gate上限。选择一个非零候选以保留DVD参与agent训练，并降低已定位的mixed更新影响；它是否提升效果需要本轮验证。',
    table(['GPU', '地图、种子', '问题'], [[0, '3s79', '是否仍保留对BM79失败轨迹的改善'],
        [1, '3s81', '是否在有限预算下至少恢复BM附近'], [2, '6h79', '是否保留首组接近BM的表现'],
        [3, '6h80', '是否解除启动明显变慢的问题']]),
    '统一t_max=5050000，比较0–5M AUC与4–5M平均胜率。已有BM与原0.25权重运行直接截取前5M作同种子对照，不重跑基线。已读训练流程中t_max只决定终止，不重新缩放epsilon/RND等绝对步数时钟。5M81用于候选筛选，不能确认10M后期回退已消除；如候选在两个地图有效，再安排少量长训练复验。',
    '```bash\n' + launch + '\n```',
    '本轮不要同时改辅助loss、学习率、RND或gate阈值，也不先扩展更多种子。若0.05在两地图都恢复且有非零DVD收益，再冻结候选补3s80/82及6h81/82配对。若只恢复BM附近而无增益，保留为稳定性消融。若6h80仍明显慢，先停止该维度细扫，转向mixed梯度路径的结构性诊断；当前不保证该候选会成功。已知失败种子保留在结果中。',
    '少量训练运行应保留逐种子结果和不确定性，参见[NeurIPS 2021评估研究](https://papers.nips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html)。',
    '## 输出',
    'all_budget_metrics.json包含16组双预算指标，6h_paired_metrics.json仅汇总已配对79/80，nomix_bm_equivalence.json保存逐点相同标签及核验范围，config_comparison.json/source_verification.json保存配置和源码核查，next_round_commands.sh保存建议命令。当前仅新增分析文件，训练代码、配置、原始记录及既有历史报告均未改动。',
]
(OUT / 'report.md').write_text('\n\n'.join(report) + '\n', encoding='utf-8')
print(json.dumps({'verified_runs': len(verification), 'nomix_equivalence': equivalence, '6h_means': means,
                  'report': str(OUT / 'report.md')}, ensure_ascii=True))
