"""Merge the completed fixed-.13 set; audit new rows and fingerprint cached rows."""
import copy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DATA = REPO.parent
FIRST = json.loads((HERE.parent / '2026-11-04_FIXED_GATE_013/config_check.json').read_text(encoding='utf-8'))
NEXT = json.loads((HERE.parent / '2026-11-05_FIXED_GATE_013_NEXT/config_check.json').read_text(encoding='utf-8'))
REFERENCES = json.loads((HERE.parent / '2026-11-03_DIRECT_COMBINATION_COMPLETE/run_summary.json').read_text(encoding='utf-8'))['rows']
CACHED = json.loads((HERE.parent / '2026-11-05_FIXED_GATE_013_RESULTS/run_summary.json').read_text(encoding='utf-8'))['rows']
CACHED_RAW = json.loads((HERE.parent / '2026-11-05_FIXED_GATE_013_RESULTS/raw_curves_and_diagnostics.json').read_text(encoding='utf-8'))

spec = importlib.util.spec_from_file_location('first_fixed013_reader', HERE.parent / '2026-11-05_FIXED_GATE_013_RESULTS/summarize.py')
first_reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(first_reader)
reader = first_reader.reader


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unchanged(row):
    return all(sha(Path(row[k]['config_path'])) == row[k]['config_sha256'] and all(sha(Path(e['path'])) == e['sha256'] for e in row[k]['event_files']) for k in ('fixed013', 'dynamic', 'bm'))


def refresh_sacred(result, raw):
    path = Path(result['config_path']).parent / 'info.json'
    try:
        info = json.loads(path.read_text(encoding='utf-8'))
        error = None
    except (OSError, ValueError) as exc:
        info, error = {}, str(exc)
    sacred = dict(zip(info.get('test_battle_won_mean_T', []), info.get('test_battle_won_mean', [])))
    wins = dict(raw['test_battle_won_mean'])
    shared = sorted(set(sacred) & set(wins))
    mismatches = [s for s in shared if abs(sacred[s] - wins[s]) > 1e-7]
    assert not mismatches
    result['checks'].update(info_error=error, shared_sacred_tb_points=len(shared), shared_sacred_win_mismatches=mismatches)


def verify(row, raw, item):
    fixed = row['fixed013']
    config = fixed['config']
    expected = item['effective_config']
    skip = {'audit_source_manifest', 'local_results_path'}
    diff = {k: [expected.get(k), config.get(k)] for k in set(expected) | set(config) if k not in skip and expected.get(k) != config.get(k)}
    assert not diff, diff
    sources = fixed['source_checks']
    assert len(sources) == 22
    for source in sources:
        current = hashlib.sha256((REPO / source['path']).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        assert current == source['recorded_sha256']
        source.update(current_lf_sha256=current, matches_current_lf=True)
    series = raw['fixed013']
    mean = [v for _, v in series['counterfactual_gate_mean']]
    std = [v for _, v in series['counterfactual_gate_std']]
    weights = [v for _, v in series['counterfactual_gate_output_weight_norm']]
    grad = [v for _, v in series['adaptive_gate_grad_norm']]
    assert max(abs(v - .13) for v in mean) < 1e-6
    assert max(std) < 1e-6 and max(weights) == 0 and max(grad) == 0
    assert set(v for _, v in series['audit_detach_mixed_hidden']) == {float(config['audit_detach_mixed_hidden'])}
    fixed['planned_config_differences'] = diff
    fixed['gate_verification'] = {'mean_range': [min(mean), max(mean)], 'maximum_within_batch_std': max(std), 'maximum_output_weight_norm': max(weights), 'maximum_gate_grad_norm': max(grad), 'point_count': len(mean)}
    default = {'adaptive_graph_mode': 'learned', 'adaptive_normalize_dvd_weights': True}
    dynamic = row['dynamic']['config']
    treatment = {k: [dynamic.get(k, default.get(k)), config.get(k)] for k in set(dynamic) | set(config) if k not in ('name', 'audit_source_manifest', 'local_results_path') and dynamic.get(k, default.get(k)) != config.get(k)}
    assert set(treatment) == {'adaptive_gate_init', 'counterfactual_route_loss_weight'}, treatment
    row['dynamic_to_fixed_training_treatment'] = treatment
    row['dynamic_minus_fixed_near_5m_pp'] = 100 * (row['dynamic']['near_5m']['win_rate'] - fixed['near_5m']['win_rate'])
    for key in ('fixed013', 'dynamic', 'bm'):
        assert row[key]['covers_5m'] and row[key]['checks']['nonfinite_simple_scalar_count'] == 0
        refresh_sacred(row[key], raw[key])


def main():
    by_pair = {(r['map'], r['seed']): r for r in FIRST['first_batch'] + NEXT['runs']}
    rows, curves = [], []
    for map_name, seeds in FIRST['fixed_development_set'].items():
        for seed in seeds:
            item = by_pair[(map_name, seed)]
            cached = next((r for r in CACHED if r['map'] == map_name and r['seed'] == seed), None)
            if cached is not None and unchanged(cached):
                row = copy.deepcopy(cached)
                raw = copy.deepcopy(next(r for r in CACHED_RAW if r['map'] == map_name and r['seed'] == seed))
                cache = True
            else:
                reference = next(r for r in REFERENCES if r['map'] == map_name and r['seed'] == seed)
                paths = {'fixed013': DATA / 'ablation_results_dvd/sacred' / map_name / item['name'] / '1/config.json', 'dynamic': Path(item['paired_dynamic_config_path']), 'bm': Path(reference['bm']['config_path'])}
                row, raw = {'map': map_name, 'seed': seed}, {'map': map_name, 'seed': seed}
                for method, path in paths.items():
                    row[method], raw[method] = reader.read_run(path)
                    row[method]['last_five_within_5_05m'] = [{'step': s, 'win_rate': v} for s, v in raw[method]['test_battle_won_mean'] if s <= 5_050_000][-5:]
                cache = False
            verify(row, raw, item)
            row['new_in_this_return'] = (map_name, seed) in {(r['map'], r['seed']) for r in NEXT['runs']}
            row['unchanged_cached_events_reused'] = cache
            rows.append(row)
            curves.append(raw)
            print(json.dumps({'map': map_name, 'seed': seed, 'new': row['new_in_this_return'], 'cached': cache, 'near5m': {k: row[k]['near_5m'] for k in ('bm', 'fixed013', 'dynamic')}, 'fixed_last': row['fixed013']['last'], 'fixed_n': row['fixed013']['point_count'], 'fixed_gate': row['fixed013']['gate_verification']}, ensure_ascii=False), flush=True)
    assert len(rows) == 8 and sum(r['new_in_this_return'] for r in rows) == 4
    output = {'review_date': '2026-11-05', 'timezone': 'Asia/Shanghai', 'new_fixed_run_count': 4, 'total_fixed_run_count': 8, 'status': 'predefined development set complete; retain per-seed observations and extend prospective independent seeds before overall conclusions', 'metric': 'raw unsmoothed evaluation nearest 5M plus actual final/tail values, no interpolation or time weighting', 'scope': '3s attached, 6h detached; original 6h subset selected during development; .13 approximates late gate amount and changes the initial prior and route supervision; no sample-adaptation-only or population-superiority claim', 'reader': 'Shared read-only TensorBoard EventAccumulator reader, scalar guidance=0. Unchanged first-batch events reused after SHA256 checks, current source fingerprints rechecked, Sacred common wins refreshed. No independent audit of TensorBoard internal CRC checks.', 'summary_fields': 'last/last_five span full logs; historical BM may reach 10M; use near_5m and last_five_within_5_05m for matched budget', 'rows': rows}
    (HERE / 'run_summary.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    with gzip.open(HERE / 'raw_curves_and_diagnostics.json.gz', 'wt', encoding='utf-8') as stream:
        json.dump(curves, stream, ensure_ascii=False)
    write_report(rows)
    write_plot(curves)
    print('WROTE', HERE / 'report.md', flush=True)


def write_report(rows):
    pct = first_reader.percent
    lines = ['# 2026-11-05：固定门控0.13既定八条补齐记录', '',
        '本次新增3s79、6h87/88/90四条，与首批合并，3s79/80/81及6h80/84/87/88/90的固定0.13集合已齐。新四条于北京时间11月5日00:12–00:13启动，均已记录到5M以上。当前结果根目录config数为104+77+91=272。', '',
        '**按用户要求：不同种子方向不一致时先不急着下总体结论，继续补充更多预先固定的种子。** 以下仅为既定开发集合内的逐seed观察，所有正反结果保留。', '',
        '## 最近5M原始胜率', '',
        '取距5,000,000最近的一次未平滑测试，不插值、不时间加权。历史BM即使跑到10M也只取5M附近；实际步数见JSON。', '',
        '| 地图/seed | 纯BM | 固定0.13 | 动态BACM | 动态−固定 | 本批新增 |', '| --- | ---: | ---: | ---: | ---: | --- |']
    for r in rows:
        name = ('3s' if r['map'].startswith('3s') else '6h') + '/' + str(r['seed'])
        lines.append(f"| {name} | {pct(r['bm']['near_5m']['win_rate'])} | {pct(r['fixed013']['near_5m']['win_rate'])} | {pct(r['dynamic']['near_5m']['win_rate'])} | {r['dynamic_minus_fixed_near_5m_pp']:+.3f} pp | {'是' if r['new_in_this_return'] else '否'} |")
    lines += ['', '本次新增四条的5M附近单点均为动态较高；首批6h80的固定较高这一反向观察继续保留。3s80在部分中前段固定较高，6h若干运行的5M单点与末次排序不同，不能用单点排列代替整段学习表现。当前不据此给总体优劣、稳定性或模块普遍必要性下结论。', '',
        '## 本批四条覆盖与末段', '', '| 地图/seed | 固定测试点数 | 固定最近5M步数 | 固定末次步数 | 固定末次 | 动态末次 |', '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:
        if not r['new_in_this_return']:
            continue
        f, d = r['fixed013'], r['dynamic']
        name = ('3s' if r['map'].startswith('3s') else '6h') + '/' + str(r['seed'])
        lines.append(f"| {name} | {f['point_count']} | {f['near_5m']['step']:,} | {f['last']['step']:,} | {pct(f['last']['win_rate'])} | {pct(d['last']['win_rate'])} |")
    lines += ['', '表中保留每条实际最后评估。6h90末次为5,050,071步，比5.05M多71步；同预算末段应读取JSON中的last_five_within_5_05m，曲线图仅显示至5.05M。3s79和6h90的TensorBoard各比Sacred多一个尾部评估，共有步数的胜率一致。', '']
    for r in rows:
        if r['new_in_this_return']:
            name = ('3s' if r['map'].startswith('3s') else '6h') + '/' + str(r['seed'])
            for key, label in [('fixed013', '固定0.13'), ('dynamic', '动态')]:
                lines.append(f"- {name} {label}最后五次：" + '、'.join(pct(p['win_rate']) for p in r[key]['last_five']) + '。')
    lines += ['', '各次评估步数并非完全相同，不把末五次逐项看成相同测试样本或独立训练重复。', '',
        '## 核验与比较范围', '',
        '- 八条固定运行均覆盖5M，实际配置符合事前命令。与各自动态配置的处理差异仍为初始gate 0.05→0.13、路由监督系数0.1→0；BM目标、RMS、learned图聚合、其他损失、RND及地图hidden路径保持。',
        '- 八条固定运行各22项源码/配置指纹与当前本地统一LF内容一致。已记录masked gate均值接近0.13，batch内标准差接近零；门控末层权重范数和门控梯度范数在已记录点均为零。原有零候选信号回退保留。',
        '- 新四条读取完整TensorBoard标量，已复核的首批只在config与event SHA256均未变化时复用数据，再更新当前源码和Sacred交叉核验。24条配对来源的共有胜率无不一致，未发现非有限simple scalar。本次未独立审计TensorBoard内部CRC实现。',
        '- Sacred RUNNING只保留为元数据，不用它判断服务器进程。日志覆盖5M足以完成当前预算比较，不据此推断当前进程状态。',
        '- 固定0.13只是近似后段融合量，不是全程/逐阶段精确匹配，也改变了初始先验和时间轨迹。当前对照不能单独证明逐样本路由或路由损失的作用。原6h集合经开发期间筛选，外推需新独立集合。', '',
        '## 下一步', '',
        '既定八条现已补齐。按用户要求，下一阶段增加预先固定的独立配对种子：既有候选3s84/85、6h92/93，每颗分别比较动态BACM、固定0.13及纯BM，共12条；正式命令前核对无重复，统一5.05M和原始评估口径。新独立验证与旧开发子集分别报告，全部方向保留，暂不凭本批给总体结论。', '',
        '当前仅完成结果整理，未准备或启动这12条新训练。继续聚焦现有两图，MMM2/新任务暂缓，QPLEX/QTRAN/Qatten等外部对照低优先级，创新点和默认主方法不改，RMS扩展暂停。', '',
        '本次未修改训练源码、原始日志或论文正文，未启动训练、提交或推送。', '',
        '[逐运行摘要](D:/论文/代码/experiments/2026-11-05_FIXED_GATE_013_COMPLETE/run_summary.json)；[压缩原始曲线与诊断JSON](D:/论文/代码/experiments/2026-11-05_FIXED_GATE_013_COMPLETE/raw_curves_and_diagnostics.json.gz)；[未平滑曲线图](D:/论文/代码/experiments/2026-11-05_FIXED_GATE_013_COMPLETE/learning_curves.png)。首批四条的 [原报告](D:/论文/代码/experiments/2026-11-05_FIXED_GATE_013_RESULTS/report.md) 继续保留为快照。', '']
    (HERE / 'report.md').write_text('\n'.join(lines), encoding='utf-8')


def write_plot(curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 2, figsize=(13, 14), sharex=True, sharey=True)
    for ax, curve in zip(axes.flat, curves):
        for key, label, color in [('bm', 'BM', '#929292'), ('fixed013', 'Fixed gate 0.13', '#d55e00'), ('dynamic', 'Dynamic BACM', '#0072b2')]:
            points = [(s / 1e6, 100 * v) for s, v in curve[key]['test_battle_won_mean'] if s <= 5_050_000]
            ax.plot(*zip(*points), label=label, color=color, alpha=.85, linewidth=1)
        ax.set_title(f"{curve['map']} / seed {curve['seed']}")
        ax.set(xlim=(0, 5.05), ylim=(-3, 103), xlabel='Environment steps (millions)', ylabel='Test win rate (%)')
        ax.grid(alpha=.18)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(.5, .978))
    fig.suptitle('Completed fixed-gate development set: raw unsmoothed evaluations', y=.997)
    fig.tight_layout(rect=(0, 0, 1, .956))
    fig.savefig(HERE / 'learning_curves.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
