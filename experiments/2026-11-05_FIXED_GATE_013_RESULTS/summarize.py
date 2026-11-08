"""Audit the returned first fixed-.13 batch and its preregistered pairs."""
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DATA = REPO.parent
PLAN_PATH = HERE.parent / '2026-11-04_FIXED_GATE_013/config_check.json'
PLAN = json.loads(PLAN_PATH.read_text(encoding='utf-8'))
REFERENCES = json.loads((HERE.parent / '2026-11-03_DIRECT_COMBINATION_COMPLETE/run_summary.json').read_text(encoding='utf-8'))['rows']

# Reuse the read-only scalar/provenance reader; do not invoke its training-free main.
spec = importlib.util.spec_from_file_location('mmm2_log_reader', HERE.parent / '2026-11-04_MMM2_RESULTS/summarize.py')
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)
EXTRA_TAGS = (
    'counterfactual_gate_mean', 'counterfactual_gate_std', 'counterfactual_gate_min',
    'counterfactual_gate_max', 'counterfactual_gate_output_weight_norm',
    'adaptive_gate_grad_norm', 'counterfactual_loss_bm', 'counterfactual_loss_dvd',
    'counterfactual_loss_route', 'counterfactual_loss_value',
    'counterfactual_diag_mae_gain_mean', 'counterfactual_diag_benefit_rate',
    'counterfactual_diag_count', 'counterfactual_diag_nonfinite_count',
)
reader.DIAGNOSTICS = tuple(dict.fromkeys(reader.DIAGNOSTICS + EXTRA_TAGS))


def percent(x):
    return f'{100 * x:.3f}'.rstrip('0').rstrip('.') + '%'


def main():
    rows, raw = [], []
    for item in PLAN['first_batch']:
        reference = next(r for r in REFERENCES if r['map'] == item['map'] and r['seed'] == item['seed'])
        row = {'map': item['map'], 'seed': item['seed']}
        curves = {'map': item['map'], 'seed': item['seed']}
        paths = {
            'fixed013': DATA / 'ablation_results_dvd/sacred' / item['map'] / item['name'] / '1/config.json',
            'dynamic': Path(item['paired_dynamic_config_path']),
            'bm': Path(reference['bm']['config_path']),
        }
        for method, path in paths.items():
            result, series = reader.read_run(path)
            wins = series['test_battle_won_mean']
            result['last_five_within_5_05m'] = [{'step': s, 'win_rate': v} for s, v in wins if s <= 5_050_000][-5:]
            row[method], curves[method] = result, series
        fixed = row['fixed013']
        actual, expected = fixed['config'], item['effective_config']
        skip = {'audit_source_manifest', 'local_results_path'}
        differences = {k: [expected.get(k), actual.get(k)] for k in set(expected) | set(actual) if k not in skip and expected.get(k) != actual.get(k)}
        assert not differences, differences
        assert len(fixed['source_checks']) == 22
        assert all(s['matches_current_lf'] for s in fixed['source_checks'])
        assert actual['adaptive_gate_init'] == .13
        assert actual['counterfactual_route_loss_weight'] == 0 and actual['counterfactual_sparse_weight'] == 0
        series = curves['fixed013']
        gate_mean = [v for _, v in series['counterfactual_gate_mean']]
        gate_std = [v for _, v in series['counterfactual_gate_std']]
        output_norm = [v for _, v in series['counterfactual_gate_output_weight_norm']]
        gradient_norm = [v for _, v in series['adaptive_gate_grad_norm']]
        assert max(abs(v - .13) for v in gate_mean) < 1e-6
        assert max(gate_std) < 1e-6
        assert max(output_norm) == 0 and max(gradient_norm) == 0
        assert set(v for _, v in series['audit_detach_mixed_hidden']) == {float(actual['audit_detach_mixed_hidden'])}
        fixed['planned_config_differences'] = differences
        fixed['gate_verification'] = {'mean_range': [min(gate_mean), max(gate_mean)], 'maximum_within_batch_std': max(gate_std), 'maximum_output_weight_norm': max(output_norm), 'maximum_gate_grad_norm': max(gradient_norm), 'point_count': len(gate_mean)}
        default = {'adaptive_graph_mode': 'learned', 'adaptive_normalize_dvd_weights': True}
        dynamic_config = row['dynamic']['config']
        treatment = {k: [dynamic_config.get(k, default.get(k)), actual.get(k)] for k in set(dynamic_config) | set(actual) if k not in ('name', 'audit_source_manifest', 'local_results_path') and dynamic_config.get(k, default.get(k)) != actual.get(k)}
        assert set(treatment) == {'adaptive_gate_init', 'counterfactual_route_loss_weight'}, treatment
        row['dynamic_to_fixed_training_treatment'] = treatment
        row['dynamic_minus_fixed_near_5m_pp'] = 100 * (row['dynamic']['near_5m']['win_rate'] - fixed['near_5m']['win_rate'])
        rows.append(row)
        raw.append(curves)
        print(json.dumps({'map': row['map'], 'seed': row['seed'], 'fixed_near5m': fixed['near_5m'], 'dynamic_near5m': row['dynamic']['near_5m'], 'fixed_last': fixed['last'], 'gate': fixed['gate_verification'], 'checks': fixed['checks']}, ensure_ascii=False), flush=True)
    summary = {
        'review_date': '2026-11-05', 'timezone': 'Asia/Shanghai', 'new_run_count': 4,
        'planned_fixed_total': 8, 'remaining_predefined_batch': PLAN['remaining_batch'],
        'metric': 'raw unsmoothed evaluation nearest 5M plus actual final evaluations; no interpolation or time weighting',
        'interpretation': 'User explicitly requests no overall superiority conclusion under mixed seed directions; record paired results and complete the predefined set, then add prospectively fixed independent seeds.',
        'scope': 'fixed .13 approximates late-stage development gate means; initial prior and temporal behavior also change; no gate-only or population-stability claim',
        'reader': 'Shared read-only TensorBoard EventAccumulator reader, scalars guidance=0; Sacred shared wins cross-check; no independent audit of TensorBoard internal CRC implementation',
        'summary_fields': 'last and last_five span each complete log; some historical BM runs reach 10M; use near_5m and last_five_within_5_05m for budget-matched comparisons',
        'rows': rows,
    }
    (HERE / 'run_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (HERE / 'raw_curves_and_diagnostics.json').write_text(json.dumps(raw, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    lines = ['# 2026-11-05：固定门控0.13首批结果与后续验证', '',
        '本次收到3s80/81 attached、6h80/84 detached四条固定门控运行，均覆盖5M。原始运行于北京时间11月4日启动，本报告于11月5日整理。三个结果根目录当前config数为104 + 77 + 87 = 268。', '',
        '**用户最新要求：不同种子方向不一致时先不急着下结论，需要更多种子实验。** 本报告只记录逐seed配对观察，既定八条尚未齐，不给出动态或固定门控总体更优、稳定性改善或模块普遍必要的结论。', '',
        '## 最近5M原始胜率', '',
        '每条取距离5,000,000最近的一次未平滑评估，不插值、不时间加权。历史BM即使训练到10M，也只取同预算的5M附近记录。', '',
        '| 地图/seed | 纯BM | 固定0.13 | 动态BACM | 动态−固定 |', '| --- | ---: | ---: | ---: | ---: |']
    for r in rows:
        label = ('3s' if r['map'].startswith('3s') else '6h') + '/' + str(r['seed'])
        lines.append(f"| {label} | {percent(r['bm']['near_5m']['win_rate'])} | {percent(r['fixed013']['near_5m']['win_rate'])} | {percent(r['dynamic']['near_5m']['win_rate'])} | {r['dynamic_minus_fixed_near_5m_pp']:+.3f} pp |")
    lines += ['', '上述单点显示3s80/81及6h84的动态更高，6h80的固定更高。这些只是本批四个配对观察，不据此概括整张地图或其他种子。', '', '## 新运行覆盖与末段', '', '| 地图/seed | 固定测试点数 | 最近5M步数 | 最后步数 | 固定末次 | 动态末次 |', '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:
        f, d = r['fixed013'], r['dynamic']
        label = ('3s' if r['map'].startswith('3s') else '6h') + '/' + str(r['seed'])
        lines.append(f"| {label} | {f['point_count']} | {f['near_5m']['step']:,} | {f['last']['step']:,} | {percent(f['last']['win_rate'])} | {percent(d['last']['win_rate'])} |")
    lines.append('')
    for r in rows:
        label = ('3s' if r['map'].startswith('3s') else '6h') + '/' + str(r['seed'])
        for key, title in [('fixed013', '固定0.13'), ('dynamic', '动态')]:
            lines.append(f"- {label} {title}最后五次：" + '、'.join(percent(p['win_rate']) for p in r[key]['last_five']) + '。')
    lines += ['', '5M单点与最后单点可能有不同排序，6h84已出现这种情况。3s80的曲线还存在固定0.13在部分中前段较高的阶段。以逐seed原始曲线及末段记录一起检查，不用最佳点或最后一点代替完整表现。', '', '## 运行与门控核验', '',
        '- 四条实际配置与事前命令一致，各22项源码/配置指纹与本地统一LF内容一致；没有修改训练算法以适配结果。',
        '- 四条整个已记录训练中的masked gate均值接近0.13，batch内标准差接近零；门控末层权重范数及门控梯度范数全程为零。实际固定行为核验通过，原有零候选信号回退规则保留。',
        '- RND、RMS、learned图聚合、BM自举目标及BM/混合/候选损失保留；3s attached、6h detached与配对动态设置一致。训练处理差异为gate初值0.05→0.13、路由监督系数0.1→0，稀疏系数仍为零。',
        '- Sacred/TensorBoard共有胜率记录一致，未发现非有限simple scalar。Sacred RUNNING只是元数据，不据此判断服务器进程状态。本次未独立审计TensorBoard内部CRC实现。',
        '- 固定0.13仅近似后段融合水平，同时改变初始先验与时间轨迹；当前比较不能单独证明逐样本路由或路由损失的作用。', '', '## 后续安排', '',
        '1. 保持配置与定义，补齐原先已预定的3s79、6h87/88/90。分批是算力安排，不根据本批结果方向决定保留或删除种子。',
        '2. 按用户要求增加独立种子验证。已有计划预留3s84/85、6h92/93；启动前核对未运行并预先锁定集合。动态BACM与固定0.13保持同seed配对，纯BM作为基础参照一并补充；该候选批次共12条，具体命令后续准备，当前不启动。',
        '3. 原6h集合属于开发期间筛选的集合，新独立验证分开报告。更多重复后仍按真实方向及不确定性表述，不预设必须得出某一方法更优的结论。',
        '4. 聚焦现有两图，MMM2/新任务暂缓，QPLEX/QTRAN/Qatten等外部对照保持低优先级；创新点、默认主方法和RMS暂停扩展的决定保持。', '',
        '本次只读取原始日志并更新本目录分析记录，未修改训练代码、原始结果或论文正文，未启动训练、提交或推送。', '',
        '[逐运行摘要](D:/论文/代码/experiments/2026-11-05_FIXED_GATE_013_RESULTS/run_summary.json)；[原始曲线和诊断序列](D:/论文/代码/experiments/2026-11-05_FIXED_GATE_013_RESULTS/raw_curves_and_diagnostics.json)；[未平滑曲线图](D:/论文/代码/experiments/2026-11-05_FIXED_GATE_013_RESULTS/learning_curves.png)。', '']
    (HERE / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
    for ax, curve in zip(axes.flat, raw):
        for key, label, color in [('bm', 'BM', '#929292'), ('fixed013', 'Fixed gate 0.13', '#d55e00'), ('dynamic', 'Dynamic BACM', '#0072b2')]:
            points = [(s / 1e6, 100 * v) for s, v in curve[key]['test_battle_won_mean'] if s <= 5_050_000]
            ax.plot(*zip(*points), label=label, color=color, alpha=.85, linewidth=1)
        ax.set_title(f"{curve['map']} / seed {curve['seed']}")
        ax.set(xlim=(0, 5.05), ylim=(-3, 103), xlabel='Environment steps (millions)', ylabel='Test win rate (%)')
        ax.grid(alpha=.18)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(.5, .965))
    fig.suptitle('First fixed-gate batch: raw evaluations, no smoothing or time weighting', y=.995)
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(HERE / 'learning_curves.png', dpi=160)
    plt.close(fig)
    print('WROTE', HERE / 'report.md', flush=True)


if __name__ == '__main__':
    main()
