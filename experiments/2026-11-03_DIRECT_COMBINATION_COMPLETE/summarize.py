"""Read-only audit of the fixed direct-combination set; no training imports."""
import hashlib
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DATA = REPO.parent
OLD = json.loads((HERE.parent / '2026-11-03_DIRECT_COMBINATION_RESULTS/run_summary.json').read_text(encoding='utf-8'))
FIELDS = OLD['rows'][0]['matched_training_fields']
NEW = {('3s5z_vs_3s6z', 79), ('6h_vs_8z', 87), ('6h_vs_8z', 88), ('6h_vs_8z', 90)}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_run(config_path):
    c = json.loads(config_path.read_text(encoding='utf-8'))
    run = json.loads((config_path.parent / 'run.json').read_text(encoding='utf-8'))
    root = config_path.parents[4]
    folders = list((root / 'tb_logs').glob(c['name'] + '__*'))
    if len(folders) != 1:
        raise ValueError(('ambiguous TensorBoard folder', c['name'], folders))
    acc = EventAccumulator(str(folders[0]), size_guidance={'scalars': 0}).Reload()
    series = {}
    nonfinite = []
    for tag in acc.Tags()['scalars']:
        events = acc.Scalars(tag)
        points = {e.step: [e.wall_time, e.value] for e in sorted(events, key=lambda e: e.wall_time)}
        series[tag] = [(step, value[1]) for step, value in sorted(points.items())]
        nonfinite.extend({'tag': tag, 'step': e.step} for e in events if not math.isfinite(e.value))
    wins = series['test_battle_won_mean']
    try:
        info = json.loads((config_path.parent / 'info.json').read_text(encoding='utf-8'))
        info_error = None
    except (OSError, ValueError) as exc:
        info, info_error = {}, str(exc)
    tb = dict(wins)
    sacred = dict(zip(info.get('test_battle_won_mean_T', []), info.get('test_battle_won_mean', [])))
    shared = sorted(set(tb) & set(sacred))
    mismatches = [s for s in shared if abs(tb[s] - sacred[s]) > 1e-7]
    sources = []
    for item in c.get('audit_source_manifest', []):
        path = REPO / item['path']
        digest = hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest() if path.is_file() else None
        sources.append({'path': item['path'], 'recorded_sha256': item['sha256'], 'local_lf_sha256': digest, 'matches_local_lf': digest == item['sha256']})
    def point(x):
        return {'step': x[0], 'win_rate': x[1]}
    diagnostics = {tag: {'step': series[tag][-1][0], 'value': series[tag][-1][1]} for tag in ('test_return_mean', 'test_ep_length_mean', 'audit_fix_td_lambda', 'direct_detach_hidden', 'loss_td', 'grad_norm') if series.get(tag)}
    start = run.get('start_time')
    start_bj = (datetime.fromisoformat(start).replace(tzinfo=timezone.utc).astimezone(timezone(timedelta(hours=8))).isoformat()) if start else None
    result = {
        'name': c['name'], 'map': c['env_args']['map_name'], 'seed': c['seed'],
        'config_path': config_path.as_posix(), 'config_sha256': sha(config_path), 'config': c,
        'start_beijing': start_bj, 'sacred_status': run.get('status'),
        'point_count': len(wins), 'covers_5m': wins[-1][0] >= 5_000_000,
        'near_5m': point(min(wins, key=lambda x: abs(x[0] - 5_000_000))),
        'last': point(wins[-1]), 'last_five': list(map(point, wins[-5:])),
        'last_five_within_5_05m': list(map(point, [x for x in wins if x[0] <= 5_050_000][-5:])),
        'positive_evaluations': sum(v > 0 for _, v in wins), 'max_win_rate': max(v for _, v in wins),
        'snapshots': {str(s): point(min(wins, key=lambda x: abs(x[0] - s))) for s in (1_000_000, 2_000_000, 3_000_000, 4_000_000, 5_000_000)},
        'source_checks': sources,
        'checks': {'sacred_info_error': info_error, 'shared_sacred_tb_points': len(shared), 'shared_sacred_win_mismatches': mismatches, 'nonfinite_simple_scalar_count': len(nonfinite), 'nonfinite_examples': nonfinite[:5], 'last_diagnostics': diagnostics, 'logged_detach_values': sorted(set(v for _, v in series.get('direct_detach_hidden', []))), 'max_logged_grad_norm': max(series.get('grad_norm', []), key=lambda x: x[1], default=None)},
        'tensorboard_files': [{'path': p.as_posix(), 'sha256': sha(p), 'bytes': p.stat().st_size} for p in sorted(folders[0].glob('events.out.tfevents.*'))],
    }
    return result, wins


def config_path(root, map_name, name):
    p = DATA / root / 'sacred' / map_name / name / '1/config.json'
    if not p.is_file():
        raise FileNotFoundError(p)
    return p


def main():
    rows, curves = [], []
    for map_name, seeds in [('3s5z_vs_3s6z', [79, 80, 81]), ('6h_vs_8z', [80, 84, 87, 88, 90])]:
        for seed in seeds:
            route = 'attached' if map_name.startswith('3s') else 'detached'
            direct = config_path('ablation_results_dvd', map_name, f'baseline_bm_dvd_direct_raw_{route}_5m_{map_name}_s{seed}')
            if map_name.startswith('3s') or seed == 80:
                bacm = config_path('ablation_results_dvd', map_name, f'diag_bacm_full_{route}_5m_{map_name}_s{seed}')
            else:
                bacm = config_path('ablation_results_10_6', map_name, f'audit_router_mix025_detachhidden_5m_{map_name}_s{seed}')
            bm = config_path('ablation_results', map_name, f'audit_bm_td_{map_name}_s{seed}') if map_name.startswith('3s') or seed == 80 else config_path('ablation_results_10_6', map_name, f'audit_bm_td_5m_{map_name}_s{seed}')
            row = {'map': map_name, 'seed': seed, 'new_direct_run': (map_name, seed) in NEW}
            curve = {'map': map_name, 'seed': seed}
            for family, p in [('direct', direct), ('bacm', bacm), ('bm', bm)]:
                row[family], curve[family] = read_run(p)
            row['matched_training_fields'] = FIELDS
            row['common_training_differences'] = {k: [row['direct']['config'].get(k), row['bacm']['config'].get(k)] for k in FIELDS if row['direct']['config'].get(k) != row['bacm']['config'].get(k)}
            row['direct_minus_bacm_5m_pp'] = 100 * (row['direct']['near_5m']['win_rate'] - row['bacm']['near_5m']['win_rate'])
            assert not row['common_training_differences'], row['common_training_differences']
            assert all(r['checks']['shared_sacred_win_mismatches'] == [] for r in (row['direct'], row['bacm'], row['bm']))
            assert len(row['direct']['source_checks']) == 22
            assert all(s['matches_local_lf'] for s in row['direct']['source_checks'])
            assert row['direct']['config']['mixer'] == 'dvd_direct'
            assert row['direct']['config']['audit_detach_mixed_hidden'] == (route == 'detached')
            assert row['direct']['checks']['logged_detach_values'] == [float(route == 'detached')]
            assert all(row[f]['covers_5m'] for f in ('direct', 'bacm', 'bm'))
            assert all(row[f]['checks']['nonfinite_simple_scalar_count'] == 0 for f in ('direct', 'bacm', 'bm'))
            rows.append(row)
            curves.append(curve)
    groups = {}
    for map_name in ['3s5z_vs_3s6z', '6h_vs_8z']:
        group = [r for r in rows if r['map'] == map_name]
        groups[map_name] = {'seeds': [r['seed'] for r in group], 'n': len(group), 'near_5m_equal_seed_mean': {f: sum(r[f]['near_5m']['win_rate'] for r in group) / len(group) for f in ('direct', 'bacm', 'bm')}, 'direct_all_logged_evaluations_zero_seeds': [r['seed'] for r in group if r['direct']['positive_evaluations'] == 0], 'direct_above_bacm_near_5m_seeds': [r['seed'] for r in group if r['direct_minus_bacm_5m_pp'] > 0]}
    result = {'review_date': '2026-11-03', 'timezone': 'Asia/Shanghai', 'new_direct_run_count': 4, 'total_direct_run_count': 8, 'metric': 'raw unsmoothed evaluation nearest 5M; no interpolation or time weighting; group means weight seeds equally', 'scope': 'fixed 3s attached and 6h detached development sets; 6h selected using prior BACM performance; no population failure-rate inference; whole-framework comparison, not gate-only ablation', 'reader': 'TensorBoard EventAccumulator; all simple scalar tags loaded with size_guidance scalars=0; Sacred cross-check; no independent audit of TensorBoard internal CRC checks', 'summary_field_scope': 'last/last_five/max_win_rate cover the complete log, including some historical 10M BM runs; use near_5m or last_five_within_5_05m for matched-budget comparisons', 'groups': groups, 'rows': rows}
    (HERE / 'run_summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (HERE / 'raw_curves.json').write_text(json.dumps(curves, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 2, figsize=(13, 14), sharex=True, sharey=True)
    for ax, r in zip(axes.flat, curves):
        for family, label, color in [('bm', 'BM', '#8a8a8a'), ('direct', 'Direct A+B', '#d55e00'), ('bacm', 'BACM', '#0072b2')]:
            pts = [(s / 1_000_000, 100 * v) for s, v in r[family] if s <= 5_050_000]
            ax.plot(*zip(*pts), color=color, label=label, alpha=.85, linewidth=1.1)
        ax.set_title(f"{r['map']} / seed {r['seed']}")
        ax.set_ylim(-3, 103)
        ax.set_xlim(0, 5.05)
        ax.grid(alpha=.18)
        ax.set_xlabel('Environment steps (millions)')
        ax.set_ylabel('Test win rate (%)')
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(.5, .98))
    fig.suptitle('Fixed paired runs: raw evaluations, no smoothing or time weighting', y=.997)
    fig.tight_layout(rect=(0, 0, 1, .958))
    fig.savefig(HERE / 'learning_curves.png', dpi=160)
    plt.close(fig)
    print(json.dumps(groups, ensure_ascii=False, indent=2))
    for r in rows:
        if r['new_direct_run']:
            print(json.dumps({'name': r['direct']['name'], 'near_5m': r['direct']['near_5m'], 'last': r['direct']['last'], 'last_five': r['direct']['last_five'], 'checks': r['direct']['checks']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
