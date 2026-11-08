"""Read the returned independent seed logs; keep both same-seed attempts."""
import gzip
import hashlib
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
ROOTS = [REPO.parent / name for name in ('ablation_results', 'ablation_results_10_6', 'ablation_results_dvd')]
CHINA = timezone(timedelta(hours=8))
WIN = 'test_battle_won_mean'
BUDGET = 5_050_000
KEEP = {
    WIN, 'test_return_mean', 'test_ep_length_mean', 'counterfactual_gate_mean',
    'counterfactual_gate_std', 'counterfactual_gate_output_weight_norm',
    'adaptive_gate_grad_norm', 'audit_detach_mixed_hidden', 'audit_fix_td_lambda',
}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path, normalize=False):
    content = path.read_bytes()
    return hashlib.sha256(content.replace(b'\r\n', b'\n') if normalize else content).hexdigest()


def read_run(root, path, expected):
    config, run = read_json(path), read_json(path.parent / 'run.json')
    start = datetime.fromisoformat(run['start_time']).replace(tzinfo=timezone.utc).astimezone(CHINA)
    stamp = start.strftime('%Y-%m-%d_%H-%M-%S')
    folder = root / 'tb_logs' / (config['name'] + '__' + stamp)
    files = sorted(folder.glob('events.out.tfevents.*'))
    assert files, folder
    skip = {'audit_source_manifest', 'local_results_path'}
    diff = {k: [expected.get(k), config.get(k)] for k in expected.keys() | config.keys()
            if k not in skip and expected.get(k) != config.get(k)}
    assert not diff, diff
    assert Path(config['local_results_path']).name == Path(expected['local_results_path']).name
    manifest = config['audit_source_manifest']
    assert len(manifest) == 22 and all(sha(REPO / m['path'], True) == m['sha256'] for m in manifest)

    accumulator = EventAccumulator(str(folder), size_guidance={'scalars': 0}).Reload()
    series = {}
    for tag in accumulator.Tags()['scalars']:
        events = accumulator.Scalars(tag)
        assert all(math.isfinite(event.value) for event in events), (folder, tag)
        series[tag] = dict(sorted({event.step: event.value
                                   for event in sorted(events, key=lambda event: event.wall_time)}.items()))
    assert set(series['audit_detach_mixed_hidden'].values()) == {float(config['audit_detach_mixed_hidden'])}
    assert set(series['audit_fix_td_lambda'].values()) == {1.0}
    wins = list(series[WIN].items())
    sacred_info = read_json(path.parent / 'info.json')
    sacred_steps = sacred_info['test_battle_won_mean_T']
    sacred_values = sacred_info[WIN]
    assert len(sacred_steps) == len(sacred_values)
    sacred = dict(zip(sacred_steps, sacred_values))
    shared = sacred.keys() & series[WIN].keys()
    assert shared and all(abs(sacred[step] - series[WIN][step]) < 1e-7 for step in shared)
    gate = series['counterfactual_gate_mean']
    checks = {
        'config_differences': diff, 'source_manifest_matches_lf': len(manifest),
        'scalar_tags_checked': len(series), 'nonfinite_scalars': 0, 'shared_sacred_tb_wins': len(shared),
        'sacred_only_wins': len(sacred.keys() - series[WIN].keys()),
        'tb_only_wins': len(series[WIN].keys() - sacred.keys()),
    }
    if config['counterfactual_route_loss_weight'] == 0.0:
        checks['fixed_gate'] = {
            'logged_gate_mean_range': [min(gate.values()), max(gate.values())],
            'maximum_logged_gate_std': max(series['counterfactual_gate_std'].values()),
            'maximum_logged_output_weight_norm': max(series['counterfactual_gate_output_weight_norm'].values()),
            'maximum_logged_gate_grad_norm': max(series['adaptive_gate_grad_norm'].values()),
        }
        assert max(abs(v - .13) for v in gate.values()) < 1e-6
        assert checks['fixed_gate']['maximum_logged_gate_std'] < 1e-6
        assert checks['fixed_gate']['maximum_logged_output_weight_norm'] == checks['fixed_gate']['maximum_logged_gate_grad_norm'] == 0
    cout = (path.parent / 'cout.txt').read_text(encoding='utf-8', errors='replace')
    return {
        'sacred_id': path.parent.name, 'config_path': path.as_posix(), 'config_sha256': sha(path),
        'event_files': [{'path': f.as_posix(), 'sha256': sha(f), 'bytes': f.stat().st_size} for f in files],
        'start_beijing': start.isoformat(), 'snapshot_sacred_status': run['status'],
        'finished_training_in_console': 'Finished Training' in cout,
        'traceback_in_console': 'Traceback (most recent call last)' in cout,
        'covers_5m': wins[-1][0] >= 5_000_000, 'point_count': len(wins),
        'positive_evaluations': sum(value > 0 for _, value in wins),
        'maximum_win_rate': max(v for _, v in wins),
        'first_positive': next(((s, v) for s, v in wins if v > 0), None),
        'near_5m': min(wins, key=lambda sv: abs(sv[0] - 5_000_000)),
        'last_five': wins[-5:], 'last_five_within_budget': [(s, v) for s, v in wins if s <= BUDGET][-5:],
        'last_diagnostics': {t: next(reversed(series[t].items())) for t in sorted(KEEP - {WIN}) if t in series},
        'checks': checks,
    }, series


def main():
    plan = read_json(HERE / 'config_check.json')
    rows, raw, curves = [], [], {}
    for item in (r for r in plan['runs'] if r['batch'] == 1):
        name = item['name']
        attempts, traces = [], []
        for root in ROOTS:
            for path in (root / 'sacred' / item['map'] / name).glob('[0-9]*/config.json'):
                data, series = read_run(root, path, item['effective_config'])
                attempts.append(data)
                traces.append(series)
                raw.append({'config_path': path.as_posix(), 'series': {t: list(series[t].items()) for t in sorted(KEEP & series.keys())}})
        order = sorted(range(len(attempts)), key=lambda i: attempts[i]['start_beijing'])
        same_scalars = len(traces) > 1 and all(t == traces[0] for t in traces[1:])
        representative = order[0] if len(order) == 1 or same_scalars else None
        if representative is not None:
            curves[(item['map'], item['method'])] = traces[representative][WIN]
        rows.append({
            **{k: item[k] for k in ('map', 'seed', 'method', 'name')},
            'attempts': [attempts[i] for i in order], 'all_attempts_have_equal_scalar_series': same_scalars,
            'representative_config_path': attempts[representative]['config_path'] if representative is not None else None,
        })
    counts = {root.name: sum(1 for _ in root.glob('sacred/*/*/*/config.json')) for root in ROOTS}
    data = {'review_date': '2026-11-05', 'run_config_counts': counts, 'rows': rows}
    (HERE / 'batch1_summary.json').write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (HERE / 'batch1_raw_curves.json.gz').write_bytes(gzip.compress(json.dumps(raw).encode('utf-8'), mtime=0))
    draw(curves, rows)
    for row in rows:
        print(row['map'], row['seed'], row['method'], row['all_attempts_have_equal_scalar_series'])
        for a in row['attempts']:
            print(a['sacred_id'], a['point_count'], a['positive_evaluations'], a['first_positive'],
                  a['near_5m'], a['last_five'], a['checks'])
    print(counts)


def draw(curves, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), sharex=True, sharey=True, constrained_layout=True)
    for ax, map_name in zip(axes, dict.fromkeys(row['map'] for row in rows)):
        seed = next(row['seed'] for row in rows if row['map'] == map_name)
        for method, label, color in [('fixed013', 'Fixed gate 0.13', '#ae5700'), ('dynamic', 'Dynamic BACM', '#146aa1')]:
            series = curves.get((map_name, method))
            if series:
                points = [(s / 1e6, 100 * v) for s, v in series.items() if s <= BUDGET]
                ax.plot(*zip(*points), label=label, color=color, lw=1, alpha=.85)
        ax.set(title=f'{map_name} · seed {seed}', xlabel='Environment steps (millions)', xlim=(0, 5.08), ylim=(-3, 103))
        ax.axvline(5, lw=.9, color='#656565', ls='--')
        ax.grid(alpha=.18)
        ax.legend(loc='upper left', frameon=False)
        if (map_name, 'fixed013') not in curves:
            ax.text(.98, .08, 'Fixed gate: no local log', transform=ax.transAxes, ha='right', color='#444444')
    axes[0].set_ylabel('Test win rate (%)')
    fig.suptitle('Independent seeds, batch 1 · raw evaluations (32 games per point)')
    fig.savefig(HERE / 'batch1_learning_curves.png', dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    main()
