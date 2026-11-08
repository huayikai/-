"""Rebuild the eight matched seed summaries from local evaluation logs."""
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
WORK = REPO.parent
WIN = 'test_battle_won_mean'
ROOTS = [WORK / name for name in (
    'ablation_results', 'ablation_results_10_6',
    'ablation_results_dvd', 'ablation_results_11_6',
)]


def read(path):
    return json.loads(path.read_text('utf-8'))


def main():
    plan = read(HERE.parent / '2026-11-05_INDEPENDENT_SEEDS/config_check.json')['runs']
    rows, interrupted = [], []
    for map_name, seeds, reference_seed in (
        ('3s5z_vs_3s6z', (84, 85, 86, 87), 85),
        ('6h_vs_8z', (92, 93, 94, 95), 93),
    ):
        for seed in seeds:
            row = {'map': map_name, 'seed': seed}
            for method in ('bm', 'dynamic', 'fixed013'):
                base = next(r for r in plan if r['seed'] == reference_seed and r['method'] == method)
                name = base['name'].replace('_s' + str(reference_seed), '_s' + str(seed))
                expected = dict(base['effective_config'], seed=seed, name=name)
                attempts, traces = [], []
                for root in ROOTS:
                    for path in sorted((root / 'sacred' / map_name / name).glob('[0-9]*/config.json')):
                        config, run = read(path), read(path.with_name('run.json'))
                        ignored = {'audit_source_manifest', 'local_results_path', 'audit_shuffle_gate'}
                        differences = [k for k in config.keys() | expected.keys()
                                       if k not in ignored and config.get(k) != expected.get(k)]
                        assert not differences, (path, differences)
                        assert not config.get('audit_shuffle_gate', False), path
                        info_error = None
                        try:
                            info = read(path.with_name('info.json'))
                            assert len(info[WIN + '_T']) == len(info[WIN])
                            sacred = dict(zip(info[WIN + '_T'], info[WIN]))
                        except json.JSONDecodeError as error:
                            sacred = {}
                            info_error = type(error).__name__
                        if sacred and max(sacred) < 5_000_000:
                            interrupted.append({'config_path': str(path.relative_to(WORK)),
                                                'status': run['status'], 'last_step': max(sacred)})
                            continue
                        start = datetime.fromisoformat(run['start_time']).replace(tzinfo=timezone.utc)
                        start = start.astimezone(timezone(timedelta(hours=8)))
                        folder = root / 'tb_logs' / (name + '__' + start.strftime('%Y-%m-%d_%H-%M-%S'))
                        if not folder.is_dir():
                            candidates = list((root / 'tb_logs').glob(name + '__*'))
                            assert len(candidates) == 1, (name, candidates)
                            timestamp = candidates[0].name.rsplit('__', 1)[1]
                            actual = datetime.strptime(timestamp, '%Y-%m-%d_%H-%M-%S').replace(tzinfo=start.tzinfo)
                            assert abs((actual - start).total_seconds()) < 2
                            folder = candidates[0]
                        acc = EventAccumulator(str(folder), size_guidance={'scalars': 0}).Reload()
                        events = sorted(acc.Scalars(WIN), key=lambda e: e.wall_time)
                        values = dict(sorted({e.step: e.value for e in events}.items()))
                        assert values and max(values) >= 5_000_000, folder
                        assert all(0 <= value <= 1 for value in values.values()), folder
                        shared = sacred.keys() & values.keys()
                        assert info_error or shared, path
                        assert all(abs(sacred[s] - values[s]) < 1e-7 for s in shared), path
                        points = list(values.items())
                        near = min(points, key=lambda p: abs(p[0] - 5_000_000))
                        last5 = [p for p in points if p[0] <= 5_050_000][-5:]
                        attempts.append({
                            'config_path': path.relative_to(WORK).as_posix(),
                            'config_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                            'sacred_id': path.parent.name, 'start_beijing': start.isoformat(),
                            'status_metadata': run['status'], 'test_point_count': len(points),
                            'positive_test_points': sum(v > 0 for v in values.values()),
                            'near_5m_step': near[0], 'near_5m_pct': near[1] * 100,
                            'last5_within_budget': [[s, v * 100] for s, v in last5],
                            'last5_mean_pct': mean(v * 100 for s, v in last5),
                            'info_error': info_error, 'shared_sacred_tb_wins': len(shared),
                            'tb_only_wins': len(values.keys() - sacred.keys()),
                        })
                        traces.append(points)
                assert attempts, name
                assert all(t == traces[0] for t in traces), (name, 'different repeated runs')
                row[method] = {'name': name, 'near_5m_pct': attempts[0]['near_5m_pct'],
                               'last5_mean_pct': attempts[0]['last5_mean_pct'], 'attempts': attempts}
            rows.append(row)
    means = {method: mean(row[method]['near_5m_pct'] for row in rows)
             for method in ('bm', 'dynamic', 'fixed013')}
    comparisons = {}
    for first, second in (('dynamic', 'bm'), ('fixed013', 'bm'), ('dynamic', 'fixed013')):
        differences = [r[first]['near_5m_pct'] - r[second]['near_5m_pct'] for r in rows]
        comparisons[first + '_minus_' + second] = {
            'mean_pp': mean(differences), 'positive': sum(d > 0 for d in differences),
            'negative': sum(d < 0 for d in differences), 'equal': sum(d == 0 for d in differences),
        }
    output = {'review_date': '2026-11-08', 'horizon': 5_000_000,
              'metric': 'nearest raw evaluation to 5M; equal weight per map/seed pair',
              'source_path_base': 'parent of this repository',
              'training_conditions': 24, 'rows': rows, 'means_pct': means,
              'paired_comparisons': comparisons, 'early_interrupted_records': interrupted}
    (HERE / 'summary.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps({'means_pct': means, 'paired_comparisons': comparisons}, ensure_ascii=False))


if __name__ == '__main__':
    main()
