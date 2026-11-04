"""Calibrate a fixed control from existing masked dynamic gate logs."""
import hashlib
import json
import statistics
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent


def main():
    references = json.loads((HERE.parent / '2026-11-03_DIRECT_COMBINATION_COMPLETE/run_summary.json').read_text(encoding='utf-8'))['rows']
    rows = []
    for reference in references:
        config_path = Path(reference['bacm']['config_path'])
        config = json.loads(config_path.read_text(encoding='utf-8'))
        info = json.loads((config_path.parent / 'info.json').read_text(encoding='utf-8'))
        folders = list((config_path.parents[4] / 'tb_logs').glob(config['name'] + '__*'))
        assert len(folders) == 1, folders
        acc = EventAccumulator(str(folders[0]), size_guidance={'scalars': 0}).Reload()
        events = sorted(acc.Scalars('counterfactual_gate_mean'), key=lambda x: x.wall_time)
        values = {event.step: event.value for event in events}
        sacred = dict(zip(info['counterfactual_gate_mean_T'], info['counterfactual_gate_mean']))
        shared = set(values) & set(sacred)
        mismatches = [step for step in shared if abs(values[step] - sacred[step]) > 1e-7]
        assert not mismatches
        periods = {}
        for low, high in [(0, 1_000_000), (1_000_000, 2_000_000), (2_000_000, 3_000_000), (3_000_000, 4_000_000), (4_000_000, 5_000_000), (0, 5_000_000)]:
            selected = [v for step, v in values.items() if low <= step < high]
            periods[f'{low // 1_000_000}-{high // 1_000_000}M'] = {'count': len(selected), 'mean_logged_masked_batch_gate': statistics.mean(selected)}
        row = {'map': reference['map'], 'seed': reference['seed'], 'dynamic_run': config['name'], 'config_path': config_path.as_posix(), 'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(), 'shared_sacred_tb_points': len(shared), 'mismatches': mismatches, 'periods': periods, 'raw_masked_batch_gate': sorted(values.items()), 'event_files': [{'path': p.as_posix(), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in folders[0].glob('events.out.tfevents.*')]}
        rows.append(row)
        print(row['map'], row['seed'], '4-5M', periods['4-5M']['mean_logged_masked_batch_gate'], flush=True)
    groups = {}
    for map_name in ('3s5z_vs_3s6z', '6h_vs_8z'):
        selected = [r for r in rows if r['map'] == map_name]
        periods = {key: statistics.mean(r['periods'][key]['mean_logged_masked_batch_gate'] for r in selected) for key in selected[0]['periods']}
        fixed = round(periods['4-5M'], 2)
        assert fixed == 0.13
        groups[map_name] = {'seeds': [r['seed'] for r in selected], 'equal_seed_period_means': periods, 'selected_fixed_gate': fixed, 'selection_rule': 'round equal-seed mean of logged masked batch means in [4M,5M) to two decimal places'}
    output = {'prepared_date': '2026-11-04', 'statistic': 'Within each run, equal arithmetic mean of logged valid-mask batch gate means; then equal weight per development seed. Not time weighted and not a count-weighted estimate across all replay transitions.', 'scope': 'Approximate late-stage fusion amount only; not exact full-training or stage-by-stage matching; masks/padding and zero-signal fallback preserved.', 'groups': groups, 'runs': rows}
    (HERE / 'gate_calibration.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(groups, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
