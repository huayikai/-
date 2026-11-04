"""Audit returned MMM2 logs without importing or running training code."""
import hashlib
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DATA = REPO.parent
PLAN = json.loads((HERE.parent / '2026-11-04_MMM2_NEW_SEEDS/config_check.json').read_text(encoding='utf-8'))
DIAGNOSTICS = (
    'test_return_mean', 'test_ep_length_mean', 'loss_td', 'grad_norm', 'td_error_abs',
    'adaptive_gate_mean', 'adaptive_gate_std', 'audit_fix_td_lambda',
    'audit_detach_mixed_hidden', 'audit_isolate_dvd_aux_hidden',
    'counterfactual_bm_loss', 'counterfactual_dvd_loss', 'counterfactual_route_loss',
    'audit_agent_bm_grad_norm', 'audit_agent_mix_grad_norm',
    'audit_agent_bm_mix_grad_cosine', 'audit_agent_weighted_mix_bm_grad_ratio',
    'intrinsic_rewards_mean', 'takeover_alpha_mean', 'takeover_schedule_scale',
)


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read_run(config_path, expected=None):
    config = json.loads(config_path.read_text(encoding='utf-8'))
    metadata = json.loads((config_path.parent / 'run.json').read_text(encoding='utf-8'))
    folders = list((config_path.parents[4] / 'tb_logs').glob(config['name'] + '__*'))
    if len(folders) != 1:
        raise ValueError(('ambiguous event folder', folders))
    accumulator = EventAccumulator(str(folders[0]), size_guidance={'scalars': 0}).Reload()
    series, nonfinite, times = {}, [], []
    for tag in accumulator.Tags()['scalars']:
        values = accumulator.Scalars(tag)
        latest = {e.step: e.value for e in sorted(values, key=lambda x: x.wall_time)}
        series[tag] = sorted(latest.items())
        nonfinite.extend({'tag': tag, 'step': e.step} for e in values if not math.isfinite(e.value))
        times.extend(e.wall_time for e in values)
    wins = series['test_battle_won_mean']
    try:
        info = json.loads((config_path.parent / 'info.json').read_text(encoding='utf-8'))
        info_error = None
    except (OSError, ValueError) as error:
        info, info_error = {}, str(error)
    sacred = dict(zip(info.get('test_battle_won_mean_T', []), info.get('test_battle_won_mean', [])))
    raw_wins = dict(wins)
    shared = sorted(set(sacred) & set(raw_wins))
    mismatches = [step for step in shared if abs(sacred[step] - raw_wins[step]) > 1e-7]
    sources = []
    for source in config.get('audit_source_manifest', []):
        local = REPO / source['path']
        current = hashlib.sha256(local.read_bytes().replace(b'\r\n', b'\n')).hexdigest() if local.is_file() else None
        sources.append({'path': source['path'], 'recorded_sha256': source['sha256'], 'current_lf_sha256': current, 'matches_current_lf': current == source['sha256']})
    skip = {'audit_source_manifest', 'local_results_path'}
    differences = {key: [expected.get(key), config.get(key)] for key in set(expected or {}) | set(config) if expected is not None and key not in skip and expected.get(key) != config.get(key)}
    def point(value):
        return {'step': value[0], 'win_rate': value[1]}
    def beijing(value):
        return datetime.fromtimestamp(value, timezone.utc).astimezone(timezone(timedelta(hours=8))).isoformat()
    cout = config_path.parent / 'cout.txt'
    console = cout.read_text(encoding='utf-8', errors='replace') if cout.is_file() else ''
    result = {
        'name': config['name'], 'seed': config['seed'], 'map': config['env_args']['map_name'],
        'config_path': config_path.as_posix(), 'config_sha256': digest(config_path), 'config': config,
        'recorded_start_utc': metadata.get('start_time'), 'sacred_status': metadata.get('status'),
        'first_scalar_beijing': beijing(min(times)), 'last_scalar_beijing': beijing(max(times)),
        'point_count': len(wins), 'covers_5m': wins[-1][0] >= 5_000_000,
        'near_5m': point(min(wins, key=lambda x: abs(x[0] - 5_000_000))),
        'last': point(wins[-1]), 'last_five': list(map(point, wins[-5:])),
        'positive_evaluations': sum(v > 0 for _, v in wins), 'maximum_win_rate': max(v for _, v in wins),
        'diagnostic_last': {tag: {'step': series[tag][-1][0], 'value': series[tag][-1][1]} for tag in DIAGNOSTICS if series.get(tag)},
        'diagnostic_4_5m_ranges': {tag: {'count': len(values), 'min': min(values), 'max': max(values)} for tag in DIAGNOSTICS if (values := [v for step, v in series.get(tag, []) if 4_000_000 <= step <= 5_000_000])},
        'maximum_logged_grad_norm': max(series.get('grad_norm', []), key=lambda x: x[1], default=None),
        'source_checks': sources, 'planned_config_differences': differences,
        'checks': {'info_error': info_error, 'shared_sacred_tb_points': len(shared), 'shared_sacred_win_mismatches': mismatches, 'nonfinite_simple_scalar_count': len(nonfinite), 'nonfinite_examples': nonfinite[:10], 'traceback_in_console': 'Traceback (most recent call last)' in console, 'finished_training_in_console': 'Finished Training' in console},
        'event_files': [{'path': p.as_posix(), 'sha256': digest(p), 'bytes': p.stat().st_size} for p in sorted(folders[0].glob('events.out.tfevents.*'))],
    }
    assert not mismatches
    assert not nonfinite
    assert result['covers_5m']
    if expected is not None:
        assert not differences, differences
        assert len(sources) == 22 and all(s['matches_current_lf'] for s in sources)
        assert config['use_rnd'] and config['audit_fix_td_lambda'] and config['use_tensorboard']
        if config['mixer'] == 'dvd_counterfactual_router':
            assert config['audit_detach_mixed_hidden']
            assert set(v for _, v in series['audit_detach_mixed_hidden']) == {1.0}
        else:
            assert config['takeover_warmup_steps'] > config['t_max']
            assert max(v for _, v in series['takeover_alpha_mean']) == 0.0
    selected = {tag: values for tag, values in series.items() if tag in DIAGNOSTICS or tag == 'test_battle_won_mean'}
    return result, selected


def main():
    runs, raw = [], []
    for item in PLAN['runs']:
        path = DATA / 'ablation_results_dvd/sacred/MMM2' / item['name'] / '1/config.json'
        result, selected = read_run(path, item['effective_config'])
        result['method'] = item['method']
        runs.append(result)
        raw.append({'name': item['name'], 'series': selected})
        print(json.dumps({'name': item['name'], 'point_count': result['point_count'], 'near_5m': result['near_5m'], 'last': result['last'], 'positive_evaluations': result['positive_evaluations'], 'diagnostic_last': result['diagnostic_last'], 'maximum_logged_grad_norm': result['maximum_logged_grad_norm'], 'checks': result['checks']}, ensure_ascii=False), flush=True)
    context = []
    for name in ['audit_bm_td_5m_MMM2_s79', 'audit_router_mix025_detachhidden_5m_MMM2_s79', 'baseline_qmix_matched_5m_MMM2_s79']:
        result, selected = read_run(DATA / 'ablation_results_10_6/sacred/MMM2' / name / '1/config.json')
        context.append(result)
        raw.append({'name': name, 'series': selected})
    output = {'review_date': '2026-11-04', 'timezone': 'Asia/Shanghai', 'new_run_count': 4, 'context_run_count': 3, 'source': 'TensorBoard EventAccumulator scalar guidance=0; readable Sacred common wins cross-checked; no independent audit of TensorBoard internal CRC checks', 'metric': 'raw unsmoothed evaluation nearest 5M and final raw evaluations; no interpolation or time-weighted metric', 'new_runs': runs, 'context_runs': context, 'conclusion': 'all four new runs and previous MMM2 seed79 BM/BACM logged zero wins throughout; shared failure under this configuration, not evidence that the gate alone causes failure; QMIX seed79 is a different-protocol reference', 'next_step': 'record negative extension; do not change default method or innovation; fixed comparable-mean gate control on already working tasks remains the next experimental priority; further MMM2 configuration/source diagnostics may be planned separately'}
    (HERE / 'run_summary.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (HERE / 'raw_diagnostics.json').write_text(json.dumps(raw, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('WROTE', HERE / 'run_summary.json', flush=True)


if __name__ == '__main__':
    main()
