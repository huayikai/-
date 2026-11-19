"""GRF Academy launcher for BACM and its first performance controls.

Run from PyMARL2 root: python src/main_football.py --config=bacm
    --scenario=3_vs_1 with seed=79
Use --football-print-config to inspect without Sacred or the GRF engine.
"""
import hashlib
import json
from pathlib import Path
import sys

import main_audit
from utils.results import prepare_results_path

ROOT = Path(__file__).resolve().parent


def load_configuration(argv):
    params = list(argv)
    selected = {}
    for flag, default in (('--config', 'bacm'), ('--scenario', '3_vs_1')):
        matches = [p for p in params if p.startswith(flag + '=')]
        if len(matches) > 1:
            raise ValueError('supply at most one ' + flag)
        selected[flag] = matches[0].split('=', 1)[1] if matches else default
        if matches:
            params.remove(matches[0])
    algorithm = selected['--config']
    if algorithm not in ('bacm', 'bm', 'qmix', 'vdn'):
        raise ValueError('football supports bacm, bm, qmix, vdn')
    profile = main_audit._read_yaml(ROOT / 'config/football.yaml')
    scenario = selected['--scenario']
    if scenario not in profile['scenarios']:
        raise ValueError('unknown scenario; choose ' + ', '.join(profile['scenarios']))
    base = 'dvd_audit_router' if algorithm == 'bacm' else 'dvd_audit_bm'
    config, _, sources = main_audit.load_configuration(
        ['--config=' + base, '--env-config=gfootball'])
    main_audit._merge(config, profile['common'])
    config['env_args'] = dict(profile['env_args'])
    config['env_args'].update(profile['scenarios'][scenario])
    if algorithm == 'bacm':
        config['audit_detach_mixed_hidden'] = True
    if algorithm in ('qmix', 'vdn'):
        config.update(learner='nq_learner', mixer=algorithm, abs=True,
                      use_rnd=False, return_priority=False)
        sources += ('learners/nq_learner.py', 'modules/mixers/vdn.py',
                    'utils/rl_utils.py',)
    config['name'] = 'football_' + algorithm + '_' + scenario
    sources += ('main_football.py', 'config/football.yaml',
                'envs/gfootball/FootballEnv.py', 'envs/gfootball/__init__.py',
                'envs/multiagentenv.py', 'envs/__init__.py',
                'runners/parallel_runner.py', 'components/action_selectors.py',
                'components/epsilon_schedules.py', 'utils/logging.py')
    sources = tuple(dict.fromkeys(sources))
    config['audit_source_manifest'] = [
        {'path': p.replace('\\', '/'),
         'sha256': hashlib.sha256((ROOT / p).read_bytes()).hexdigest()}
        for p in sources]
    config = main_audit.inspect_overrides(config, params)
    # The launcher owns map/agent coupling. Select a preset rather than passing
    # a different map with the old number of controlled players.
    for key, value in profile['scenarios'][scenario].items():
        if config['env_args'].get(key) != value:
            raise ValueError('select --scenario instead of overriding env_args.' + key)
    if config['env'] != 'gfootball' or config['runner'] != 'parallel':
        raise ValueError('football protocol requires gfootball and parallel runner')
    if config.get('use_per'):
        raise ValueError('football launcher currently verifies uniform replay only')
    return config, params, sources


def main(argv=None):
    config, params, sources = load_configuration(sys.argv[1:] if argv is None else argv)
    params = prepare_results_path(config, params, ROOT.parent)
    if '--football-print-config' in params:
        print(json.dumps(config, ensure_ascii=False, indent=2))
        return
    try:
        import gfootball.env
    except ImportError as exc:
        raise RuntimeError('GRF is unavailable. From the PyMARL2 root, activate '
                           'pymarl and run bash install_gfootball.sh; '
                           'then run the native smoke_env.py check.') from exc
    import main as entry
    from learners import REGISTRY
    from learners.dvd_audit_learner import DVDNQLearner
    from sacred.observers import FileStorageObserver

    REGISTRY['dvd_audit_learner'] = DVDNQLearner
    ex = entry.ex
    main_audit.register_file_records(ex, sources)
    ex.add_config(config)
    map_name = config['env_args']['map_name']
    observer = Path(config['local_results_path']) / 'sacred' / map_name / config['name']
    ex.observers.append(FileStorageObserver.create(str(observer)))
    ex.run_commandline([str(ROOT / 'main_football.py')] + params)


if __name__ == '__main__':
    main()
