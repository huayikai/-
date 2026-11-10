"""Recorded external-baseline entry point, independent of BACM audit code."""
import hashlib
import json
from pathlib import Path
import sys
import yaml
from utils.results import prepare_results_path

ROOT = Path(__file__).resolve().parent


def merge(base, update):
    for k,v in update.items():
        if isinstance(v,dict) and isinstance(base.get(k),dict):merge(base[k],v)
        else:base[k]=v
    return base


def load_configuration(argv):
    params=list(argv)
    selected={}
    for flag in ('--config','--env-config'):
        matches=[p for p in params if p.startswith(flag+'=')]
        if len(matches)!=1:raise ValueError('supply exactly one '+flag)
        selected[flag]=matches[0].split('=',1)[1];params.remove(matches[0])
    if selected['--config'] not in ('qplex','qtran') or selected['--env-config']!='sc2':
        raise ValueError('this verified baseline entry supports qplex/qtran with sc2')
    algorithm=selected['--config']
    config={}
    for name in ('config/default.yaml','config/envs/sc2.yaml','config/algs/'+algorithm+'.yaml'):
        merge(config,yaml.safe_load((ROOT/name).read_text(encoding='utf-8')))
    sources=(
        'main_baseline.py','main.py','utils/results.py','utils/logging.py',
        'run/run.py','runners/parallel_runner.py','components/episode_buffer.py',
        'components/action_selectors.py','components/epsilon_schedules.py',
        'controllers/basic_controller.py','modules/agents/rnn_agent.py',
        'utils/rl_utils.py','utils/th_utils.py',
        'config/default.yaml','config/envs/sc2.yaml','config/algs/'+algorithm+'.yaml',
    )
    sources += (('learners/dmaq_qatten_learner.py','modules/mixers/dmaq_general.py',
                 'modules/mixers/dmaq_si_weight.py') if algorithm=='qplex' else
                ('learners/qtran_learner.py','modules/mixers/qtran.py'))
    config['baseline_source_manifest']=[{'path':p,'sha256':hashlib.sha256((ROOT/p).read_bytes()).hexdigest()} for p in sources]
    return config,params,sources


def inspect_overrides(config,params):
    for arg in params:
        if '=' not in arg or arg.startswith('--'):continue
        key,value=arg.split('=',1);parts=key.split('.');target=config
        for part in parts[:-1]:target=target.setdefault(part,{})
        target[parts[-1]]=yaml.safe_load(value)
    return config


def main(argv=None):
    config,params,sources=load_configuration(sys.argv[1:] if argv is None else argv)
    params=prepare_results_path(config,params,ROOT.parent)
    if '--baseline-print-config' in params:
        params.remove('--baseline-print-config')
        print(json.dumps(inspect_overrides(config,params),ensure_ascii=False,indent=2));return
    # Delayed imports allow verification without SC2/Sacred installed.
    import main as entry
    from sacred.observers import FileStorageObserver
    ex=entry.ex
    for name in sources:
        if name.endswith('.py'):ex.add_source_file(str(ROOT/name))
    @ex.pre_run_hook
    def archive_inputs(_run):
        for name in sources:
            if not name.endswith('.py'):_run.add_resource(str(ROOT/name))
    ex.add_config(config)
    map_name=entry.parse_command(params,'env_args.map_name',config['env_args']['map_name'])
    name=entry.parse_command(params,'name',config['name'])
    ex.observers.append(FileStorageObserver.create(str(Path(config['local_results_path'])/'sacred'/map_name/name)))
    ex.run_commandline([str(ROOT/'main_baseline.py')]+params)


if __name__=='__main__':main()
