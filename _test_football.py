"""Football contracts and a real BACM update, without the native GRF engine."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import gym
import numpy as np
import torch as th
import main_football
from components.episode_buffer import EpisodeBatch
from components.transforms import OneHot
from controllers.n_controller import NMAC

ROOT = Path(__file__).resolve().parent


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeFootball:
    def __init__(self, n):
        self.observation_space = gym.spaces.Box(-1., 1., shape=(n, 115), dtype=np.float32)
        self.action_space = gym.spaces.MultiDiscrete([19] * n)
        self.n = n
        self.result = (0.1, False, 0.)
        self.unwrapped = self
        self._config = {}

    def reset(self):
        return np.zeros((self.n, 115), dtype=np.float32)

    def step(self, actions):
        self.actions = actions
        reward, done, score = self.result
        return self.reset(), np.full(self.n, reward), done, {'score_reward': score}

    def close(self):
        pass

    def render(self):
        return 'rendered'


class FootballTests(unittest.TestCase):
    def make_env(self, n=3, **kwargs):
        fake = FakeFootball(n)
        package = ModuleType('_grf_test'); package.__path__ = []
        subpackage = ModuleType('_grf_test.gfootball'); subpackage.__path__ = []
        base = load_file('_grf_test.multiagentenv', 'envs/multiagentenv.py')
        grf = ModuleType('gfootball'); grf.__path__ = []
        engine = ModuleType('gfootball.env')
        engine.observation_preprocessing = SimpleNamespace(SMM_WIDTH=96, SMM_HEIGHT=72)
        def create(**options):
            self.options = options
            return fake
        engine.create_environment = create
        grf.env = engine
        with patch.dict(sys.modules, {'_grf_test': package,
                '_grf_test.gfootball': subpackage, '_grf_test.multiagentenv': base,
                'gfootball': grf, 'gfootball.env': engine}):
            module = load_file('_grf_test.gfootball.FootballEnv', 'envs/gfootball/FootballEnv.py')
        env = module.GoogleFootballEnv(num_agents=n, seed=79, **kwargs)
        env.reset()
        return env, fake

    def test_all_presets_and_baselines(self):
        for scenario, n in [('3_vs_1', 3), ('counterattack_easy', 4),
                            ('counterattack_hard', 4), ('pass_and_shoot', 2)]:
            for algorithm in ('bacm', 'bm', 'qmix', 'vdn'):
                c, _, sources = main_football.load_configuration(
                    ['--config=' + algorithm, '--scenario=' + scenario, 'with', 'seed=80'])
                self.assertEqual(c['env_args']['num_agents'], n)
                self.assertEqual(c['seed'], 80)
                self.assertIn('envs/gfootball/FootballEnv.py', sources)
                self.assertEqual(c['use_rnd'], algorithm in ('bacm', 'bm'))
                self.assertEqual(c['env_args']['team_reward'], 'mean')

    def test_map_agent_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            main_football.load_configuration(['env_args.num_agents=11'])

    def test_checkpoint_reward_is_not_success_and_not_agent_scaled(self):
        env, _ = self.make_env(team_reward='mean')
        reward, done, info = env.step([0, 1, 2])
        self.assertAlmostEqual(reward, 0.1)
        self.assertFalse(done)
        self.assertFalse(info['battle_won'])
        self.assertEqual(env.get_state().shape, (345,))
        self.assertEqual(env.get_env_info()['obs_shape'], 115)

    def test_score_and_reset(self):
        env, fake = self.make_env(team_reward='mean')
        fake.result = (2., True, 1.)
        reward, done, info = env.step(th.tensor([1, 2, 3]))
        self.assertTrue(done and info['battle_won'])
        self.assertFalse(info['episode_limit'])
        self.assertEqual(reward, 2.)
        env.reset(); fake.result = (0.1, True, 0.)
        self.assertFalse(env.step([0, 0, 0])[2]['battle_won'])

    def test_truncation_and_natural_terminal(self):
        env, fake = self.make_env(time_limit=1)
        self.assertTrue(env.step([0, 0, 0])[2]['episode_limit'])
        env.reset(); fake.result = (0., True, 0.)
        self.assertFalse(env.step([0, 0, 0])[2]['episode_limit'])

    def test_seed_reaches_engine_and_render_remains_callable(self):
        env, _ = self.make_env()
        self.assertEqual(self.options['other_config_options']['game_engine_random_seed'], 79)
        self.assertEqual(env.seed(80), [80])
        self.assertEqual(env.render(), 'rendered')

    def test_legacy_sum_and_action_validation(self):
        env, fake = self.make_env()
        self.assertAlmostEqual(env.step([0, 0, 0])[0], 0.3)
        with self.assertRaises(ValueError): env.step([0, 1])
        with self.assertRaises(ValueError): env.step([0, 1, 19])
        with self.assertRaises(ValueError): env.step([0, 1, 1.5])

    def test_parallel_worker_seeds_and_sc2_protocol(self):
        registry = ModuleType('envs')
        registry.REGISTRY = {'gfootball': lambda **kw: None, 'sc2': lambda **kw: None}
        with patch.dict(sys.modules, {'envs': registry}):
            runner_module = load_file('_grf_parallel_test', 'runners/parallel_runner.py')
        class Connection:
            def send(self, value): pass
            def recv(self): return {'episode_limit':400}
        class Process:
            def __init__(self, **kwargs): self.kwargs = kwargs
            def start(self): pass
        for env_name in ('gfootball', 'sc2'):
            args = SimpleNamespace(env=env_name, env_args={'seed':79}, batch_size_run=4)
            with patch.object(runner_module, 'Process', Process), patch.object(
                    runner_module, 'Pipe', lambda: (Connection(), Connection())):
                runner = runner_module.ParallelRunner(args, None)
            seeds = [p.kwargs['args'][1].x.keywords['seed'] for p in runner.ps]
            self.assertEqual(seeds, [79,80,81,82] if env_name == 'gfootball' else [79]*4)
            self.assertEqual(args.env_args, {'seed':79})

    def test_real_bacm_football_shape_update(self):
        from _test_dvd_audit import audit
        th.set_num_threads(1)
        th.manual_seed(79)
        c, _, _ = main_football.load_configuration([])
        c.update(n_agents=3, n_actions=19, state_shape=345,
                 device='cpu', use_cuda=False, learner_log_interval=1)
        args = SimpleNamespace(**c)
        scheme = {'state': {'vshape':345}, 'obs': {'vshape':115, 'group':'agents'},
                  'actions': {'vshape':(1,), 'group':'agents', 'dtype':th.long},
                  'avail_actions': {'vshape':(19,), 'group':'agents', 'dtype':th.int},
                  'reward': {'vshape':(1,)}, 'terminated': {'vshape':(1,), 'dtype':th.uint8}}
        batch = EpisodeBatch(scheme, {'agents':3}, 2, 5,
            preprocess={'actions':('actions_onehot',[OneHot(out_dim=19)])}, device='cpu')
        for t in range(5):
            batch.update({'state':th.randn(2,345), 'obs':th.randn(2,3,115),
                          'actions':th.randint(19,(2,3,1)),
                          'avail_actions':th.ones(2,3,19,dtype=th.int)}, ts=t)
            if t < 4:
                batch.update({'reward':th.ones(2,1)*0.1,
                              'terminated':th.tensor([[t == 3], [False]])}, ts=t)
        class Logger:
            console_logger = SimpleNamespace(info=lambda *a: None)
            def log_stat(self, key, value, t):
                if not np.isfinite(float(value)): raise AssertionError(key)
        mac = NMAC(batch.scheme, {'agents':3}, args)
        learner = audit.DVDNQLearner(mac, batch.scheme, Logger(), args)
        before = [p.detach().clone() for p in learner.params]
        for t in range(3): learner.train(batch, 10000*t, t)
        self.assertTrue(any(not th.equal(p,q) for p,q in zip(before,learner.params)))
        self.assertTrue(all(th.isfinite(p).all() for p in learner.params))


if __name__ == '__main__':
    unittest.main(verbosity=2)
