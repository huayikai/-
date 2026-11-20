"""Football contracts and a real BACM update, without the native GRF engine."""
import importlib.util
import random
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
        self.last_engine_seed = (self._config['game_engine_random_seed']
            if 'game_engine_random_seed' in self._config else random.randint(0, 2000000000))
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
            fake._config.update(options.get('other_config_options', {}))
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
                self.assertEqual(c['env_args']['engine_seed_mode'], 'mappo')

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
        self.assertNotIn('game_engine_random_seed', self.options['other_config_options'])
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

    def test_episode_engine_seeds_vary_and_reseed_reproduces(self):
        env, fake = self.make_env(engine_seed_mode='mappo')
        # make_env already reset once; explicitly rewind to compare sequences.
        def collect():
            result = []
            for _ in range(8):
                env.reset()
                result.append(fake.last_engine_seed)
            return result
        env.seed(80)
        first = collect()
        self.assertEqual(len(set(first)), 8)
        env.seed(80)
        self.assertEqual(first, collect())
        env.seed(81)
        self.assertNotEqual(first, collect())

    def test_episode_seed_stream_does_not_consume_global_rng(self):
        env, _ = self.make_env(engine_seed_mode='mappo')
        np.random.seed(123)
        before = np.random.get_state()
        env.reset()
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        self.assertTrue(np.array_equal(before[1], after[1]))
        self.assertEqual(before[2:], after[2:])

    def test_fixed_mode_retains_legacy_engine_seed(self):
        env, fake = self.make_env(engine_seed_mode='fixed')
        env.seed(80)
        for _ in range(8):
            env.reset()
            self.assertEqual(fake._config['game_engine_random_seed'], 80)

    def test_separate_eval_worker_streams_and_switching(self):
        registry = ModuleType('envs')
        registry.REGISTRY = {'gfootball': lambda **kw: None}
        with patch.dict(sys.modules, {'envs': registry}):
            module = load_file('_grf_mappo_parallel_test', 'runners/parallel_runner.py')
        class Connection:
            def __init__(self): self.commands = []
            def send(self, value): self.commands.append(value[0])
            def recv(self): return {'episode_limit':400}
        class Process:
            def __init__(self, **kwargs): self.kwargs = kwargs
            def start(self): pass
        args = SimpleNamespace(env='gfootball', env_args={'seed':80},
                               batch_size_run=4, football_eval_separate_envs=True)
        with patch.object(module, 'Process', Process), patch.object(
                module, 'Pipe', lambda: (Connection(), Connection())):
            runner = module.ParallelRunner(args, None)
        seeds = [p.kwargs['args'][1].x.keywords['seed'] for p in runner.ps]
        self.assertEqual(seeds[:4], [80,1080,2080,3080])
        self.assertEqual(seeds[4:], [4000000,4010000,4020000,4030000])
        for test_mode, expected in [(True,runner.eval_parent_conns),
                                    (False,runner.train_parent_conns)]:
            with patch.object(runner, 'reset', side_effect=RuntimeError('stop at reset')):
                with self.assertRaisesRegex(RuntimeError, 'stop at reset'):
                    runner.run(test_mode=test_mode)
            self.assertIs(runner.parent_conns, expected)
        runner.close_env()
        self.assertTrue(all(p.commands[-1] == 'close' for p in
            runner.train_parent_conns + runner.eval_parent_conns))

    def test_evaluation_aggregation_over_100_episodes(self):
        registry = ModuleType('envs'); registry.REGISTRY = {}
        with patch.dict(sys.modules, {'envs': registry}):
            module = load_file('_grf_eval_count_test', 'runners/parallel_runner.py')
        # Exercise real runner rollouts and logging, using one-step fake workers.
        class Connection:
            def __init__(self, won): self.won = won
            def send(self, command): self.command = command[0]
            def recv(self):
                if self.command == 'get_stats': return {}
                data = {'state':np.zeros(6), 'obs':np.zeros((3,5)),
                        'avail_actions':np.ones((3,19))}
                if self.command == 'step':
                    data.update(reward=float(self.won), terminated=True,
                                info={'battle_won':self.won})
                return data
        class Logger:
            def __init__(self): self.stats = {}
            def log_stat(self, key, value, t): self.stats[key] = value
        class MAC:
            def init_hidden(self, **kw): pass
            def select_actions(self, batch, bs, **kw):
                return th.zeros((len(bs),3),dtype=th.long)
        runner = module.ParallelRunner.__new__(module.ParallelRunner)
        runner.args = SimpleNamespace(env='gfootball', test_nepisode=100,
                                     device='cpu', runner_log_interval=10000)
        runner.logger = Logger(); runner.batch_size = 4
        runner.episode_limit = 1; runner.t_env = 123
        runner.test_returns = []; runner.train_returns = []
        runner.test_stats = {}; runner.train_stats = {}
        runner.train_parent_conns = tuple(Connection(False) for _ in range(4))
        runner.eval_parent_conns = tuple(Connection(i<2) for i in range(4))
        scheme = {'state': {'vshape':6}, 'obs': {'vshape':5, 'group':'agents'},
                  'actions': {'vshape':(1,), 'group':'agents', 'dtype':th.long},
                  'avail_actions': {'vshape':(19,), 'group':'agents', 'dtype':th.int},
                  'reward': {'vshape':(1,)}, 'terminated': {'vshape':(1,), 'dtype':th.uint8}}
        runner.setup(scheme, {'agents':3}, {}, MAC())
        for i in range(25):
            runner.run(test_mode=True)
            if i < 24: self.assertNotIn('test_n_episodes',runner.logger.stats)
        self.assertEqual(runner.logger.stats['test_n_episodes'],100)
        self.assertEqual(runner.logger.stats['test_won_count'],50)
        self.assertEqual(runner.logger.stats['test_battle_won_mean'],0.5)
        self.assertEqual(runner.t_env,123)
        self.assertEqual(runner.test_returns,[])

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
