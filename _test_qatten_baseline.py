"""Exercise the actual standalone Qatten learner without an SC2 installation."""
import importlib.util
from itertools import product
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch as th

from _test_external_baseline import BaselineTests, Logger
from controllers.n_controller import NMAC
import main_baseline
from modules.mixers.qatten import QattenMixer


spec = importlib.util.spec_from_file_location(
    'qatten_cpu_learner', Path(__file__).resolve().parent / 'learners/nq_learner.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class QattenTests(unittest.TestCase):
    def args(self):
        config, _, _ = main_baseline.load_configuration(['--config=qatten', '--env-config=sc2'])
        config.update(n_agents=3, n_actions=4, state_shape=6,
                      agent_own_state_size=2, use_cuda=False, device='cpu')
        return SimpleNamespace(**config)

    def test_real_updates_and_target_sync(self):
        th.manual_seed(272)
        np.random.seed(272)
        args = self.args()
        batch = BaselineTests().batch(args)
        mac = NMAC(batch.scheme, {'agents': args.n_agents}, args)
        logger = Logger()
        learner = module.NQLearner(mac, batch.scheme, logger, args)
        self.assertIsInstance(learner.mixer, QattenMixer)
        for i in range(3):
            before = [p.detach().clone() for p in learner.params]
            learner.train(batch, 10000 * i, 201 * (i + 1))
            split = len(list(mac.parameters()))
            changed = [not th.equal(a, b) for a, b in zip(before, learner.params)]
            self.assertTrue(any(changed[:split]) and any(changed[split:]))
            self.assertTrue(all(th.isfinite(p).all() for p in learner.params))
            self.assertTrue(all(np.isfinite(float(v)) for v in logger.stats.values()))
            self.assertIsInstance(logger.stats['grad_norm'], float)
        for online, target in ((mac.agent, learner.target_mac.agent),
                               (learner.mixer, learner.target_mixer)):
            for key, value in online.state_dict().items():
                self.assertTrue(th.equal(value, target.state_dict()[key]))

    def test_greedy_joint_action_maximizes_weighted_qatten(self):
        th.manual_seed(273)
        args = self.args()
        args.state_shape = 9  # Three two-feature ally blocks, then other state features.
        mixer = QattenMixer(args)
        q = th.randn(2, 4, 3, 2)
        states = th.randn(2, 4, 9)
        best = mixer(q.max(dim=-1).values, states)
        for actions in product(range(2), repeat=3):
            index = th.tensor(actions).view(1, 1, 3, 1).expand(2, 4, 3, 1)
            chosen = q.gather(-1, index).squeeze(-1)
            self.assertTrue((mixer(chosen, states) <= best + 1e-5).all())

    def test_reference_recipe_and_sources(self):
        config, _, sources = main_baseline.load_configuration(['--config=qatten', '--env-config=sc2'])
        self.assertEqual((config['mac'], config['agent'], config['learner'], config['mixer']),
                         ('n_mac', 'n_rnn', 'nq_learner', 'qatten'))
        self.assertEqual((config['optimizer'], config['lr'], config['td_lambda'],
                          config['type'], config['n_attention_head']), ('adam', .001, .6, 'weighted', 4))
        self.assertEqual((config['test_nepisode'], config['test_interval']), (32, 10000))
        self.assertIs(config['agent_own_state_size'], True)
        self.assertFalse(config['env_args']['obs_instead_of_state'])
        self.assertIn('modules/mixers/qatten.py', sources)
        self.assertNotIn('learners/dmaq_qatten_learner.py', sources)


if __name__ == '__main__':
    th.set_num_threads(1)
    unittest.main(verbosity=2)
