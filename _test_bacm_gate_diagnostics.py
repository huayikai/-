"""Paired gate metrics, masking/ties, and observational-training invariance."""
import copy
import unittest
from unittest.mock import patch

import torch as th
import _test_dvd_audit as fixture

diagnostics = fixture.audit._masked_gate_diagnostics


def shaped(values, grad=False):
    return th.tensor(values, dtype=th.float32).reshape(1, -1, 1).requires_grad_(grad)


def assert_equal(test, expected, actual):
    if isinstance(expected, th.Tensor):
        test.assertTrue(th.equal(expected, actual))
    elif isinstance(expected, dict):
        test.assertEqual(expected.keys(), actual.keys())
        for key in expected:
            assert_equal(test, expected[key], actual[key])
    elif isinstance(expected, (tuple, list)):
        test.assertEqual(len(expected), len(actual))
        for a, b in zip(expected, actual):
            assert_equal(test, a, b)
    else:
        test.assertEqual(expected, actual)


class GateMetricTests(unittest.TestCase):
    def test_hand_calculated_endpoint_groups_and_actual_mixing_gain(self):
        g = shaped([.05 * (i + 1) for i in range(10)])
        stats = diagnostics(g, shaped([2] * 10),
                            shaped([1, 3, 1, 3, 2, 3, 1, 3, 1, 3]),
                            shaped([1, 3, 2, 2, 1, 2, 1, 4, 1, 2]),
                            shaped([0, 0, 0, 0, 0, 0, .1, 0, .2, 0]),
                            shaped([1] * 10))
        expected = {'count': 10, 'nonfinite_count': 0, 'dvd_better_count': 4,
                    'bm_better_equal_count': 6, 'dvd_better_gate_mean': .25,
                    'bm_better_equal_gate_mean': 1.75 / 6,
                    'route_positive_count': 2, 'route_positive_gate_mean': .4,
                    'mae_gain_mean': .1, 'mse_gain_mean': -.5,
                    'benefit_rate': .4, 'harm_rate': .2, 'unchanged_rate': .4}
        for key, value in expected.items():
            self.assertAlmostEqual(stats[key].item(), value, places=6, msg=key)
        # Each quintile contains two gates; independent expected gain/rate.
        for i, (gain, rate) in enumerate([(0, .5), (0, 0), (.5, .5), (-.5, .5), (.5, .5)]):
            self.assertEqual(stats['gate_bin%d_count' % i].item(), 2)
            self.assertAlmostEqual(stats['gate_bin%d_mae_gain_mean' % i].item(), gain)
            self.assertAlmostEqual(stats['gate_bin%d_benefit_rate' % i].item(), rate)
        self.assertLess(stats['endpoint_gate_gap'].item(), 0)
        self.assertGreater(stats['route_gate_gap'].item(), 0)

    def test_equal_gates_are_not_split_and_constant_gate_has_one_group(self):
        gates = [0, 0, 0, .2, .2, .2, .4, .4, .4, .4]
        stats = diagnostics(shaped(gates), shaped([2] * 10), shaped([1] * 10),
                            shaped([1] * 10), shaped([1] * 10), shaped([1] * 10))
        self.assertEqual([stats['gate_bin%d_count' % i].item() for i in range(5)], [3, 3, 0, 4, 0])
        self.assertNotIn('gate_bin2_gate_mean', stats)
        self.assertNotIn('bm_better_equal_gate_mean', stats)
        self.assertNotIn('endpoint_gate_gap', stats)
        self.assertNotIn('route_gate_gap', stats)
        constant = diagnostics(shaped([.1] * 10), shaped([2] * 10), shaped([2] * 10),
                               shaped([2] * 10), shaped([0] * 10), shaped([1] * 10))
        self.assertEqual([constant['gate_bin%d_count' % i].item() for i in range(5)], [10, 0, 0, 0, 0])
        self.assertEqual(constant['dvd_better_count'].item(), 0)
        self.assertNotIn('dvd_better_gate_mean', constant)
        self.assertEqual(constant['unchanged_rate'].item(), 1)

    def test_padding_and_nonfinite_observations_are_counted_without_fake_means(self):
        stats = diagnostics(shaped([.1, float('nan'), float('inf'), .9]),
                            shaped([2, 2, float('nan'), 900]), shaped([1, 1, 1000, 0]),
                            shaped([1, 1, float('inf'), 0]), shaped([1, 1, 0, 1]),
                            shaped([1, 1, 0, 0]))
        self.assertEqual(stats['masked_count'].item(), 2)
        self.assertEqual(stats['nonfinite_count'].item(), 1)
        self.assertEqual(stats['count'].item(), 1)
        self.assertEqual(stats['mae_gain_mean'].item(), 1)
        self.assertAlmostEqual(stats['dvd_better_gate_mean'].item(), .1)
        for value in stats.values():
            self.assertTrue(th.isfinite(value).all())
        for mask, gate in [([0], [float('nan')]), ([1], [float('nan')])]:
            empty = diagnostics(shaped(gate), shaped([1]), shaped([1]), shaped([1]), shaped([0]), shaped(mask))
            self.assertEqual(empty['count'].item(), 0)
            self.assertFalse(any(key.endswith('_mean') for key in empty))
            self.assertEqual(sum(empty['gate_bin%d_count' % i].item() for i in range(5)), 0)

    def test_no_rng_input_mutation_or_autograd_and_large_finite_errors(self):
        values = [shaped(v, grad=True) for v in ([.1, .5], [2, 3e30], [1, 1e30], [1, 2e30], [.1, .2])]
        before = [v.detach().clone() for v in values]
        rng = th.get_rng_state().clone()
        stats = diagnostics(*values, shaped([1, 1]))
        self.assertTrue(th.equal(rng, th.get_rng_state()))
        for value in stats.values():
            self.assertFalse(value.requires_grad)
            self.assertTrue(th.isfinite(value).all())
        for original, value in zip(before, values):
            self.assertTrue(th.equal(original, value))
            self.assertIsNone(value.grad)
        values[0].square().sum().backward()
        self.assertTrue(th.equal(values[0].grad, 2 * values[0].detach()))


class GateLearnerTests(unittest.TestCase):
    def test_real_learner_uses_masked_pre_update_pairs_only_at_log_interval(self):
        args = fixture.args_for()
        args.learner_log_interval = 2000
        th.manual_seed(823)
        mac, batch, logger = fixture.FakeMAC(args), fixture.FakeBatch(), fixture.FakeLogger()
        learner = fixture.audit.DVDNQLearner(mac, {}, logger, args)
        captured = {}
        def record(*inputs):
            captured['inputs'] = [v.detach().clone() for v in inputs]
            return diagnostics(*inputs)
        with patch.object(fixture.audit, '_masked_gate_diagnostics', side_effect=record) as spy:
            learner.train(batch, 0, 0)
            self.assertEqual(spy.call_count, 1)
            g, bm, dvd, mixed, target, mask = captured['inputs']
            expected_mask = batch['filled'][:, :-1].clone()
            expected_mask[:, 1:] *= 1 - batch['terminated'][:, :-2]
            self.assertTrue(th.equal(mask, expected_mask))
            self.assertEqual(logger.stats['counterfactual_diag_count'], 12)
            valid = mask.bool()
            gain = [b - m for b, m in zip(bm[valid].tolist(), mixed[valid].tolist())]
            self.assertAlmostEqual(logger.stats['counterfactual_diag_mae_gain_mean'], sum(gain) / 12, places=8)
            self.assertAlmostEqual(logger.stats['counterfactual_diag_benefit_rate'], sum(x > 0 for x in gain) / 12)
            old_logs = copy.deepcopy(logger.stats)
            learner.train(batch, 1, 1)
            self.assertEqual(spy.call_count, 1)
            self.assertEqual(logger.stats, old_logs)
            learner.train(batch, 2000, 2)
            self.assertEqual(spy.call_count, 2)

    def test_logged_diagnostics_preserve_updates_gradients_targets_rnd_and_old_logs(self):
        def run(detach, mode, aux_weight, collect):
            args = fixture.args_for()
            args.audit_detach_mixed_hidden = detach
            args.adaptive_graph_mode = mode
            args.counterfactual_dvd_loss_weight = aux_weight
            args.audit_log_agent_gradients = True
            args.use_rnd = True
            th.manual_seed(824)
            mac, batch, logger = fixture.FakeMAC(args), fixture.FakeBatch(), fixture.FakeLogger()
            learner = fixture.audit.DVDNQLearner(mac, {}, logger, args)
            function = diagnostics if collect else lambda *inputs: {}
            with patch.object(fixture.audit, '_masked_gate_diagnostics', side_effect=function):
                for step in range(3):
                    learner.train(batch, step, step)
            return {
                'params': [p.detach().clone() for p in learner.params],
                'grads': [None if p.grad is None else p.grad.detach().clone() for p in learner.params],
                'optimizer': copy.deepcopy(learner.optimiser.state_dict()),
                'targets': [p.detach().clone() for p in list(learner.target_mac.parameters()) + list(learner.target_mixer.parameters())],
                'rnd': copy.deepcopy(learner.rnd.state_dict()),
                'rnd_optimizer': copy.deepcopy(learner.rnd_optimizer.state_dict()),
                'rnd_ms': copy.deepcopy(vars(learner.rnd_ms)),
                'rng': th.get_rng_state().clone(),
                'logs': {k: v for k, v in logger.stats.items() if not k.startswith('counterfactual_diag_')},
            }
        for detach in (False, True):
            for mode, aux_weight in [('learned', .1), ('identity', .1), ('learned', 0.)]:
                with self.subTest(detach=detach, mode=mode, aux_weight=aux_weight):
                    assert_equal(self, run(detach, mode, aux_weight, False), run(detach, mode, aux_weight, True))


if __name__ == '__main__':
    unittest.main(verbosity=2)
