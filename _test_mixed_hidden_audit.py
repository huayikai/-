"""CPU checks for independent mixed-hidden detachment and gradient logging."""
import copy
import math
import unittest
from types import SimpleNamespace

import torch as th
import torch.nn as nn

import _test_dvd_audit as existing
from modules.mixers.dvd_counterfactual_router_audit import CounterfactualRouterAuditMixer


class MixedHiddenTests(unittest.TestCase):
    def args(self, detach=True, aux=True, diagnostics=False):
        args = existing.args_for(aux=aux)
        args.audit_detach_mixed_hidden = detach
        args.audit_log_agent_gradients = diagnostics
        return args

    def forward(self, detach=True, aux=True):
        th.manual_seed(301)
        mixer = CounterfactualRouterAuditMixer(self.args(detach, aux))
        q = th.randn(2, 3, 3, requires_grad=True)
        hidden = th.randn(2, 3, 3, 8, requires_grad=True)
        state = th.randn(2, 3, 6)
        output = mixer(q, state, hidden)
        return mixer, q, hidden, output

    def test_only_direct_hidden_path_is_cut(self):
        for detach in (False, True):
            with self.subTest(detach=detach):
                mixer, q, hidden, output = self.forward(detach)
                output.square().mean().backward()
                self.assertGreater(q.grad.abs().sum().item(), 0)
                if detach:
                    self.assertIsNone(hidden.grad)
                else:
                    self.assertGreater(hidden.grad.abs().sum().item(), 0)
                for module in (mixer.bm_mixer, mixer.gat, mixer.hyper_dvd_w1):
                    self.assertGreater(sum(p.grad.abs().sum().item()
                                           for p in module.parameters() if p.grad is not None), 0)
                self.assertTrue(all(p.grad is None for p in mixer.gate_net.parameters()))

    def test_auxiliary_switch_is_independent(self):
        for detach in (False, True):
            for aux in (False, True):
                with self.subTest(detach=detach, aux=aux):
                    mixer, q, hidden, output = self.forward(detach, aux)
                    mixer.last_counterfactual_dvd_q.square().mean().backward()
                    self.assertIsNone(q.grad)
                    self.assertEqual(hidden.grad is None, aux)
                    if not aux:
                        self.assertGreater(hidden.grad.abs().sum().item(), 0)
                    self.assertTrue(all(p.grad is None for p in mixer.bm_mixer.parameters()))
                    self.assertGreater(mixer.gat.W.weight.grad.abs().sum().item(), 0)
                    self.assertGreater(mixer.hyper_dvd_w1.weight.grad.abs().sum().item(), 0)

    def test_forward_values_and_zero_signal_fallback_are_preserved(self):
        for aux in (False, True):
            off = CounterfactualRouterAuditMixer(self.args(False, aux))
            on = CounterfactualRouterAuditMixer(self.args(True, aux))
            on.load_state_dict(off.state_dict())
            q, state = th.randn(2, 3, 3), th.randn(2, 3, 6)
            for hidden in (th.randn(2, 3, 3, 8), th.zeros(2, 3, 3, 8)):
                for no_grad in (False, True):
                    with self.subTest(aux=aux, zero=not hidden.any(), no_grad=no_grad):
                        with th.set_grad_enabled(not no_grad):
                            self.assertTrue(th.equal(off(q, state, hidden), on(q, state, hidden)))
                            for attr in ('last_counterfactual_bm_q', 'last_counterfactual_dvd_q',
                                         'last_counterfactual_gate'):
                                self.assertTrue(th.equal(getattr(off, attr), getattr(on, attr)))
                            if not hidden.any():
                                self.assertTrue(th.equal(on(q, state, hidden), on.bm_mixer(q, state)))

    def test_agent_still_updates_and_logging_is_observational(self):
        th.manual_seed(302)
        mac = existing.FakeMAC(self.args())
        other_mac = copy.deepcopy(mac)
        batch = existing.FakeBatch()
        th.manual_seed(303)
        off = existing.audit.DVDNQLearner(mac, {}, existing.FakeLogger(), self.args())
        th.manual_seed(303)
        on = existing.audit.DVDNQLearner(other_mac, {}, existing.FakeLogger(),
                                        self.args(diagnostics=True))
        before = [p.detach().clone() for p in mac.parameters()]
        off.train(batch, 0, 0)
        on.train(batch, 0, 0)
        self.assertTrue(any(not th.equal(a, b) for a, b in zip(before, mac.parameters())))
        for a, b in zip(off.params, on.params):
            self.assertTrue(th.equal(a, b))
            self.assertEqual(a.grad is None, b.grad is None)
            if a.grad is not None:
                self.assertTrue(th.equal(a.grad, b.grad))
            self.assertTrue(th.isfinite(b).all())
        for a, b in zip(off.target_mac.parameters(), on.target_mac.parameters()):
            self.assertTrue(th.equal(a, b))
        for a, b in zip(off.target_mixer.parameters(), on.target_mixer.parameters()):
            self.assertTrue(th.equal(a, b))
        stats = on.logger.stats
        self.assertEqual(stats['audit_detach_mixed_hidden'], 1)
        self.assertGreater(stats['audit_agent_bm_grad_norm'], 0)
        self.assertGreater(stats['audit_agent_mix_grad_norm'], 0)
        self.assertLessEqual(abs(stats['audit_agent_bm_mix_grad_cosine']), 1)
        self.assertTrue(all(math.isfinite(value) for value in stats.values()))
        self.assertNotIn('audit_agent_bm_grad_norm', off.logger.stats)

    def test_gradient_statistics_match_known_vectors_without_accumulating(self):
        learner = existing.audit.DVDNQLearner.__new__(existing.audit.DVDNQLearner)
        learner.mac = nn.Linear(2, 1, bias=False)
        learner.counterfactual_mix_loss_weight = 0.25
        learner.counterfactual_bm_loss_weight = 1.0
        weights = learner.mac.weight
        stats = learner._agent_gradient_stats(2 * weights[0, 0],
                                             -weights[0, 0] + 3 * weights[0, 1])
        self.assertAlmostEqual(stats['audit_agent_bm_grad_norm'], 2)
        self.assertAlmostEqual(stats['audit_agent_mix_grad_norm'], math.sqrt(10), places=6)
        self.assertAlmostEqual(stats['audit_agent_bm_mix_grad_cosine'], -1 / math.sqrt(10), places=6)
        self.assertAlmostEqual(stats['audit_agent_weighted_mix_bm_grad_ratio'], math.sqrt(10) / 8, places=6)
        self.assertIsNone(weights.grad)

    def test_entry_accepts_explicit_experiment_controls(self):
        config, params, _ = existing.main_audit.load_configuration([
            '--config=dvd_audit_router', '--env-config=sc2', 'with',
            'audit_detach_mixed_hidden=True', 'audit_log_agent_gradients=True',
            'counterfactual_mix_loss_weight=0.25', 't_max=5050000',
        ])
        self.assertFalse(config['audit_detach_mixed_hidden'])
        self.assertFalse(config['audit_log_agent_gradients'])
        preview = existing.main_audit.inspect_overrides(config, params)
        self.assertTrue(preview['audit_detach_mixed_hidden'])
        self.assertTrue(preview['audit_log_agent_gradients'])
        self.assertEqual(preview['t_max'], 5050000)


if __name__ == '__main__':
    unittest.main(verbosity=2)
