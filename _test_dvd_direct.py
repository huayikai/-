"""CPU checks for the preregistered direct BM + trajectory control."""
import copy
import unittest
from unittest import mock

import torch as th
import torch.nn.functional as F

import _test_dvd_audit as fixture
import main_audit
from modules.mixers.dvd_adaptive_takeover import AdaptiveTakeoverDVDMixer
from modules.mixers.dvd_direct import DirectDVDMixer


class DirectMixerTests(unittest.TestCase):
    def args(self, detach=False):
        args = fixture.args_for(mixer="dvd_direct")
        args.audit_detach_mixed_hidden = detach
        return args

    def test_shared_initialization_and_downstream_rng_match_bacm(self):
        for seed in (79, 80, 81):
            with self.subTest(seed=seed):
                th.manual_seed(seed)
                reference = AdaptiveTakeoverDVDMixer(self.args())
                expected_rng = th.get_rng_state().clone()
                th.manual_seed(seed)
                direct = DirectDVDMixer(self.args())
                self.assertTrue(th.equal(th.get_rng_state(), expected_rng))
                for name in ("hyper_b1", "hyper_w2", "hyper_b2"):
                    for a, b in zip(getattr(direct, name).parameters(),
                                    getattr(reference.bm_mixer, name).parameters()):
                        self.assertTrue(th.equal(a, b))
                for name in ("gat", "hyper_dvd_w1"):
                    for a, b in zip(getattr(direct, name).parameters(),
                                    getattr(reference, name).parameters()):
                        self.assertTrue(th.equal(a, b))
                self.assertFalse(any("gate" in name or "hyper_w1" in name
                                     for name, _ in direct.named_parameters()))

    def test_raw_weight_formula_and_zero_signal_without_bm_fallback(self):
        th.manual_seed(611)
        direct = DirectDVDMixer(self.args())
        q, state, hidden = th.randn(2, 4, 3), th.randn(2, 4, 6), th.randn(2, 4, 3, 8)
        for zero in (False, True):
            h = th.zeros_like(hidden) if zero else hidden
            flat_s = state.reshape(-1, 6)
            graph = direct.gat(h.reshape(-1, 3, 8))
            hyper = direct.hyper_dvd_w1(flat_s).view(-1, 2, 8, 4)
            w1 = (hyper @ graph.transpose(-1, -2)).mean(1).transpose(-1, -2)
            b1 = direct.hyper_b1(flat_s).view(-1, 1, 8)
            w2 = direct.hyper_w2(flat_s).view(-1, 8, 1)
            b2 = direct.hyper_b2(flat_s).view(-1, 1, 1)
            expected = (F.elu(q.reshape(-1, 1, 3) @ w1 + b1) @ w2 + b2).view(2, 4, 1)
            th.testing.assert_close(direct(q, state, h), expected)
            if zero:
                self.assertEqual(w1.abs().sum().item(), 0)

    def test_hidden_detachment_keeps_q_and_graph_parameter_gradients(self):
        for detach in (False, True):
            with self.subTest(detach=detach):
                th.manual_seed(612)
                direct = DirectDVDMixer(self.args(detach))
                q = th.randn(2, 4, 3, requires_grad=True)
                h = th.randn(2, 4, 3, 8, requires_grad=True)
                direct(q, th.randn(2, 4, 6), h).square().mean().backward()
                self.assertGreater(q.grad.abs().sum().item(), 0)
                if detach:
                    self.assertIsNone(h.grad)
                else:
                    self.assertGreater(h.grad.abs().sum().item(), 0)
                for module in (direct.gat, direct.hyper_dvd_w1, direct.hyper_w2):
                    self.assertGreater(sum(p.grad.abs().sum().item() for p in module.parameters()
                                           if p.grad is not None), 0)


class DirectLearnerTests(unittest.TestCase):
    def test_own_sarsa_target_and_single_td_objective(self):
        args = fixture.args_for(mixer="dvd_direct")
        args.audit_detach_mixed_hidden = False
        th.manual_seed(613)
        mac, batch, logger = fixture.FakeMAC(args), fixture.FakeBatch(), fixture.FakeLogger()
        learner = fixture.audit.DVDNQLearner(mac, {}, logger, args)
        self.assertFalse(learner.is_counterfactual_router)
        with th.no_grad():
            learner.target_mac.init_hidden(batch.batch_size)
            values, hidden = [], []
            for t in range(batch.max_seq_length):
                values.append(learner.target_mac.forward(batch, t))
                hidden.append(learner.target_mac.hidden_states.clone())
            values = th.stack(values, 1)[:, 1:]
            chosen = th.gather(values, 3, batch["actions"][:, 1:]).squeeze(3)
            expected_next = learner.target_mixer(chosen, batch["state"][:, 1:],
                                                th.stack(hidden, 1)[:, 1:])
        before = [p.detach().clone() for p in mac.parameters()]
        builder = fixture.audit.build_td_lambda_targets_audit
        with mock.patch.object(fixture.audit, "build_td_lambda_targets_audit", wraps=builder) as spy:
            learner.train(batch, 0, 0)
        spy.assert_called_once()
        th.testing.assert_close(spy.call_args[0][3], expected_next)
        self.assertFalse(any(key.startswith("counterfactual_") for key in logger.stats))
        self.assertTrue(any(not th.equal(a, b) for a, b in zip(before, mac.parameters())))
        self.assertEqual(logger.stats["direct_detach_hidden"], 0)
        self.assertTrue(all(th.isfinite(p).all() for p in learner.params))
        self.assertTrue(all(th.isfinite(p.grad).all() for p in learner.params if p.grad is not None))

    def test_rnd_and_both_gradient_modes_complete_finite_updates(self):
        for detach in (False, True):
            with self.subTest(detach=detach):
                args = fixture.args_for(mixer="dvd_direct")
                args.audit_detach_mixed_hidden = detach
                args.use_rnd = True
                args.rnd_lr = 0.0005
                th.manual_seed(614)
                learner = fixture.audit.DVDNQLearner(fixture.FakeMAC(args), {}, fixture.FakeLogger(), args)
                learner.train(fixture.FakeBatch(), 0, 0)
                self.assertTrue(all(th.isfinite(p).all() for p in learner.params))
                self.assertTrue(all(th.isfinite(p.grad).all() for p in learner.params if p.grad is not None))

    def test_entry_records_the_new_sources_and_preserves_protocol(self):
        config, params, sources = main_audit.load_configuration([
            "--config=dvd_audit_direct", "--env-config=sc2", "with",
            "env_args.map_name=6h_vs_8z", "seed=84", "audit_detach_mixed_hidden=True",
        ])
        preview = main_audit.inspect_overrides(copy.deepcopy(config), params)
        self.assertEqual(preview["mixer"], "dvd_direct")
        self.assertEqual(preview["learner"], "dvd_audit_learner")
        self.assertTrue(preview["audit_detach_mixed_hidden"])
        self.assertTrue(preview["audit_fix_td_lambda"])
        self.assertTrue(preview["use_rnd"])
        self.assertEqual(preview["t_max"], 5050000)
        self.assertIn("modules/mixers/dvd_direct.py", sources)
        self.assertFalse(any(key.startswith("counterfactual_") for key in preview))


if __name__ == "__main__":
    unittest.main(verbosity=2)
