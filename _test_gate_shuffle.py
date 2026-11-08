"""CPU regressions for the opt-in, training-only gate permutation."""
import copy
import tempfile
import unittest
from pathlib import Path

import torch as th
from unittest import mock

import _test_dvd_audit as fixture
from modules.mixers.dvd_counterfactual_router_audit import CounterfactualRouterAuditMixer


def arguments(enabled=True, detach=False, seed=85):
    args = fixture.args_for()
    args.seed = seed
    args.audit_shuffle_gate = enabled
    args.audit_detach_mixed_hidden = detach
    args.audit_log_agent_gradients = True
    return args


class GateShuffleTests(unittest.TestCase):
    @staticmethod
    def reject_generator_copy(generator, memo):
        # PyTorch 1.12.1 raises this error when a Generator is deep-copied.
        raise TypeError("cannot pickle 'torch._C.Generator' object")

    def inputs(self):
        th.manual_seed(901)
        q = th.randn(3, 5, 3, requires_grad=True)
        states = th.randn(3, 5, 6)
        hidden = th.randn(3, 5, 3, 8, requires_grad=True)
        with th.no_grad():
            hidden[0, 0] = 0
        mask = th.tensor([
            [1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0],
            [1, 0, 0, 0, 0],
        ]).unsqueeze(-1)
        return q, states, hidden, mask

    def mixer(self, enabled=True, detach=False, seed=85):
        th.manual_seed(902)
        mixer = CounterfactualRouterAuditMixer(arguments(enabled, detach, seed))
        with th.no_grad():
            mixer.gate_net[2].weight.normal_(0, .4)
        return mixer

    def test_real_forward_preserves_distribution_mask_and_fallback(self):
        q, states, hidden, mask = self.inputs()
        mixer = self.mixer()
        reference = self.mixer(False)
        actual = mixer(q, states, hidden, valid_mask=mask)
        normal = reference(q, states, hidden)
        original = mixer.last_counterfactual_gate
        applied = mixer.last_audit_routing_gate.view_as(original)
        eligible = mask.bool() & mixer.last_adaptive_signal_mask.view_as(mask)
        th.testing.assert_close(original, reference.last_counterfactual_gate, rtol=0, atol=0)
        self.assertTrue(th.equal(original[~eligible], applied[~eligible]))
        self.assertTrue(th.equal(original[eligible].sort().values, applied[eligible].sort().values))
        self.assertFalse(th.equal(original[eligible], applied[eligible]))
        self.assertEqual(applied[0, 0].item(), 0)
        self.assertTrue(th.equal(actual[0, 0], mixer.last_counterfactual_bm_q[0, 0]))
        self.assertFalse(th.equal(actual, normal))
        self.assertEqual(mixer.last_audit_shuffle_stats['eligible_count'].item(), 8)
        self.assertEqual(mixer.last_audit_shuffle_stats['fallback_count'].item(), 1)
        self.assertGreater(mixer.last_audit_shuffle_stats['changed_fraction'].item(), 0)
        self.assertLess(abs(mixer.last_audit_shuffle_stats['mean_delta'].item()), 1e-7)
        self.assertLess(abs(mixer.last_audit_shuffle_stats['std_delta'].item()), 1e-7)

        # Independently reconstruct the mixed value using the applied gates.
        _, bm, dvd, _ = mixer._weights(states.reshape(-1, 6), hidden.reshape(-1, 3, 8))
        flat_states = states.reshape(-1, 6)
        weights = (1 - applied.reshape(-1, 1, 1)) * bm + applied.reshape(-1, 1, 1) * dvd
        expected = mixer._mix_value(
            q.reshape(-1, 1, 3), weights,
            mixer.bm_mixer.hyper_b1(flat_states).view(-1, 1, 8),
            mixer.bm_mixer.hyper_w2(flat_states).view(-1, 8, 1),
            mixer.bm_mixer.hyper_b2(flat_states).view(-1, 1, 1),
        ).view_as(actual)
        th.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_permutation_uses_its_own_random_stream(self):
        q, states, hidden, mask = self.inputs()
        mixer = self.mixer()
        global_before = th.get_rng_state().clone()
        private_before = mixer.gate_shuffle_generator.get_state().clone()
        mixer(q, states, hidden, valid_mask=mask)
        self.assertTrue(th.equal(global_before, th.get_rng_state()))
        self.assertFalse(th.equal(private_before, mixer.gate_shuffle_generator.get_state()))
        same = self.mixer()
        same(q, states, hidden, valid_mask=mask)
        self.assertTrue(th.equal(mixer.last_audit_routing_gate, same.last_audit_routing_gate))

    def test_deepcopy_preserves_advanced_private_stream_without_pickling_generator(self):
        disabled = self.mixer(False)
        mixer = self.mixer()
        th.randperm(17, generator=mixer.gate_shuffle_generator)
        before = mixer.gate_shuffle_generator.get_state().clone()
        global_before = th.get_rng_state().clone()
        generator_type = type(mixer.gate_shuffle_generator)
        # nn.Module has no __getstate__ in PyTorch 1.12.1 / Python 3.8.
        parent_getstate = th.nn.Module.__dict__.get('__getstate__')
        with mock.patch.dict(copy._deepcopy_dispatch,
                             {generator_type: self.reject_generator_copy}):
            if parent_getstate is not None:
                del th.nn.Module.__getstate__
            try:
                with self.assertRaisesRegex(TypeError, 'cannot pickle'):
                    copy.deepcopy(mixer.gate_shuffle_generator)
                target = copy.deepcopy(mixer)
                disabled = copy.deepcopy(disabled)
            finally:
                if parent_getstate is not None:
                    th.nn.Module.__getstate__ = parent_getstate
        self.assertIsNone(disabled.gate_shuffle_generator)
        self.assertIsNot(mixer.gate_shuffle_generator, target.gate_shuffle_generator)
        self.assertTrue(th.equal(before, target.gate_shuffle_generator.get_state()))
        self.assertTrue(th.equal(global_before, th.get_rng_state()))
        for source_param, target_param in zip(mixer.parameters(), target.parameters()):
            self.assertTrue(th.equal(source_param, target_param))
            self.assertNotEqual(source_param.data_ptr(), target_param.data_ptr())
        expected = th.randperm(31, generator=mixer.gate_shuffle_generator)
        self.assertTrue(th.equal(before, target.gate_shuffle_generator.get_state()))
        actual = th.randperm(31, generator=target.gate_shuffle_generator)
        self.assertTrue(th.equal(expected, actual))

    def test_learner_trains_when_generators_cannot_be_pickled(self):
        generator_type = type(th.Generator(device='cpu'))
        with mock.patch.dict(copy._deepcopy_dispatch,
                             {generator_type: self.reject_generator_copy}):
            for detach in (False, True):
                args = arguments(detach=detach)
                args.use_rnd = True
                mac = fixture.FakeMAC(args)
                batch, logger = fixture.FakeBatch(), fixture.FakeLogger()
                learner = fixture.audit.DVDNQLearner(mac, {}, logger, args)
                target_before = learner.target_mixer.gate_shuffle_generator.get_state().clone()
                for step in range(3):
                    learner.train(batch, step, step)
                self.assertTrue(th.equal(target_before,
                    learner.target_mixer.gate_shuffle_generator.get_state()))
                self.assertTrue(all(th.isfinite(p).all() for p in learner.params))
                self.assertEqual(logger.stats['audit_gate_shuffle_distribution_error'], 0)

    def test_eval_and_target_forwards_do_not_permute_or_advance_rng(self):
        q, states, hidden, mask = self.inputs()
        mixer = self.mixer()
        reference = self.mixer(False)
        before = mixer.gate_shuffle_generator.get_state().clone()
        mixer.eval()
        th.testing.assert_close(mixer(q, states, hidden), reference(q, states, hidden), rtol=0, atol=0)
        mixer.train()
        with th.no_grad():
            th.testing.assert_close(mixer(q, states, hidden), reference(q, states, hidden), rtol=0, atol=0)
        self.assertTrue(th.equal(before, mixer.gate_shuffle_generator.get_state()))
        self.assertIsNone(mixer.last_audit_shuffle_stats)

    def test_empty_single_and_constant_gate_batches(self):
        q, states, hidden, mask = self.inputs()
        mixer = self.mixer()
        mixer(q, states, hidden, valid_mask=th.zeros_like(mask))
        self.assertEqual(mixer.last_audit_shuffle_stats['eligible_count'].item(), 0)
        self.assertEqual(mixer.last_audit_shuffle_stats['changed_fraction'].item(), 0)
        single = th.zeros_like(mask)
        single[0, 1] = 1
        mixer = self.mixer()
        before = mixer.gate_shuffle_generator.get_state().clone()
        mixer(q, states, hidden, valid_mask=single)
        self.assertEqual(mixer.last_audit_shuffle_stats['eligible_count'].item(), 1)
        self.assertTrue(th.equal(before, mixer.gate_shuffle_generator.get_state()))
        with th.no_grad():
            mixer.gate_net[2].weight.zero_()
        mixer(q, states, hidden, valid_mask=mask)
        self.assertEqual(mixer.last_audit_shuffle_stats['changed_fraction'].item(), 0)

    def test_routing_supervision_keeps_original_gate_gradients(self):
        q, states, hidden, mask = self.inputs()
        enabled, disabled = self.mixer(), self.mixer(False)
        mixed = enabled(q, states, hidden, valid_mask=mask)
        disabled(q, states, hidden)
        # The mixed TD term must never train the gate, including via shuffle.
        mixed.square().mean().backward(retain_graph=True)
        self.assertTrue(all(p.grad is None for p in enabled.gate_net.parameters()))
        target = th.linspace(0, 1, mask.numel()).view_as(mask)
        for mixer in (enabled, disabled):
            route_loss = ((mixer.last_counterfactual_gate - target).square() * mask).sum() / mask.sum()
            route_loss.backward()
        for a, b in zip(enabled.gate_net.parameters(), disabled.gate_net.parameters()):
            th.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)

    def test_disabled_switch_keeps_initialization_rng_and_three_updates(self):
        learners = []
        for flag in (None, False):
            args = arguments(False)
            if flag is None:
                del args.audit_shuffle_gate
            th.manual_seed(903)
            mac, batch, logger = fixture.FakeMAC(args), fixture.FakeBatch(), fixture.FakeLogger()
            learner = fixture.audit.DVDNQLearner(mac, {}, logger, args)
            for step in range(3):
                learner.train(batch, step, step)
            learners.append((learner, th.get_rng_state().clone()))
        a, b = [pair[0] for pair in learners]
        self.assertTrue(th.equal(learners[0][1], learners[1][1]))
        for x, y in zip(a.params, b.params):
            self.assertTrue(th.equal(x, y))
        self.assertEqual(list(a.mixer.state_dict()), list(b.mixer.state_dict()))

    def test_real_learner_updates_both_hidden_modes_and_logs_masks(self):
        for detach in (False, True):
            th.manual_seed(904)
            args = arguments(detach=detach)
            args.use_rnd = True
            mac, batch, logger = fixture.FakeMAC(args), fixture.FakeBatch(), fixture.FakeLogger()
            learner = fixture.audit.DVDNQLearner(mac, {}, logger, args)
            target_rng = learner.target_mixer.gate_shuffle_generator.get_state().clone()
            before = learner.mixer.hyper_dvd_w1.weight.detach().clone()
            for step in range(3):
                learner.train(batch, step, step)
            self.assertFalse(th.equal(before, learner.mixer.hyper_dvd_w1.weight))
            self.assertTrue(th.equal(target_rng, learner.target_mixer.gate_shuffle_generator.get_state()))
            self.assertTrue(all(th.isfinite(p).all() for p in learner.params))
            self.assertEqual(logger.stats['audit_shuffle_gate'], 1)
            self.assertEqual(logger.stats['audit_gate_shuffle_valid_count'], 12)
            self.assertLessEqual(abs(logger.stats['audit_gate_shuffle_mean_delta']), 1e-7)
            self.assertLessEqual(abs(logger.stats['audit_gate_shuffle_std_delta']), 1e-7)
            self.assertGreater(logger.stats['audit_gate_shuffle_changed_fraction'], 0)
            self.assertEqual(logger.stats['audit_gate_shuffle_distribution_error'], 0)

    def test_missing_mask_and_invalid_switch_fail_early(self):
        q, states, hidden, mask = self.inputs()
        mixer = self.mixer()
        with self.assertRaises(ValueError):
            mixer(q, states, hidden)
        with self.assertRaises(ValueError):
            mixer(q, states, hidden, valid_mask=mask[:, :-1])
        args = arguments()
        args.audit_shuffle_gate = 'False'
        with self.assertRaises(ValueError):
            CounterfactualRouterAuditMixer(args)
        args = arguments()
        args.mixer = 'dvd_takeover'
        with self.assertRaises(ValueError):
            fixture.audit.DVDNQLearner(fixture.FakeMAC(args), {}, fixture.FakeLogger(), args)

    def test_shuffle_state_is_saved_and_restored_with_models(self):
        args = arguments()
        learner = fixture.audit.DVDNQLearner(fixture.FakeMAC(args), {}, fixture.FakeLogger(), args)
        q, states, hidden, mask = self.inputs()
        learner.mixer(q, states, hidden, valid_mask=mask)
        before = learner.mixer.gate_shuffle_generator.get_state().clone()
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(learner.mac, 'save_models', create=True):
                learner.save_models(directory)
            saved = Path(directory) / 'gate_shuffle_rng.th'
            self.assertTrue(saved.is_file())
            learner.mixer(q, states, hidden, valid_mask=mask)
            with mock.patch.object(learner.mac, 'load_models', create=True), mock.patch.object(learner.target_mac, 'load_models', create=True):
                learner.load_models(directory)
            self.assertTrue(th.equal(before, learner.mixer.gate_shuffle_generator.get_state()))


if __name__ == '__main__':
    th.set_num_threads(1)
    unittest.main(verbosity=2)
