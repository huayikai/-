"""CPU forward/backward checks for BACM component ablations; no SC2 run."""
import copy
import unittest

import torch as th
import torch.nn.functional as F

import _test_dvd_audit as fixture
from modules.mixers.dvd_counterfactual_router_audit import CounterfactualRouterAuditMixer


class ComponentTests(unittest.TestCase):
    def args(self, mode='learned', normalize=True):
        args = fixture.args_for()
        args.adaptive_graph_mode = mode
        args.adaptive_normalize_dvd_weights = normalize
        return args

    def test_identity_and_uniform_have_expected_features_and_dependency(self):
        for mode in ('identity', 'uniform'):
            mixer = CounterfactualRouterAuditMixer(self.args(mode))
            h = th.randn(2, 3, 8, requires_grad=True)
            projected = mixer.gat.W(h).view(2, 3, 2, 4).permute(0, 2, 1, 3)
            expected = projected if mode == 'identity' else projected.mean(2, keepdim=True).expand_as(projected)
            features, attention = mixer._graph_features(h)
            th.testing.assert_close(features, F.elu(expected))
            features[:, :, 0].sum().backward()
            self.assertGreater(h.grad[:, 0].abs().sum().item(), 0)
            if mode == 'identity':
                self.assertEqual(h.grad[:, 1:].abs().sum().item(), 0)
            else:
                self.assertGreater(h.grad[:, 1:].abs().sum().item(), 0)
            self.assertIsNone(mixer.gat.att_a.grad)
            entropy, disagreement, _ = mixer._attention_statistics(attention)
            th.testing.assert_close(entropy, th.full_like(entropy, 0. if mode=='identity' else 1.))
            th.testing.assert_close(disagreement, th.zeros_like(disagreement))

    def test_normalization_and_auxiliary_use_the_same_candidate(self):
        for mode in ('learned', 'identity', 'uniform'):
            for normalize in (True, False):
                with self.subTest(mode=mode, normalize=normalize):
                    mixer = CounterfactualRouterAuditMixer(self.args(mode, normalize))
                    state, h = th.randn(6, 6), th.randn(6, 3, 8)
                    _, bm, candidate, _ = mixer._weights(state, h)
                    features, _ = mixer._graph_features(h)
                    hyper = mixer.hyper_dvd_w1(state).view(-1, 2, 8, 4)
                    raw = th.matmul(hyper, features.permute(0,1,3,2)).mean(1).permute(0,2,1)
                    if normalize:
                        th.testing.assert_close(candidate.square().mean((1,2)), bm.square().mean((1,2)))
                    else:
                        self.assertTrue(th.equal(candidate, raw))
                    self.assertTrue(th.equal(candidate, mixer._auxiliary_dvd_weights(state, h)))
                    self.assertAlmostEqual(mixer.last_dvd_raw_w1_rms.item(), raw.square().mean((1,2)).sqrt().mean().item())

    def test_zero_hidden_is_exact_bm_with_finite_gradients_for_all_modes(self):
        for mode in ('learned', 'identity', 'uniform'):
            for normalize in (True, False):
                mixer = CounterfactualRouterAuditMixer(self.args(mode, normalize))
                q = th.randn(2, 3, 3, requires_grad=True)
                h = th.zeros(2, 3, 3, 8, requires_grad=True)
                state = th.randn(2, 3, 6)
                output = mixer(q, state, h)
                self.assertTrue(th.equal(output, mixer.bm_mixer(q, state)))
                self.assertTrue(th.equal(output, mixer.last_counterfactual_dvd_q))
                self.assertEqual(mixer.last_counterfactual_gate.abs().sum().item(), 0)
                (output.square().mean()+mixer.last_counterfactual_dvd_q.square().mean()).backward()
                self.assertTrue(all(th.isfinite(p.grad).all() for p in mixer.parameters() if p.grad is not None))
                self.assertTrue(th.isfinite(q.grad).all())
                self.assertTrue(th.isfinite(h.grad).all())

    def test_new_modes_do_not_change_initialization_or_rng(self):
        for mode in ('learned', 'identity', 'uniform'):
            for normalize in (True, False):
                th.manual_seed(809)
                baseline = CounterfactualRouterAuditMixer(self.args())
                expected_rng = th.get_rng_state()
                th.manual_seed(809)
                variant = CounterfactualRouterAuditMixer(self.args(mode, normalize))
                self.assertTrue(th.equal(expected_rng, th.get_rng_state()))
                self.assertEqual(list(baseline.state_dict()), list(variant.state_dict()))
                for key, tensor in baseline.state_dict().items():
                    self.assertTrue(th.equal(tensor, variant.state_dict()[key]))

    def test_ablation_learner_updates_and_zero_aux_keeps_mixed_training(self):
        for mode, normalize, weight in [('identity',True,.1),('learned',False,.1),('learned',True,0.)]:
            for detach in (False, True):
                with self.subTest(mode=mode,normalize=normalize,weight=weight,detach=detach):
                    args = self.args(mode, normalize)
                    args.audit_detach_mixed_hidden = detach
                    args.counterfactual_dvd_loss_weight = weight
                    args.counterfactual_mix_loss_weight = .25
                    th.manual_seed(810)
                    mac, batch, logger = fixture.FakeMAC(args), fixture.FakeBatch(), fixture.FakeLogger()
                    learner = fixture.audit.DVDNQLearner(mac,{},logger,args)
                    before = learner.mixer.gat.W.weight.detach().clone()
                    learner.train(batch, 0, 0)
                    self.assertFalse(th.equal(before, learner.mixer.gat.W.weight))
                    self.assertGreater(learner.mixer.hyper_dvd_w1.weight.grad.abs().sum().item(),0)
                    self.assertTrue(all(th.isfinite(p).all() for p in learner.params))
                    self.assertTrue(all(th.isfinite(p.grad).all() for p in learner.params if p.grad is not None))
                    self.assertEqual(logger.stats['counterfactual_dvd_loss_weight'],weight)
                    if mode=='identity':
                        self.assertIsNone(learner.mixer.gat.att_a.grad)

    def test_zero_aux_learner_matches_value_plus_route_gradient_oracle(self):
        args = self.args()
        args.counterfactual_dvd_loss_weight = 0.
        args.counterfactual_route_loss_weight = .1
        args.counterfactual_sparse_weight = 0.
        args.grad_norm_clip = 1e20
        th.manual_seed(811)
        mac, batch, logger = fixture.FakeMAC(args), fixture.FakeBatch(), fixture.FakeLogger()
        learner = fixture.audit.DVDNQLearner(mac,{},logger,args)
        captured = {}
        def targets(*values, **kwargs):
            result = fixture.build_td_lambda_targets_audit(*values,**kwargs)
            captured['target'] = result.detach()
            return result
        from unittest.mock import patch
        with patch.object(fixture.audit,'build_td_lambda_targets_audit',side_effect=targets), patch.object(learner.optimiser,'step'), patch.object(learner,'_update_targets_soft'):
            learner.train(batch,0,0)
        observed = [None if p.grad is None else p.grad.detach().clone() for p in learner.params]
        learner.optimiser.zero_grad()
        mac.init_hidden(batch.batch_size)
        qs, hs = [], []
        for t in range(batch.max_seq_length):
            qs.append(mac.forward(batch,t=t)); hs.append(mac.hidden_states.clone())
        qs=th.stack(qs,dim=1)
        chosen=th.gather(qs[:,:-1],3,batch['actions'][:,:-1]).squeeze(3)
        out=learner.mixer(chosen,batch['state'][:,:-1],th.stack(hs,dim=1)[:,:-1])
        mask=batch['filled'][:,:-1].float().clone()
        mask[:,1:]*=1-batch['terminated'][:,:-2].float()
        error=out-captured['target']
        bm_error=learner.mixer.last_counterfactual_bm_q-captured['target']
        dvd_error=learner.mixer.last_counterfactual_dvd_q-captured['target']
        route=fixture.audit.build_counterfactual_route_target(bm_error.detach().abs(),dvd_error.detach().abs(),learner.mixer.gate_max,learner.counterfactual_route_margin,learner.counterfactual_route_min_abs_advantage,learner.counterfactual_route_eps)[0]
        expected=((.25*(mask*error).square().sum()+(mask*bm_error).square().sum())/1.25+.1*(mask*(learner.mixer.last_counterfactual_gate-route)).square().sum())/mask.sum()
        expected.backward()
        for p, actual in zip(learner.params,observed):
            if actual is None:
                self.assertIsNone(p.grad)
            else:
                th.testing.assert_close(p.grad,actual,rtol=2e-5,atol=2e-7)

    def test_invalid_switches_fail_early(self):
        for mode, normalize in [('none',True),('learned','False')]:
            with self.assertRaises(ValueError):
                CounterfactualRouterAuditMixer(self.args(mode,normalize))


if __name__ == '__main__':
    unittest.main(verbosity=2)
