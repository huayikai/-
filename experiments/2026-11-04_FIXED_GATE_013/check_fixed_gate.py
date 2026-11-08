"""Check fixed-gate controls on the real learner with a small CPU batch."""
import copy
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

import torch as th
import _test_dvd_audit as fixture


def main():
    th.set_num_threads(1)
    checks = []
    for seed, detach, gradient_logging in [(80, False, False), (81, False, True), (80, True, True), (84, True, True)]:
        args = fixture.args_for()
        args.adaptive_gate_init = 0.13
        args.counterfactual_mix_loss_weight = 0.25
        args.counterfactual_bm_loss_weight = 1.0
        args.counterfactual_dvd_loss_weight = 0.1
        args.counterfactual_route_loss_weight = 0.0
        args.counterfactual_sparse_weight = 0.0
        args.audit_detach_mixed_hidden = detach
        args.audit_log_agent_gradients = gradient_logging
        args.adaptive_normalize_dvd_weights = True
        args.adaptive_graph_mode = 'learned'
        args.use_rnd = True
        args.rnd_lr = 0.0005
        th.manual_seed(seed)
        learner = fixture.audit.DVDNQLearner(fixture.FakeMAC(args), {}, fixture.FakeLogger(), args)
        rng_after_fixed_init = th.get_rng_state().clone()
        dynamic_args = copy.deepcopy(args)
        dynamic_args.adaptive_gate_init = 0.05
        dynamic_args.counterfactual_route_loss_weight = 0.1
        th.manual_seed(seed)
        dynamic = fixture.audit.DVDNQLearner(fixture.FakeMAC(dynamic_args), {}, fixture.FakeLogger(), dynamic_args)
        assert th.equal(rng_after_fixed_init, th.get_rng_state())
        for name, value in learner.mac.state_dict().items():
            assert th.equal(value, dynamic.mac.state_dict()[name]), name
        for name, value in learner.mixer.state_dict().items():
            if name != 'gate_net.2.bias':
                assert th.equal(value, dynamic.mixer.state_dict()[name]), name
        gate_before = {name: p.detach().clone() for name, p in learner.mixer.gate_net.named_parameters()}
        agent_before = [p.detach().clone() for p in learner.mac.parameters()]
        candidate_before = [p.detach().clone() for p in learner.mixer.hyper_dvd_w1.parameters()]
        batch = fixture.FakeBatch()
        for step in range(3):
            learner.train(batch, step, step)
            for name, p in learner.mixer.gate_net.named_parameters():
                assert th.equal(p.detach(), gate_before[name]), name
                assert p.grad is None or th.count_nonzero(p.grad).item() == 0, name
        assert any(not th.equal(p, before) for p, before in zip(learner.mac.parameters(), agent_before))
        assert any(not th.equal(p, before) for p, before in zip(learner.mixer.hyper_dvd_w1.parameters(), candidate_before))
        assert all(th.isfinite(p).all() for p in learner.params)
        assert th.count_nonzero(learner.mixer.gate_net[2].weight).item() == 0
        assert abs(learner.mixer.gate_net[2].bias.item() - math.log(.13 / .87)) < 1e-6
        with th.no_grad():
            q = th.randn(2, 2, args.n_agents)
            state = th.randn(2, 2, args.state_shape)
            hidden = th.randn(2, 2, args.n_agents, args.rnn_hidden_dim)
            learner.mixer(q, state, hidden)
            gate = learner.mixer.last_counterfactual_gate
            assert th.allclose(gate, th.full_like(gate, .13), atol=1e-7)
            learner.mixer(q, state, th.zeros_like(hidden))
            assert th.count_nonzero(learner.mixer.last_counterfactual_gate).item() == 0
        checks.append({'seed': seed, 'detach': detach, 'gradient_logging': gradient_logging, 'updates': 3, 'gate_parameters_unchanged': True, 'gate_gradients_zero': True, 'common_initialization_and_rng_match': True, 'agent_and_candidate_updated': True, 'nonzero_signal_gate': 0.13, 'zero_signal_gate': 0.0, 'finite_parameters': True})
    result = {'date': '2026-11-04', 'torch_version': th.__version__, 'device': 'CPU', 'scope': 'Small synthetic batch on the existing real learner; validates control behavior, not SC2 performance.', 'cases': checks}
    (HERE / 'cpu_check.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
