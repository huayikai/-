# -*- coding: utf-8 -*-
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch as th

from modules.mixers.dvd_counterfactual_router import (
    CounterfactualRouterDVDMixer,
    build_counterfactual_route_target,
)


def make_args():
    return SimpleNamespace(
        n_agents=5,
        state_shape=24,
        mixing_embed_dim=32,
        hypernet_embed=64,
        rnn_hidden_dim=64,
        dvd_heads=8,
        gat_embed_dim=16,
        use_orthogonal=False,
        qmix_pos_func="abs",
        adaptive_bm_abs=False,
        adaptive_gate_hidden_dim=32,
        adaptive_gate_max=1.0,
        adaptive_gate_init=0.05,
        adaptive_norm_eps=1e-6,
    )


args = make_args()
batch_size, sequence_length = 4, 6
th.manual_seed(1234)
mixer = CounterfactualRouterDVDMixer(args)

agent_qs = th.randn(
    batch_size, sequence_length, args.n_agents, requires_grad=True
)
states = th.randn(batch_size, sequence_length, args.state_shape)
hiddens = th.randn(
    batch_size,
    sequence_length,
    args.n_agents,
    args.rnn_hidden_dim,
)

# One forward pass must expose all counterfactual branches with matching shape.
mixed_q = mixer(agent_qs, states, hiddens)
bm_q = mixer.last_counterfactual_bm_q
dvd_q = mixer.last_counterfactual_dvd_q
gate = mixer.last_counterfactual_gate
expected_shape = (batch_size, sequence_length, 1)
assert mixed_q.shape == expected_shape
assert bm_q.shape == expected_shape
assert dvd_q.shape == expected_shape
assert gate.shape == expected_shape
assert th.all(gate > 0.0)
assert th.all(gate <= args.adaptive_gate_max)
assert abs(gate.mean().item() - args.adaptive_gate_init) < 1e-6
assert th.equal(bm_q, mixer.bm_mixer(agent_qs, states))

# The mixed TD path trains both experts but must not directly train the gate.
mixer.zero_grad(set_to_none=True)
agent_qs.grad = None
mixed_q.mean().backward()
for parameter in mixer.gate_net.parameters():
    assert parameter.grad is None
assert mixer.hyper_dvd_w1.weight.grad.abs().sum().item() > 0.0
assert mixer.bm_mixer.hyper_w1[0].weight.grad.abs().sum().item() > 0.0

# A DVD-only auxiliary loss must not update agent inputs or the BM hypernets.
mixer.zero_grad(set_to_none=True)
agent_qs.grad = None
mixed_q = mixer(agent_qs, states, hiddens)
dvd_q = mixer.last_counterfactual_dvd_q
dvd_q.mean().backward()
assert agent_qs.grad is None
assert mixer.hyper_dvd_w1.weight.grad.abs().sum().item() > 0.0
assert mixer.gat.W.weight.grad.abs().sum().item() > 0.0
for parameter in mixer.bm_mixer.parameters():
    assert parameter.grad is None

# The real combined objective must train BM, DVD, and gate with finite grads.
mixer.zero_grad(set_to_none=True)
agent_qs.grad = None
mixed_q = mixer(agent_qs, states, hiddens)
bm_q = mixer.last_counterfactual_bm_q
dvd_q = mixer.last_counterfactual_dvd_q
gate = mixer.last_counterfactual_gate
target = th.randn_like(mixed_q)
bm_error = (bm_q.detach() - target).abs()
dvd_error = (dvd_q.detach() - target).abs()
advantage = (bm_error - dvd_error) / (bm_error + dvd_error + 1e-6)
route_target, _, _, _ = build_counterfactual_route_target(
    bm_error,
    dvd_error,
    args.adaptive_gate_max,
    0.05,
    0.01,
    1e-6,
)
loss = (
    0.2 * (mixed_q - target).pow(2).mean()
    + 0.8 * (bm_q - target).pow(2).mean()
    + 0.1 * (dvd_q - target).pow(2).mean()
    + 0.1 * (gate - route_target.detach()).pow(2).mean()
)
loss.backward()
assert agent_qs.grad is not None
assert mixer.bm_mixer.hyper_w1[0].weight.grad.abs().sum().item() > 0.0
assert mixer.hyper_dvd_w1.weight.grad.abs().sum().item() > 0.0
assert mixer.gat.W.weight.grad.abs().sum().item() > 0.0
assert mixer.gate_net[2].weight.grad.abs().sum().item() > 0.0
for parameter in mixer.parameters():
    if parameter.grad is not None:
        assert th.isfinite(parameter.grad).all()

# Relative wins caused only by tiny absolute errors must not open the route.
tiny_bm_error = th.tensor([[[0.0040]]])
tiny_dvd_error = th.tensor([[[0.0020]]])
tiny_route_target, _, _, tiny_reliable = build_counterfactual_route_target(
    tiny_bm_error,
    tiny_dvd_error,
    args.adaptive_gate_max,
    0.05,
    0.01,
    1e-6,
)
assert tiny_route_target.item() == 0.0
assert not tiny_reliable.item()

# A material DVD improvement must still produce a positive route target.
clear_bm_error = th.tensor([[[0.1000]]])
clear_dvd_error = th.tensor([[[0.0500]]])
clear_route_target, _, _, clear_reliable = build_counterfactual_route_target(
    clear_bm_error,
    clear_dvd_error,
    args.adaptive_gate_max,
    0.05,
    0.01,
    1e-6,
)
assert clear_route_target.item() > 0.0
assert clear_reliable.item()

# No DVD signal means exact BM fallback and a zero effective route.
zero_hiddens = th.zeros_like(hiddens)
zero_output = mixer(agent_qs.detach(), states, zero_hiddens)
assert th.equal(zero_output, mixer.bm_mixer(agent_qs.detach(), states))
assert mixer.last_counterfactual_gate.max().item() == 0.0

print(
    "OK shape=%s gate=%.6f bm_error=%.6f dvd_error=%.6f"
    % (
        tuple(mixed_q.shape),
        gate.mean().item(),
        bm_error.mean().item(),
        dvd_error.mean().item(),
    )
)
