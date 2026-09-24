# -*- coding: utf-8 -*-
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch as th

from modules.mixers.dvd_adaptive_takeover import AdaptiveTakeoverDVDMixer
from modules.mixers.nmix import Mixer


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
        adaptive_gate_init=0.1,
        adaptive_norm_eps=1e-6,
    )


args = make_args()
batch_size, sequence_length = 4, 6

# Extra DVD and gate modules must not alter BM or downstream RNG state.
th.manual_seed(1234)
baseline_mixer = Mixer(args, abs=False)
expected_next_random = th.randn(16)
th.manual_seed(1234)
mixer = AdaptiveTakeoverDVDMixer(args)
actual_next_random = th.randn(16)
assert th.equal(actual_next_random, expected_next_random)
for baseline_parameter, adaptive_parameter in zip(
    baseline_mixer.parameters(), mixer.bm_mixer.parameters()
):
    assert th.equal(baseline_parameter, adaptive_parameter)

agent_qs = th.randn(batch_size, sequence_length, args.n_agents)
states = th.randn(batch_size, sequence_length, args.state_shape)
hiddens = th.randn(
    batch_size, sequence_length, args.n_agents, args.rnn_hidden_dim
)

# No hand-authored time schedule: t_env cannot change an unchanged forward pass.
mixer.set_t_env(0)
out_at_zero = mixer(agent_qs, states, hiddens)
mixer.set_t_env(9000000)
out_at_nine_million = mixer(agent_qs, states, hiddens)
assert th.equal(out_at_zero, out_at_nine_million)
assert abs(mixer.last_adaptive_gate_mean.item() - args.adaptive_gate_init) < 1e-6

# All-zero recurrent states have no DVD signal and must be exact BM, including
# finite backward gradients and a zero effective gate.
zero_hiddens = th.zeros_like(hiddens)
mixer.zero_grad(set_to_none=True)
zero_hidden_out = mixer(agent_qs, states, zero_hiddens)
bm_out = mixer.bm_mixer(agent_qs, states)
assert th.equal(zero_hidden_out, bm_out)
assert mixer.last_adaptive_gate_mean.item() == 0.0
assert mixer.last_adaptive_candidate_distance.item() == 0.0
zero_hidden_out.pow(2).mean().backward()
for parameter in mixer.parameters():
    if parameter.grad is not None:
        assert th.isfinite(parameter.grad).all()

# With a graph signal, the gate makes an RMS-matched convex interpolation.
mixer.zero_grad(set_to_none=True)
adaptive_out = mixer(agent_qs, states, hiddens)
flat_states = states.reshape(-1, args.state_shape)
flat_hidden = hiddens.reshape(-1, args.n_agents, args.rnn_hidden_dim)
w1, bm_w1, dvd_w1, alpha = mixer._weights(flat_states, flat_hidden)
assert th.all(alpha > 0.0)
assert th.all(alpha <= args.adaptive_gate_max)
assert th.allclose(
    w1,
    (1.0 - alpha.unsqueeze(-1)) * bm_w1 + alpha.unsqueeze(-1) * dvd_w1,
    atol=1e-7,
)
assert abs(mixer.last_adaptive_dvd_w1_ratio.item() - 1.0) < 2e-3

reduce_dims = (1, 2)
bm_rms = th.sqrt(bm_w1.pow(2).mean(dim=reduce_dims)).clamp_min(
    args.adaptive_norm_eps
)
shift_rms = th.sqrt((w1 - bm_w1).pow(2).mean(dim=reduce_dims))
assert th.all(shift_rms <= 2.0 * alpha.view(-1) * bm_rms + 2e-5)

loss = adaptive_out.pow(2).mean()
loss.backward()
assert mixer.hyper_dvd_w1.weight.grad.abs().sum().item() > 0.0
assert mixer.gat.W.weight.grad.abs().sum().item() > 0.0
assert mixer.bm_mixer.hyper_w1[0].weight.grad.abs().sum().item() > 0.0
assert mixer.gate_net[2].weight.grad.abs().sum().item() > 0.0
for value in (
    mixer.last_adaptive_gate_mean,
    mixer.last_adaptive_gate_std,
    mixer.last_adaptive_shift_ratio,
    mixer.last_adaptive_bm_dvd_cosine,
    mixer.last_adaptive_candidate_distance,
    mixer.last_attention_entropy_mean,
    mixer.last_head_disagreement_mean,
):
    assert th.isfinite(value)

print(
    "OK shape=%s gate=%.6f dvd_ratio=%.6f shift=%.6f"
    % (
        tuple(adaptive_out.shape),
        mixer.last_adaptive_gate_mean.item(),
        mixer.last_adaptive_dvd_w1_ratio.item(),
        mixer.last_adaptive_shift_ratio.item(),
    )
)
