# -*- coding: utf-8 -*-
import sys
from types import SimpleNamespace

sys.path.insert(0, "D:/src")

import torch as th

from modules.mixers.dvd_takeover import TakeoverDVDMixer
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
        takeover_bm_abs=False,
        takeover_warmup_steps=100,
        takeover_ramp_steps=200,
        takeover_max=1.0,
        takeover_reliability_floor=0.5,
        takeover_norm_eps=1e-6,
    )


args = make_args()
batch_size, sequence_length = 4, 6

# Extra DVD modules must not change the RNG seen by downstream RND creation.
th.manual_seed(1234)
baseline_mixer = Mixer(args, abs=False)
expected_next_random = th.randn(16)
th.manual_seed(1234)
mixer = TakeoverDVDMixer(args)
actual_next_random = th.randn(16)
assert th.equal(actual_next_random, expected_next_random)
for baseline_parameter, takeover_parameter in zip(
    baseline_mixer.parameters(), mixer.bm_mixer.parameters()
):
    assert th.equal(baseline_parameter, takeover_parameter)

assert mixer._schedule_scale() == 0.0
mixer.set_t_env(200)
assert mixer._schedule_scale() == 0.5
mixer.set_t_env(300)
assert mixer._schedule_scale() == 1.0

agent_qs = th.randn(batch_size, sequence_length, args.n_agents)
states = th.randn(batch_size, sequence_length, args.state_shape)
hiddens = th.randn(
    batch_size, sequence_length, args.n_agents, args.rnn_hidden_dim
)

# Reliability distinguishes diffuse, agreed, and head-disagreeing attention.
uniform = th.full(
    (2, args.dvd_heads, args.n_agents, args.n_agents), 1.0 / args.n_agents
)
entropy, disagreement, confidence = mixer._attention_reliability(uniform)
assert th.allclose(entropy, th.ones_like(entropy), atol=1e-6)
assert th.allclose(disagreement, th.zeros_like(disagreement), atol=1e-6)
assert th.allclose(confidence, th.zeros_like(confidence), atol=1e-6)

agreed = th.zeros_like(uniform)
agreed[:, :, :, 0] = 1.0
entropy, disagreement, confidence = mixer._attention_reliability(agreed)
assert th.allclose(entropy, th.zeros_like(entropy), atol=1e-6)
assert th.allclose(disagreement, th.zeros_like(disagreement), atol=1e-6)
assert th.allclose(confidence, th.ones_like(confidence), atol=1e-6)

split = th.zeros_like(uniform)
for head in range(args.dvd_heads):
    split[:, head, :, head % args.n_agents] = 1.0
_, disagreement, confidence = mixer._attention_reliability(split)
assert th.all(disagreement > 0.0)
assert th.all(confidence < 1.0)

# alpha=0 is exactly the internal matched BM, including all shared biases/W2.
mixer.set_t_env(0)
takeover_out = mixer(agent_qs, states, hiddens)
bm_out = mixer.bm_mixer(agent_qs, states)
assert th.equal(takeover_out, bm_out)
assert mixer.last_takeover_alpha_mean.item() == 0.0
assert mixer.last_takeover_shift_ratio.item() == 0.0

# Regression: initial recurrent states are exactly zero. They must neither
# enter sqrt(0) backward nor receive an arbitrary normalized DVD direction.
zero_hiddens = th.zeros_like(hiddens)
mixer.set_t_env(300)
mixer.zero_grad(set_to_none=True)
zero_hidden_out = mixer(agent_qs, states, zero_hiddens)
assert th.allclose(zero_hidden_out, bm_out, atol=1e-6)
zero_hidden_out.pow(2).mean().backward()
for parameter in mixer.parameters():
    if parameter.grad is not None:
        assert th.isfinite(parameter.grad).all()
mixer.zero_grad(set_to_none=True)

# After the ramp, DVD has meaningful capacity and RMS-matched W1 weights.
mixer.set_t_env(300)
takeover_out = mixer(agent_qs, states, hiddens)
assert not th.allclose(takeover_out, bm_out)
assert 0.0 < mixer.last_takeover_alpha_mean.item() <= args.takeover_max
assert abs(mixer.last_takeover_dvd_w1_ratio.item() - 1.0) < 2e-3

flat_states = states.reshape(-1, args.state_shape)
flat_hidden = hiddens.reshape(-1, args.n_agents, args.rnn_hidden_dim)
w1, bm_w1, dvd_w1, alpha = mixer._weights(flat_states, flat_hidden)
reduce_dims = (1, 2)
bm_rms = th.sqrt(bm_w1.pow(2).mean(dim=reduce_dims)).clamp_min(
    args.takeover_norm_eps
)
shift_rms = th.sqrt((w1 - bm_w1).pow(2).mean(dim=reduce_dims))

# Triangle inequality with RMS matching gives ||W-W_BM|| <= 2 alpha ||W_BM||.
assert th.all(shift_rms <= 2.0 * alpha * bm_rms + 2e-5)
assert th.allclose(
    w1,
    (1.0 - alpha.view(-1, 1, 1)) * bm_w1
    + alpha.view(-1, 1, 1) * dvd_w1,
    atol=1e-7,
)

loss = takeover_out.pow(2).mean()
loss.backward()
assert mixer.hyper_dvd_w1.weight.grad.abs().sum().item() > 0.0
assert mixer.gat.W.weight.grad.abs().sum().item() > 0.0
assert mixer.bm_mixer.hyper_w1[0].weight.grad.abs().sum().item() > 0.0
for gradient_name, gradient_value in mixer.gradient_stats().items():
    assert gradient_value > 0.0, (gradient_name, gradient_value)

for value in (
    mixer.last_takeover_alpha_mean,
    mixer.last_takeover_dvd_w1_ratio,
    mixer.last_takeover_shift_ratio,
    mixer.last_attention_entropy_mean,
    mixer.last_head_disagreement_mean,
    mixer.last_attention_confidence_mean,
):
    assert th.isfinite(value)

print(
    "OK shape=%s alpha=%.6f dvd_ratio=%.6f shift_ratio=%.6f"
    % (
        tuple(takeover_out.shape),
        mixer.last_takeover_alpha_mean.item(),
        mixer.last_takeover_dvd_w1_ratio.item(),
        mixer.last_takeover_shift_ratio.item(),
    )
)
