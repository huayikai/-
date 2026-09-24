# -*- coding: utf-8 -*-
import sys
from types import SimpleNamespace

sys.path.insert(0, "D:/src")

import torch as th

from modules.mixers.dvd_credit import CreditDVDMixer


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
        credit_bm_abs=False,
        credit_delta_center=True,
        credit_delta_ratio=0.25,
        credit_norm_eps=1e-6,
        credit_gate_hidden_dim=32,
        credit_gate_init=0.2,
        credit_gate_max=0.5,
        credit_reliability_floor=0.25,
        credit_gate_warmup_steps=100,
        credit_gate_ramp_steps=200,
    )


batch_size, sequence_length = 4, 6
args = make_args()
agent_qs = th.randn(batch_size, sequence_length, args.n_agents)
states = th.randn(batch_size, sequence_length, args.state_shape)
hiddens = th.randn(
    batch_size, sequence_length, args.n_agents, args.rnn_hidden_dim
)

mixer = CreditDVDMixer(args)

# Reliability must distinguish diffuse, agreed, and head-disagreeing attention.
uniform_attention = th.full(
    (2, args.dvd_heads, args.n_agents, args.n_agents), 1.0 / args.n_agents
)
entropy, disagreement, confidence = mixer._attention_reliability(uniform_attention)
assert th.allclose(entropy, th.ones_like(entropy), atol=1e-6)
assert th.allclose(disagreement, th.zeros_like(disagreement), atol=1e-6)
assert th.allclose(confidence, th.zeros_like(confidence), atol=1e-6)

agreed_attention = th.zeros_like(uniform_attention)
agreed_attention[:, :, :, 0] = 1.0
entropy, disagreement, confidence = mixer._attention_reliability(agreed_attention)
assert th.allclose(entropy, th.zeros_like(entropy), atol=1e-6)
assert th.allclose(disagreement, th.zeros_like(disagreement), atol=1e-6)
assert th.allclose(confidence, th.ones_like(confidence), atol=1e-6)

split_attention = th.zeros_like(uniform_attention)
for head in range(args.dvd_heads):
    split_attention[:, head, :, head % args.n_agents] = 1.0
_, disagreement, confidence = mixer._attention_reliability(split_attention)
assert th.all(disagreement > 0.0)
assert th.all(confidence < 1.0)

# During warm-up the new method must exactly equal its BM backbone.
mixer.set_t_env(0)
credit_out = mixer(agent_qs, states, hiddens)
bm_out = mixer.bm_mixer(agent_qs, states)
assert credit_out.shape == (batch_size, sequence_length, 1)
assert th.allclose(credit_out, bm_out, atol=1e-6)
assert mixer.last_credit_gate_mean.item() == 0.0
assert mixer.last_credit_delta_ratio_mean.item() == 0.0

# After the ramp, DVD must affect W1 while respecting the configured norm cap.
mixer.set_t_env(300)
credit_out = mixer(agent_qs, states, hiddens)
assert not th.allclose(credit_out, bm_out)
assert 0.0 < mixer.last_credit_gate_mean.item() <= args.credit_gate_max
assert mixer.last_credit_delta_ratio_mean.item() <= (
    args.credit_gate_max * args.credit_delta_ratio + 1e-4
)

loss = credit_out.pow(2).mean()
loss.backward()
assert mixer.hyper_delta_w1.weight.grad is not None
assert mixer.hyper_delta_w1.weight.grad.abs().sum().item() > 0.0
assert mixer.gat.W.weight.grad is not None
assert mixer.gat.W.weight.grad.abs().sum().item() > 0.0
assert mixer.bm_mixer.hyper_w1[0].weight.grad is not None
gradient_stats = mixer.gradient_stats()
for gradient_name, gradient_value in gradient_stats.items():
    assert gradient_value > 0.0, (gradient_name, gradient_value)

for value in (
    mixer.last_attention_entropy_mean,
    mixer.last_head_disagreement_mean,
    mixer.last_attention_confidence_mean,
    mixer.last_credit_delta_ratio_mean,
):
    assert th.isfinite(value)

print(
    "OK shape=%s gate=%.6f entropy=%.6f disagreement=%.6f delta_ratio=%.6f"
    % (
        tuple(credit_out.shape),
        mixer.last_credit_gate_mean.item(),
        mixer.last_attention_entropy_mean.item(),
        mixer.last_head_disagreement_mean.item(),
        mixer.last_credit_delta_ratio_mean.item(),
    )
)
