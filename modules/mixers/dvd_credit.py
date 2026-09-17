import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from modules.mixers.dvd import MultiHeadGAT
from modules.mixers.nmix import Mixer


class CreditDVDMixer(nn.Module):
    """Reliability-aware DVD modulation of BM's first-layer credit weights.

    Unlike the output-level residual mixer, this module has a single Q_total
    path. BM generates the backbone weights and DVD only supplies a bounded,
    normalized redistribution of the first-layer agent credits.
    """

    def __init__(self, args):
        super(CreditDVDMixer, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.embed_dim = args.mixing_embed_dim
        self.rnn_hidden_dim = args.rnn_hidden_dim
        self.n_heads = getattr(args, "dvd_heads", 8)
        self.gat_dim = getattr(args, "gat_embed_dim", 32)

        self.bm_mixer = Mixer(args, abs=getattr(args, "credit_bm_abs", False))
        self.gat = MultiHeadGAT(self.rnn_hidden_dim, self.gat_dim, self.n_heads)
        self.hyper_delta_w1 = nn.Linear(
            self.state_dim, self.n_heads * self.embed_dim * self.gat_dim
        )

        gate_hidden_dim = getattr(args, "credit_gate_hidden_dim", self.embed_dim)
        self.gate_net = nn.Sequential(
            nn.Linear(self.state_dim + 2, gate_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(gate_hidden_dim, 1),
        )

        self.gate_max = float(getattr(args, "credit_gate_max", 0.5))
        self.gate_init = float(getattr(args, "credit_gate_init", 0.2))
        self.reliability_floor = float(getattr(args, "credit_reliability_floor", 0.25))
        self.delta_ratio = float(getattr(args, "credit_delta_ratio", 0.25))
        self.center_delta = bool(getattr(args, "credit_delta_center", True))
        self.norm_eps = float(getattr(args, "credit_norm_eps", 1e-6))
        self.gate_warmup_steps = int(getattr(args, "credit_gate_warmup_steps", 1000000))
        self.gate_ramp_steps = int(getattr(args, "credit_gate_ramp_steps", 1000000))
        self.current_t_env = 0

        self._validate_hyperparameters()
        self._init_credit_modules()
        self._clear_stats()

    def _validate_hyperparameters(self):
        if not 0.0 < self.gate_max <= 1.0:
            raise ValueError("credit_gate_max must be in (0, 1]")
        if not 0.0 < self.gate_init < self.gate_max:
            raise ValueError("credit_gate_init must be in (0, credit_gate_max)")
        if not 0.0 <= self.reliability_floor <= 1.0:
            raise ValueError("credit_reliability_floor must be in [0, 1]")
        if self.delta_ratio <= 0.0:
            raise ValueError("credit_delta_ratio must be positive")
        if self.norm_eps <= 0.0:
            raise ValueError("credit_norm_eps must be positive")
        if self.gate_warmup_steps < 0 or self.gate_ramp_steps < 0:
            raise ValueError("credit gate warmup/ramp steps must be non-negative")

    def _init_credit_modules(self):
        nn.init.xavier_uniform_(self.hyper_delta_w1.weight)
        self.hyper_delta_w1.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.gate_net[0].weight)
        self.gate_net[0].bias.data.fill_(0)

        # Make the initial learned gate state-independent. Reliability and the
        # time schedule still suppress uncertain/early DVD corrections.
        initial_probability = self.gate_init / self.gate_max
        gate_bias = math.log(initial_probability / (1.0 - initial_probability))
        self.gate_net[2].weight.data.fill_(0)
        self.gate_net[2].bias.data.fill_(gate_bias)

    def _clear_stats(self):
        self.last_credit_gate_mean = None
        self.last_credit_schedule_scale = None
        self.last_attention_entropy_mean = None
        self.last_head_disagreement_mean = None
        self.last_attention_confidence_mean = None
        self.last_credit_delta_ratio_mean = None
        self.last_bm_w1_rms = None
        self.last_credit_delta_rms = None

    def set_t_env(self, t_env):
        self.current_t_env = t_env

    @staticmethod
    def _module_grad_norm(module):
        squared_norm = None
        for parameter in module.parameters():
            if parameter.grad is None:
                continue
            value = parameter.grad.detach().pow(2).sum()
            squared_norm = value if squared_norm is None else squared_norm + value
        if squared_norm is None:
            return 0.0
        return th.sqrt(squared_norm).item()

    def gradient_stats(self):
        return {
            "credit_bm_grad_norm": self._module_grad_norm(self.bm_mixer),
            "credit_gat_grad_norm": self._module_grad_norm(self.gat),
            "credit_delta_grad_norm": self._module_grad_norm(self.hyper_delta_w1),
            "credit_gate_grad_norm": self._module_grad_norm(self.gate_net),
        }

    def _schedule_scale(self):
        if self.current_t_env < self.gate_warmup_steps:
            return 0.0
        if self.gate_ramp_steps == 0:
            return 1.0
        progress = (
            self.current_t_env - self.gate_warmup_steps
        ) / float(self.gate_ramp_steps)
        return min(max(progress, 0.0), 1.0)

    def _attention_reliability(self, attention):
        eps = self.norm_eps
        log_agents = max(math.log(float(self.n_agents)), eps)

        entropy = -(attention * th.log(attention.clamp_min(eps))).sum(dim=-1)
        entropy = entropy.mean(dim=(1, 2)) / log_agents

        mean_attention = attention.mean(dim=1, keepdim=True)
        disagreement = (
            attention
            * (
                th.log(attention.clamp_min(eps))
                - th.log(mean_attention.clamp_min(eps))
            )
        ).sum(dim=-1)
        disagreement = disagreement.mean(dim=(1, 2)) / log_agents

        entropy = entropy.clamp(0.0, 1.0)
        disagreement = disagreement.clamp(0.0, 1.0)
        confidence = (1.0 - entropy) * (1.0 - disagreement)
        return entropy, disagreement, confidence

    def forward(self, agent_qs, states, hidden_states):
        batch_size, sequence_length, _ = agent_qs.size()
        flat_states = states.reshape(-1, self.state_dim)
        flat_qs = agent_qs.reshape(-1, 1, self.n_agents)
        flat_hidden = hidden_states.reshape(
            -1, self.n_agents, self.rnn_hidden_dim
        )

        # BM backbone. These are the only bias/final-weight/value networks used
        # to produce Q_total, so there is no independently scaled DVD output.
        bm_w1 = self.bm_mixer.hyper_w1(flat_states).view(
            -1, self.n_agents, self.embed_dim
        )
        b1 = self.bm_mixer.hyper_b1(flat_states).view(-1, 1, self.embed_dim)
        w2 = self.bm_mixer.hyper_w2(flat_states).view(-1, self.embed_dim, 1)
        b2 = self.bm_mixer.hyper_b2(flat_states).view(-1, 1, 1)
        if self.bm_mixer.abs:
            bm_w1 = self.bm_mixer.pos_func(bm_w1)
            w2 = self.bm_mixer.pos_func(w2)

        graph_features, attention = self.gat(flat_hidden, return_attention=True)
        entropy, disagreement, confidence = self._attention_reliability(attention)

        delta_hyper = self.hyper_delta_w1(flat_states).view(
            -1, self.n_heads, self.embed_dim, self.gat_dim
        )
        delta_heads = th.matmul(
            delta_hyper, graph_features.permute(0, 1, 3, 2)
        )
        delta_raw = delta_heads.mean(dim=1).permute(0, 2, 1)
        if self.center_delta:
            delta_raw = delta_raw - delta_raw.mean(dim=1, keepdim=True)

        # Match the correction scale to the BM weights, then apply a strict
        # relative bound. Detaching BM's scale prevents correction gradients
        # from changing the backbone merely to enlarge the allowed correction.
        reduce_dims = (1, 2)
        bm_rms = th.sqrt(bm_w1.pow(2).mean(dim=reduce_dims, keepdim=True) + self.norm_eps)
        raw_delta_rms = th.sqrt(
            delta_raw.pow(2).mean(dim=reduce_dims, keepdim=True) + self.norm_eps
        )
        delta_w1 = (
            delta_raw / raw_delta_rms * bm_rms.detach() * self.delta_ratio
        )

        # Reliability statistics are detached in the gate path: the GAT cannot
        # game the gate by becoming artificially sharp or making heads agree.
        gate_features = th.cat(
            [
                flat_states,
                (1.0 - entropy).detach().unsqueeze(-1),
                (1.0 - disagreement).detach().unsqueeze(-1),
            ],
            dim=-1,
        )
        learned_gate = self.gate_max * th.sigmoid(self.gate_net(gate_features))
        reliability = self.reliability_floor + (
            1.0 - self.reliability_floor
        ) * confidence.detach().unsqueeze(-1)
        schedule_scale = self._schedule_scale()
        gate = learned_gate * reliability * schedule_scale
        gate_3d = gate.unsqueeze(-1)

        effective_delta = gate_3d * delta_w1
        w1 = bm_w1 + effective_delta
        hidden = F.elu(th.bmm(flat_qs, w1) + b1)
        q_tot = th.bmm(hidden, w2) + b2

        effective_delta_rms = th.sqrt(effective_delta.pow(2).mean(dim=reduce_dims))
        bm_rms_flat = bm_rms.view(-1)
        self.last_credit_gate_mean = gate.detach().mean()
        self.last_credit_schedule_scale = schedule_scale
        self.last_attention_entropy_mean = entropy.detach().mean()
        self.last_head_disagreement_mean = disagreement.detach().mean()
        self.last_attention_confidence_mean = confidence.detach().mean()
        self.last_credit_delta_ratio_mean = (
            effective_delta_rms.detach() / bm_rms_flat.detach().clamp_min(self.norm_eps)
        ).mean()
        self.last_bm_w1_rms = bm_rms_flat.detach().mean()
        self.last_credit_delta_rms = effective_delta_rms.detach().mean()

        return q_tot.view(batch_size, sequence_length, 1)
