import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from modules.mixers.dvd import MultiHeadGAT
from modules.mixers.nmix import Mixer


class TakeoverDVDMixer(nn.Module):
    """Gradually replace BM's first-layer credits with DVD credits.

    BM owns the complete value path. DVD generates an alternative W1 with the
    same RMS scale, and a scheduled reliability coefficient interpolates the
    two. Therefore alpha=0 is exactly BM, while alpha=1 permits full takeover.
    """

    def __init__(self, args):
        super(TakeoverDVDMixer, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.embed_dim = args.mixing_embed_dim
        self.rnn_hidden_dim = args.rnn_hidden_dim
        self.n_heads = getattr(args, "dvd_heads", 8)
        self.gat_dim = getattr(args, "gat_embed_dim", 32)

        # Construct BM first so it has exactly the same initialization as a
        # matched BM run. fork_rng prevents the extra DVD modules from moving
        # the global RNG state and changing the subsequently created RND model.
        self.bm_mixer = Mixer(args, abs=getattr(args, "takeover_bm_abs", False))
        with th.random.fork_rng(devices=[]):
            self.gat = MultiHeadGAT(
                self.rnn_hidden_dim, self.gat_dim, self.n_heads
            )
            self.hyper_dvd_w1 = nn.Linear(
                self.state_dim, self.n_heads * self.embed_dim * self.gat_dim
            )
            nn.init.xavier_uniform_(self.hyper_dvd_w1.weight)
            nn.init.zeros_(self.hyper_dvd_w1.bias)

        self.takeover_max = float(getattr(args, "takeover_max", 1.0))
        self.reliability_floor = float(
            getattr(args, "takeover_reliability_floor", 0.5)
        )
        self.norm_eps = float(getattr(args, "takeover_norm_eps", 1e-6))
        self.warmup_steps = int(getattr(args, "takeover_warmup_steps", 1000000))
        self.ramp_steps = int(getattr(args, "takeover_ramp_steps", 4000000))
        self.current_t_env = 0

        self._validate_hyperparameters()
        self._clear_stats()

    def _validate_hyperparameters(self):
        if not 0.0 < self.takeover_max <= 1.0:
            raise ValueError("takeover_max must be in (0, 1]")
        if not 0.0 <= self.reliability_floor <= 1.0:
            raise ValueError("takeover_reliability_floor must be in [0, 1]")
        if self.norm_eps <= 0.0:
            raise ValueError("takeover_norm_eps must be positive")
        if self.warmup_steps < 0 or self.ramp_steps < 0:
            raise ValueError("takeover warmup/ramp steps must be non-negative")

    def _clear_stats(self):
        self.last_takeover_alpha_mean = None
        self.last_takeover_schedule_scale = None
        self.last_takeover_dvd_w1_ratio = None
        self.last_takeover_shift_ratio = None
        self.last_attention_entropy_mean = None
        self.last_head_disagreement_mean = None
        self.last_attention_confidence_mean = None
        self.last_bm_w1_rms = None
        self.last_dvd_w1_rms = None

    def set_t_env(self, t_env):
        self.current_t_env = t_env

    def _schedule_scale(self):
        if self.current_t_env < self.warmup_steps:
            return 0.0
        if self.ramp_steps == 0:
            return 1.0
        progress = (self.current_t_env - self.warmup_steps) / float(
            self.ramp_steps
        )
        return min(max(progress, 0.0), 1.0)

    def _attention_reliability(self, attention):
        log_agents = max(math.log(float(self.n_agents)), self.norm_eps)

        entropy = -(attention * th.log(attention.clamp_min(self.norm_eps))).sum(
            dim=-1
        )
        entropy = entropy.mean(dim=(1, 2)) / log_agents

        mean_attention = attention.mean(dim=1, keepdim=True)
        disagreement = (
            attention
            * (
                th.log(attention.clamp_min(self.norm_eps))
                - th.log(mean_attention.clamp_min(self.norm_eps))
            )
        ).sum(dim=-1)
        disagreement = disagreement.mean(dim=(1, 2)) / log_agents

        entropy = entropy.clamp(0.0, 1.0)
        disagreement = disagreement.clamp(0.0, 1.0)
        confidence = (1.0 - entropy) * (1.0 - disagreement)
        return entropy, disagreement, confidence

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
            "takeover_bm_grad_norm": self._module_grad_norm(self.bm_mixer),
            "takeover_gat_grad_norm": self._module_grad_norm(self.gat),
            "takeover_dvd_grad_norm": self._module_grad_norm(self.hyper_dvd_w1),
        }

    def _weights(self, flat_states, flat_hidden):
        bm_w1 = self.bm_mixer.hyper_w1(flat_states).view(
            -1, self.n_agents, self.embed_dim
        )
        if self.bm_mixer.abs:
            bm_w1 = self.bm_mixer.pos_func(bm_w1)

        graph_features, attention = self.gat(flat_hidden, return_attention=True)
        dvd_hyper = self.hyper_dvd_w1(flat_states).view(
            -1, self.n_heads, self.embed_dim, self.gat_dim
        )
        dvd_heads = th.matmul(
            dvd_hyper, graph_features.permute(0, 1, 3, 2)
        )
        dvd_raw = dvd_heads.mean(dim=1).permute(0, 2, 1)

        reduce_dims = (1, 2)
        bm_mean_square = bm_w1.pow(2).mean(
            dim=reduce_dims, keepdim=True
        )
        dvd_raw_mean_square = dvd_raw.pow(2).mean(
            dim=reduce_dims, keepdim=True
        )

        # Clamp the mean square *before* sqrt. sqrt(x).clamp_min(eps)
        # still differentiates through sqrt(0), whose backward is infinite.
        # Initial recurrent states are exactly zero, so this distinction is
        # essential even while alpha=0 (0 * inf otherwise becomes NaN).
        min_mean_square = self.norm_eps * self.norm_eps
        bm_scale = th.sqrt(bm_mean_square.clamp_min(min_mean_square))
        dvd_raw_scale = th.sqrt(
            dvd_raw_mean_square.clamp_min(min_mean_square)
        )
        dvd_scaled = dvd_raw / dvd_raw_scale * bm_scale.detach()

        # A zero graph representation contains no DVD credit information and
        # has no meaningful direction to RMS-normalize. Fall back to BM for
        # those rows; the detached mask also gives them a zero DVD gradient.
        has_dvd_signal = (dvd_raw_mean_square > min_mean_square).detach()
        dvd_w1 = th.where(has_dvd_signal, dvd_scaled, bm_w1.detach())

        entropy, disagreement, confidence = self._attention_reliability(attention)
        reliability = self.reliability_floor + (
            1.0 - self.reliability_floor
        ) * confidence.detach()
        schedule_scale = self._schedule_scale()
        alpha = self.takeover_max * schedule_scale * reliability
        alpha_3d = alpha.view(-1, 1, 1)
        w1 = (1.0 - alpha_3d) * bm_w1 + alpha_3d * dvd_w1

        # Metrics are detached before sqrt so exact zero shifts can be logged
        # without adding epsilon or creating another backward path at zero.
        bm_rms_flat = th.sqrt(
            bm_w1.detach().pow(2).mean(dim=reduce_dims)
        )
        dvd_rms = th.sqrt(
            dvd_w1.detach().pow(2).mean(dim=reduce_dims)
        )
        shift_rms = th.sqrt(
            (w1.detach() - bm_w1.detach()).pow(2).mean(dim=reduce_dims)
        )

        self.last_takeover_alpha_mean = alpha.detach().mean()
        self.last_takeover_schedule_scale = schedule_scale
        self.last_takeover_dvd_w1_ratio = (
            dvd_rms.detach() / bm_rms_flat.detach().clamp_min(self.norm_eps)
        ).mean()
        self.last_takeover_shift_ratio = (
            shift_rms.detach() / bm_rms_flat.detach().clamp_min(self.norm_eps)
        ).mean()
        self.last_attention_entropy_mean = entropy.detach().mean()
        self.last_head_disagreement_mean = disagreement.detach().mean()
        self.last_attention_confidence_mean = confidence.detach().mean()
        self.last_bm_w1_rms = bm_rms_flat.detach().mean()
        self.last_dvd_w1_rms = dvd_rms.detach().mean()
        return w1, bm_w1, dvd_w1, alpha

    def forward(self, agent_qs, states, hidden_states):
        batch_size, sequence_length, _ = agent_qs.size()
        flat_states = states.reshape(-1, self.state_dim)
        flat_qs = agent_qs.reshape(-1, 1, self.n_agents)
        flat_hidden = hidden_states.reshape(
            -1, self.n_agents, self.rnn_hidden_dim
        )

        w1, _, _, _ = self._weights(flat_states, flat_hidden)
        b1 = self.bm_mixer.hyper_b1(flat_states).view(-1, 1, self.embed_dim)
        w2 = self.bm_mixer.hyper_w2(flat_states).view(-1, self.embed_dim, 1)
        b2 = self.bm_mixer.hyper_b2(flat_states).view(-1, 1, 1)
        if self.bm_mixer.abs:
            w2 = self.bm_mixer.pos_func(w2)

        hidden = F.elu(th.bmm(flat_qs, w1) + b1)
        q_tot = th.bmm(hidden, w2) + b2
        return q_tot.view(batch_size, sequence_length, 1)
