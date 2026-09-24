import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from modules.mixers.dvd import MultiHeadGAT
from modules.mixers.nmix import Mixer


class AdaptiveTakeoverDVDMixer(nn.Module):
    """Learn when to use BM or DVD first-layer credit weights.

    BM remains the complete value backbone. DVD supplies an RMS-matched
    alternative W1, and a state-conditioned gate chooses their convex mixture.
    The gate never receives t_env, so there is no hand-authored warm-up, ramp,
    or release schedule. Diagnostic inputs to the gate are detached to prevent
    either expert from manipulating the routing statistics.
    """

    def __init__(self, args):
        super(AdaptiveTakeoverDVDMixer, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.embed_dim = args.mixing_embed_dim
        self.rnn_hidden_dim = args.rnn_hidden_dim
        self.n_heads = getattr(args, "dvd_heads", 8)
        self.gat_dim = getattr(args, "gat_embed_dim", 32)

        # Construct BM first so a matched BM run sees identical initialization.
        # Extra modules are isolated from the global RNG used by downstream RND.
        self.bm_mixer = Mixer(args, abs=getattr(args, "adaptive_bm_abs", False))
        with th.random.fork_rng(devices=[]):
            self.gat = MultiHeadGAT(
                self.rnn_hidden_dim, self.gat_dim, self.n_heads
            )
            self.hyper_dvd_w1 = nn.Linear(
                self.state_dim, self.n_heads * self.embed_dim * self.gat_dim
            )

            # Gate inputs: state, attention entropy, head disagreement,
            # BM/DVD cosine similarity, and their normalized RMS distance.
            gate_hidden_dim = int(
                getattr(args, "adaptive_gate_hidden_dim", self.embed_dim)
            )
            self.gate_net = nn.Sequential(
                nn.Linear(self.state_dim + 4, gate_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(gate_hidden_dim, 1),
            )
            self._init_adaptive_modules()

        self.gate_max = float(getattr(args, "adaptive_gate_max", 1.0))
        self.gate_init = float(getattr(args, "adaptive_gate_init", 0.1))
        self.norm_eps = float(getattr(args, "adaptive_norm_eps", 1e-6))
        self.current_t_env = 0  # Logged for compatibility; never used by gate.

        self._validate_hyperparameters()
        self._reset_gate_output_bias()
        self._clear_stats()

    def _init_adaptive_modules(self):
        nn.init.xavier_uniform_(self.hyper_dvd_w1.weight)
        nn.init.zeros_(self.hyper_dvd_w1.bias)
        nn.init.xavier_uniform_(self.gate_net[0].weight)
        nn.init.zeros_(self.gate_net[0].bias)
        # A zero final weight makes the initial prior state-independent. The
        # first TD update trains the final layer; state conditioning follows.
        nn.init.zeros_(self.gate_net[2].weight)

    def _validate_hyperparameters(self):
        if not 0.0 < self.gate_max <= 1.0:
            raise ValueError("adaptive_gate_max must be in (0, 1]")
        if not 0.0 < self.gate_init < self.gate_max:
            raise ValueError(
                "adaptive_gate_init must be in (0, adaptive_gate_max)"
            )
        if self.norm_eps <= 0.0:
            raise ValueError("adaptive_norm_eps must be positive")

    def _reset_gate_output_bias(self):
        initial_probability = self.gate_init / self.gate_max
        initial_logit = math.log(
            initial_probability / (1.0 - initial_probability)
        )
        nn.init.constant_(self.gate_net[2].bias, initial_logit)

    def _clear_stats(self):
        self.last_adaptive_gate_mean = None
        self.last_adaptive_gate_std = None
        self.last_adaptive_gate_min = None
        self.last_adaptive_gate_max = None
        self.last_adaptive_shift_ratio = None
        self.last_adaptive_dvd_w1_ratio = None
        self.last_adaptive_bm_dvd_cosine = None
        self.last_adaptive_candidate_distance = None
        self.last_attention_entropy_mean = None
        self.last_head_disagreement_mean = None
        self.last_attention_confidence_mean = None
        self.last_bm_w1_rms = None
        self.last_dvd_w1_rms = None

    def set_t_env(self, t_env):
        # Deliberately not consumed by the gate. Keeping this method preserves
        # the learner interface and makes the absence of a schedule explicit.
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
            "adaptive_bm_grad_norm": self._module_grad_norm(self.bm_mixer),
            "adaptive_gat_grad_norm": self._module_grad_norm(self.gat),
            "adaptive_dvd_grad_norm": self._module_grad_norm(
                self.hyper_dvd_w1
            ),
            "adaptive_gate_grad_norm": self._module_grad_norm(self.gate_net),
        }

    def _attention_statistics(self, attention):
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
        min_mean_square = self.norm_eps * self.norm_eps
        bm_mean_square = bm_w1.pow(2).mean(dim=reduce_dims, keepdim=True)
        dvd_raw_mean_square = dvd_raw.pow(2).mean(
            dim=reduce_dims, keepdim=True
        )
        bm_scale = th.sqrt(bm_mean_square.clamp_min(min_mean_square))
        dvd_raw_scale = th.sqrt(
            dvd_raw_mean_square.clamp_min(min_mean_square)
        )
        dvd_scaled = dvd_raw / dvd_raw_scale * bm_scale.detach()

        # Exact BM fallback for all-zero recurrent representations. Besides
        # avoiding an arbitrary normalized direction, this gives alpha=0 and
        # preserves the full BM gradient on those rows.
        has_dvd_signal = (dvd_raw_mean_square > min_mean_square).detach()
        dvd_w1 = th.where(has_dvd_signal, dvd_scaled, bm_w1.detach())

        entropy, disagreement, confidence = self._attention_statistics(attention)
        flat_bm = bm_w1.reshape(bm_w1.size(0), -1)
        flat_dvd = dvd_w1.reshape(dvd_w1.size(0), -1)
        cosine = F.cosine_similarity(flat_bm, flat_dvd, dim=-1, eps=self.norm_eps)
        candidate_mean_square = (dvd_w1 - bm_w1).pow(2).mean(
            dim=reduce_dims
        )
        # Keep an exact zero for identical candidates, but clamp before sqrt
        # on the active branch. This remains finite even if a future change
        # makes the diagnostic differentiable instead of detached.
        candidate_rms = th.where(
            candidate_mean_square > min_mean_square,
            th.sqrt(candidate_mean_square.clamp_min(min_mean_square)),
            th.zeros_like(candidate_mean_square),
        )
        candidate_distance = candidate_rms / th.sqrt(
            bm_w1.pow(2).mean(dim=reduce_dims).clamp_min(min_mean_square)
        )

        gate_features = th.cat(
            [
                flat_states,
                entropy.detach().unsqueeze(-1),
                disagreement.detach().unsqueeze(-1),
                cosine.detach().unsqueeze(-1),
                candidate_distance.detach().unsqueeze(-1),
            ],
            dim=-1,
        )
        learned_gate = self.gate_max * th.sigmoid(self.gate_net(gate_features))
        signal_mask = has_dvd_signal.view(-1, 1).to(learned_gate.dtype)
        alpha = learned_gate * signal_mask
        alpha_3d = alpha.unsqueeze(-1)
        w1 = (1.0 - alpha_3d) * bm_w1 + alpha_3d * dvd_w1

        with th.no_grad():
            bm_rms = th.sqrt(bm_w1.pow(2).mean(dim=reduce_dims))
            dvd_rms = th.sqrt(dvd_w1.pow(2).mean(dim=reduce_dims))
            shift_rms = th.sqrt((w1 - bm_w1).pow(2).mean(dim=reduce_dims))
            alpha_flat = alpha.view(-1)

            self.last_adaptive_gate_mean = alpha_flat.mean()
            self.last_adaptive_gate_std = alpha_flat.std(unbiased=False)
            self.last_adaptive_gate_min = alpha_flat.min()
            self.last_adaptive_gate_max = alpha_flat.max()
            self.last_adaptive_shift_ratio = (
                shift_rms / bm_rms.clamp_min(self.norm_eps)
            ).mean()
            self.last_adaptive_dvd_w1_ratio = (
                dvd_rms / bm_rms.clamp_min(self.norm_eps)
            ).mean()
            self.last_adaptive_bm_dvd_cosine = cosine.mean()
            self.last_adaptive_candidate_distance = candidate_distance.mean()
            self.last_attention_entropy_mean = entropy.mean()
            self.last_head_disagreement_mean = disagreement.mean()
            self.last_attention_confidence_mean = confidence.mean()
            self.last_bm_w1_rms = bm_rms.mean()
            self.last_dvd_w1_rms = dvd_rms.mean()

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
