"""Direct trajectory weights on the matched BM output backbone.

This control has no gate, RMS rescaling, BM fallback, or auxiliary output.
The temporary reference in __init__ only reproduces BACM's initialization;
its gate and unused BM W1 are not retained in this model or its optimizer.
"""
import copy

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from modules.mixers.dvd_adaptive_takeover import AdaptiveTakeoverDVDMixer


class DirectDVDMixer(nn.Module):
    def __init__(self, args):
        super(DirectDVDMixer, self).__init__()
        reference_args = copy.copy(args)
        # Match the existing BACM protocol's initialization even though no
        # gate is kept or evaluated by this control.
        reference_args.adaptive_gate_init = 0.05
        reference_args.adaptive_bm_abs = False
        reference_args.adaptive_graph_mode = "learned"
        reference = AdaptiveTakeoverDVDMixer(reference_args)
        self.n_agents = reference.n_agents
        self.state_dim = reference.state_dim
        self.embed_dim = reference.embed_dim
        self.rnn_hidden_dim = reference.rnn_hidden_dim
        self.n_heads = reference.n_heads
        self.gat_dim = reference.gat_dim
        self.detach_hidden = bool(getattr(args, "audit_detach_mixed_hidden", False))

        self.hyper_b1 = reference.bm_mixer.hyper_b1
        self.hyper_w2 = reference.bm_mixer.hyper_w2
        self.hyper_b2 = reference.bm_mixer.hyper_b2
        self.gat = reference.gat
        self.hyper_dvd_w1 = reference.hyper_dvd_w1

    def forward(self, agent_qs, states, hidden_states):
        batch_size, sequence_length, _ = agent_qs.size()
        flat_states = states.reshape(-1, self.state_dim)
        flat_qs = agent_qs.reshape(-1, 1, self.n_agents)
        if self.detach_hidden:
            hidden_states = hidden_states.detach()
        flat_hidden = hidden_states.reshape(-1, self.n_agents, self.rnn_hidden_dim)

        graph_features = self.gat(flat_hidden)
        dvd_hyper = self.hyper_dvd_w1(flat_states).view(
            -1, self.n_heads, self.embed_dim, self.gat_dim
        )
        dvd_heads = th.matmul(dvd_hyper, graph_features.permute(0, 1, 3, 2))
        w1 = dvd_heads.mean(dim=1).permute(0, 2, 1)
        b1 = self.hyper_b1(flat_states).view(-1, 1, self.embed_dim)
        w2 = self.hyper_w2(flat_states).view(-1, self.embed_dim, 1)
        b2 = self.hyper_b2(flat_states).view(-1, 1, 1)
        value = F.elu(th.bmm(flat_qs, w1) + b1).bmm(w2) + b2
        return value.view(batch_size, sequence_length, 1)
