import torch.nn.functional as F

from modules.mixers.dvd_adaptive_takeover import AdaptiveTakeoverDVDMixer


class CounterfactualRouterDVDMixer(AdaptiveTakeoverDVDMixer):
    """BM-protected routing between BM and DVD credit weights.

    The architecture remains the same convex BM/DVD interpolation used by
    AdaptiveTakeoverDVDMixer. In addition to the mixed value, every forward
    pass exposes a pure BM value, a pure DVD value, and the effective gate so
    that the learner can construct counterfactual routing supervision.

    The pure DVD tensor deliberately detaches agent Q-values and the BM
    hypernet outputs. Its auxiliary TD loss therefore trains only the GAT and
    DVD first-layer hypernet instead of pulling the agent or BM backbone toward
    a possibly unreliable DVD solution.
    """

    def __init__(self, args):
        super(CounterfactualRouterDVDMixer, self).__init__(args)
        self.last_counterfactual_bm_q = None
        self.last_counterfactual_dvd_q = None
        self.last_counterfactual_gate = None

    @staticmethod
    def _mix_value(flat_qs, w1, b1, w2, b2):
        hidden = F.elu(flat_qs.bmm(w1) + b1)
        return hidden.bmm(w2) + b2

    def forward(self, agent_qs, states, hidden_states):
        batch_size, sequence_length, _ = agent_qs.size()
        flat_states = states.reshape(-1, self.state_dim)
        flat_qs = agent_qs.reshape(-1, 1, self.n_agents)
        flat_hidden = hidden_states.reshape(
            -1, self.n_agents, self.rnn_hidden_dim
        )

        mixed_w1, bm_w1, dvd_w1, gate = self._weights(
            flat_states, flat_hidden
        )
        b1 = self.bm_mixer.hyper_b1(flat_states).view(
            -1, 1, self.embed_dim
        )
        w2 = self.bm_mixer.hyper_w2(flat_states).view(
            -1, self.embed_dim, 1
        )
        b2 = self.bm_mixer.hyper_b2(flat_states).view(-1, 1, 1)
        if self.bm_mixer.abs:
            w2 = self.bm_mixer.pos_func(w2)

        mixed_q = self._mix_value(flat_qs, mixed_w1, b1, w2, b2)
        bm_q = self._mix_value(flat_qs, bm_w1, b1, w2, b2)

        # Numerically this is the pure DVD counterfactual. The detach pattern
        # restricts its auxiliary loss to the DVD/GAT expert parameters.
        dvd_q = self._mix_value(
            flat_qs.detach(),
            dvd_w1,
            b1.detach(),
            w2.detach(),
            b2.detach(),
        )

        output_shape = (batch_size, sequence_length, 1)
        self.last_counterfactual_bm_q = bm_q.view(*output_shape)
        self.last_counterfactual_dvd_q = dvd_q.view(*output_shape)
        self.last_counterfactual_gate = gate.view(*output_shape)
        return mixed_q.view(*output_shape)
