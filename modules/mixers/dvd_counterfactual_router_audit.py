"""Optional auxiliary-gradient isolation; the existing router is unchanged."""
import torch as th

from modules.mixers.dvd_counterfactual_router import CounterfactualRouterDVDMixer


class CounterfactualRouterAuditMixer(CounterfactualRouterDVDMixer):
    """Keep mixed-value gradients, but detach the auxiliary expert's inputs.

No additional parameters or random initialization are introduced. With the
flag disabled this is exactly the existing router, including its gradients.
The isolated branch recomputes DVD weights from detached hidden states;
detaching only the final DVD weights would also stop training the GAT.
"""

    def __init__(self, args):
        super(CounterfactualRouterAuditMixer, self).__init__(args)
        self.audit_isolate_dvd_aux_hidden = bool(
            getattr(args, "audit_isolate_dvd_aux_hidden", True)
        )

    def _isolated_dvd_weights(self, states, hidden_states):
        # The auxiliary branch must train only GAT and the DVD hypernetwork.
        # The mixed branch has already used the original, attached hidden states.
        with th.no_grad():
            bm_w1 = self.bm_mixer.hyper_w1(states).view(
                -1, self.n_agents, self.embed_dim
            )
            if self.bm_mixer.abs:
                bm_w1 = self.bm_mixer.pos_func(bm_w1)
            min_mean_square = self.norm_eps * self.norm_eps
            bm_scale = th.sqrt(
                bm_w1.pow(2).mean(dim=(1, 2), keepdim=True).clamp_min(min_mean_square)
            )

        graph_features = self.gat(hidden_states.detach())
        dvd_hyper = self.hyper_dvd_w1(states).view(
            -1, self.n_heads, self.embed_dim, self.gat_dim
        )
        dvd_heads = th.matmul(dvd_hyper, graph_features.permute(0, 1, 3, 2))
        dvd_raw = dvd_heads.mean(dim=1).permute(0, 2, 1)
        dvd_mean_square = dvd_raw.pow(2).mean(dim=(1, 2), keepdim=True)
        dvd_scale = th.sqrt(dvd_mean_square.clamp_min(min_mean_square))
        dvd_scaled = dvd_raw / dvd_scale * bm_scale
        return th.where(dvd_mean_square > min_mean_square, dvd_scaled, bm_w1)

    def forward(self, agent_qs, states, hidden_states):
        mixed_q = super(CounterfactualRouterAuditMixer, self).forward(
            agent_qs, states, hidden_states
        )
        if not self.audit_isolate_dvd_aux_hidden or not th.is_grad_enabled():
            return mixed_q

        flat_states = states.detach().reshape(-1, self.state_dim)
        flat_hidden = hidden_states.detach().reshape(
            -1, self.n_agents, self.rnn_hidden_dim
        )
        dvd_w1 = self._isolated_dvd_weights(flat_states, flat_hidden)
        with th.no_grad():
            b1 = self.bm_mixer.hyper_b1(flat_states).view(-1, 1, self.embed_dim)
            w2 = self.bm_mixer.hyper_w2(flat_states).view(-1, self.embed_dim, 1)
            b2 = self.bm_mixer.hyper_b2(flat_states).view(-1, 1, 1)
            if self.bm_mixer.abs:
                w2 = self.bm_mixer.pos_func(w2)
        dvd_q = self._mix_value(
            agent_qs.detach().reshape(-1, 1, self.n_agents), dvd_w1, b1, w2, b2
        )
        self.last_counterfactual_dvd_q = dvd_q.view(
            agent_qs.size(0), agent_qs.size(1), 1
        )
        return mixed_q
