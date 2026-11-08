"""Optional hidden-gradient controls; the original router is unchanged."""
import torch as th

from modules.mixers.dvd_counterfactual_router import CounterfactualRouterDVDMixer


class CounterfactualRouterAuditMixer(CounterfactualRouterDVDMixer):
    """Control mixed and auxiliary hidden paths independently.

    Detaching mixed hidden inputs leaves agent Q and GAT/DVD parameter
    gradients active. It introduces no parameters or random initialization.
    With the mixed flag off, the previous audit behavior is preserved.
    """

    def __init__(self, args):
        super(CounterfactualRouterAuditMixer, self).__init__(args)
        self.audit_isolate_dvd_aux_hidden = bool(
            getattr(args, "audit_isolate_dvd_aux_hidden", True)
        )
        self.audit_detach_mixed_hidden = bool(
            getattr(args, "audit_detach_mixed_hidden", False)
        )
        self.audit_shuffle_gate = getattr(args, "audit_shuffle_gate", False)
        if not isinstance(self.audit_shuffle_gate, bool):
            raise ValueError("audit_shuffle_gate must be a boolean")
        self.last_audit_routing_gate = None
        self.last_audit_shuffle_stats = None
        self.gate_shuffle_generator = None
        if self.audit_shuffle_gate:
            # A CPU generator also works for CUDA tensors and leaves the
            # sampling, exploration, and RND random streams alone.
            self.gate_shuffle_generator = th.Generator(device="cpu")
            self.gate_shuffle_generator.manual_seed(
                int(getattr(args, "seed", 0)) ^ 0x5A17
            )

    def __getstate__(self):
        # Older PyTorch versions cannot pickle/deepcopy a Generator. Copy its
        # CPU RNG state instead, including when the learner builds its target.
        parent_getstate = getattr(
            super(CounterfactualRouterAuditMixer, self), "__getstate__", None
        )
        state = (parent_getstate() if parent_getstate is not None
                 else self.__dict__).copy()
        generator = state.get("gate_shuffle_generator")
        if generator is not None:
            state["_gate_shuffle_rng_state"] = generator.get_state()
            state["gate_shuffle_generator"] = None
        return state

    def __setstate__(self, state):
        state = state.copy()
        rng_state = state.pop("_gate_shuffle_rng_state", None)
        super(CounterfactualRouterAuditMixer, self).__setstate__(state)
        if rng_state is not None:
            self.gate_shuffle_generator = th.Generator(device="cpu")
            self.gate_shuffle_generator.set_state(rng_state)

    def _routing_gate(self, gate, valid_mask):
        routing_gate = gate.detach()
        self.last_audit_shuffle_stats = None
        if self.audit_shuffle_gate and self.training and th.is_grad_enabled():
            if valid_mask is None or valid_mask.numel() != gate.numel():
                raise ValueError("gate shuffling requires the aligned training mask")
            valid = valid_mask.detach().reshape(-1).bool()
            eligible = valid & self.last_adaptive_signal_mask.reshape(-1)
            indices = eligible.nonzero(as_tuple=False).flatten()
            count = indices.numel()
            routing_gate = routing_gate.clone()
            if count > 1:
                order = th.randperm(
                    count, generator=self.gate_shuffle_generator, device="cpu"
                )
                order = order.to(indices.device)
                routing_gate[indices] = gate.detach()[indices[order]]

            # Measure the actual assignment without changing the original
            # differentiable gate used by routing supervision in the learner.
            zero = routing_gate.new_zeros(())
            stats = {
                "valid_count": valid.sum(),
                "eligible_count": zero.new_tensor(count),
                "fallback_count": (valid & ~eligible).sum(),
                "changed_fraction": zero,
                "mean_abs_change": zero,
                "mean_delta": zero,
                "std_delta": zero,
            }
            if count:
                before = gate.detach()[indices].flatten()
                after = routing_gate[indices].flatten()
                stats.update({
                    "changed_fraction": (before != after).float().mean(),
                    "mean_abs_change": (after - before).abs().mean(),
                    "mean_delta": after.mean() - before.mean(),
                    "std_delta": after.std(unbiased=False) - before.std(unbiased=False),
                })
            self.last_audit_shuffle_stats = stats
        self.last_audit_routing_gate = routing_gate
        return routing_gate

    def _auxiliary_dvd_weights(self, states, hidden_states):
        # Caller controls hidden isolation; auxiliary BM/Q inputs stay detached.
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

        graph_features, _ = self._graph_features(hidden_states)
        dvd_hyper = self.hyper_dvd_w1(states).view(
            -1, self.n_heads, self.embed_dim, self.gat_dim
        )
        dvd_heads = th.matmul(dvd_hyper, graph_features.permute(0, 1, 3, 2))
        dvd_raw = dvd_heads.mean(dim=1).permute(0, 2, 1)
        dvd_mean_square = dvd_raw.pow(2).mean(dim=(1, 2), keepdim=True)
        dvd_scale = th.sqrt(dvd_mean_square.clamp_min(min_mean_square))
        dvd_scaled = (
            dvd_raw / dvd_scale * bm_scale
            if self.normalize_dvd_weights else dvd_raw
        )
        return th.where(dvd_mean_square > min_mean_square, dvd_scaled, bm_w1)

    def forward(self, agent_qs, states, hidden_states, valid_mask=None):
        mixed_hidden = (
            hidden_states.detach() if self.audit_detach_mixed_hidden else hidden_states
        )
        mixed_q = super(CounterfactualRouterAuditMixer, self).forward(
            agent_qs, states, mixed_hidden, valid_mask=valid_mask
        )
        if not th.is_grad_enabled():
            return mixed_q
        if not self.audit_isolate_dvd_aux_hidden and not self.audit_detach_mixed_hidden:
            return mixed_q

        # Recompute the auxiliary expert from its own hidden input. In
        # particular, mixed detachment must not implicitly isolate auxiliary
        # hidden gradients when the auxiliary switch is disabled.
        flat_states = states.detach().reshape(-1, self.state_dim)
        aux_hidden = (
            hidden_states.detach() if self.audit_isolate_dvd_aux_hidden else hidden_states
        )
        flat_hidden = aux_hidden.reshape(-1, self.n_agents, self.rnn_hidden_dim)
        dvd_w1 = self._auxiliary_dvd_weights(flat_states, flat_hidden)
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
