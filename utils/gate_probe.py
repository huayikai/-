"""Sparse, RNG-free pre-update snapshots for offline gate fitting checks."""
from pathlib import Path
import numpy as np


class GateProbeRecorder:
    def __init__(self, args):
        self.enabled = getattr(args, 'audit_save_gate_probe', False)
        if not isinstance(self.enabled, bool):
            raise ValueError('audit_save_gate_probe must be boolean')
        self.steps = list(getattr(args, 'audit_gate_probe_steps', [1000000, 2000000, 3000000, 4000000, 5000000]))
        self.batches = int(getattr(args, 'audit_gate_probe_batches', 3))
        if self.enabled and (not self.steps or self.steps != sorted(set(self.steps))
                             or min(self.steps) <= 0 or self.batches < 1):
            raise ValueError('gate probe requires increasing positive steps and positive batches')
        self.stage, self.count = 0, 0
        self.args = args

    def due(self, t_env):
        return self.enabled and self.stage < len(self.steps) and t_env >= self.steps[self.stage]

    def save(self, t_env, mixer, chosen_q, states, hidden, mask, targets, route_target, mixed_q):
        # All data come from this update before backward/Adam. No extra agent
        # forward, replay sampling, or global random draws are performed here.
        if not self.due(t_env):
            return None
        import torch as th
        with th.no_grad():
            valid = mask.reshape(-1).bool()
            if not valid.any():
                return None
            # Frozen _weights is recomputed only to obtain sufficient statistics.
            # It contains no dropout/BN/random operations. Restore diagnostic
            # attributes so the observer cannot overwrite training log values.
            saved = {k: v for k, v in mixer.__dict__.items() if k.startswith('last_')}
            try:
                s = states.detach().reshape(-1, mixer.state_dim)
                h = hidden.detach().reshape(-1, mixer.n_agents, mixer.rnn_hidden_dim)
                q = chosen_q.detach().reshape(-1, 1, mixer.n_agents)
                _, wb, wd, _ = mixer._weights(s, h)
                b = mixer.bm_mixer.hyper_b1(s).reshape(-1, 1, mixer.embed_dim)
                w2 = mixer.bm_mixer.hyper_w2(s).reshape(-1, mixer.embed_dim)
                if mixer.bm_mixer.abs:
                    w2 = mixer.bm_mixer.pos_func(w2)
                b2 = mixer.bm_mixer.hyper_b2(s).reshape(-1)
                pb = (q.bmm(wb) + b).reshape(-1, mixer.embed_dim)
                pd = (q.bmm(wd) + b).reshape_as(pb)
                eligible = mixer.last_adaptive_signal_mask.reshape(-1).bool()
            finally:
                for key, value in saved.items():
                    setattr(mixer, key, value)
            tensors = {
                'pre_bm': pb, 'pre_dvd': pd, 'w2': w2, 'b2': b2,
                'target': targets.detach().reshape(-1),
                'gate': mixer.last_counterfactual_gate.detach().reshape(-1),
                'applied_gate': mixer.last_audit_routing_gate.detach().reshape(-1),
                'route_target': route_target.detach().reshape(-1),
                'bm_q': mixer.last_counterfactual_bm_q.detach().reshape(-1),
                'dvd_q': mixer.last_counterfactual_dvd_q.detach().reshape(-1),
                'mixed_q': mixed_q.detach().reshape(-1), 'eligible': eligible,
            }
            arrays = {k: v[valid].cpu().numpy().copy() for k, v in tensors.items()}
            arrays['flat_indices'] = valid.nonzero(as_tuple=False).reshape(-1).cpu().numpy()
            arrays['batch_shape'] = np.asarray(mask.shape[:2], dtype=np.int64)
            arrays['t_env'] = np.asarray(t_env)
            arrays['planned_step'] = np.asarray(self.steps[self.stage])
            arrays['gate_max'] = np.asarray(mixer.gate_max)
            arrays['seed'] = np.asarray(getattr(self.args, 'seed', 0))
            arrays['schema_version'] = np.asarray(1)
            arrays['name'] = np.asarray(self.args.name)
        directory = Path(self.args.local_results_path) / 'gate_probes' / self.args.name
        directory.mkdir(parents=True, exist_ok=True)
        # Exclusive creation preserves earlier files on accidental re-launch.
        # A collision fails explicitly rather than mixing separate trajectories.
        path = directory / ('stage_%d_batch_%d_t_%d.npz' % (self.steps[self.stage], self.count, t_env))
        with path.open('xb') as stream:
            np.savez_compressed(stream, **arrays)
        self.count += 1
        if self.count == self.batches:
            self.stage += 1
            self.count = 0
        return path
