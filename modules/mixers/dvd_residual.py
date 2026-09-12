import torch as th
import torch.nn as nn
import numpy as np

from modules.mixers.dvd import DVDMixer
from modules.mixers.nmix import Mixer


class ResidualDVDMixer(nn.Module):
    """BM backbone with a gated DVD residual.

    The motivation is to avoid forcing every state through the DVD deconfounding
    path. The BM branch provides a non-monotonic baseline, while the DVD branch
    can add a bounded state-dependent correction when it is useful.
    """

    def __init__(self, args):
        super(ResidualDVDMixer, self).__init__()
        self.args = args
        self.state_dim = int(np.prod(args.state_shape))

        self.bm_mixer = Mixer(args, abs=getattr(args, "residual_bm_abs", False))
        self.dvd_mixer = DVDMixer(args)

        gate_hidden_dim = getattr(args, "residual_gate_hidden_dim", args.mixing_embed_dim)
        gate_init = getattr(args, "residual_gate_init", 0.05)
        gate_init = min(max(gate_init, 1e-4), 1.0 - 1e-4)
        gate_bias = np.log(gate_init / (1.0 - gate_init))

        self.gate_net = nn.Sequential(
            nn.Linear(self.state_dim, gate_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(gate_hidden_dim, 1),
        )
        nn.init.xavier_uniform_(self.gate_net[0].weight)
        self.gate_net[0].bias.data.fill_(0)
        self.gate_net[2].weight.data.fill_(0)
        self.gate_net[2].bias.data.fill_(gate_bias)

        self.residual_scale = getattr(args, "residual_dvd_scale", 1.0)
        self.residual_mode = getattr(args, "residual_dvd_mode", "delta")
        self.gate_warmup_steps = getattr(args, "residual_gate_warmup_steps", 0)
        self.gate_ramp_steps = getattr(args, "residual_gate_ramp_steps", 1)
        self.current_t_env = 0
        self.last_gate_mean = None
        self.last_bm_q_mean = None
        self.last_dvd_q_mean = None
        self.last_residual_mean = None

    def set_t_env(self, t_env):
        self.current_t_env = t_env

    def forward(self, agent_qs, states, hidden_states):
        q_bm = self.bm_mixer(agent_qs, states)
        q_dvd = self.dvd_mixer(agent_qs, states, hidden_states)

        flat_states = states.reshape(-1, self.state_dim)
        gate = th.sigmoid(self.gate_net(flat_states)).view(states.size(0), -1, 1)
        if self.current_t_env < self.gate_warmup_steps:
            gate = gate * 0.0
        elif self.gate_ramp_steps > 0:
            progress = (self.current_t_env - self.gate_warmup_steps) / float(self.gate_ramp_steps)
            gate = gate * min(max(progress, 0.0), 1.0)

        if self.residual_mode == "delta":
            residual = q_dvd - q_bm.detach()
        else:
            residual = q_dvd

        q_tot = q_bm + gate * self.residual_scale * residual

        self.last_gate_mean = gate.detach().mean()
        self.last_bm_q_mean = q_bm.detach().mean()
        self.last_dvd_q_mean = q_dvd.detach().mean()
        self.last_residual_mean = residual.detach().mean()

        return q_tot
