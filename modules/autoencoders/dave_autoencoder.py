import numpy as np
import torch as th
import torch.nn as nn


class DAVEStateActionAutoEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.state_dim = int(np.prod(args.state_shape))
        self.joint_action_dim = self.n_agents * self.n_actions

        hidden_dim = getattr(args, "dave_recon_hidden_dim", 128)
        latent_dim = getattr(args, "dave_recon_latent_dim", 64)

        self.encoder = nn.Sequential(
            nn.Linear(self.state_dim + self.joint_action_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.state_dim + self.joint_action_dim),
        )

    def forward(self, states, joint_action_onehot):
        encoded = self.encoder(th.cat([states, joint_action_onehot], dim=-1))
        decoded = self.decoder(encoded)
        state_recon = decoded[:, : self.state_dim]
        action_logits = decoded[:, self.state_dim :].view(-1, self.n_agents, self.n_actions)
        return state_recon, action_logits
