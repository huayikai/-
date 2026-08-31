import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm

from utils.th_utils import orthogonal_init_


class DAVERNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super().__init__()
        self.args = args
        self.hidden_dim = args.rnn_hidden_dim

        self.alter_fc1 = nn.Linear(input_shape, self.hidden_dim)
        self.alter_rnn = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        self.alter_head = nn.Linear(self.hidden_dim, args.n_actions)

        self.ego_fc1 = nn.Linear(input_shape, self.hidden_dim)
        self.ego_rnn = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        self.ego_head = nn.Linear(self.hidden_dim, args.n_actions)

        self.use_layer_norm = getattr(args, "use_layer_norm", False)
        if self.use_layer_norm:
            self.alter_layer_norm = LayerNorm(self.hidden_dim)
            self.ego_layer_norm = LayerNorm(self.hidden_dim)

        if getattr(args, "use_orthogonal", False):
            orthogonal_init_(self.alter_fc1)
            orthogonal_init_(self.alter_head, gain=args.gain)
            orthogonal_init_(self.ego_fc1)
            orthogonal_init_(self.ego_head, gain=args.gain)

    def init_hidden(self):
        hidden = self.alter_fc1.weight.new_zeros(1, self.hidden_dim)
        return hidden, hidden.clone()

    def alter_parameters(self):
        return list(self.alter_fc1.parameters()) + list(self.alter_rnn.parameters()) + list(self.alter_head.parameters())

    def ego_parameters(self):
        return list(self.ego_fc1.parameters()) + list(self.ego_rnn.parameters()) + list(self.ego_head.parameters())

    def forward(self, inputs, hidden_state):
        alter_hidden, ego_hidden = hidden_state
        batch_size, n_agents, input_dim = inputs.size()

        flat_inputs = inputs.reshape(-1, input_dim)
        alter_h_in = alter_hidden.reshape(-1, self.hidden_dim)
        ego_h_in = ego_hidden.reshape(-1, self.hidden_dim)

        alter_x = F.relu(self.alter_fc1(flat_inputs), inplace=True)
        ego_x = F.relu(self.ego_fc1(flat_inputs), inplace=True)

        alter_h = self.alter_rnn(alter_x, alter_h_in)
        ego_h = self.ego_rnn(ego_x, ego_h_in)

        alter_feat = self.alter_layer_norm(alter_h) if self.use_layer_norm else alter_h
        ego_feat = self.ego_layer_norm(ego_h) if self.use_layer_norm else ego_h

        alter_q = self.alter_head(alter_feat)
        ego_logits = self.ego_head(ego_feat)

        return (
            alter_q.view(batch_size, n_agents, -1),
            ego_logits.view(batch_size, n_agents, -1),
            (
                alter_h.view(batch_size, n_agents, -1),
                ego_h.view(batch_size, n_agents, -1),
            ),
        )
