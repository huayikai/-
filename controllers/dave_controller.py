import torch as th

from components.action_selectors import REGISTRY as action_REGISTRY
from modules.agents import REGISTRY as agent_REGISTRY


class DAVEMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        self.agent_output_type = args.agent_output_type

        input_shape = self._get_input_shape(scheme)
        self.agent = agent_REGISTRY[args.agent](input_shape, args)
        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.alter_hidden_states = None
        self.ego_hidden_states = None
        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        ego_probs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        return self.action_selector.select_action(ego_probs[bs], avail_actions[bs], t_env, test_mode=test_mode)

    def forward(self, ep_batch, t, test_mode=False, return_info=False):
        if test_mode:
            self.agent.eval()

        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]

        alter_qs, ego_logits, hidden_states = self.agent(
            agent_inputs,
            (self.alter_hidden_states, self.ego_hidden_states),
        )
        self.alter_hidden_states, self.ego_hidden_states = hidden_states
        self.hidden_states = self.alter_hidden_states

        masked_logits = ego_logits.clone()
        masked_logits[avail_actions == 0] = -1e10
        ego_probs = th.softmax(masked_logits, dim=-1)

        if return_info:
            return {
                "alter_qs": alter_qs,
                "ego_logits": ego_logits,
                "ego_probs": ego_probs,
                "alter_hidden": self.alter_hidden_states,
                "ego_hidden": self.ego_hidden_states,
            }

        return ego_probs

    def init_hidden(self, batch_size):
        alter_hidden, ego_hidden = self.agent.init_hidden()
        self.alter_hidden_states = alter_hidden.unsqueeze(0).expand(batch_size, self.n_agents, -1)
        self.ego_hidden_states = ego_hidden.unsqueeze(0).expand(batch_size, self.n_agents, -1)
        self.hidden_states = self.alter_hidden_states

    def parameters(self):
        return self.agent.parameters()

    def alter_parameters(self):
        return self.agent.alter_parameters()

    def ego_parameters(self):
        return self.agent.ego_parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_inputs(self, batch, t):
        bs = batch.batch_size
        inputs = [batch["obs"][:, t]]

        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t - 1])

        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        return th.cat([x.reshape(bs, self.n_agents, -1) for x in inputs], dim=-1)

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]

        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]

        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape
