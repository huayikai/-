import numpy as np
import random
import gfootball.env as football_env
from gfootball.env import observation_preprocessing
from ..multiagentenv import MultiAgentEnv
import gym
import torch as th


class GoogleFootballEnv(MultiAgentEnv):

    def __init__(
        self,
        dense_reward=False,
        write_full_episode_dumps=False,
        write_goal_dumps=False,
        dump_freq=0,
        render=False,
        num_agents=4,
        time_limit=200,
        time_step=0,
        map_name='academy_counterattack_hard',
        stacked=False,
        representation="simple115",
        rewards='scoring,checkpoints',
        logdir='football_dumps',
        write_video=False,
        number_of_right_players_agent_controls=0,
        seed=0,
        team_reward='sum',
    ):
        self.dense_reward = dense_reward
        self.write_full_episode_dumps = write_full_episode_dumps
        self.write_goal_dumps = write_goal_dumps
        self.dump_freq = dump_freq
        self._render = render
        self.n_agents = num_agents
        self.episode_limit = time_limit
        self.time_step = time_step
        self.env_name = map_name
        self.stacked = stacked
        self.representation = representation
        self.rewards = rewards
        self.logdir = logdir
        self.write_video = write_video
        self.number_of_right_players_agent_controls = number_of_right_players_agent_controls
        self._seed = int(seed)
        # GRF's engine seed is a configuration option; Gym's inherited seed()
        # alone does not seed the football simulator. Scenario Python code also
        # uses process-local random generators (each runner owns a process).
        random.seed(self._seed)
        np.random.seed(self._seed)
        if team_reward not in ('mean', 'sum'):
            raise ValueError('team_reward must be mean or sum')
        if num_agents < 1 or time_limit < 1:
            raise ValueError('num_agents and time_limit must be positive')
        if number_of_right_players_agent_controls:
            raise ValueError('cooperative wrapper only controls left-side players')
        if representation not in ('simple115', 'simple115v2') or stacked:
            raise ValueError('vector agents require unstacked simple115/simple115v2')
        self.team_reward = team_reward
        self._score_reward = 0.0

        self.env = football_env.create_environment(
            write_full_episode_dumps=self.write_full_episode_dumps,
            write_goal_dumps=self.write_goal_dumps,
            env_name=self.env_name,
            stacked=self.stacked,
            representation=self.representation,
            rewards=self.rewards,
            logdir=self.logdir,
            render=self._render,
            write_video=self.write_video,
            dump_frequency=self.dump_freq,
            number_of_left_players_agent_controls=self.n_agents,
            number_of_right_players_agent_controls=self.number_of_right_players_agent_controls,
            channel_dimensions=(observation_preprocessing.SMM_WIDTH, observation_preprocessing.SMM_HEIGHT),
            other_config_options={'game_engine_random_seed': self._seed})

        obs_space_low = self.env.observation_space.low.reshape(self.n_agents, -1)[0]
        obs_space_high = self.env.observation_space.high.reshape(self.n_agents, -1)[0]

        action_counts = np.asarray(getattr(self.env.action_space, 'nvec',
                                         [getattr(self.env.action_space, 'n', 0)]))
        if action_counts.size not in (1, self.n_agents) or not np.all(action_counts == action_counts[0]):
            raise ValueError('expected homogeneous discrete football action spaces')
        self.action_space = [gym.spaces.Discrete(int(action_counts[0]))
                             for _ in range(self.n_agents)]
        self.observation_space = [
            gym.spaces.Box(low=obs_space_low, high=obs_space_high, dtype=self.env.observation_space.dtype) for _ in range(self.n_agents)
        ]

        self.n_actions = self.action_space[0].n
        self.obs = None

    def step(self, _actions):
        """Returns reward, terminated, info."""
        if th.is_tensor(_actions):
            actions = _actions.cpu().numpy()
        else:
            actions = np.asarray(_actions)
        actions = np.asarray(actions).reshape(-1)
        if actions.size != self.n_agents or np.any(actions != actions.astype(np.int64)):
            raise ValueError('one integer action is required per agent')
        if np.any(actions < 0) or np.any(actions >= self.n_actions):
            raise ValueError('football action outside action space')
        self.time_step += 1
        obs, rewards, done, infos = self.env.step(actions.tolist())

        self.obs = np.asarray(obs, dtype=np.float32).reshape(self.n_agents, -1)
        # Checkpoint rewards can be positive without a goal. Only the engine's
        # unshaped scoring signal defines Academy success.
        if 'score_reward' not in infos:
            raise RuntimeError('gfootball must provide info[score_reward]')
        self._score_reward += float(infos['score_reward'])
        natural_done = bool(done)

        if self.time_step >= self.episode_limit:
            done = True

        info = dict(infos)
        info['episode_limit'] = bool(done and not natural_done)
        info['battle_won'] = bool(done and self._score_reward > 0)
        info['score_reward'] = self._score_reward
        # GRF duplicates team reward for controlled players. Mean keeps its
        # scale independent of the number of agents; sum retains legacy runs.
        rewards = np.asarray(rewards, dtype=np.float32)
        reward = rewards.mean() if self.team_reward == 'mean' else rewards.sum()
        return float(reward), bool(done), info

    def get_obs(self):
        """Returns all agent observations in a list."""
        return self.obs.reshape(self.n_agents, -1)

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id."""
        return self.obs[agent_id].reshape(-1)

    def get_obs_size(self):
        """Returns the size of the observation."""
        return int(np.prod(self.env.observation_space.shape) // self.n_agents)

    def get_global_state(self):
        return self.obs.flatten()

    def get_state(self):
        """Returns the global state."""
        return self.get_global_state()

    def get_state_size(self):
        """Returns the size of the global state."""
        return self.get_obs_size() * self.n_agents

    def get_avail_actions(self):
        """Returns the available actions of all agents in a list."""
        return [[1 for _ in range(self.n_actions)] for agent_id in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id."""
        return self.get_avail_actions()[agent_id]

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take."""
        return self.action_space[0].n

    def reset(self):
        """Returns initial observations and states."""
        self.time_step = 0
        self._score_reward = 0.0
        self.obs = np.asarray(self.env.reset(), dtype=np.float32).reshape(self.n_agents, -1)

        return self.get_obs(), self.get_global_state()

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

    def seed(self, seed=None):
        if seed is not None:
            self._seed = int(seed)
        random.seed(self._seed)
        np.random.seed(self._seed)
        self.env.unwrapped._config['game_engine_random_seed'] = self._seed
        return [self._seed]

    def save_replay(self):
        """Save a replay."""
        pass

    def get_stats(self):
        return  {}
