# src — Codebase Digest

> Auto-generated context document. Load this instead of re-reading source files.
> Generated: 2026-07-31  |  Source: D:\src

## Overview

PyMARL-based multi-agent reinforcement learning (MARL) framework supporting value decomposition methods (QMIX, VDN, QPLEX, DVD, Kaleidoscope), policy gradient methods (COMA, PPO, LICA, DOP), and hybrid approaches (RODE, DAVE, FMAC). Primary environments are StarCraft II micromanagement (SMAC), Google Football, Stag Hunt, and matrix games. Entry point is `main.py` which uses Sacred for experiment configuration and dispatches to a training run loop.

## Architecture

```
main.py (Sacred entry)
  └─> run/REGISTRY[run_mode]  (training loop)
        ├─> runners/REGISTRY[runner]  (episode collection)
        │     └─> envs/REGISTRY[env]  (environment)
        ├─> controllers/REGISTRY[mac]  (multi-agent controller)
        │     └─> modules/agents/REGISTRY[agent]  (agent network)
        ├─> learners/REGISTRY[learner]  (training algorithm)
        │     ├─> modules/mixers/*  (value mixing)
        │     └─> modules/critics/*  (centralized critics)
        └─> components/episode_buffer  (replay storage)
```

All major subsystems use a **REGISTRY** dict pattern — config YAML specifies a string key, the framework looks up the corresponding class at runtime.

## Module Details

### config/

| File | Purpose |
|------|---------|
| `config/default.yaml` | Global defaults: runner, MAC, env, hyperparams (gamma, lr, buffer_size, etc.) |
| `config/algs/*.yaml` | Per-algorithm overrides (qmix, vdn, coma, ppo, dvd, rode, etc.) |
| `config/envs/*.yaml` | Per-environment configs (sc2, gfootball, stag_hunt, matrix_game) |

Config is loaded by Sacred in `main.py` via `_get_config()` and deep-merged with `recursive_dict_update()`.

### components/

| File | Purpose | Key Components |
|------|---------|----------------|
| `action_selectors.py` | Action selection strategies during rollout | `EpsilonGreedyActionSelector`, `MultinomialActionSelector`, `GumbelSoftmaxMultinomialActionSelector`, `GaussianActionSelector` |
| `episode_buffer.py` | Pre-allocated episode storage and replay | `EpisodeBatch`, `ReplayBuffer`, `PrioritizedReplayBuffer` |
| `epsilon_schedules.py` | Annealing schedules for exploration | `DecayThenFlatSchedule`, `LinearIncreaseSchedule` |
| `segment_tree.py` | O(log N) data structures for PER sampling | `SumSegmentTree`, `MinSegmentTree` |
| `transforms.py` | Data transforms on buffer insertion | `OneHot` |

### controllers/

Multi-Agent Controllers (MAC) manage agent network execution during rollout. All expose `select_actions()`, `forward()`, `init_hidden()`, `save_models()`, `load_models()`.

| File | Purpose | Key Components |
|------|---------|----------------|
| `basic_controller.py` | Decentralized MAC with local obs | `BasicMAC` |
| `basic_central_controller.py` | Centralized MAC using global state | `CentralBasicMAC` |
| `n_controller.py` | Raw Q-value output (no softmax) | `NMAC` |
| `ppo_controller.py` | Dual-head (policy + value) for PPO | `PPOMAC` |
| `conv_controller.py` | Frame-stacking for image obs | `ConvMAC` |
| `dave_controller.py` | Dual alter/ego streams for DAVE | `DAVEMAC` |
| `dop_controller.py` | Epsilon-blended policy for DOP | `DOPMAC` |
| `lica_controller.py` | Raw logits for Gumbel-Softmax | `LICAMAC` |
| `rode_controller.py` | Role assignment + action space restriction | `RODEMAC` |

### envs/

| File | Purpose | Key Components |
|------|---------|----------------|
| `multiagentenv.py` | Abstract base interface | `MultiAgentEnv` |
| `starcraft/StarCraft2Env.py` | SMAC micromanagement via SC2 client | `StarCraft2Env` |
| `gfootball/FootballEnv.py` | Google Football wrapper | `GoogleFootballEnv` |
| `stag_hunt/stag_hunt.py` | Grid-world cooperative hunting | `StagHunt` |
| `matrix_game/one_step_matrix_game.py` | 1-step test for value decomposition | `OneStepMatrixGame` |

All envs implement: `reset()`, `step(actions)`, `get_obs()`, `get_state()`, `get_avail_actions()`, `get_env_info()`.

### learners/

Training algorithms. All expose `train(batch, t_env, episode_num)` and manage their own optimizers and target networks.

| File | Purpose | Algorithm Family |
|------|---------|-----------------|
| `q_learner.py` | 1-step DQN with VDN/QMIX | Value decomposition |
| `nq_learner.py` | N-step TD(λ) with double-Q, PER support | Value decomposition |
| `nq_learner_with_sarsa.py` | SARSA targets + RND intrinsic reward | Value decomposition |
| `dvd_learner.py` | DVD mixer with hidden-state conditioning | Value decomposition |
| `dvd_nq_learner_with_sarsa.py` | DVD + SARSA + RND + soft target updates | Value decomposition |
| `dmaq_qatten_learner.py` | QPLEX dueling V+A decomposition | Value decomposition |
| `qplex_dvd.py` | QPLEX + DVD hidden-state advantage | Value decomposition |
| `qtran_learner.py` | QTran triple-loss (td + opt + nopt) | Value decomposition |
| `max_q_learner.py` | OW-QMIX / CW-QMIX weighted loss | Value decomposition |
| `Kaleidoscope_learner.py` | Masked-network diversity + dead-weight reset | Value decomposition |
| `Kaleidoscope_DVD_learner.py` | Kaleidoscope + DVD mixer | Value decomposition |
| `coma_learner.py` | COMA counterfactual policy gradient | Policy gradient |
| `lica_learner.py` | LICA centralized critic + Gumbel actor | Policy gradient |
| `offpg_learner.py` | DOP off-policy PG with importance weighting | Policy gradient |
| `ppo_learner.py` | Multi-agent PPO with GAE + value clipping | Policy gradient |
| `policy_gradient_v2.py` | IAC/VDN-style PG with TD(λ) value targets | Policy gradient |
| `fmac_learner.py` | Factored critic + differentiable Q actor update | Actor-critic |
| `dave_learner.py` | DAVE: alter Q + ego policy + autoencoder | Hybrid |
| `rode_learner.py` | RODE: role Q + action Q + action encoder | Hybrid |

### modules/agents/

Neural network architectures for individual agents. All implement `forward(inputs, hidden_state)` → `(outputs, new_hidden)`.

| File | Purpose |
|------|---------|
| `rnn_agent.py` | Basic GRU → Q-values (default) |
| `n_rnn_agent.py` | GRU + optional LayerNorm + orthogonal init |
| `rnn_ppo_agent.py` | GRU → (policy logits, value) dual-head |
| `atten_rnn_agent.py` | GRU + multi-head self-attention |
| `central_rnn_agent.py` | GRU → action embeddings for centralized mixing |
| `conv_agent.py` | 1D Conv for temporal frame stacks |
| `dave_rnn_agent.py` | Dual GRU (alter Q + ego logits) |
| `ff_agent.py` | Feed-forward (no recurrence) |
| `mlp_agent.py` | MLP with optional tanh output |
| `noisy_agents.py` | GRU + NoisyNet output layer |
| `KaleidoscopeAgent.py` | Per-agent binary masks over shared weights (STE) |
| `rode_agent.py` | GRU encoder only (no Q-head, for role decomposition) |

### modules/mixers/

Value mixing networks that combine individual Q-values into Q_tot. All implement `forward(agent_qs, states, ...)` → `Q_tot`.

| File | Purpose |
|------|---------|
| `vdn.py` | Sum of individual Q-values |
| `qmix.py` | State-conditioned hypernetwork (monotonic) |
| `nmix.py` | Enhanced QMIX with configurable positivity constraints |
| `qatten.py` | Multi-head attention mixing weights |
| `dvd.py` | Graph Attention Network on hidden states |
| `dvd_qplex.py` | DVD + QPLEX dueling decomposition |
| `dmaq_general.py` | QPLEX (V + advantage with SI-weights) |
| `dmaq_si_weight.py` | State-action importance weights for QPLEX |
| `qmix_central_no_hyper.py` | Direct MLP mixer (no hypernetworks) |
| `qtran.py` | QTran joint Q + V networks |

### modules/critics/

Centralized critics for actor-critic methods.

| File | Purpose |
|------|---------|
| `coma.py` | COMA per-action Q conditioned on state + other actions |
| `centralv.py` | Centralized V(s) critic |
| `lica.py` | LICA hypernetwork critic mixing action probs |
| `offpg.py` | Dueling (V+A) critic for off-policy PG |
| `fmac_critic.py` | Factored Q(obs, action) critic |

### modules/other

| File | Purpose |
|------|---------|
| `exploration/rnd.py` | RND intrinsic curiosity (frozen target + trainable predictor) |
| `layer/self_atten.py` | Multi-head self-attention layer |
| `layer/MaskedLinear.py` | Per-agent masked weight layer (STE) |
| `action_encoders/obs_reward_encoder.py` | Learns action embeddings via obs/reward prediction (RODE) |
| `autoencoders/dave_autoencoder.py` | State-action reconstruction for DAVE auxiliary loss |

### run/

Training loop variants.

| File | Purpose |
|------|---------|
| `run.py` | Default loop: collect → sample → train → test → save |
| `per_run.py` | PER variant with priority-weighted sampling |
| `on_off_run.py` | Dual buffer (on-policy + off-policy) with alternating training |
| `dop_run.py` | DOP variant with separate critic/actor training phases |

### runners/

| File | Purpose |
|------|---------|
| `episode_runner.py` | Sequential single-env episode collection |
| `parallel_runner.py` | Multi-process parallel env collection |

### utils/

| File | Purpose |
|------|---------|
| `rl_utils.py` | `build_td_lambda_targets`, `build_gae_targets`, `build_q_lambda_targets`, `RunningMeanStd` |
| `logging.py` | Logger with TensorBoard + Sacred + console backends |
| `value_norm.py` | PopArt-style running value normalization |
| `th_utils.py` | Tensor clipping, param counting, weight init |
| `noisy_liner.py` | NoisyNet linear layer |
| `timehelper.py` | ETA and progress display |
| `dict2namedtuple.py` | Dict → namedtuple conversion |

## Data Flow

1. **Config** → `main.py` loads YAML (default + alg + env), merges, seeds RNG
2. **Dispatch** → `run/REGISTRY[args.run]` selects training loop
3. **Init** → training loop creates Runner, ReplayBuffer, MAC (with agent network), Learner (with mixer/critic)
4. **Collect** → Runner calls `env.reset()` + `env.step()` loop, MAC produces actions via `select_actions()`, transitions stored in `EpisodeBatch`
5. **Buffer** → completed episodes inserted into `ReplayBuffer` (or `PrioritizedReplayBuffer`)
6. **Sample** → random batch drawn from buffer
7. **Train** → Learner computes loss (TD error, policy gradient, etc.), updates agent + mixer/critic networks
8. **Target update** → periodic hard copy or soft Polyak averaging of target networks
9. **Test/Save** → periodic evaluation episodes + model checkpoint saves

## Key Interfaces

```python
# Environment
env.reset() -> None
env.step(actions) -> (reward, terminated, info)
env.get_obs() -> List[np.array]
env.get_state() -> np.array
env.get_avail_actions() -> List[List[int]]
env.get_env_info() -> dict  # {state_shape, obs_shape, n_actions, n_agents, episode_limit}

# Multi-Agent Controller
mac.select_actions(ep_batch, t_ep, t_env, test_mode) -> actions
mac.forward(ep_batch, t) -> agent_outs
mac.init_hidden(batch_size) -> None

# Learner
learner.train(batch: EpisodeBatch, t_env: int, episode_num: int) -> None

# Mixer
mixer.forward(agent_qs, states) -> Q_tot  # [bs, T, 1]
```

## External Dependencies

| Library | Usage |
|---------|-------|
| PyTorch | All neural networks, optimizers, tensor ops |
| Sacred | Experiment configuration, logging, reproducibility |
| numpy | Array operations, random sampling |
| PySC2 / SMAC | StarCraft II environment interface |
| gfootball | Google Research Football environment |
| scikit-learn | KMeans clustering (RODE action space partitioning) |
| tensorboard_logger | Optional TensorBoard logging |
| pygame | Optional Stag Hunt rendering |
| PyYAML | Config file parsing |

## Notes

- All registries (`REGISTRY` dicts) use string keys from YAML config — typos silently fail at lookup time.
- Agent hidden states have shape `[batch, n_agents, hidden_dim]` — flattened to `[batch*n_agents, hidden_dim]` before GRU, then reshaped back.
- Mixers enforce monotonicity via `abs()` or `softplus()` on hypernetwork weights — this is the core QMIX constraint.
- The `EpisodeBatch` pre-allocates all tensors at construction; `buffer_cpu_only=True` (default) keeps replay on CPU, moves to GPU only for training.
- RODE uses variable-interval role selection (every `role_interval` steps) — role Q-values aggregate rewards over that interval.
- Kaleidoscope uses Straight-Through Estimator for binary mask gradients — masks are hard 0/1 in forward, continuous in backward.
- DVD mixer conditions on agent hidden states (not just global state) via Graph Attention, enabling agent-specific credit assignment.
- `on_off_run` and `dop_run` maintain separate buffers for different training objectives within the same learner.
