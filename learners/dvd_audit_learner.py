"""Frozen experimental copy of DVDNQLearner; existing learner is untouched.

Optional TD(lambda), auxiliary isolation, and mixed-hidden controls are
contained here. Disabled mixed controls preserve the previous audit behavior.
The original TD(lambda) helper below is deliberately retained for parity.
"""
import copy
import torch as th
import numpy as np
from torch.optim import Adam, RMSprop
from components.episode_buffer import EpisodeBatch
from modules.mixers.nmix import Mixer
from modules.mixers.dvd import DVDMixer
from modules.mixers.dvd_residual import ResidualDVDMixer
from modules.mixers.dvd_credit import CreditDVDMixer
from modules.mixers.dvd_takeover import TakeoverDVDMixer
from modules.mixers.dvd_adaptive_takeover import AdaptiveTakeoverDVDMixer
from modules.mixers.dvd_counterfactual_router import build_counterfactual_route_target
from modules.mixers.dvd_counterfactual_router_audit import (
    CounterfactualRouterAuditMixer as CounterfactualRouterDVDMixer,
)
from utils.td_lambda_audit import build_td_lambda_targets_audit
from modules.exploration.rnd import RNDModel

class RunningMeanStd:
    # 不需要保存全部的数据，就可以通过一个新的数据来计算整体的均值和方差
    def __init__(self, shape=()):
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4

    def update(self, x):
        batch_mean = x.mean()
        batch_var = x.var()
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        # 防止除0
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        new_var = m_2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

# TD(lambda) 计算函数
# 计算的是td-error的目标
# 这个其实/home/zhangbei/pymarl2/src/utils/rl_utils.py里面有
def build_td_lambda_targets(rewards, terminated, mask, target_qvals, n_agents, gamma, td_lambda):
    ret = target_qvals.new_zeros(*target_qvals.shape)
    ret[:, -1] = target_qvals[:, -1] * (1 - terminated[:, -1])
    for t in range(ret.shape[1] - 2, -1, -1):
        ret[:, t] = td_lambda * gamma * ret[:, t + 1] + (1 - td_lambda) * gamma * target_qvals[:, t]
        ret[:, t] = rewards[:, t] + ret[:, t] * (1 - terminated[:, t])
    return ret


class DVDNQLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.audit_fix_td_lambda = bool(getattr(args, "audit_fix_td_lambda", True))
        self.mac = mac
        self.logger = logger
        self.device = th.device('cuda' if args.use_cuda else 'cpu')

        self.params = list(mac.parameters())
        self.last_target_update_episode = 0

        # 软更新的参数
        self.target_update_tau = 0.01

        # Mixer 初始化
        if args.mixer == "dvd":
            self.mixer = DVDMixer(args) # DVD 结构
        elif args.mixer == "dvd_residual":
            self.mixer = ResidualDVDMixer(args) # BM 主干 + DVD 门控残差
        elif args.mixer == "dvd_credit":
            self.mixer = CreditDVDMixer(args) # BM 权重主干 + 可靠性感知 DVD 信用修正
        elif args.mixer == "dvd_takeover":
            self.mixer = TakeoverDVDMixer(args) # BM 到 DVD 信用权重的渐进接管
        elif args.mixer == "dvd_adaptive_takeover":
            self.mixer = AdaptiveTakeoverDVDMixer(args) # TD loss 学习的 BM/DVD 自适应路由
        elif args.mixer == "dvd_counterfactual_router":
            self.mixer = CounterfactualRouterDVDMixer(args) # BM 锚定的反事实路由
        elif args.mixer == "qmix_without_abs":
            self.mixer = Mixer(args)# Unconstrained Mixer
        else:
            raise ValueError(f"Mixer {args.mixer} not supported in DVDNQLearner")

        self.is_counterfactual_router = (
            args.mixer == "dvd_counterfactual_router"
        )
        if self.is_counterfactual_router:
            self.counterfactual_mix_loss_weight = float(
                getattr(args, "counterfactual_mix_loss_weight", 0.25)
            )
            self.counterfactual_bm_loss_weight = float(
                getattr(args, "counterfactual_bm_loss_weight", 1.0)
            )
            self.counterfactual_dvd_loss_weight = float(
                getattr(args, "counterfactual_dvd_loss_weight", 0.1)
            )
            self.counterfactual_route_loss_weight = float(
                getattr(args, "counterfactual_route_loss_weight", 0.1)
            )
            self.counterfactual_sparse_weight = float(
                getattr(args, "counterfactual_sparse_weight", 0.0)
            )
            self.counterfactual_route_margin = float(
                getattr(args, "counterfactual_route_margin", 0.05)
            )
            self.counterfactual_route_min_abs_advantage = float(
                getattr(
                    args,
                    "counterfactual_route_min_abs_advantage",
                    0.01,
                )
            )
            self.counterfactual_route_eps = float(
                getattr(args, "counterfactual_route_eps", 1e-6)
            )
            loss_weights = (
                self.counterfactual_mix_loss_weight,
                self.counterfactual_bm_loss_weight,
                self.counterfactual_dvd_loss_weight,
                self.counterfactual_route_loss_weight,
                self.counterfactual_sparse_weight,
            )
            if any(weight < 0.0 for weight in loss_weights):
                raise ValueError("counterfactual loss weights must be non-negative")
            if (
                self.counterfactual_mix_loss_weight
                + self.counterfactual_bm_loss_weight
                <= 0.0
            ):
                raise ValueError(
                    "counterfactual mix and BM loss weights cannot both be zero"
                )
            if not 0.0 <= self.counterfactual_route_margin < 1.0:
                raise ValueError("counterfactual_route_margin must be in [0, 1)")
            if self.counterfactual_route_eps <= 0.0:
                raise ValueError("counterfactual_route_eps must be positive")
            if self.counterfactual_route_min_abs_advantage < 0.0:
                raise ValueError(
                    "counterfactual_route_min_abs_advantage must be non-negative"
                )

        self.target_mixer = copy.deepcopy(self.mixer)
        self.params += list(self.mixer.parameters())

        # 优化器
        if self.args.optimizer == 'adam':
            self.optimiser = Adam(params=self.params, lr=args.lr, weight_decay=getattr(args, "weight_decay", 0))
        else:
            self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        self.target_mac = copy.deepcopy(mac)
        self.log_stats_t = -self.args.learner_log_interval - 1

        # RND 探索模块
        self.use_rnd = getattr(args, "use_rnd", False)
        if self.use_rnd:
            self.rnd = RNDModel(args).to(self.device)
            self.rnd_optimizer = Adam(self.rnd.predictor.parameters(), lr=getattr(args, "rnd_lr", 5e-4))
            self.rnd_ms = RunningMeanStd()

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        if hasattr(self.mixer, "set_t_env"):
            self.mixer.set_t_env(t_env)
        if hasattr(self.target_mixer, "set_t_env"):
            self.target_mixer.set_t_env(t_env)

        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        # 统计量
        rnd_loss_item = 0
        intrinsic_rewards_mean = 0

        '''
        之前的代码，rnd的奖励全程存在
        '''
        # if self.use_rnd:
        #     state_inputs = batch["state"][:, 1:]
        #     # 1. 计算预测误差 (batch, T, 1)
        #     rnd_predict_error = self.rnd(state_inputs)
        #     intrinsic_rewards = rnd_predict_error.detach()

        #     # 2. 更新统计量并归一化
        #     mask_rnd = mask[:, :intrinsic_rewards.shape[1]]
        #     valid_rewards = intrinsic_rewards[mask_rnd == 1]

        #     if valid_rewards.numel() > 0:
        #         self.rnd_ms.update(valid_rewards.cpu().numpy())

        #     # 使用 Running Mean/Std 归一化
        #     # 这一步非常关键，它保证了即使原始 error 变小，归一化后的 reward 分布依然稳定
        #     intrinsic_rewards_norm = (intrinsic_rewards - self.rnd_ms.mean) / (self.rnd_ms.var**0.5 + 1e-8)

        #     # 裁剪，防止离群值破坏 TD update
        #     intrinsic_rewards_norm = th.clamp(intrinsic_rewards_norm, -5, 5)

        #     # 3. 加权求和
        #     # 使用归一化后的奖励，而不是原始奖励
        #     min_len = min(rewards.shape[1], intrinsic_rewards.shape[1])
        #     rewards = rewards[:, :min_len] + self.args.rnd_beta * intrinsic_rewards_norm[:, :min_len]

        #     # 记录用于 Tensorboard
        #     intrinsic_rewards_mean = intrinsic_rewards_norm.mean().item()

        #     # RND 更新
        #     rnd_loss = (rnd_predict_error[:, :min_len] * mask[:, :min_len]).sum() / mask[:, :min_len].sum()
        #     self.rnd_optimizer.zero_grad()
        #     rnd_loss.backward()
        #     self.rnd_optimizer.step()
        #     rnd_loss_item = rnd_loss.item()

        '''
        添加了动态的rnd机制
        '''
        # 计算当前的 rnd_beta (线性衰减逻辑)
        # 建议设置衰减周期，例如在 6M 步时衰减到 0
        rnd_decay_steps = 6000000
        if t_env < rnd_decay_steps:
            progress = t_env / rnd_decay_steps
            current_rnd_beta = self.args.rnd_beta * (1.0 - progress)
        else:
            current_rnd_beta = 0.0  # 后期彻底关闭内在奖励

        # 只有在需要时才计算 RND
        if self.use_rnd and current_rnd_beta > 1e-5:
            state_inputs = batch["state"][:, 1:]

            # 1. 计算预测误差
            rnd_predict_error = self.rnd(state_inputs)
            intrinsic_rewards = rnd_predict_error.detach()

            # 2. 更新统计量
            mask_rnd = mask[:, :intrinsic_rewards.shape[1]]
            valid_rewards = intrinsic_rewards[mask_rnd == 1]

            if valid_rewards.numel() > 0:
                self.rnd_ms.update(valid_rewards.cpu().numpy())

            # 3. 归一化 (带方差保护)
            if self.rnd_ms.var < 1e-6:
                intrinsic_rewards_norm = th.zeros_like(intrinsic_rewards)
            else:
                intrinsic_rewards_norm = (intrinsic_rewards - self.rnd_ms.mean) / (self.rnd_ms.var**0.5 + 1e-6)

            # 裁剪防止离群值
            intrinsic_rewards_norm = th.clamp(intrinsic_rewards_norm, -5, 5)

            # 4. 更新 rewards
            min_len = min(rewards.shape[1], intrinsic_rewards.shape[1])
            # 注意：这里我们修改的是 rewards 变量，它随后会被传入 build_td_lambda_targets
            rewards = rewards[:, :min_len] + current_rnd_beta * intrinsic_rewards_norm[:, :min_len]

            # 记录日志
            intrinsic_rewards_mean = intrinsic_rewards_norm.mean().item()

            # 5. 更新 RND 网络
            # 计算 loss 时要用 mask 遮盖
            rnd_loss = (rnd_predict_error[:, :min_len] * mask[:, :min_len]).sum() / mask[:, :min_len].sum()
            self.rnd_optimizer.zero_grad()
            rnd_loss.backward()
            self.rnd_optimizer.step()
            rnd_loss_item = rnd_loss.item()

        # 2. Online Network 前向传播 (收集 Hidden States 用于 DVD)
        self.mac.agent.train()
        mac_out = []
        hidden_states_list = [] # 用于存储 hidden states
        self.mac.init_hidden(batch.batch_size) # 初始化隐藏层

        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
            # 收集 hidden states: (batch, n_agents, rnn_dim)
            cur_h = self.mac.hidden_states.view(batch.batch_size, self.args.n_agents, -1).clone()
            hidden_states_list.append(cur_h)

        mac_out = th.stack(mac_out, dim=1)
        whole_hidden_states = th.stack(hidden_states_list, dim=1) # (batch, T_max, n_agents, rnn_dim)

        # 截取 t=0 到 T-1 的 hidden states 用于 online mixer
        hidden_states_main = whole_hidden_states[:, :-1]

        # 获取选定动作的 Q 值
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)

        # 3. Target Network 前向传播
        with th.no_grad():
            self.target_mac.agent.train()
            target_mac_out = []
            target_hidden_states_list = []
            self.target_mac.init_hidden(batch.batch_size)

            for t in range(batch.max_seq_length):
                target_agent_outs = self.target_mac.forward(batch, t=t)
                target_mac_out.append(target_agent_outs)
                # 收集 Target Hidden States
                cur_target_h = self.target_mac.hidden_states.view(batch.batch_size, self.args.n_agents, -1).clone()
                target_hidden_states_list.append(cur_target_h)

            target_mac_out = th.stack(target_mac_out, dim=1)
            whole_target_hidden_states = th.stack(target_hidden_states_list, dim=1)

            # SARSA Target 计算
            # 1. 获取 Target Q-values (t=1 到 T)
            target_qvals_next = target_mac_out[:, 1:]
            # 2. 获取实际执行的 Next Actions (来自 Replay Buffer)
            # batch["actions"] 包含了历史轨迹中真实执行的动作
            # 我们需要的是 t+1 时刻的动作
            next_actions = batch["actions"][:, 1:]
            # 3. 安全对齐长度
            # 确保 Q值 和 动作 的时间维度长度一致
            target_len = min(target_qvals_next.shape[1], next_actions.shape[1])
            target_qvals_next = target_qvals_next[:, :target_len]
            next_actions = next_actions[:, :target_len]
            # 4. 根据实际动作获取 Target Q 值
            # 不再使用 .max(dim=3)，而是使用 .gather
            # y = r + gamma * Q_target(s', a'_actual)
            target_chosen_qvals = th.gather(target_qvals_next, 3, next_actions).squeeze(3)

            # # double_DQN 计算
            # # 1. 准备 Next Q-values (Online Network)
            # mac_out_next = mac_out[:, 1:].clone().detach()
            # # 2. 准备 Next Available Actions
            # # 必须使用 avail_actions 来屏蔽动作，而不是 mask(filled)
            # avail_actions_next = batch["avail_actions"][:, 1:]
            # # 3. 安全对齐长度
            # target_len = min(mac_out_next.shape[1], avail_actions_next.shape[1])
            # mac_out_next = mac_out_next[:, :target_len]
            # avail_actions_next = avail_actions_next[:, :target_len]
            # # 4. 屏蔽非法动作
            # mac_out_next[avail_actions_next == 0] = -9999999
            # # 5. 选动作 (Greedy Action)
            # cur_next_actions = mac_out_next.max(dim=3, keepdim=True)[1]
            # # 6. 准备 Target Q-values (Target Network)
            # target_qvals_next = target_mac_out[:, 1:]
            # # 同样对齐长度
            # target_qvals_next = target_qvals_next[:, :target_len]
            # # 7. 根据选出的动作获取 Q 值
            # target_chosen_qvals = th.gather(target_qvals_next, 3, cur_next_actions).squeeze(3)

            # Target Mixer 前向传播
            # 如果是 DVD Mixer，需要传入 hidden states
            if self.args.mixer in ["dvd", "dvd_residual", "dvd_credit", "dvd_takeover", "dvd_adaptive_takeover", "dvd_counterfactual_router"]:
                # target hidden states 也要取 t=1 到 T
                target_hs_next = whole_target_hidden_states[:, 1:1+target_len]
                target_q_tot = self.target_mixer(target_chosen_qvals, batch["state"][:, 1:1+target_len], target_hs_next)
                if self.is_counterfactual_router:
                    # The target gate is not allowed to drift together with
                    # the online gate. Pure target-BM is the Bellman anchor.
                    target_q_tot = self.target_mixer.last_counterfactual_bm_q
            else:
                target_q_tot = self.target_mixer(target_chosen_qvals, batch["state"][:, 1:1+target_len])

        # 4. TD(lambda) Targets
        # 确保 Online Q_tot 计算所需的数据长度一致
        T_min = min(chosen_action_qvals.shape[1], target_q_tot.shape[1], hidden_states_main.shape[1])

        # 截断数据
        chosen_action_qvals = chosen_action_qvals[:, :T_min]
        hidden_states_main = hidden_states_main[:, :T_min]
        target_q_tot = target_q_tot[:, :T_min]
        rewards = rewards[:, :T_min]
        mask = mask[:, :T_min]
        terminated = terminated[:, :T_min]

        # 计算 TD Targets
        target_builder = (
            build_td_lambda_targets_audit
            if self.audit_fix_td_lambda else build_td_lambda_targets
        )
        targets = target_builder(rewards, terminated, mask, target_q_tot,
                                 self.args.n_agents, self.args.gamma, self.args.td_lambda)

        # 5. Online Mixer 前向传播
        if self.args.mixer in ["dvd", "dvd_residual", "dvd_credit", "dvd_takeover", "dvd_adaptive_takeover", "dvd_counterfactual_router"]:
            # DVD: 传入 Q, State, Hidden States
            online_q_tot = self.mixer(chosen_action_qvals, batch["state"][:, :T_min], hidden_states_main)
        else:
            # Normal: 传入 Q, State
            online_q_tot = self.mixer(chosen_action_qvals, batch["state"][:, :T_min])

        # 6. 计算 Loss
        detached_targets = targets.detach()
        td_error = (online_q_tot - detached_targets)
        masked_td_error = mask * td_error
        loss_td = (masked_td_error ** 2).sum() / mask.sum()

        total_loss = loss_td
        counterfactual_stats = None
        if self.is_counterfactual_router:
            bm_q = self.mixer.last_counterfactual_bm_q
            dvd_q = self.mixer.last_counterfactual_dvd_q
            gate = self.mixer.last_counterfactual_gate

            bm_error = bm_q - detached_targets
            dvd_error = dvd_q - detached_targets
            loss_bm = ((mask * bm_error) ** 2).sum() / mask.sum()
            loss_dvd = ((mask * dvd_error) ** 2).sum() / mask.sum()

            # Routing criterion: use DVD only when its fitting error to the
            # same BM-anchored bootstrap target is lower by the margin. This
            # is not an estimate of error to the unknown true return.
            bm_error_abs = bm_error.detach().abs()
            dvd_error_abs = dvd_error.detach().abs()
            (
                route_target,
                relative_advantage,
                absolute_advantage,
                route_is_reliable,
            ) = build_counterfactual_route_target(
                bm_error_abs,
                dvd_error_abs,
                self.mixer.gate_max,
                self.counterfactual_route_margin,
                self.counterfactual_route_min_abs_advantage,
                self.counterfactual_route_eps,
            )
            route_error = gate - route_target
            loss_route = ((mask * route_error) ** 2).sum() / mask.sum()
            loss_sparse = (mask * gate).sum() / mask.sum()

            # Keep the overall value-loss scale fixed while making the BM path
            # the dominant agent-learning signal. The default 0.25:1 ratio is
            # 20% mixed and 80% pure BM after normalization.
            value_weight_sum = (
                self.counterfactual_mix_loss_weight
                + self.counterfactual_bm_loss_weight
            )
            loss_value = (
                self.counterfactual_mix_loss_weight * loss_td
                + self.counterfactual_bm_loss_weight * loss_bm
            ) / value_weight_sum
            total_loss = (
                loss_value
                + self.counterfactual_dvd_loss_weight * loss_dvd
                + self.counterfactual_route_loss_weight * loss_route
                + self.counterfactual_sparse_weight * loss_sparse
            )

            with th.no_grad():
                valid_gate = gate.detach()[mask.bool()]
                gate_mean = (mask * gate.detach()).sum() / mask.sum()
                gate_variance = (
                    mask * (gate.detach() - gate_mean).pow(2)
                ).sum() / mask.sum()
                valid_route_target = route_target[mask.bool()]
                gate_centered = valid_gate - valid_gate.mean()
                route_target_centered = (
                    valid_route_target - valid_route_target.mean()
                )
                gate_target_correlation = (
                    (gate_centered * route_target_centered).mean()
                    / (
                        th.sqrt(gate_centered.pow(2).mean())
                        * th.sqrt(route_target_centered.pow(2).mean())
                        + self.counterfactual_route_eps
                    )
                )
                counterfactual_stats = {
                    "loss_value": loss_value.detach(),
                    "loss_bm": loss_bm.detach(),
                    "loss_dvd": loss_dvd.detach(),
                    "loss_route": loss_route.detach(),
                    "loss_sparse": loss_sparse.detach(),
                    "route_target_mean": (
                        mask * route_target
                    ).sum() / mask.sum(),
                    "route_positive_rate": (
                        mask * (route_target > 0.0).to(mask.dtype)
                    ).sum() / mask.sum(),
                    "route_reliable_rate": (
                        mask * route_is_reliable.to(mask.dtype)
                    ).sum() / mask.sum(),
                    "relative_advantage_mean": (
                        mask * relative_advantage
                    ).sum() / mask.sum(),
                    "absolute_advantage_mean": (
                        mask * absolute_advantage
                    ).sum() / mask.sum(),
                    "bm_error_mean": (
                        mask * bm_error_abs
                    ).sum() / mask.sum(),
                    "dvd_error_mean": (
                        mask * dvd_error_abs
                    ).sum() / mask.sum(),
                    "gate_mean": gate_mean,
                    "gate_std": th.sqrt(gate_variance.clamp_min(0.0)),
                    "gate_min": valid_gate.min(),
                    "gate_max": valid_gate.max(),
                    "gate_target_correlation": gate_target_correlation,
                    "gate_output_weight_norm": (
                        self.mixer.gate_net[2].weight.detach().norm()
                    ),
                }

        agent_gradient_stats = None
        if (self.is_counterfactual_router
                and bool(getattr(self.args, "audit_log_agent_gradients", False))
                and t_env - self.log_stats_t >= self.args.learner_log_interval):
            agent_gradient_stats = self._agent_gradient_stats(loss_bm, loss_td)

        # 7. 反向传播与更新
        self.optimiser.zero_grad()
        total_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(
            self.params,
            self.args.grad_norm_clip,
            error_if_nonfinite=self.args.mixer in ["dvd_takeover", "dvd_adaptive_takeover", "dvd_counterfactual_router"],
        )
        self.optimiser.step()

        # 下面这个是硬更新
        # 更新 Target Network
        # if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
        #     self._update_targets()
        #     self.last_target_update_episode = episode_num

        # 每一步进行软更新
        self._update_targets_soft()

        # 8. 日志记录
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("audit_fix_td_lambda", int(self.audit_fix_td_lambda), t_env)
            self.logger.log_stat("audit_isolate_dvd_aux_hidden", int(
                self.is_counterfactual_router and
                self.mixer.audit_isolate_dvd_aux_hidden
            ), t_env)
            self.logger.log_stat("loss_td", loss_td.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask.sum().item()), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask.sum().item()), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask.sum().item()), t_env)

            if counterfactual_stats is not None:
                self.logger.log_stat("audit_detach_mixed_hidden", int(
                    self.mixer.audit_detach_mixed_hidden), t_env)
                if agent_gradient_stats is not None:
                    for stat_name, stat_value in agent_gradient_stats.items():
                        self.logger.log_stat(stat_name, stat_value, t_env)
                self.logger.log_stat("counterfactual_loss_mix", loss_td.item(), t_env)
                self.logger.log_stat("counterfactual_loss_value", counterfactual_stats["loss_value"].item(), t_env)
                self.logger.log_stat("counterfactual_loss_bm", counterfactual_stats["loss_bm"].item(), t_env)
                self.logger.log_stat("counterfactual_loss_dvd", counterfactual_stats["loss_dvd"].item(), t_env)
                self.logger.log_stat("counterfactual_loss_route", counterfactual_stats["loss_route"].item(), t_env)
                self.logger.log_stat("counterfactual_loss_sparse", counterfactual_stats["loss_sparse"].item(), t_env)
                self.logger.log_stat("counterfactual_route_target_mean", counterfactual_stats["route_target_mean"].item(), t_env)
                self.logger.log_stat("counterfactual_route_positive_rate", counterfactual_stats["route_positive_rate"].item(), t_env)
                self.logger.log_stat("counterfactual_route_reliable_rate", counterfactual_stats["route_reliable_rate"].item(), t_env)
                self.logger.log_stat("counterfactual_relative_advantage_mean", counterfactual_stats["relative_advantage_mean"].item(), t_env)
                self.logger.log_stat("counterfactual_absolute_advantage_mean", counterfactual_stats["absolute_advantage_mean"].item(), t_env)
                self.logger.log_stat("counterfactual_bm_error_mean", counterfactual_stats["bm_error_mean"].item(), t_env)
                self.logger.log_stat("counterfactual_dvd_error_mean", counterfactual_stats["dvd_error_mean"].item(), t_env)
                self.logger.log_stat("counterfactual_gate_mean", counterfactual_stats["gate_mean"].item(), t_env)
                self.logger.log_stat("counterfactual_gate_std", counterfactual_stats["gate_std"].item(), t_env)
                self.logger.log_stat("counterfactual_gate_min", counterfactual_stats["gate_min"].item(), t_env)
                self.logger.log_stat("counterfactual_gate_max", counterfactual_stats["gate_max"].item(), t_env)
                self.logger.log_stat("counterfactual_gate_target_correlation", counterfactual_stats["gate_target_correlation"].item(), t_env)
                self.logger.log_stat("counterfactual_gate_output_weight_norm", counterfactual_stats["gate_output_weight_norm"].item(), t_env)

            if self.use_rnd:
                self.logger.log_stat("rnd_loss", rnd_loss_item, t_env)
                self.logger.log_stat("intrinsic_rewards_mean", intrinsic_rewards_mean, t_env)

            if hasattr(self.mixer, "last_gate_mean") and self.mixer.last_gate_mean is not None:
                self.logger.log_stat("residual_gate_mean", self.mixer.last_gate_mean.item(), t_env)
                self.logger.log_stat("residual_bm_q_mean", self.mixer.last_bm_q_mean.item(), t_env)
                self.logger.log_stat("residual_dvd_q_mean", self.mixer.last_dvd_q_mean.item(), t_env)
                if hasattr(self.mixer, "last_raw_residual_mean") and self.mixer.last_raw_residual_mean is not None:
                    self.logger.log_stat("residual_raw_delta_mean", self.mixer.last_raw_residual_mean.item(), t_env)
                self.logger.log_stat("residual_delta_mean", self.mixer.last_residual_mean.item(), t_env)

            if hasattr(self.mixer, "last_credit_gate_mean") and self.mixer.last_credit_gate_mean is not None:
                self.logger.log_stat("credit_gate_mean", self.mixer.last_credit_gate_mean.item(), t_env)
                self.logger.log_stat("credit_schedule_scale", self.mixer.last_credit_schedule_scale, t_env)
                self.logger.log_stat("attention_entropy_mean", self.mixer.last_attention_entropy_mean.item(), t_env)
                self.logger.log_stat("head_disagreement_mean", self.mixer.last_head_disagreement_mean.item(), t_env)
                self.logger.log_stat("attention_confidence_mean", self.mixer.last_attention_confidence_mean.item(), t_env)
                self.logger.log_stat("credit_delta_ratio_mean", self.mixer.last_credit_delta_ratio_mean.item(), t_env)
                self.logger.log_stat("bm_w1_rms", self.mixer.last_bm_w1_rms.item(), t_env)
                self.logger.log_stat("credit_delta_rms", self.mixer.last_credit_delta_rms.item(), t_env)
                for stat_name, stat_value in self.mixer.gradient_stats().items():
                    self.logger.log_stat(stat_name, stat_value, t_env)

            if hasattr(self.mixer, "last_takeover_alpha_mean") and self.mixer.last_takeover_alpha_mean is not None:
                self.logger.log_stat("takeover_alpha_mean", self.mixer.last_takeover_alpha_mean.item(), t_env)
                self.logger.log_stat("takeover_schedule_scale", self.mixer.last_takeover_schedule_scale, t_env)
                self.logger.log_stat("takeover_dvd_w1_ratio", self.mixer.last_takeover_dvd_w1_ratio.item(), t_env)
                self.logger.log_stat("takeover_shift_ratio", self.mixer.last_takeover_shift_ratio.item(), t_env)
                self.logger.log_stat("takeover_attention_entropy", self.mixer.last_attention_entropy_mean.item(), t_env)
                self.logger.log_stat("takeover_head_disagreement", self.mixer.last_head_disagreement_mean.item(), t_env)
                self.logger.log_stat("takeover_attention_confidence", self.mixer.last_attention_confidence_mean.item(), t_env)
                self.logger.log_stat("takeover_bm_w1_rms", self.mixer.last_bm_w1_rms.item(), t_env)
                self.logger.log_stat("takeover_dvd_w1_rms", self.mixer.last_dvd_w1_rms.item(), t_env)
                for stat_name, stat_value in self.mixer.gradient_stats().items():
                    self.logger.log_stat(stat_name, stat_value, t_env)

            if hasattr(self.mixer, "last_adaptive_gate_mean") and self.mixer.last_adaptive_gate_mean is not None:
                self.logger.log_stat("adaptive_gate_mean", self.mixer.last_adaptive_gate_mean.item(), t_env)
                self.logger.log_stat("adaptive_gate_std", self.mixer.last_adaptive_gate_std.item(), t_env)
                self.logger.log_stat("adaptive_gate_min", self.mixer.last_adaptive_gate_min.item(), t_env)
                self.logger.log_stat("adaptive_gate_max", self.mixer.last_adaptive_gate_max.item(), t_env)
                self.logger.log_stat("adaptive_shift_ratio", self.mixer.last_adaptive_shift_ratio.item(), t_env)
                self.logger.log_stat("adaptive_dvd_w1_ratio", self.mixer.last_adaptive_dvd_w1_ratio.item(), t_env)
                self.logger.log_stat("adaptive_bm_dvd_cosine", self.mixer.last_adaptive_bm_dvd_cosine.item(), t_env)
                self.logger.log_stat("adaptive_candidate_distance", self.mixer.last_adaptive_candidate_distance.item(), t_env)
                self.logger.log_stat("adaptive_attention_entropy", self.mixer.last_attention_entropy_mean.item(), t_env)
                self.logger.log_stat("adaptive_head_disagreement", self.mixer.last_head_disagreement_mean.item(), t_env)
                self.logger.log_stat("adaptive_attention_confidence", self.mixer.last_attention_confidence_mean.item(), t_env)
                self.logger.log_stat("adaptive_bm_w1_rms", self.mixer.last_bm_w1_rms.item(), t_env)
                self.logger.log_stat("adaptive_dvd_w1_rms", self.mixer.last_dvd_w1_rms.item(), t_env)
                for stat_name, stat_value in self.mixer.gradient_stats().items():
                    self.logger.log_stat(stat_name, stat_value, t_env)

            self.log_stats_t = t_env

        return {}

    def _agent_gradient_stats(self, loss_bm, loss_mix):
        """Raw same-batch gradients on agent parameters, before global clipping.

        autograd.grad does not populate .grad or step an optimizer. These
        diagnostics describe loss gradients, not Adam's actual update ratio.
        """
        agent_params = [p for p in self.mac.parameters() if p.requires_grad]
        bm_grads = th.autograd.grad(
            loss_bm, agent_params, retain_graph=True, allow_unused=True
        )
        mix_grads = th.autograd.grad(
            loss_mix, agent_params, retain_graph=True, allow_unused=True
        )
        bm_square = loss_bm.detach().new_zeros(())
        mix_square = bm_square.clone()
        dot = bm_square.clone()
        for bm_grad, mix_grad in zip(bm_grads, mix_grads):
            if bm_grad is not None:
                bm_square += bm_grad.detach().square().sum()
            if mix_grad is not None:
                mix_square += mix_grad.detach().square().sum()
            if bm_grad is not None and mix_grad is not None:
                dot += (bm_grad.detach() * mix_grad.detach()).sum()
        bm_norm, mix_norm = bm_square.sqrt(), mix_square.sqrt()
        cosine = dot / (bm_norm * mix_norm).clamp_min(1e-12)
        # The common value-loss normalizer cancels from this norm ratio.
        weighted_ratio = (
            self.counterfactual_mix_loss_weight * mix_norm
            / (self.counterfactual_bm_loss_weight * bm_norm).clamp_min(1e-12)
        )
        return {
            "audit_agent_bm_grad_norm": bm_norm.item(),
            "audit_agent_mix_grad_norm": mix_norm.item(),
            "audit_agent_bm_mix_grad_cosine": cosine.clamp(-1.0, 1.0).item(),
            "audit_agent_weighted_mix_bm_grad_ratio": weighted_ratio.item(),
        }

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def _update_targets_soft(self):
        # 对 Mac (Agent) 和 Mixer 同时进行软更新
        for target_param, param in zip(self.target_mac.parameters(), self.mac.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.target_update_tau) + param.data * self.target_update_tau
            )

        if self.mixer is not None:
            for target_param, param in zip(self.target_mixer.parameters(), self.mixer.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - self.target_update_tau) + param.data * self.target_update_tau
                )

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))

AUDIT_REFERENCE_SHA256 = '759e77244e30d9522ac08b0736d39823e1706f0b2cad3fca82ee98b29a995045'
