import copy

import torch as th
import torch.nn.functional as F
from torch.optim import Adam, RMSprop

from components.episode_buffer import EpisodeBatch
from components.epsilon_schedules import DecayThenFlatSchedule
from modules.autoencoders.dave_autoencoder import DAVEStateActionAutoEncoder
from modules.mixers.nmix import Mixer
from modules.mixers.qmix import QMixer
from modules.mixers.vdn import VDNMixer


class DAVELearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.device = th.device("cuda" if args.use_cuda else "cpu")

        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.state_dim = int(th.prod(th.tensor(args.state_shape)).item())
        self.num_samples = getattr(args, "dave_num_samples", 100)
        self.sample_chunk = getattr(args, "dave_sample_chunk", 16)

        self.last_target_update_episode = 0
        self.log_stats_t = -self.args.learner_log_interval - 1

        self.mixer = self._build_mixer(args)
        self.target_mixer = copy.deepcopy(self.mixer) if self.mixer is not None else None
        self.target_mac = copy.deepcopy(mac)
        self.autoencoder = DAVEStateActionAutoEncoder(args)

        self.value_params = list(self.mac.alter_parameters())
        if self.mixer is not None:
            self.value_params += list(self.mixer.parameters())
        self.ego_params = list(self.mac.ego_parameters())
        self.recon_params = list(self.autoencoder.parameters())

        self.value_optimiser = self._build_optimiser(self.value_params, getattr(args, "dave_value_lr", args.lr))
        self.ego_optimiser = self._build_optimiser(self.ego_params, getattr(args, "dave_ego_lr", args.lr))
        self.recon_optimiser = self._build_optimiser(self.recon_params, getattr(args, "dave_recon_lr", args.lr))

        self.lambda_schedule = DecayThenFlatSchedule(
            getattr(args, "dave_lambda_start", 0.5),
            getattr(args, "dave_lambda_finish", 0.0),
            max(1, getattr(args, "dave_lambda_anneal_time", 500000)),
            decay="linear",
        )

    def _build_optimiser(self, params, lr):
        if self.args.optimizer == "adam":
            return Adam(params=params, lr=lr, weight_decay=getattr(self.args, "weight_decay", 0))
        return RMSprop(params=params, lr=lr, alpha=self.args.optim_alpha, eps=self.args.optim_eps)

    def _build_mixer(self, args):
        if args.mixer is None:
            return None
        if args.mixer == "vdn":
            return VDNMixer()
        if args.mixer == "qmix":
            return QMixer(args)
        if args.mixer == "qmix_without_abs":
            # Keep the paper's comparison setting aligned with standard QMIX and
            # remove only the monotonicity constraint via args.abs = False.
            return QMixer(args)
        raise ValueError("Mixer {} not recognised.".format(args.mixer))

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        actions_no_last_dim = actions.squeeze(-1)
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        states = batch["state"].reshape(batch.batch_size, batch.max_seq_length, -1)

        # Stage 1: update alter ego value function and IGM-free mixer.
        online_out = self._rollout_mac(self.mac, batch)
        target_out = self._rollout_mac(self.target_mac, batch)

        online_alter_qs = online_out["alter_qs"]
        online_ego_probs = online_out["ego_probs"]
        target_alter_qs = target_out["alter_qs"]

        chosen_action_qvals = th.gather(online_alter_qs[:, :-1], dim=3, index=actions).squeeze(3)
        chosen_qtot = self._mix_agent_qs(chosen_action_qvals, states[:, :-1], self.mixer)

        with th.no_grad():
            next_joint_actions = self._sample_joint_actions(
                online_ego_probs[:, 1:].detach(),
                avail_actions[:, 1:],
                self.num_samples,
            )
            target_joint_qs = self._gather_joint_action_qs(target_alter_qs[:, 1:], next_joint_actions)
            target_qtot_samples = self._mix_joint_q_samples(
                target_joint_qs,
                states[:, 1:],
                self.target_mixer if self.target_mixer is not None else None,
            )
            target_expected_q = target_qtot_samples.mean(dim=2)
            targets = rewards + self.args.gamma * (1 - terminated) * target_expected_q

        td_error = chosen_qtot - targets.detach()
        masked_td_error = td_error * mask
        value_loss = 0.5 * (masked_td_error ** 2).sum() / mask.sum()

        self.value_optimiser.zero_grad()
        value_loss.backward()
        value_grad_norm = th.nn.utils.clip_grad_norm_(self.value_params, self.args.grad_norm_clip)
        self.value_optimiser.step()

        # Stage 2: refresh labels with the updated alter ego network, then
        # supervise the ego policy to put more mass on the best sampled action
        # and the most novel anti-ego action.
        online_out = self._rollout_mac(self.mac, batch)
        online_alter_qs = online_out["alter_qs"]
        online_ego_logits = online_out["ego_logits"]
        online_ego_probs = online_out["ego_probs"]

        with th.no_grad():
            candidate_joint_actions = self._sample_joint_actions(
                online_ego_probs[:, :-1].detach(),
                avail_actions[:, :-1],
                self.num_samples,
            )
            candidate_joint_qs = self._gather_joint_action_qs(
                online_alter_qs[:, :-1].detach(),
                candidate_joint_actions,
            )
            candidate_qtot = self._mix_joint_q_samples(
                candidate_joint_qs,
                states[:, :-1],
                self.mixer,
            ).squeeze(-1)
            best_joint_actions = self._select_best_joint_actions(candidate_joint_actions, candidate_qtot)

            anti_probs = self._build_anti_policy(online_ego_logits[:, :-1].detach(), avail_actions[:, :-1])
            anti_joint_actions = self._sample_joint_actions(anti_probs, avail_actions[:, :-1], self.num_samples)
            novelty_scores = self._score_novelty(states[:, :-1], anti_joint_actions)
            novel_joint_actions = self._select_best_joint_actions(anti_joint_actions, novelty_scores)

        lambda_t = self.lambda_schedule.eval(t_env)
        best_log_prob = self._joint_log_prob(online_ego_probs[:, :-1], best_joint_actions)
        novel_log_prob = self._joint_log_prob(online_ego_probs[:, :-1], novel_joint_actions)
        ego_loss_terms = -(best_log_prob + lambda_t * novel_log_prob)
        ego_loss = (ego_loss_terms * mask.squeeze(-1)).sum() / mask.sum()

        self.ego_optimiser.zero_grad()
        ego_loss.backward()
        ego_grad_norm = th.nn.utils.clip_grad_norm_(self.ego_params, self.args.grad_norm_clip)
        self.ego_optimiser.step()

        # Stage 3: update the autoencoder using the replayed state-action pairs.
        recon_loss = self._reconstruction_loss(states[:, :-1], actions_no_last_dim, mask)
        self.recon_optimiser.zero_grad()
        recon_loss.backward()
        recon_grad_norm = th.nn.utils.clip_grad_norm_(self.recon_params, self.args.grad_norm_clip)
        self.recon_optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            mask_elems = mask.sum().item()
            self.logger.log_stat("loss_td", value_loss.item(), t_env)
            self.logger.log_stat("loss_ego", ego_loss.item(), t_env)
            self.logger.log_stat("loss_recon", recon_loss.item(), t_env)
            self.logger.log_stat("dave_lambda", lambda_t, t_env)
            self.logger.log_stat("best_log_prob_mean", (best_log_prob * mask.squeeze(-1)).sum().item() / mask_elems, t_env)
            self.logger.log_stat("novel_log_prob_mean", (novel_log_prob * mask.squeeze(-1)).sum().item() / mask_elems, t_env)
            self.logger.log_stat("grad_norm_value", value_grad_norm, t_env)
            self.logger.log_stat("grad_norm_ego", ego_grad_norm, t_env)
            self.logger.log_stat("grad_norm_recon", recon_grad_norm, t_env)
            self.logger.log_stat("td_error_abs", masked_td_error.abs().sum().item() / mask_elems, t_env)
            self.logger.log_stat("q_taken_mean", (chosen_qtot * mask).sum().item() / mask_elems, t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item() / mask_elems, t_env)
            self.log_stats_t = t_env

    def _rollout_mac(self, mac, batch):
        alter_qs = []
        ego_logits = []
        ego_probs = []

        mac.agent.train()
        mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            out = mac.forward(batch, t=t, return_info=True)
            alter_qs.append(out["alter_qs"])
            ego_logits.append(out["ego_logits"])
            ego_probs.append(out["ego_probs"])

        return {
            "alter_qs": th.stack(alter_qs, dim=1),
            "ego_logits": th.stack(ego_logits, dim=1),
            "ego_probs": th.stack(ego_probs, dim=1),
        }

    def _mix_agent_qs(self, agent_qs, states, mixer):
        if mixer is None:
            return agent_qs.sum(dim=-1, keepdim=True)
        return mixer(agent_qs, states)

    def _sample_joint_actions(self, probs, avail_actions, num_samples):
        batch_size, seq_len, _, _ = probs.shape
        flat_probs = probs.reshape(-1, self.n_actions).clamp_min(1e-10)
        flat_probs = flat_probs / flat_probs.sum(dim=-1, keepdim=True)
        sampled = th.distributions.Categorical(flat_probs).sample((num_samples,))
        sampled = sampled.view(num_samples, batch_size, seq_len, self.n_agents).permute(1, 2, 0, 3)
        return sampled

    def _gather_joint_action_qs(self, alter_qs, joint_actions):
        expanded_qs = alter_qs.unsqueeze(2).expand(-1, -1, joint_actions.size(2), -1, -1)
        return th.gather(expanded_qs, dim=4, index=joint_actions.unsqueeze(-1)).squeeze(-1)

    def _mix_joint_q_samples(self, joint_agent_qs, states, mixer):
        if mixer is None:
            return joint_agent_qs.sum(dim=-1, keepdim=True)

        batch_size, seq_len, num_samples, _ = joint_agent_qs.shape
        mixed_chunks = []

        for start in range(0, num_samples, self.sample_chunk):
            end = min(num_samples, start + self.sample_chunk)
            chunk_qs = joint_agent_qs[:, :, start:end].permute(0, 2, 1, 3).reshape(batch_size * (end - start), seq_len, self.n_agents)
            chunk_states = states.unsqueeze(1).expand(-1, end - start, -1, -1).reshape(batch_size * (end - start), seq_len, self.state_dim)
            chunk_qtot = mixer(chunk_qs, chunk_states)
            chunk_qtot = chunk_qtot.view(batch_size, end - start, seq_len, 1).permute(0, 2, 1, 3)
            mixed_chunks.append(chunk_qtot)

        return th.cat(mixed_chunks, dim=2)

    def _select_best_joint_actions(self, joint_actions, scores):
        best_idx = scores.argmax(dim=2, keepdim=True)
        gather_index = best_idx.unsqueeze(-1).expand(-1, -1, 1, self.n_agents)
        return th.gather(joint_actions, dim=2, index=gather_index).squeeze(2)

    def _build_anti_policy(self, ego_logits, avail_actions):
        anti_logits = -ego_logits
        anti_logits = anti_logits.masked_fill(avail_actions == 0, -1e10)
        return th.softmax(anti_logits, dim=-1)

    def _score_novelty(self, states, joint_actions):
        batch_size, seq_len, num_samples, _ = joint_actions.shape
        scores = []

        for start in range(0, num_samples, self.sample_chunk):
            end = min(num_samples, start + self.sample_chunk)
            chunk_actions = joint_actions[:, :, start:end]
            chunk_state_inputs = states.unsqueeze(2).expand(-1, -1, end - start, -1).reshape(-1, self.state_dim)
            chunk_action_inputs = chunk_actions.reshape(-1, self.n_agents)
            with th.no_grad():
                chunk_scores = self._novelty_score_flat(chunk_state_inputs, chunk_action_inputs)
            scores.append(chunk_scores.view(batch_size, seq_len, end - start))

        return th.cat(scores, dim=2)

    def _novelty_score_flat(self, states, joint_actions):
        joint_action_onehot = F.one_hot(joint_actions.long(), num_classes=self.n_actions).float().view(-1, self.n_agents * self.n_actions)
        state_recon, action_logits = self.autoencoder(states, joint_action_onehot)
        state_loss = F.mse_loss(state_recon, states, reduction="none").mean(dim=-1)
        action_loss = F.cross_entropy(
            action_logits.reshape(-1, self.n_actions),
            joint_actions.reshape(-1),
            reduction="none",
        ).view(-1, self.n_agents).sum(dim=-1)
        return state_loss + action_loss

    def _joint_log_prob(self, ego_probs, joint_actions):
        action_probs = th.gather(ego_probs, dim=3, index=joint_actions.unsqueeze(-1)).squeeze(-1).clamp_min(1e-10)
        return action_probs.log().sum(dim=-1)

    def _reconstruction_loss(self, states, joint_actions, mask):
        state_inputs = states.reshape(-1, self.state_dim)
        joint_action_inputs = joint_actions.reshape(-1, self.n_agents)
        mask_inputs = mask.reshape(-1)

        joint_action_onehot = F.one_hot(joint_action_inputs.long(), num_classes=self.n_actions).float().view(-1, self.n_agents * self.n_actions)
        state_recon, action_logits = self.autoencoder(state_inputs, joint_action_onehot)

        state_loss = F.mse_loss(state_recon, state_inputs, reduction="none").mean(dim=-1)
        action_loss = F.cross_entropy(
            action_logits.reshape(-1, self.n_actions),
            joint_action_inputs.reshape(-1),
            reduction="none",
        ).view(-1, self.n_agents).sum(dim=-1)

        return ((state_loss + action_loss) * mask_inputs).sum() / mask_inputs.sum()

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        self.autoencoder.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.autoencoder.state_dict(), "{}/dave_autoencoder.th".format(path))
        th.save(self.value_optimiser.state_dict(), "{}/opt_value.th".format(path))
        th.save(self.ego_optimiser.state_dict(), "{}/opt_ego.th".format(path))
        th.save(self.recon_optimiser.state_dict(), "{}/opt_recon.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.autoencoder.load_state_dict(th.load("{}/dave_autoencoder.th".format(path), map_location=lambda storage, loc: storage))
        self.value_optimiser.load_state_dict(th.load("{}/opt_value.th".format(path), map_location=lambda storage, loc: storage))
        self.ego_optimiser.load_state_dict(th.load("{}/opt_ego.th".format(path), map_location=lambda storage, loc: storage))
        self.recon_optimiser.load_state_dict(th.load("{}/opt_recon.th".format(path), map_location=lambda storage, loc: storage))
