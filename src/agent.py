from src.model import Actor, Critic, SACActorModel
from src.buffer import PERBuffer, ReplayBuffer, HERBuffer, PERBufferSumTree
from src.utils import AgentConfig
import torch
from torch.optim import AdamW, lr_scheduler
import os
import numpy as np
import torch.nn as nn


class TD3Agent:
    def __init__(
        self, obs_dim: int, ac_dim: int, config: AgentConfig, weights: str, nenvs: int
    ):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config = config

        self.actor = Actor(
            obs_dim, self.config.hidden_dim, ac_dim, self.config.layer_count
        ).to(self.device)
        self.target_actor = Actor(
            obs_dim, self.config.hidden_dim, ac_dim, self.config.layer_count
        ).to(self.device)

        self.critic_1 = Critic(
            obs_dim + ac_dim, self.config.hidden_dim, self.config.layer_count
        ).to(self.device)
        self.critic_2 = Critic(
            obs_dim + ac_dim, self.config.hidden_dim, self.config.layer_count
        ).to(self.device)
        self.target_critic_1 = Critic(
            obs_dim + ac_dim, self.config.hidden_dim, self.config.layer_count
        ).to(self.device)
        self.target_critic_2 = Critic(
            obs_dim + ac_dim, self.config.hidden_dim, self.config.layer_count
        ).to(self.device)

        self.actor_opt = AdamW(self.actor.parameters(), self.config.actor_lr)
        self.critic_1_opt = AdamW(self.critic_1.parameters(), self.config.critic_lr)
        self.critic_2_opt = AdamW(self.critic_2.parameters(), self.config.critic_lr)

        self.actor_scheduler = lr_scheduler.CosineAnnealingLR(
            self.actor_opt,
            T_max=self.config.ac_scheduler_steps,
            eta_min=self.config.actor_lr_min,
        )
        self.critic_1_scheduler = lr_scheduler.CosineAnnealingLR(
            self.critic_1_opt,
            T_max=self.config.cr_scheduler_steps,
            eta_min=self.config.critic_lr_min,
        )
        self.critic_2_scheduler = lr_scheduler.CosineAnnealingLR(
            self.critic_2_opt,
            T_max=self.config.cr_scheduler_steps,
            eta_min=self.config.critic_lr_min,
        )

        if self.config.buffer_type == "PER":
            self.buffer = PERBuffer(self.config.max_len, self.config.alpha)
        elif self.config.buffer_type == "PER_SUMTREE":
            self.buffer = PERBufferSumTree(self.config.max_len, self.config.alpha)
        elif self.config.buffer_type == "REPLAY":
            self.buffer = ReplayBuffer(self.config.max_len)
        elif self.config.buffer_type == "HER":
            self.buffer = HERBuffer(
                self.config.max_len,
                self.config.max_eps_len,
                nenvs,
                k_future=self.config.k_future,
            )
        else:
            raise ValueError(
                f"[ERROR] Invalid Buffer type. Received {self.config.buffer_type}."
            )

        self.noise_std = self.config.noise_std
        self.noise_clamp = self.config.noise_clamp

        self.beta = self.config.beta
        self.beta_start = self.config.beta
        self.beta_max = 1.0
        self.beta_end = self.config.beta_end

        self.policy_noise = self.config.policy_noise
        self.gamma = self.config.gamma
        self.batch_size = self.config.batch_size
        self.ac_update_freq = self.config.ac_update_freq
        self.grad_clip = self.config.grad_clip
        self.tau = self.config.tau

        if weights:
            self.actor.load(os.path.join(weights, "actor.pth"), self.device)
            self.critic_1.load(os.path.join(weights, "critic_1.pth"), self.device)
            self.critic_2.load(os.path.join(weights, "critic_2.pth"), self.device)
            # Load log_alpha if exists
            log_alpha_path = os.path.join(weights, "log_alpha.pth")
            if os.path.exists(log_alpha_path):
                self.log_alpha = torch.load(log_alpha_path, map_location=self.device)
                self.alpha = self.log_alpha.exp()
                self.alpha_opt = AdamW([self.log_alpha], lr=self.config.alpha_lr)
            self.update_target_network()
        else:
            self.update_target_network()

    def update_target_network(self):
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

    def update_actor(self, tau: float = 0.005):
        for target_param, param in zip(
            self.target_actor.parameters(), self.actor.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def update_critic(self, tau: float = 0.005):
        for target_param, param in zip(
            self.target_critic_1.parameters(), self.critic_1.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for target_param, param in zip(
            self.target_critic_2.parameters(), self.critic_2.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def beta_scheduler(self, step: int):
        ratio = step / self.beta_end
        self.beta = min(
            self.beta_max, self.beta_start + ratio * (self.beta_max - self.beta_start)
        )

    def get_gradient_norm(self, model: torch.nn.Module):
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm**0.5
        return total_norm

    def actor_update(self, states: torch.Tensor):
        actions = self.actor(states)
        actor_loss = -self.critic_1(torch.cat([states, actions], dim=-1)).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_clip)
        actor_grad_norm = self.get_gradient_norm(self.actor)

        self.actor_opt.step()

        self.actor_scheduler.step()

        return actor_loss.item(), actor_grad_norm

    def critic_update(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
        weights: torch.Tensor = None,
    ):
        with torch.no_grad():
            noise = torch.clamp(
                torch.randn_like(action) * self.policy_noise,
                -self.noise_clamp,
                self.noise_clamp,
            )
            next_action = torch.clamp(self.target_actor(next_state) + noise, -1, 1)

            target_critic_input = torch.concat([next_state, next_action], dim=-1)
            target_Q1_value = self.target_critic_1(target_critic_input)
            target_Q2_value = self.target_critic_2(target_critic_input)
            target_Q_value = torch.min(target_Q1_value, target_Q2_value)

            target = reward + self.gamma * (1 - done) * target_Q_value

        current_critic_input = torch.concat([state, action], dim=-1)
        current_q1_value = self.critic_1(current_critic_input)
        current_q2_value = self.critic_2(current_critic_input)

        self.critic_1_opt.zero_grad()
        if weights is not None and weights.numel() > 0:
            critic_1_loss = torch.nn.functional.smooth_l1_loss(
                current_q1_value, target, reduction="none"
            )
            critic_1_loss = (weights * critic_1_loss).mean()
        else:
            critic_1_loss = torch.nn.functional.smooth_l1_loss(current_q1_value, target)
        critic_1_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.critic_1.parameters(), self.grad_clip)
        critic_1_grad_norm = self.get_gradient_norm(self.critic_1)
        self.critic_1_opt.step()

        self.critic_2_opt.zero_grad()
        if weights is not None and weights.numel() > 0:
            critic_2_loss = torch.nn.functional.smooth_l1_loss(
                current_q2_value, target, reduction="none"
            )
            critic_2_loss = (weights * critic_2_loss).mean()
        else:
            critic_2_loss = torch.nn.functional.smooth_l1_loss(current_q2_value, target)
        critic_2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_2.parameters(), self.grad_clip)
        critic_2_grad_norm = self.get_gradient_norm(self.critic_2)
        self.critic_2_opt.step()

        self.critic_1_scheduler.step()
        self.critic_2_scheduler.step()

        td_error_1 = torch.abs(current_q1_value - target).detach()
        td_error_2 = torch.abs(current_q2_value - target).detach()

        q_value = (
            torch.cat([current_q1_value, current_q2_value], dim=-1)
            .mean()
            .detach()
            .cpu()
            .item()
        )

        if weights is not None and weights.numel() > 0:
            td_error = torch.maximum(td_error_1, td_error_2).cpu().numpy()
            return (
                critic_1_loss.item(),
                critic_2_loss.item(),
                td_error,
                q_value,
                critic_1_grad_norm,
                critic_2_grad_norm,
            )
        else:
            td_error = torch.mean(torch.maximum(td_error_1, td_error_2)).cpu().numpy()
            return (
                critic_1_loss.item(),
                critic_2_loss.item(),
                td_error,
                q_value,
                critic_1_grad_norm,
                critic_2_grad_norm,
            )

    def select_action(self, obs_tensor: np.array, eval_action: bool = False):
        self.set_eval()
        if not eval_action:
            obs_tensor = torch.as_tensor(obs_tensor, dtype=torch.float32).to(
                self.device
            )
            with torch.no_grad():
                action = torch.tanh(self.actor(obs_tensor)).detach().cpu().numpy()
            noise = np.random.normal(0, self.noise_std, size=action.shape)
            action = np.clip(action + noise, -1, 1)
            return action
        else:
            obs_tensor = torch.as_tensor(obs_tensor, dtype=torch.float32).to(
                self.device
            )
            with torch.no_grad():
                return np.clip(torch.tanh(self.actor(obs_tensor)).cpu().numpy(), -1, 1)

    def push(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def push_her(
        self, idx, state, action, next_state, reward, done, desired_goal, achieved_goal
    ):
        self.buffer.push(
            idx, state, action, next_state, reward, done, desired_goal, achieved_goal
        )

    def update(self, step: int):
        self.set_train()
        if isinstance(self.buffer, PERBuffer) or isinstance(
            self.buffer, PERBufferSumTree
        ):
            states, actions, rewards, next_states, dones, weights, indices = (
                self.buffer.sample(self.batch_size, self.beta)
            )
            q1_loss, q2_loss, td_error, q_value, critic_1_grad, critic_2_grad = (
                self.critic_update(
                    states, actions, rewards, next_states, dones, weights
                )
            )
            self.buffer.update_priorities(indices=indices, priorities=td_error)
        else:
            states, actions, rewards, next_states, dones = self.buffer.sample(
                self.batch_size
            )
            q1_loss, q2_loss, td_error, q_value, critic_1_grad, critic_2_grad = (
                self.critic_update(states, actions, rewards, next_states, dones)
            )

        self.beta_scheduler(step)
        self.update_critic(self.tau)
        if step % self.ac_update_freq == 0:
            ac_loss, ac_grad = self.actor_update(states)
            self.update_actor(self.tau)
            return (
                q1_loss,
                q2_loss,
                ac_loss,
                td_error,
                q_value,
                critic_1_grad,
                critic_2_grad,
                ac_grad,
            )
        else:
            return q1_loss, q2_loss, td_error, q_value, critic_1_grad, critic_2_grad

    def save_weights(self, path: str):
        self.actor.save(os.path.join(path, "actor.pth"))
        self.critic_1.save(os.path.join(path, "critic_1.pth"))
        self.critic_2.save(os.path.join(path, "critic_2.pth"))

    def is_buffer_filled(self):
        return len(self.buffer) >= self.batch_size

    def set_train(self):
        self.actor.train()
        self.critic_1.train()
        self.critic_2.train()
        self.target_actor.eval()
        self.target_critic_1.eval()
        self.target_critic_2.eval()

    def set_eval(self):
        self.actor.eval()
        self.critic_1.eval()
        self.critic_2.eval()
        self.target_actor.eval()
        self.target_critic_1.eval()
        self.target_critic_2.eval()

    def update_normalizers(self, obs_list, dg_list, normalize):
        if hasattr(self.buffer, "obs_normalizer") and obs_list and normalize:
            combined_obs = np.concatenate(obs_list, axis=0)
            self.buffer.obs_normalizer.update(combined_obs)

        if hasattr(self.buffer, "dg_normalizer") and dg_list and normalize:
            combined_dg = np.concatenate(dg_list, axis=0)
            self.buffer.dg_normalizer.update(combined_dg)

    def normalize_state_batch(self, obs_batch, dg_batch, normalize):
        if hasattr(self.buffer, "obs_normalizer"):
            normalized_obs = self.normalize_obs(obs_batch, normalize)
        else:
            normalized_obs = obs_batch

        if hasattr(self.buffer, "dg_normalizer") and normalize:
            normalized_dg = self.normalize_goal(dg_batch, normalize)
        else:
            normalized_dg = dg_batch

        return np.concatenate([normalized_obs, normalized_dg], axis=-1)

    def normalize_obs(self, obs: np.array, normalize: bool):
        if hasattr(self.buffer, "obs_normalizer") and normalize:
            normalized_obs = self.buffer.obs_normalizer.normalize(obs)
        else:
            normalized_obs = obs
        return normalized_obs

    def normalize_goal(self, goal: np.array, normalize: bool):
        if hasattr(self.buffer, "dg_normalizer") and normalize:
            normalized_obs = self.buffer.dg_normalizer.normalize(goal)
        else:
            normalized_obs = goal
        return normalized_obs

    def reset(self):
        self.actor.reset()
        self.target_actor.reset()
        self.critic_1.reset()
        self.critic_2.reset()
        self.target_critic_1.reset()
        self.target_critic_2.reset()


class SACAgent:
    def __init__(
        self, obs_dim: int, ac_dim: int, config: AgentConfig, weights: str, nenvs: int
    ):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config = config

        self.actor = SACActorModel(
            obs_dim, self.config.hidden_dim, ac_dim, self.config.layer_count
        ).to(self.device)

        self.critic_1 = Critic(
            obs_dim + ac_dim, self.config.hidden_dim, self.config.layer_count
        ).to(self.device)
        self.critic_2 = Critic(
            obs_dim + ac_dim, self.config.hidden_dim, self.config.layer_count
        ).to(self.device)
        self.target_critic_1 = Critic(
            obs_dim + ac_dim, self.config.hidden_dim, self.config.layer_count
        ).to(self.device)
        self.target_critic_2 = Critic(
            obs_dim + ac_dim, self.config.hidden_dim, self.config.layer_count
        ).to(self.device)

        self.actor_opt = AdamW(self.actor.parameters(), self.config.actor_lr)
        self.critic_1_opt = AdamW(self.critic_1.parameters(), self.config.critic_lr)
        self.critic_2_opt = AdamW(self.critic_2.parameters(), self.config.critic_lr)

        self.target_entropy = -ac_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_opt = AdamW([self.log_alpha], lr=self.config.alpha_lr)

        self.actor_scheduler = lr_scheduler.CosineAnnealingLR(
            self.actor_opt,
            T_max=self.config.ac_scheduler_steps,
            eta_min=self.config.actor_lr_min,
        )
        self.critic_1_scheduler = lr_scheduler.CosineAnnealingLR(
            self.critic_1_opt,
            T_max=self.config.cr_scheduler_steps,
            eta_min=self.config.critic_lr_min,
        )
        self.critic_2_scheduler = lr_scheduler.CosineAnnealingLR(
            self.critic_2_opt,
            T_max=self.config.cr_scheduler_steps,
            eta_min=self.config.critic_lr_min,
        )

        if self.config.buffer_type == "PER":
            self.buffer = PERBuffer(self.config.max_len, self.config.alpha)
        elif self.config.buffer_type == "PER_SUMTREE":
            self.buffer = PERBufferSumTree(self.config.max_len, self.config.alpha)
        elif self.config.buffer_type == "REPLAY":
            self.buffer = ReplayBuffer(self.config.max_len)
        elif self.config.buffer_type == "HER":
            self.buffer = HERBuffer(
                self.config.max_len,
                self.config.max_eps_len,
                nenvs,
                k_future=self.config.k_future,
            )
        else:
            raise ValueError(
                f"[ERROR] Invalid Buffer type. Received {self.config.buffer_type}."
            )

        self.gamma = self.config.gamma
        self.batch_size = self.config.batch_size
        self.ac_update_freq = self.config.ac_update_freq
        self.grad_clip = self.config.grad_clip
        self.tau = self.config.tau

        self.beta = self.config.beta
        self.beta_start = self.config.beta
        self.beta_max = 1.0
        self.beta_end = self.config.beta_end

        if weights:
            self.actor.load(os.path.join(weights, "actor.pth"), self.device)
            self.critic_1.load(os.path.join(weights, "critic_1.pth"), self.device)
            self.critic_2.load(os.path.join(weights, "critic_2.pth"), self.device)
            self.update_target_network()
        else:
            self.update_target_network()

    def update_target_network(self):
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

    def update_critic(self, tau: float = 0.005):
        for target_param, param in zip(
            self.target_critic_1.parameters(), self.critic_1.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for target_param, param in zip(
            self.target_critic_2.parameters(), self.critic_2.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def beta_scheduler(self, step: int):
        ratio = step / self.beta_end
        self.beta = min(
            self.beta_max, self.beta_start + ratio * (self.beta_max - self.beta_start)
        )

    def get_gradient_norm(self, model: torch.nn.Module):
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm**0.5
        return total_norm

    def actor_update(self, states: torch.Tensor):
        actions, log_probs = self.actor.sample(states)

        q1_values = self.critic_1(torch.cat([states, actions], dim=-1))
        q2_values = self.critic_2(torch.cat([states, actions], dim=-1))
        min_q_values = torch.min(q1_values, q2_values)

        actor_loss = (self.alpha.detach() * log_probs - min_q_values).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_clip)
        actor_grad_norm = self.get_gradient_norm(self.actor)
        self.actor_opt.step()
        self.actor_scheduler.step()

        return actor_loss.item(), actor_grad_norm, log_probs.detach()

    def alpha_update(self, log_probs: torch.Tensor):
        alpha_loss = -(
            self.log_alpha * (log_probs + self.target_entropy).detach()
        ).mean()

        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        self.alpha = self.log_alpha.exp()
        return alpha_loss.item()

    def critic_update(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
        weights: torch.Tensor = None,
    ):
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_state)

            target_q1 = self.target_critic_1(
                torch.cat([next_state, next_actions], dim=-1)
            )
            target_q2 = self.target_critic_2(
                torch.cat([next_state, next_actions], dim=-1)
            )
            target_q = torch.min(target_q1, target_q2)

            target_q = target_q - self.alpha * next_log_probs
            target = reward + self.gamma * (1 - done) * target_q

        current_critic_input = torch.cat([state, action], dim=-1)
        current_q1_value = self.critic_1(current_critic_input)
        current_q2_value = self.critic_2(current_critic_input)

        self.critic_1_opt.zero_grad()
        if weights is not None and weights.numel() > 0:
            critic_1_loss = torch.nn.functional.mse_loss(
                current_q1_value, target, reduction="none"
            )
            critic_1_loss = (weights * critic_1_loss).mean()
        else:
            critic_1_loss = torch.nn.functional.mse_loss(current_q1_value, target)
        critic_1_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.critic_1.parameters(), max_norm=self.grad_clip
        )
        critic_1_grad_norm = self.get_gradient_norm(self.critic_1)
        self.critic_1_opt.step()

        self.critic_2_opt.zero_grad()
        if weights is not None and weights.numel() > 0:
            critic_2_loss = torch.nn.functional.mse_loss(
                current_q2_value, target, reduction="none"
            )
            critic_2_loss = (weights * critic_2_loss).mean()
        else:
            critic_2_loss = torch.nn.functional.mse_loss(current_q2_value, target)
        critic_2_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.critic_2.parameters(), max_norm=self.grad_clip
        )
        critic_2_grad_norm = self.get_gradient_norm(self.critic_2)
        self.critic_2_opt.step()

        self.critic_1_scheduler.step()
        self.critic_2_scheduler.step()

        td_error_1 = torch.abs(current_q1_value - target).detach()
        td_error_2 = torch.abs(current_q2_value - target).detach()

        q_value = (
            torch.cat([current_q1_value, current_q2_value], dim=-1)
            .mean()
            .detach()
            .cpu()
            .item()
        )

        if weights is not None and weights.numel() > 0:
            td_error = torch.maximum(td_error_1, td_error_2).cpu().numpy()
            return (
                critic_1_loss.item(),
                critic_2_loss.item(),
                td_error,
                q_value,
                critic_1_grad_norm,
                critic_2_grad_norm,
            )
        else:
            td_error = torch.mean(torch.maximum(td_error_1, td_error_2)).cpu().numpy()
            return (
                critic_1_loss.item(),
                critic_2_loss.item(),
                td_error,
                q_value,
                critic_1_grad_norm,
                critic_2_grad_norm,
            )

    def select_action(self, obs_tensor: np.array, eval_action: bool = False):
        self.set_eval()
        obs_tensor = torch.as_tensor(obs_tensor, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            action, _ = self.actor.sample(obs_tensor, deterministic=eval_action)
            return action.cpu().numpy()

    def push(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def push_her(
        self, idx, state, action, next_state, reward, done, desired_goal, achieved_goal
    ):
        self.buffer.push(
            idx, state, action, next_state, reward, done, desired_goal, achieved_goal
        )

    def update(self, step: int):
        self.set_train()
        if isinstance(self.buffer, PERBuffer) or isinstance(
            self.buffer, PERBufferSumTree
        ):
            states, actions, rewards, next_states, dones, weights, indices = (
                self.buffer.sample(self.batch_size, self.beta)
            )
            q1_loss, q2_loss, td_error, q_value, critic_1_grad, critic_2_grad = (
                self.critic_update(
                    states, actions, rewards, next_states, dones, weights
                )
            )
            self.buffer.update_priorities(indices=indices, priorities=td_error)
        else:
            states, actions, rewards, next_states, dones = self.buffer.sample(
                self.batch_size
            )
            q1_loss, q2_loss, td_error, q_value, critic_1_grad, critic_2_grad = (
                self.critic_update(states, actions, rewards, next_states, dones)
            )

        self.beta_scheduler(step)
        self.update_critic(self.tau)

        if step % self.ac_update_freq == 0:
            ac_loss, ac_grad, log_probs = self.actor_update(states)
            alpha_loss = self.alpha_update(log_probs)
            return (
                q1_loss,
                q2_loss,
                ac_loss,
                td_error,
                q_value,
                critic_1_grad,
                critic_2_grad,
                ac_grad,
                alpha_loss,
            )
        else:
            return q1_loss, q2_loss, td_error, q_value, critic_1_grad, critic_2_grad

    def save_weights(self, path: str):
        self.actor.save(os.path.join(path, "actor.pth"))
        self.critic_1.save(os.path.join(path, "critic_1.pth"))
        self.critic_2.save(os.path.join(path, "critic_2.pth"))
        torch.save(self.log_alpha, os.path.join(path, "log_alpha.pth"))

    def is_buffer_filled(self):
        return len(self.buffer) >= self.batch_size

    def set_train(self):
        self.actor.train()
        self.critic_1.train()
        self.critic_2.train()
        self.target_critic_1.eval()
        self.target_critic_2.eval()

    def set_eval(self):
        self.actor.eval()
        self.critic_1.eval()
        self.critic_2.eval()
        self.target_critic_1.eval()
        self.target_critic_2.eval()

    def update_normalizers(self, obs_list, dg_list, normalize):
        if hasattr(self.buffer, "obs_normalizer") and obs_list and normalize:
            combined_obs = np.concatenate(obs_list, axis=0)
            self.buffer.obs_normalizer.update(combined_obs)

        if hasattr(self.buffer, "dg_normalizer") and dg_list and normalize:
            combined_dg = np.concatenate(dg_list, axis=0)
            self.buffer.dg_normalizer.update(combined_dg)

    def normalize_state_batch(self, obs_batch, dg_batch, normalize):
        if hasattr(self.buffer, "obs_normalizer"):
            normalized_obs = self.normalize_obs(obs_batch, normalize)
        else:
            normalized_obs = obs_batch

        if hasattr(self.buffer, "dg_normalizer") and normalize:
            normalized_dg = self.normalize_goal(dg_batch, normalize)
        else:
            normalized_dg = dg_batch

        return np.concatenate([normalized_obs, normalized_dg], axis=-1)

    def normalize_obs(self, obs: np.array, normalize: bool):
        if hasattr(self.buffer, "obs_normalizer") and normalize:
            normalized_obs = self.buffer.obs_normalizer.normalize(obs)
        else:
            normalized_obs = obs
        return normalized_obs

    def normalize_goal(self, goal: np.array, normalize: bool):
        if hasattr(self.buffer, "dg_normalizer") and normalize:
            normalized_obs = self.buffer.dg_normalizer.normalize(goal)
        else:
            normalized_obs = goal
        return normalized_obs

    def reset(self):
        self.actor.reset()
        self.critic_1.reset()
        self.critic_2.reset()
        self.target_critic_1.reset()
        self.target_critic_2.reset()

        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_opt = AdamW([self.log_alpha], lr=self.config.alpha_lr)


class TQCAgent:
    def __init__(
        self, obs_dim: int, ac_dim: int, config: AgentConfig, weights: str, nenvs: int
    ):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config = config

        # Number of critics for TQC (typically 5-10)
        self.num_critics = getattr(config, "num_critics", 5)
        self.top_quantiles_to_drop = getattr(config, "top_quantiles_to_drop", 2)

        self.actor = SACActorModel(
            obs_dim, self.config.hidden_dim, ac_dim, self.config.layer_count
        ).to(self.device)

        # Create multiple critics
        self.critics = nn.ModuleList(
            [
                Critic(
                    obs_dim + ac_dim, self.config.hidden_dim, self.config.layer_count
                ).to(self.device)
                for _ in range(self.num_critics)
            ]
        )

        self.target_critics = nn.ModuleList(
            [
                Critic(
                    obs_dim + ac_dim, self.config.hidden_dim, self.config.layer_count
                ).to(self.device)
                for _ in range(self.num_critics)
            ]
        )

        self.actor_opt = AdamW(self.actor.parameters(), self.config.actor_lr)
        self.critic_opts = [
            AdamW(critic.parameters(), self.config.critic_lr) for critic in self.critics
        ]

        self.target_entropy = -ac_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_opt = AdamW([self.log_alpha], lr=self.config.alpha_lr)

        self.actor_scheduler = lr_scheduler.CosineAnnealingLR(
            self.actor_opt,
            T_max=self.config.ac_scheduler_steps,
            eta_min=self.config.actor_lr_min,
        )
        self.critic_schedulers = [
            lr_scheduler.CosineAnnealingLR(
                opt,
                T_max=self.config.cr_scheduler_steps,
                eta_min=self.config.critic_lr_min,
            )
            for opt in self.critic_opts
        ]

        if self.config.buffer_type == "PER":
            self.buffer = PERBuffer(self.config.max_len, self.config.alpha)
        elif self.config.buffer_type == "PER_SUMTREE":
            self.buffer = PERBufferSumTree(self.config.max_len, self.config.alpha)
        elif self.config.buffer_type == "REPLAY":
            self.buffer = ReplayBuffer(self.config.max_len)
        elif self.config.buffer_type == "HER":
            self.buffer = HERBuffer(
                self.config.max_len,
                self.config.max_eps_len,
                nenvs,
                k_future=self.config.k_future,
            )
        else:
            raise ValueError(
                f"[ERROR] Invalid Buffer type. Received {self.config.buffer_type}."
            )

        self.gamma = self.config.gamma
        self.batch_size = self.config.batch_size
        self.ac_update_freq = self.config.ac_update_freq
        self.grad_clip = self.config.grad_clip
        self.tau = self.config.tau

        self.beta = self.config.beta
        self.beta_start = self.config.beta
        self.beta_max = 1.0
        self.beta_end = self.config.beta_end

        if weights:
            self.actor.load(os.path.join(weights, "actor.pth"), self.device)
            for i, critic in enumerate(self.critics):
                critic_path = os.path.join(weights, f"critic_{i}.pth")
                if os.path.exists(critic_path):
                    critic.load(critic_path, self.device)
            # Load log_alpha if exists
            log_alpha_path = os.path.join(weights, "log_alpha.pth")
            if os.path.exists(log_alpha_path):
                self.log_alpha = torch.load(log_alpha_path, map_location=self.device)
                self.alpha = self.log_alpha.exp()
                self.alpha_opt = AdamW([self.log_alpha], lr=self.config.alpha_lr)
            self.update_target_network()
        else:
            self.update_target_network()

    def update_target_network(self):
        for critic, target_critic in zip(self.critics, self.target_critics):
            target_critic.load_state_dict(critic.state_dict())

    def update_critics(self, tau: float = 0.005):
        for critic, target_critic in zip(self.critics, self.target_critics):
            for target_param, param in zip(
                target_critic.parameters(), critic.parameters()
            ):
                target_param.data.copy_(
                    tau * param.data + (1 - tau) * target_param.data
                )

    def beta_scheduler(self, step: int):
        ratio = step / self.beta_end
        self.beta = min(
            self.beta_max, self.beta_start + ratio * (self.beta_max - self.beta_start)
        )

    def get_gradient_norm(self, model: torch.nn.Module):
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm**0.5
        return total_norm

    def actor_update(self, states: torch.Tensor):
        actions, log_probs = self.actor.sample(states)

        critic_inputs = torch.cat([states, actions], dim=-1)
        q_values = torch.stack([critic(critic_inputs) for critic in self.critics])

        if self.top_quantiles_to_drop > 0:
            q_values_sorted, _ = torch.sort(q_values, dim=0)
            q_values_truncated = q_values_sorted[: -self.top_quantiles_to_drop]
            min_q_values = q_values_truncated.mean(dim=0)
        else:
            min_q_values = q_values.mean(dim=0)

        actor_loss = (self.alpha.detach() * log_probs - min_q_values).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_clip)
        actor_grad_norm = self.get_gradient_norm(self.actor)
        self.actor_opt.step()
        self.actor_scheduler.step()

        return actor_loss.item(), actor_grad_norm, log_probs.detach()

    def alpha_update(self, log_probs: torch.Tensor):
        alpha_loss = -(
            self.log_alpha * (log_probs + self.target_entropy).detach()
        ).mean()

        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        self.alpha = self.log_alpha.exp()
        return alpha_loss.item()

    def critic_update(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
        weights: torch.Tensor = None,
    ):
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_state)

            # Get target Q-values from all target critics
            target_critic_inputs = torch.cat([next_state, next_actions], dim=-1)
            target_q_values = torch.stack(
                [
                    target_critic(target_critic_inputs)
                    for target_critic in self.target_critics
                ]
            )

            # Use truncated mean for target (conservative estimate)
            if self.top_quantiles_to_drop > 0:
                target_q_sorted, _ = torch.sort(target_q_values, dim=0)
                target_q_truncated = target_q_sorted[: -self.top_quantiles_to_drop]
                target_q = target_q_truncated.mean(dim=0)
            else:
                target_q = target_q_values.mean(dim=0)

            target_q = target_q - self.alpha * next_log_probs
            target = reward + self.gamma * (1 - done) * target_q

        # Update all critics
        critic_losses = []
        critic_grads = []
        all_td_errors = []

        current_critic_input = torch.cat([state, action], dim=-1)

        for i, (critic, opt, scheduler) in enumerate(
            zip(self.critics, self.critic_opts, self.critic_schedulers)
        ):
            current_q_value = critic(current_critic_input)

            opt.zero_grad()
            if weights is not None and weights.numel() > 0:
                critic_loss = torch.nn.functional.mse_loss(
                    current_q_value, target, reduction="none"
                )
                critic_loss = (weights * critic_loss).mean()
            else:
                critic_loss = torch.nn.functional.mse_loss(current_q_value, target)

            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=self.grad_clip)
            critic_grad_norm = self.get_gradient_norm(critic)
            opt.step()
            scheduler.step()

            critic_losses.append(critic_loss.item())
            critic_grads.append(critic_grad_norm)

            # Calculate TD errors for this critic
            td_error = torch.abs(current_q_value - target).detach()
            all_td_errors.append(td_error)

        # Aggregate metrics
        avg_critic_loss = np.mean(critic_losses)
        avg_critic_grad = np.mean(critic_grads)

        # Calculate average Q-value across all critics
        all_q_values = torch.stack(
            [critic(current_critic_input) for critic in self.critics]
        )
        q_value = all_q_values.mean().detach().cpu().item()

        # Use max TD error across critics for priority updates
        max_td_errors = torch.stack(all_td_errors).max(dim=0)[0]

        if weights is not None and weights.numel() > 0:
            td_error = max_td_errors.cpu().numpy()
            return (
                avg_critic_loss,
                avg_critic_loss,
                td_error,
                q_value,
                avg_critic_grad,
                avg_critic_grad,
            )
        else:
            td_error = torch.mean(max_td_errors).cpu().numpy()
            return (
                avg_critic_loss,
                avg_critic_loss,
                td_error,
                q_value,
                avg_critic_grad,
                avg_critic_grad,
            )

    def select_action(self, obs_tensor: np.array, eval_action: bool = False):
        self.set_eval()
        obs_tensor = torch.as_tensor(obs_tensor, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            action, _ = self.actor.sample(obs_tensor, deterministic=eval_action)
            return action.cpu().numpy()

    def push(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def push_her(
        self, idx, state, action, next_state, reward, done, desired_goal, achieved_goal
    ):
        self.buffer.push(
            idx, state, action, next_state, reward, done, desired_goal, achieved_goal
        )

    def update(self, step: int):
        self.set_train()
        if isinstance(self.buffer, PERBuffer) or isinstance(
            self.buffer, PERBufferSumTree
        ):
            states, actions, rewards, next_states, dones, weights, indices = (
                self.buffer.sample(self.batch_size, self.beta)
            )
            q1_loss, q2_loss, td_error, q_value, critic_1_grad, critic_2_grad = (
                self.critic_update(
                    states, actions, rewards, next_states, dones, weights
                )
            )
            self.buffer.update_priorities(indices=indices, priorities=td_error)
        else:
            states, actions, rewards, next_states, dones = self.buffer.sample(
                self.batch_size
            )
            q1_loss, q2_loss, td_error, q_value, critic_1_grad, critic_2_grad = (
                self.critic_update(states, actions, rewards, next_states, dones)
            )

        self.beta_scheduler(step)
        self.update_critics(self.tau)

        if step % self.ac_update_freq == 0:
            ac_loss, ac_grad, log_probs = self.actor_update(states)
            alpha_loss = self.alpha_update(log_probs)
            return (
                q1_loss,
                q2_loss,
                ac_loss,
                td_error,
                q_value,
                critic_1_grad,
                critic_2_grad,
                ac_grad,
                alpha_loss,
            )
        else:
            return q1_loss, q2_loss, td_error, q_value, critic_1_grad, critic_2_grad

    def save_weights(self, path: str):
        self.actor.save(os.path.join(path, "actor.pth"))
        for i, critic in enumerate(self.critics):
            critic.save(os.path.join(path, f"critic_{i}.pth"))
        torch.save(self.log_alpha, os.path.join(path, "log_alpha.pth"))

    def is_buffer_filled(self):
        return len(self.buffer) >= self.batch_size

    def set_train(self):
        self.actor.train()
        for critic in self.critics:
            critic.train()
        for target_critic in self.target_critics:
            target_critic.eval()

    def set_eval(self):
        self.actor.eval()
        for critic in self.critics:
            critic.eval()
        for target_critic in self.target_critics:
            target_critic.eval()

    def update_normalizers(self, obs_list, dg_list, normalize):
        if hasattr(self.buffer, "obs_normalizer") and obs_list and normalize:
            combined_obs = np.concatenate(obs_list, axis=0)
            self.buffer.obs_normalizer.update(combined_obs)

        if hasattr(self.buffer, "dg_normalizer") and dg_list and normalize:
            combined_dg = np.concatenate(dg_list, axis=0)
            self.buffer.dg_normalizer.update(combined_dg)

    def normalize_state_batch(self, obs_batch, dg_batch, normalize):
        if hasattr(self.buffer, "obs_normalizer"):
            normalized_obs = self.normalize_obs(obs_batch, normalize)
        else:
            normalized_obs = obs_batch

        if hasattr(self.buffer, "dg_normalizer") and normalize:
            normalized_dg = self.normalize_goal(dg_batch, normalize)
        else:
            normalized_dg = dg_batch

        return np.concatenate([normalized_obs, normalized_dg], axis=-1)

    def normalize_obs(self, obs: np.array, normalize: bool):
        if hasattr(self.buffer, "obs_normalizer") and normalize:
            normalized_obs = self.buffer.obs_normalizer.normalize(obs)
        else:
            normalized_obs = obs
        return normalized_obs

    def normalize_goal(self, goal: np.array, normalize: bool):
        if hasattr(self.buffer, "dg_normalizer") and normalize:
            normalized_obs = self.buffer.dg_normalizer.normalize(goal)
        else:
            normalized_obs = goal
        return normalized_obs

    def reset(self):
        self.actor.reset()
        for critic in self.critics:
            critic.reset()
        for target_critic in self.target_critics:
            target_critic.reset()

        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_opt = AdamW([self.log_alpha], lr=self.config.alpha_lr)


class DDPG:
    def __init__(
        self, obs_dim: int, ac_dim: int, config: AgentConfig, weights: str, nenvs: int
    ):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config = config

        self.actor = Actor(
            obs_dim, self.config.hidden_dim, ac_dim, self.config.layer_count
        ).to(self.device)
        self.target_actor = Actor(
            obs_dim, self.config.hidden_dim, ac_dim, self.config.layer_count
        ).to(self.device)
        self.critic = Critic(
            obs_dim + ac_dim, self.config.hidden_dim, self.config.layer_count
        ).to(self.device)
        self.target_critic = Critic(
            obs_dim + ac_dim, self.config.hidden_dim, self.config.layer_count
        ).to(self.device)

        self.actor_opt = AdamW(self.actor.parameters(), self.config.actor_lr)
        self.critic_opt = AdamW(self.critic.parameters(), self.config.critic_lr)
        self.actor_scheduler = lr_scheduler.CosineAnnealingLR(
            self.actor_opt,
            T_max=self.config.ac_scheduler_steps,
            eta_min=self.config.actor_lr_min,
        )
        self.critic_scheduler = lr_scheduler.CosineAnnealingLR(
            self.critic_opt,
            T_max=self.config.cr_scheduler_steps,
            eta_min=self.config.critic_lr_min,
        )

        if self.config.buffer_type == "PER":
            self.buffer = PERBuffer(self.config.max_len, self.config.alpha)
        elif self.config.buffer_type == "PER_SUMTREE":
            self.buffer = PERBufferSumTree(self.config.max_len, self.config.alpha)
        elif self.config.buffer_type == "REPLAY":
            self.buffer = ReplayBuffer(self.config.max_len)
        elif self.config.buffer_type == "HER":
            self.buffer = HERBuffer(
                self.config.max_len,
                self.config.max_eps_len,
                nenvs,
                k_future=self.config.k_future,
            )
        else:
            raise ValueError(
                f"[ERROR] Invalid Buffer type. Received {self.config.buffer_type}."
            )

        self.noise_std = self.config.noise_std
        self.gamma = self.config.gamma
        self.batch_size = self.config.batch_size
        self.ac_update_freq = self.config.ac_update_freq
        self.grad_clip = self.config.grad_clip
        self.tau = self.config.tau

        self.beta = self.config.beta
        self.beta_start = self.config.beta
        self.beta_max = 1.0
        self.beta_end = self.config.beta_end

        if weights:
            self.actor.load(os.path.join(weights, "actor.pth"), self.device)
            # Try to load from both possible names for backward compatibility
            critic_path = os.path.join(weights, "critic.pth")
            if os.path.exists(critic_path):
                self.critic.load(critic_path, self.device)
            else:
                self.critic.load(os.path.join(weights, "critic_1.pth"), self.device)
            self.update_target_network()
        else:
            self.update_target_network()

    def update_target_network(self, hard_update: bool = True, tau: float = 0.005):
        if hard_update:
            self.target_actor.load_state_dict(self.actor.state_dict())
            self.target_critic.load_state_dict(self.critic.state_dict())
        else:
            for target_param, param in zip(
                self.target_actor.parameters(), self.actor.parameters()
            ):
                target_param.data.copy_(
                    tau * param.data + (1 - tau) * target_param.data
                )

            for target_param, param in zip(
                self.target_critic.parameters(), self.critic.parameters()
            ):
                target_param.copy_(tau * param.data + (1 - tau) * target_param.data)

    def beta_scheduler(self, step: int):
        ratio = step / self.beta_end
        self.beta = min(
            self.beta_max, self.beta_start + ratio * (self.beta_max - self.beta_start)
        )

    def get_gradient_norm(self, model: torch.nn.Module):
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm**0.5
        return total_norm

    def actor_update(self, states: torch.Tensor):
        actions = self.actor(states)
        q_values = self.critic(torch.cat([states, actions], dim=-1))
        actor_loss = -q_values.mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.actor_opt.step()
        self.actor_scheduler.step()

        return actor_loss.item(), self.get_gradient_norm(self.actor)

    def critic_update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        weights: torch.Tensor = None,
    ):
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            target_q = self.target_critic(
                torch.cat([next_states, next_actions], dim=-1)
            )
            y = rewards + self.gamma * (1.0 - dones) * target_q

        current_q = self.critic(torch.cat([states, actions], dim=-1))
        td_errors = torch.abs(y - current_q).detach()

        if weights is not None and weights.numel() > 0:
            critic_loss = torch.nn.functional.mse_loss(current_q, y, reduction="none")
            critic_loss = (weights * critic_loss).mean()
        else:
            critic_loss = torch.nn.functional.mse_loss(current_q, y)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        critic_grad_norm = self.get_gradient_norm(self.critic)
        self.critic_opt.step()
        self.critic_scheduler.step()

        q_value = current_q.mean().detach().cpu().item()

        if weights is not None and weights.numel() > 0:
            td_error = td_errors.cpu().numpy()
            return critic_loss.item(), td_error, q_value, critic_grad_norm
        else:
            td_error = torch.mean(td_errors).cpu().numpy()
            return critic_loss.item(), td_error, q_value, critic_grad_norm

    def select_action(self, obs_tensor: np.array, eval_action: bool = False):
        self.set_eval()
        if not eval_action:
            obs_tensor = torch.as_tensor(obs_tensor, dtype=torch.float32).to(
                self.device
            )
            with torch.no_grad():
                action = torch.tanh(self.actor(obs_tensor)).detach().cpu().numpy()
            noise = np.random.normal(0, self.noise_std, size=action.shape)
            action = np.clip(action + noise, -1, 1)
            return action
        else:
            obs_tensor = torch.as_tensor(obs_tensor, dtype=torch.float32).to(
                self.device
            )
            with torch.no_grad():
                return np.clip(torch.tanh(self.actor(obs_tensor)).cpu().numpy(), -1, 1)

    def push(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def push_her(
        self, idx, state, action, next_state, reward, done, desired_goal, achieved_goal
    ):
        self.buffer.push(
            idx, state, action, next_state, reward, done, desired_goal, achieved_goal
        )

    def update(self, step: int):
        self.set_train()
        if isinstance(self.buffer, PERBuffer) or isinstance(
            self.buffer, PERBufferSumTree
        ):
            states, actions, rewards, next_states, dones, weights, indices = (
                self.buffer.sample(self.batch_size, self.beta)
            )
            critic_loss, td_error, q_value, critic_grad = self.critic_update(
                states, actions, rewards, next_states, dones, weights
            )
            self.buffer.update_priorities(indices=indices, priorities=td_error)
        else:
            states, actions, rewards, next_states, dones = self.buffer.sample(
                self.batch_size
            )
            critic_loss, td_error, q_value, critic_grad = self.critic_update(
                states, actions, rewards, next_states, dones
            )

        self.beta_scheduler(step)
        self.update_target_network(hard_update=False, tau=self.tau)

        if step % self.ac_update_freq == 0:
            ac_loss, ac_grad = self.actor_update(states)
            return critic_loss, ac_loss, td_error, q_value, critic_grad, ac_grad
        else:
            return critic_loss, td_error, q_value, critic_grad

    def save_weights(self, path: str):
        self.actor.save(os.path.join(path, "actor.pth"))
        self.critic.save(os.path.join(path, "critic.pth"))

    def is_buffer_filled(self):
        return len(self.buffer) >= self.batch_size

    def set_train(self):
        self.actor.train()
        self.critic.train()
        self.target_actor.eval()
        self.target_critic.eval()

    def set_eval(self):
        self.actor.eval()
        self.critic.eval()
        self.target_actor.eval()
        self.target_critic.eval()

    def update_normalizers(self, obs_list, dg_list, normalize):
        if hasattr(self.buffer, "obs_normalizer") and obs_list and normalize:
            combined_obs = np.concatenate(obs_list, axis=0)
            self.buffer.obs_normalizer.update(combined_obs)

        if hasattr(self.buffer, "dg_normalizer") and dg_list and normalize:
            combined_dg = np.concatenate(dg_list, axis=0)
            self.buffer.dg_normalizer.update(combined_dg)

    def normalize_state_batch(self, obs_batch, dg_batch, normalize):
        if hasattr(self.buffer, "obs_normalizer"):
            normalized_obs = self.normalize_obs(obs_batch, normalize)
        else:
            normalized_obs = obs_batch

        if hasattr(self.buffer, "dg_normalizer") and normalize:
            normalized_dg = self.normalize_goal(dg_batch, normalize)
        else:
            normalized_dg = dg_batch

        return np.concatenate([normalized_obs, normalized_dg], axis=-1)

    def normalize_obs(self, obs: np.array, normalize: bool):
        if hasattr(self.buffer, "obs_normalizer") and normalize:
            normalized_obs = self.buffer.obs_normalizer.normalize(obs)
        else:
            normalized_obs = obs
        return normalized_obs

    def normalize_goal(self, goal: np.array, normalize: bool):
        if hasattr(self.buffer, "dg_normalizer") and normalize:
            normalized_obs = self.buffer.dg_normalizer.normalize(goal)
        else:
            normalized_obs = goal
        return normalized_obs

    def reset(self):
        self.actor.reset()
        self.target_actor.reset()
        self.critic.reset()
        self.target_critic.reset()
