import os
import logging
from collections import deque
from typing import Optional, Any
import cv2
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import wandb
import random
import panda_gym
from src.agent import TD3Agent, SACAgent, DDPG, TQCAgent
from src.buffer import HERBuffer
from src.utils import (
    load_config,
    load_her_config,
    set_seed,
    RunningNormalizer,
    TimeFeatureWrapperDictObs,
)


class GoalEnvHER:
    """A single RL training interface that transparently supports both
    *standard* and *Hindsight Experience Replay (HER)* pipelines.

    The class automatically detects whether the provided configuration (or the
    optional ``her`` flag) requires HER and switches to the appropriate
    training loop and buffer interactions.
    """

    def __init__(
        self,
        env_id: str,
        seed: int,
        config: str,
        num_envs: int,
        *,
        weights: Optional[str] = None,
        verbose: bool = True,
        use_wandb: bool = True,
        agent_type: str = "TD3",
        her: Optional[bool] = None,
    ) -> None:
        base_config = load_config(config, agent_type)
        self.agent_config = base_config.agent
        self.is_her = her if her is not None else self.agent_config.buffer_type == "HER"

        if self.is_her:
            her_cfg = load_her_config(config, agent_type)
            self.max_episode = her_cfg.max_episode
            self.max_cycle = her_cfg.max_cycle
            self.max_epoch = her_cfg.max_epoch
            self.config = her_cfg
        else:
            self.config = base_config

        self.num_envs = num_envs
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.save_freq = self.config.save_freq
        self.video_freq = self.config.video_freq
        self.gradient_step = self.config.gradient_step
        self.reset_freq = self.config.reset_freq
        self.verbose = verbose
        self.best_reward = 0.0
        self.grad_counter = 1
        self.env_id = env_id
        self.seed = seed
        self.use_wandb = use_wandb
        self.agent_type = agent_type
        self.g_normalize = self.config.g_normalize
        self.obs_normalize = self.config.obs_normalize

        self.env = gym.vector.AsyncVectorEnv(
            [
                lambda: TimeFeatureWrapperDictObs(gym.make(env_id))
                for _ in range(self.num_envs)
            ],
            autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP,
        )

        obs_space = self.env.observation_space
        act_space = self.env.action_space
        self.ag_dim = obs_space["achieved_goal"].shape[1]
        self.dg_dim = obs_space["desired_goal"].shape[1]
        self.obs_dim = obs_space["observation"].shape[1]
        self.ac_dim = act_space.shape[1]

        self.agent = self._build_agent(weights)

        if hasattr(self.agent.buffer, "obs_normalizer"):
            self.agent.buffer.obs_normalizer = RunningNormalizer(size=self.obs_dim)
            self.agent.buffer.dg_normalizer = RunningNormalizer(size=self.dg_dim)
            if weights:
                self.agent.buffer.obs_normalizer.load(os.path.join(weights, "obs.yaml"))
                self.agent.buffer.dg_normalizer.load(os.path.join(weights, "dg.yaml"))

        if self.is_her:
            assert isinstance(self.agent.buffer, HERBuffer), (
                "[ERROR] HER requested but agent buffer is not HERBuffer. "
                "Set buffer_type = 'HER' in config or pass her=False."
            )
            self.agent.buffer.compute_reward = gym.make(env_id).unwrapped.compute_reward

        self._init_history()

        set_seed(self.seed, self.env)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.logger = logging.getLogger(__name__)

    def _build_agent(self, weights: Optional[str]):
        """Factory for the selected agent type."""
        common_kwargs = dict(
            config=self.agent_config,
            obs_dim=self.obs_dim + self.dg_dim,
            ac_dim=self.ac_dim,
            weights=weights,
            nenvs=self.num_envs,
            gradient_step=self.gradient_step,
        )
        if self.agent_type == "SAC":
            return SACAgent(**common_kwargs)
        if self.agent_type == "TQC":
            return TQCAgent(**common_kwargs)
        if self.agent_type == "DDPG":
            return DDPG(**common_kwargs)
        return TD3Agent(**common_kwargs)

    def _init_history(self):
        tpl = dict(maxlen=self.config.window_size)
        self.history = {
            "reward": deque(**tpl),
            "q_value": deque(**tpl),
            "td_error": deque(**tpl),
            "success_rate": deque(**tpl),
            "q1_loss": deque(**tpl),
            "q2_loss": deque(**tpl),
            "ac_loss": deque(**tpl),
            "critic_1_grad": deque(**tpl),
            "critic_2_grad": deque(**tpl),
            "actor_grad": deque(**tpl),
            "alpha_loss": deque(**tpl),
            "reward_history": [],
            "q_value_history": [],
            "td_error_history": [],
            "success_rate_history": [],
            "q1_loss_history": [],
            "q2_loss_history": [],
            "ac_loss_history": [],
            "critic_1_grad_history": [],
            "critic_2_grad_history": [],
            "actor_grad_history": [],
            "alpha_loss_history": [],
        }

    def _process_step(self, state, actions, next_obs_raw, rewards, dones):
        """Normalise observations/goals & push transitions to replay."""
        self.agent.update_normalizers(
            [state["observation"], next_obs_raw["observation"]],
            [
                state["desired_goal"],
                next_obs_raw["desired_goal"],
                state["achieved_goal"],
                next_obs_raw["achieved_goal"],
            ],
            obs_normalize=self.obs_normalize,
            g_normalize=self.g_normalize,
        )

        obs_np = self.agent.normalize_state_batch(
            state["observation"],
            state["desired_goal"],
            self.obs_normalize,
            self.g_normalize,
        )
        next_np = self.agent.normalize_state_batch(
            next_obs_raw["observation"],
            next_obs_raw["desired_goal"],
            self.obs_normalize,
            self.g_normalize,
        )
        obs_batch = torch.from_numpy(obs_np).float().to(self.device)
        next_obs_batch = torch.from_numpy(next_np).float().to(self.device)

        for i in range(self.num_envs):
            self._push_to_buffer(
                i,
                obs_batch,
                actions,
                rewards,
                next_obs_batch,
                dones,
                next_obs_raw,
            )

    def _push_to_buffer(
        self,
        i: int,
        obs_batch: torch.Tensor,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs_batch: torch.Tensor,
        dones: np.ndarray,
        state: Any,
    ) -> None:
        """Abstracted buffer interaction. Chooses HER or standard push."""
        if self.is_her:
            self.agent.push_her(
                i,
                obs_batch[i].detach(),
                actions[i],
                next_obs_batch[i].detach(),
                rewards[i],
                dones[i],
                self.agent.normalize_goal(state["desired_goal"][i], self.g_normalize),
                self.agent.normalize_goal(state["achieved_goal"][i], self.g_normalize),
            )
        else:
            self.agent.push(
                obs_batch[i].detach(),
                actions[i],
                rewards[i],
                next_obs_batch[i].detach(),
                dones[i],
            )

    def warmup(self, warmup_steps: int = 200_000):
        """Fill replay buffer with random policy interactions."""
        if self.is_her:
            self.logger.warning("Warm‑up skipped – HER collects on‑policy episodes.")
            return
        self.logger.info(f"Starting warmup process. Total samples: {warmup_steps}")
        state, _ = self.env.reset()
        iterations = warmup_steps // self.num_envs
        with tqdm(total=warmup_steps, desc="Warm Up (samples)") as pbar:
            for _ in range(iterations):
                actions = self.env.action_space.sample()
                next_obs_raw, rewards, terminateds, truncateds, _ = self.env.step(
                    actions
                )
                self._process_step(state, actions, next_obs_raw, rewards, terminateds)
                state = next_obs_raw
                pbar.update(self.num_envs)

    def train(self, path: str, callback=None):
        """Dispatch to standard or HER training routine."""
        if self.use_wandb:
            wandb.init(
                project=f"FetchAgents-{self.agent_type}_{self.env_id}",
                name=f"{self.env_id}_{self.agent_type}_seed_{self.seed}_{random.randint(1000, 9999)}",
                tags=[
                    self.env_id,
                    self.agent_type,
                    "HER" if self.is_her else "Standard",
                ],
            )
        if self.is_her:
            return self._train_her(path, callback)
        return self._train_standard(path)

    def _train_standard(self, path: str):
        self._log_start("Standard", path)
        os.makedirs(path, exist_ok=True)

        total_frames = 0
        state, _ = self.env.reset()
        episode_rewards = np.zeros(self.num_envs, dtype=float)
        pbar = tqdm(total=self.config.max_frames, desc="Frames")

        while total_frames < self.config.max_frames:
            state_input = self.agent.normalize_state_batch(
                state["observation"],
                state["desired_goal"],
                self.obs_normalize,
                self.g_normalize,
            )
            actions = self.agent.select_action(state_input)
            actions = np.asarray(actions, dtype=np.float32)

            next_obs_raw, rewards, terminateds, truncateds, _ = self.env.step(actions)
            dones = np.logical_or(terminateds, truncateds)
            self._process_step(state, actions, next_obs_raw, rewards, terminateds)

            for i in range(self.num_envs):
                episode_rewards[i] += rewards[i]
                if dones[i]:
                    self.history["reward"].append(episode_rewards[i])
                    episode_rewards[i] = 0.0
                total_frames += 1

                if total_frames % self.save_freq == 0:
                    self._checkpoint(path, total_frames)
                if total_frames % self.video_freq == 0:
                    self.test(os.path.join(path, "checkpoint"), 10)

            if self.agent.is_buffer_filled():
                for _ in range(self.gradient_step):
                    self._update_and_log()

            success_rate = np.mean(
                np.linalg.norm(
                    next_obs_raw["achieved_goal"] - next_obs_raw["desired_goal"], axis=1
                )
                < 0.05
            )
            self.history["success_rate"].append(success_rate)
            pbar.update(self.num_envs)

            self._track_best(path)

            self._append_global_metrics()

            if self.use_wandb and total_frames % 1000 == 0:
                self._wandb_log_standard(total_frames)

            pbar.set_postfix(
                reward=f"{self.history['reward_history'][-1]:.4f}",
                ac_loss=f"{self.history['ac_loss_history'][-1]:.4f}",
                success_rate=f"{success_rate}",
            )
            state = next_obs_raw

        pbar.close()
        self._finalise_training(path)
        return float(np.mean(np.array(self.history["success_rate_history"])))

    def _train_her(self, path: str, callback):
        self._log_start("HER", path)
        os.makedirs(path, exist_ok=True)

        state, _ = self.env.reset()
        episode_rewards = np.zeros(self.num_envs, dtype=float)
        total_frames = 0

        for epoch in tqdm(range(1, self.max_epoch + 1), desc="Epoch", position=0):
            for cycle in tqdm(
                range(1, self.max_cycle + 1), desc="Cycle", position=1, leave=False
            ):
                episode_count = 0
                while episode_count < self.max_episode:
                    state_input = self.agent.normalize_state_batch(
                        state["observation"],
                        state["desired_goal"],
                        self.obs_normalize,
                        self.g_normalize,
                    )
                    actions = self.agent.select_action(state_input)
                    actions = np.asarray(actions, dtype=np.float32)

                    next_obs_raw, rewards, terminateds, truncateds, _ = self.env.step(
                        actions
                    )
                    dones = np.logical_or(terminateds, truncateds)
                    success_rate = np.mean(
                        np.linalg.norm(
                            next_obs_raw["achieved_goal"]
                            - next_obs_raw["desired_goal"],
                            axis=1,
                        )
                        < 0.05
                    )
                    self.history["success_rate"].append(success_rate)

                    self._process_step(
                        state, actions, next_obs_raw, rewards, terminateds
                    )

                    for i in range(self.num_envs):
                        episode_rewards[i] += rewards[i]
                        total_frames += 1
                        if dones[i]:
                            self.history["reward"].append(episode_rewards[i])
                            episode_rewards[i] = 0.0
                            episode_count += 1
                    state = next_obs_raw

                for _ in range(self.gradient_step):
                    self._update_and_log()

                self._append_global_metrics()
                self._track_best(path)

                if self.use_wandb:
                    self._wandb_log_her(epoch, cycle)

            if epoch % self.save_freq == 0:
                self._checkpoint(path, epoch, epoch_based=True)
            if epoch % self.video_freq == 0:
                self.test(os.path.join(path, "checkpoint"), 10)
            if epoch % self.reset_freq == 0:
                self.agent.reset()
                if self.verbose:
                    self.logger.info("Resetting network weights (soft).")

            if callback is not None:
                callback(epoch, np.mean(np.array(self.history["success_rate_history"])))

        self._finalise_training(path)
        return float(np.mean(np.array(self.history["success_rate_history"])))

    def _log_start(self, mode: str, path: str):
        self.logger.info(f"Starting {mode} training. Weights → {path}")
        self.logger.info(f"Environment: {self.env_id} | Action Space: {self.ac_dim}")
        if not self.is_her:
            self.logger.info(
                f"Update:Sample ratio ≈ {(1 / ((self.num_envs) / self.gradient_step)):.4f}:1"
            )
            est_grad = (
                round(self.config.max_frames / self.num_envs) * self.gradient_step
            )
            est_samples = est_grad * self.agent_config.batch_size
            self.logger.info(f"Expected Gradient Steps: {est_grad}")
            self.logger.info(f"Expected Samples Read: {est_samples}")
        else:
            self.logger.info(
                f"Expected Gradient Steps: {self.max_epoch * self.max_cycle * self.gradient_step}"
            )
            self.logger.info(
                f"Expected Episodes Read: {self.max_epoch * self.max_cycle * self.max_episode}"
            )
        self.logger.info(f"Seed: {self.seed}")

    def _checkpoint(self, path: str, step: int, epoch_based: bool = False):
        ckpt_dir = os.path.join(path, "checkpoint")
        self.agent.save_weights(ckpt_dir)
        self.plot_history(ckpt_dir)
        if hasattr(self.agent.buffer, "obs_normalizer"):
            self.agent.buffer.obs_normalizer.save(os.path.join(ckpt_dir, "obs.yaml"))
        if hasattr(self.agent.buffer, "dg_normalizer"):
            self.agent.buffer.dg_normalizer.save(os.path.join(ckpt_dir, "dg.yaml"))
        if self.verbose:
            unit = "epoch" if epoch_based else "frame"
            self.logger.info(f"Checkpoint saved at {unit} {step}")

    def _update_and_log(self):
        """Run one update of the agent and write raw metrics to trackers."""
        info = self.agent.update(step=self.grad_counter)

        q1_loss = q2_loss = ac_loss = td_error = q_value = 0.0
        critic_1_grad = critic_2_grad = ac_grad = alpha_loss = 0.0
        if self.agent_type in {"SAC", "TQC"}:
            if len(info) == 9:
                (
                    q1_loss,
                    q2_loss,
                    ac_loss,
                    td_error,
                    q_value,
                    critic_1_grad,
                    critic_2_grad,
                    ac_grad,
                    alpha_loss,
                ) = info
            else:
                (
                    q1_loss,
                    q2_loss,
                    td_error,
                    q_value,
                    critic_1_grad,
                    critic_2_grad,
                ) = info
        elif self.agent_type == "TD3":
            if len(info) == 8:
                (
                    q1_loss,
                    q2_loss,
                    ac_loss,
                    td_error,
                    q_value,
                    critic_1_grad,
                    critic_2_grad,
                    ac_grad,
                ) = info
            else:
                (
                    q1_loss,
                    q2_loss,
                    td_error,
                    q_value,
                    critic_1_grad,
                    critic_2_grad,
                ) = info
        elif self.agent_type == "DDPG":
            if len(info) == 6:
                (
                    critic_loss,
                    ac_loss,
                    td_error,
                    q_value,
                    critic_grad,
                    ac_grad,
                ) = info
                q1_loss = critic_loss
                critic_1_grad = critic_grad
            else:
                critic_loss, td_error, q_value, critic_grad = info
                q1_loss = critic_loss
                critic_1_grad = critic_grad
        else:
            raise ValueError(f"Unknown agent_type {self.agent_type}")

        self.history["q1_loss"].append(q1_loss)
        self.history["q2_loss"].append(q2_loss)
        self.history["td_error"].append(td_error)
        self.history["q_value"].append(q_value)
        self.history["critic_1_grad"].append(critic_1_grad)
        self.history["critic_2_grad"].append(critic_2_grad)
        self.history["ac_loss"].append(ac_loss)
        self.history["actor_grad"].append(ac_grad)
        self.history["alpha_loss"].append(alpha_loss)
        self.grad_counter += 1

    def _append_global_metrics(self):
        """Collapse windowed deques into scalar series for plotting."""

        def mean_or_zero(key: str):
            return np.mean(self.history[key]) if len(self.history[key]) > 0 else 0.0

        self.history["reward_history"].append(mean_or_zero("reward"))
        self.history["ac_loss_history"].append(mean_or_zero("ac_loss"))
        self.history["q1_loss_history"].append(mean_or_zero("q1_loss"))
        self.history["q2_loss_history"].append(mean_or_zero("q2_loss"))
        self.history["td_error_history"].append(mean_or_zero("td_error"))
        self.history["q_value_history"].append(mean_or_zero("q_value"))
        self.history["success_rate_history"].append(mean_or_zero("success_rate"))
        self.history["critic_1_grad_history"].append(mean_or_zero("critic_1_grad"))
        self.history["critic_2_grad_history"].append(mean_or_zero("critic_2_grad"))
        self.history["actor_grad_history"].append(mean_or_zero("actor_grad"))
        self.history["alpha_loss_history"].append(mean_or_zero("alpha_loss"))

    def _track_best(self, path: str):
        if len(self.history["reward"]) >= self.config.window_size:
            recent_reward_avg = np.mean(self.history["reward"])
            if recent_reward_avg > self.best_reward:
                self.best_reward = recent_reward_avg
                self.agent.save_weights(os.path.join(path, "best_weights"))
                self.test(os.path.join(path, "video"), num_episodes=1)
                if self.verbose:
                    self.logger.info(
                        f"New best model saved! Avg reward: {recent_reward_avg:.2f}"
                    )

    def _wandb_log_standard(self, total_frames: int):
        best_reward_window = (
            max(self.history["reward"]) if self.history["reward"] else 0.0
        )
        log_dict = {
            "reward": self.history["reward_history"][-1],
            "q1_loss": self.history["q1_loss_history"][-1],
            "q2_loss": self.history["q2_loss_history"][-1],
            "actor_loss": self.history["ac_loss_history"][-1],
            "td_error": self.history["td_error_history"][-1],
            "q_value": self.history["q_value_history"][-1],
            "frames": total_frames,
            "best_reward": best_reward_window,
            "success_rate": self.history["success_rate_history"][-1],
            "critic_1_grad": self.history["critic_1_grad_history"][-1],
            "critic_2_grad": self.history["critic_2_grad_history"][-1],
            "actor_grad": self.history["actor_grad_history"][-1],
        }
        if self.agent_type == "SAC":
            log_dict.update(
                {
                    "alpha_loss": self.history["alpha_loss_history"][-1],
                    "alpha": (
                        self.agent.alpha.item() if hasattr(self.agent, "alpha") else 0.0
                    ),
                }
            )
        wandb.log(log_dict, step=total_frames)

    def _wandb_log_her(self, epoch: int, cycle: int):
        best_reward_window = (
            max(self.history["reward"]) if self.history["reward"] else 0.0
        )
        log_dict = {
            "reward": self.history["reward_history"][-1],
            "actor_loss": self.history["ac_loss_history"][-1],
            "td_error": self.history["td_error_history"][-1],
            "q_value": self.history["q_value_history"][-1],
            "success_rate": self.history["success_rate_history"][-1],
            "best_reward": best_reward_window,
            "epoch": epoch,
            "cycle": cycle,
            "actor_grad": self.history["actor_grad_history"][-1],
        }
        if self.agent_type in {"SAC", "TQC"}:
            log_dict.update(
                {
                    "q1_loss": self.history["q1_loss_history"][-1],
                    "q2_loss": self.history["q2_loss_history"][-1],
                    "critic_1_grad": self.history["critic_1_grad_history"][-1],
                    "critic_2_grad": self.history["critic_2_grad_history"][-1],
                    "alpha_loss": self.history["alpha_loss_history"][-1],
                    "alpha": (
                        self.agent.alpha.item() if hasattr(self.agent, "alpha") else 0.0
                    ),
                }
            )
        elif self.agent_type == "TD3":
            log_dict.update(
                {
                    "q1_loss": self.history["q1_loss_history"][-1],
                    "q2_loss": self.history["q2_loss_history"][-1],
                    "critic_1_grad": self.history["critic_1_grad_history"][-1],
                    "critic_2_grad": self.history["critic_2_grad_history"][-1],
                }
            )
        elif self.agent_type == "DDPG":
            log_dict.update(
                {
                    "critic_loss": self.history["q1_loss_history"][-1],
                    "critic_grad": self.history["critic_1_grad_history"][-1],
                }
            )
        wandb.log(log_dict, step=(epoch - 1) * self.max_cycle + cycle)

    def _finalise_training(self, path: str):
        self.logger.info("Training complete. Saving final weights…")
        final_dir = os.path.join(path, "final")
        self.agent.save_weights(final_dir)
        self.plot_history(final_dir)
        if hasattr(self.agent.buffer, "obs_normalizer"):
            self.agent.buffer.obs_normalizer.save(os.path.join(final_dir, "obs.yaml"))
        if hasattr(self.agent.buffer, "dg_normalizer"):
            self.agent.buffer.dg_normalizer.save(os.path.join(final_dir, "dg.yaml"))
        self.logger.info(f"Final weights stored at '{final_dir}'")

    def test(self, path: str, num_episodes: int):
        """Render policies to MP4; identical to original implementation."""
        os.makedirs(path, exist_ok=True)
        env = TimeFeatureWrapperDictObs(gym.make(self.env_id, render_mode="rgb_array"))
        self.agent.actor.eval()
        state, _ = env.reset()
        frame = env.render()
        height, width, _ = frame.shape
        video = cv2.VideoWriter(
            os.path.join(path, "panda_manipulation.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            20,
            (width, height),
        )
        total_rewards = total_steps = 0
        for ep in range(num_episodes):
            state, _ = env.reset()
            done, rewards, steps = False, 0, 0
            while not done:
                frame = env.render()
                normalized_state = self.agent.normalize_state_batch(
                    state["observation"].reshape(1, -1),
                    state["desired_goal"].reshape(1, -1),
                    self.obs_normalize,
                    self.g_normalize,
                )
                action = self.agent.select_action(
                    normalized_state, eval_action=True
                ).squeeze(0)
                video.write(frame)
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                rewards += reward
                steps += 1
                if done:
                    for _ in range(20):
                        video.write(frame)
            if self.verbose:
                self.logger.info(
                    f"Episode {ep + 1} – Reward: {rewards:.2f} – Steps: {steps}"
                )
            total_rewards += rewards
            total_steps += steps
        avg_reward = total_rewards / num_episodes
        avg_steps = total_steps / num_episodes
        if self.verbose:
            self.logger.info(
                f"Average reward: {avg_reward:.2f} | Average steps: {avg_steps:.2f}"
            )
            self.logger.info(
                f"Video saved → {os.path.join(path, 'panda_manipulation.mp4')}"
            )
        video.release()
        del env

    def close(self):
        self.env.close()
        del self.agent
        torch.cuda.empty_cache()
        if self.use_wandb:
            wandb.finish()

    def plot_history(self, path: str):
        os.makedirs(path, exist_ok=True)

        plt.figure(figsize=(8, 5))
        plt.plot(self.history["reward_history"], label="Episode Reward")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Reward Curve")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(path, "reward_curve.png"))
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(self.history["q1_loss_history"], label="Q1 Loss")
        plt.plot(self.history["q2_loss_history"], label="Q2 Loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Critic Losses")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(path, "critic_losses.png"))
        plt.close()

        if len(self.history["ac_loss_history"]):
            plt.figure(figsize=(8, 5))
            plt.plot(self.history["ac_loss_history"], label="Actor Loss")
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.title("Actor Loss")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(path, "actor_loss.png"))
            plt.close()
        if self.verbose:
            self.logger.info(f"Training plots saved to '{path}'")
