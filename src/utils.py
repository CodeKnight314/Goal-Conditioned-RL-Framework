import os
import random
import numpy as np
import torch
import gymnasium as gym
import yaml
from pydantic import BaseModel, Field


class BaseAgentConfig(BaseModel):
    hidden_dim: int = Field(..., ge=1)
    layer_count: int = Field(..., ge=1)
    actor_lr: float = Field(..., gt=0)
    actor_lr_min: float = Field(..., gt=0)
    ac_scheduler_steps: int = Field(..., ge=1)
    critic_lr: float = Field(..., gt=0)
    critic_lr_min: float = Field(..., gt=0)
    cr_scheduler_steps: int = Field(..., ge=1)
    buffer_type: str
    max_len: int = Field(..., ge=1)
    alpha: float = Field(..., ge=0)
    batch_size: int = Field(..., ge=1)
    gamma: float = Field(..., ge=0, le=1)
    ac_update_freq: int = Field(..., ge=1)
    noise_std: float = Field(..., ge=0)
    noise_clamp: float = Field(..., ge=0)
    policy_noise: float = Field(..., ge=0)
    grad_clip: float = Field(..., ge=0)
    beta: float = Field(..., ge=0)
    beta_end: int = Field(..., ge=1)
    k_future: int = Field(..., ge=0)
    max_eps_len: int = Field(..., ge=1)
    tau: float = Field(..., ge=0)


class SACAgentConfig(BaseAgentConfig):
    alpha_lr: float = Field(default=0.0003, gt=0)
    alpha_min: float = Field(default=0.05, gt=0)
    alpha_min_steps: float = Field(..., ge=0)


class Config(BaseModel):
    max_frames: int = Field(..., ge=1)
    save_freq: int = Field(..., ge=1)
    video_freq: int = Field(..., ge=1)
    window_size: int = Field(..., ge=1)
    gradient_step: int = Field(..., ge=1)
    reset_freq: int = Field(..., ge=1)
    g_normalize: bool = Field(default=True)
    obs_normalize: bool = Field(default=True)
    agent: BaseAgentConfig | SACAgentConfig


class HERConfig(BaseModel):
    max_episode: int = Field(..., ge=1)
    max_cycle: int = Field(..., ge=1)
    max_epoch: int = Field(..., ge=1)
    save_freq: int = Field(..., ge=1)
    video_freq: int = Field(..., ge=1)
    window_size: int = Field(..., ge=1)
    gradient_step: int = Field(..., ge=1)
    reset_freq: int = Field(..., ge=1)
    g_normalize: bool = Field(default=False)
    obs_normalize: bool = Field(default=True)
    agent: BaseAgentConfig | SACAgentConfig


class RunningNormalizer:
    def __init__(self, size, clip_range=5.0, eps=1e-8):
        self.mean = np.zeros(size)
        self.var = np.ones(size)
        self.count = eps
        self.clip_range = clip_range

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, mean, var, count):
        total_count = self.count + count
        delta = mean - self.mean

        new_mean = self.mean + delta * count / total_count
        m_a = self.var * self.count
        m_b = var * count
        M2 = m_a + m_b + np.square(delta) * self.count * count / total_count
        new_var = M2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def normalize(self, x):
        norm_x = (x - self.mean) / (np.sqrt(self.var) + 1e-8)
        return np.clip(norm_x, -self.clip_range, self.clip_range)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "mean": self.mean.tolist(),
            "var": self.var.tolist(),
            "count": float(self.count),
            "clip_range": float(self.clip_range),
        }
        with open(path, "w") as f:
            yaml.dump(data, f)

    def load(self, path: str):
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        self.mean = np.array(data["mean"], dtype=np.float32)
        self.var = np.array(data["var"], dtype=np.float32)
        self.count = float(data["count"])
        self.clip_range = float(data["clip_range"])


class TerminateOnAchieve(gym.Wrapper):
    def __init__(self, env, threshold: float = 0.05):
        super().__init__(env)
        self.threshold = threshold

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        terminated = self._compute_custom_termination(obs)
        return obs, reward, terminated, truncated, info

    def _compute_custom_termination(self, state):
        achieved_goals = state["achieved_goal"]
        desired_goals = state["desired_goal"]
        distances = np.linalg.norm(achieved_goals - desired_goals, axis=-1)
        return distances < self.threshold


class TimeFeatureWrapperDictObs(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        assert isinstance(env.observation_space, gym.spaces.Dict)
        assert "observation" in env.observation_space.spaces

        self.max_steps = env.spec.max_episode_steps
        self.current_step = 0

        obs_space = env.observation_space.spaces["observation"]
        assert isinstance(obs_space, gym.spaces.Box)

        low = np.append(obs_space.low, 0.0)
        high = np.append(obs_space.high, 1.0)

        self.observation_space = gym.spaces.Dict(
            {
                **env.observation_space.spaces,
                "observation": gym.spaces.Box(low=low, high=high, dtype=np.float64),
            }
        )

    def observation(self, observation):
        obs = observation.copy()
        time_feature = np.array([self.current_step / self.max_steps], dtype=np.float32)
        obs["observation"] = np.concatenate((obs["observation"], time_feature), axis=-1)
        return obs

    def step(self, action):
        self.current_step += 1
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.observation(obs), reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.current_step = 0
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info


def load_config(path: str, agent_type: str) -> Config:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if agent_type in ["SAC", "TQC"]:
        cfg["agent"] = SACAgentConfig(**cfg["agent"])
    else:
        cfg["agent"] = BaseAgentConfig(**cfg["agent"])
    return Config(**cfg)


def load_her_config(path: str, agent_type: str) -> HERConfig:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if agent_type in ["SAC", "TQC"]:
        cfg["agent"] = SACAgentConfig(**cfg["agent"])
    else:
        cfg["agent"] = BaseAgentConfig(**cfg["agent"])
    return HERConfig(**cfg)


def set_seed(seed: int, env: gym.vector.AsyncVectorEnv = None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if env is not None:
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
