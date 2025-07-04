import random
import numpy as np
import torch
import gymnasium as gym
import yaml
from pydantic import BaseModel, Field

class AgentConfig(BaseModel):
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
    noise_std_min: float = Field(..., ge=0)
    noise_std_end: int = Field(..., ge=1)
    noise_clamp: float = Field(..., ge=0)
    noise_clamp_min: float = Field(..., ge=0)
    noise_clamp_end: int = Field(..., ge=1)
    policy_noise: float = Field(..., ge=0)
    smooth_coeff: float = Field(..., ge=0)
    grad_clip: float = Field(..., ge=0)
    beta: float = Field(..., ge=0)
    beta_end: int = Field(..., ge=1)
    k_future: int = Field(..., ge=0)
    max_eps_len: int = Field(..., ge=1)

class Config(BaseModel):
    max_frames: int = Field(..., ge=1)
    save_freq: int = Field(..., ge=1)
    video_freq: int = Field(..., ge=1)
    window_size: int = Field(..., ge=1)
    gradient_step: int = Field(..., ge=1)
    reset_freq: int = Field(..., ge=1)
    agent: AgentConfig

def load_config(path: str) -> Config:
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return Config(**cfg)

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
