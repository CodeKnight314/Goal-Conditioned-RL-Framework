import torch 
import gymnasium as gym
from src.agent import PandaAgent
import yaml
import os
import logging
from tqdm import tqdm
import panda_gym
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import cv2

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class PandasEnv(): 
    def __init__(self, env_id: str, config: str, num_envs: int, weights: str = None, verbose: bool = True):
        with open(config, 'r') as f: 
            self.config = yaml.safe_load(f)
        
        self.agent_config = self.config["agent"]
        self.num_envs = num_envs
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.save_freq = self.config["save_freq"]
        self.verbose = verbose
        self.best_reward = 0.0
        self.env_id = env_id
        
        self.env = gym.vector.AsyncVectorEnv(
            [lambda: gym.make(env_id) for _ in range(self.num_envs)],
            autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP
        )
        
        obs_space = self.env.observation_space
        act_space = self.env.action_space
        
        ag_dim = obs_space['achieved_goal'].shape[1]
        dg_dim = obs_space['desired_goal'].shape[1]
        obs_dim = obs_space['observation'].shape[1]
        
        ac_dim = act_space.shape[1]
        
        self.obs_dim = obs_dim
        self.ag_dim = ag_dim
        self.dg_dim = dg_dim
        self.ac_dim = ac_dim
        
        self.agent = PandaAgent(
            config=self.agent_config,
            obs_dim=self.obs_dim + self.dg_dim + self.ag_dim,
            ac_dim=self.ac_dim,
            weights=weights
        )
        
        self.history= {
            "reward": deque(maxlen=self.config["window_size"]),
            "q1_loss": deque(maxlen=self.config["window_size"]),
            "q2_loss": deque(maxlen=self.config["window_size"]),
            "ac_loss": deque(maxlen=self.config["window_size"]),
        }
        
    def set_seed(self, seed: int):
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    def train(self, path: str):
        logger.info(f"Starting training process. Model will be saved to: {path}")
        os.makedirs(path, exist_ok=True)
        
        total_frames = 0
        state, _ = self.env.reset()  
        episode_rewards = np.zeros(self.num_envs, dtype=float)
        pbar = tqdm(total=self.config["max_frames"], desc="Frames")
        
        while total_frames < self.config["max_frames"]:
            if not self.agent.is_buffer_filled():
                actions = self.env.action_space.sample()
            else:
                state_input = np.concatenate([state["observation"], state["desired_goal"], state["achieved_goal"]], axis=-1)
                actions = self.agent.select_action(state_input)
                actions = np.array(actions, dtype=np.float32)
                
            next_obs_raw, rewards, terminateds, truncateds, _ = self.env.step(actions)
            dones = np.logical_or(terminateds, truncateds)
            obs_np = np.concatenate([state["observation"], state["desired_goal"], state["achieved_goal"]], axis=-1)
            next_np = np.concatenate([next_obs_raw["observation"], next_obs_raw["desired_goal"], next_obs_raw["achieved_goal"]], axis=-1)

            obs_batch = torch.from_numpy(obs_np).float().to(self.device)
            next_obs_batch = torch.from_numpy(next_np).float().to(self.device)
            
            for i in range(self.num_envs):
                self.agent.push(
                    obs_batch[i],
                    actions[i],
                    rewards[i],
                    next_obs_batch[i],
                    dones[i],
                )
                episode_rewards[i] += rewards[i]
                
                if dones[i]:
                    self.history["reward"].append(episode_rewards[i])
                    episode_rewards[i] = 0.0
                    
                total_frames += 1
                
                if total_frames % self.save_freq == 0:
                    self.agent.save_weights(os.path.join(path, f"checkpoint"))
                    if self.verbose:
                        logger.info(f"Checkpoint saved at frame {total_frames}")
                        
            if self.agent.is_buffer_filled(): 
                info = self.agent.update(step=total_frames//self.num_envs)
                if len(info) == 3: 
                    q1_loss, q2_loss, ac_loss = info
                    self.history["ac_loss"].append(ac_loss)
                else: 
                    q1_loss, q2_loss = info
                self.history["q1_loss"].append(q1_loss)
                self.history["q2_loss"].append(q2_loss)
                        
            pbar.update(self.num_envs)
            
            if len(self.history["reward"]) >= self.config["window_size"]:
                recent_reward_avg = np.mean(self.history["reward"])
                if recent_reward_avg > self.best_reward:
                    self.best_reward = recent_reward_avg
                    self.agent.save_weights(os.path.join(path, "best_weights"))
                    self.test(os.path.join(path, "video"), num_episodes=1)

                    if self.verbose:
                        logger.info(f"New best model saved! Average reward: {recent_reward_avg:.2f}")

            state = next_obs_raw

            pbar_rewards = np.mean(self.history['reward']) if len(self.history["reward"]) > 0 else 0.0
            pbar_loss = np.mean(self.history['ac_loss']) if len(self.history["ac_loss"]) > 0 else 0.0
            pbar.set_postfix(
                reward=f"{pbar_rewards:.4f}", 
                loss=f"{pbar_loss:.4f}", 
            )
        
        pbar.close()
        logger.info("Training completed. Saving final model weights...")
        self.agent.save_weights(os.path.join(path, "final"))
        logger.info(f"Final model weights saved to: {os.path.join(path, 'final')}")
        
        return np.mean(self.history['reward'])
                    
    def test(self, path: str, num_episodes: int, smooth: bool = True, alpha: float = 0.65):
        os.makedirs(path, exist_ok=True)

        env = gym.make(self.env_id, render_mode="rgb_array", render_width=1280, render_height=720)
        self.agent.actor.eval()

        state, _ = env.reset()
        frame = env.render()
        height, width, _ = frame.shape

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_path = os.path.join(path, "panda_manipulation.mp4")
        FPS = 20
        video = cv2.VideoWriter(video_path, fourcc, FPS, (width, height))

        total_rewards = 0
        total_steps = 0

        for i in range(num_episodes):
            state, _ = env.reset()
            prev_action = None
            done = False
            rewards = 0
            steps = 0
            
            while not done:
                frame = env.render()

                action = self.agent.select_action(
                    np.concatenate([state["observation"], state["desired_goal"]], axis=-1), 
                    eval_action=True)

                if smooth and prev_action is not None: 
                    action = alpha * prev_action + (1 - alpha) * action
                video.write(frame)

                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                prev_action = action
                rewards += reward
                steps += 1
                
                if done: 
                    for i in range(FPS):
                        video.write(frame)

            if self.verbose:
                logger.info(f"Episode {i + 1} - Reward: {rewards:.2f} - Steps: {steps}")
                
            total_rewards += rewards    
            total_steps += steps

        avg_reward = total_rewards / num_episodes
        avg_steps = total_steps / num_episodes

        if self.verbose:
            logger.info(f"Average reward: {avg_reward:.2f} - Average steps: {avg_steps:.2f}")

        video.release()
        if self.verbose:
            logger.info(f"Video saved to: {video_path}")
        del env

    def close(self):
        self.env.close() 
        del self.agent
        torch.cuda.empty_cache()

    def plot_history(self, path: str):
        os.makedirs(path, exist_ok=True)

        plt.figure(figsize=(8, 5))
        plt.plot(self.history["reward"], label="Episode Reward")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Reward Curve")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(path, "reward_curve.png"))
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(self.history["q1_loss"], label="Q1 Loss")
        plt.plot(self.history["q2_loss"], label="Q2 Loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Critic Losses (Q1 & Q2)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(path, "critic_losses.png"))
        plt.close()

        if len(self.history["ac_loss"]) > 0:
            plt.figure(figsize=(8, 5))
            plt.plot(self.history["ac_loss"], label="Actor Loss")
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.title("Actor Loss")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(path, "actor_loss.png"))
            plt.close()

        if self.verbose:
            logger.info(f"Training plots saved to: {path}")