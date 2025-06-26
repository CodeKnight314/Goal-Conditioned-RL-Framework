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
            [lambda: gym.make(env_id, render_mode=None) for _ in range(self.num_envs)],
            autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP
        )
        
        self.obs_dim = self.env.single_observation_space.shape[0]
        self.ac_dim = self.env.single_action_space.shape[0]
        
        self.agent = PandaAgent(
            config=self.agent_config,
            obs_dim=self.obs_dim,
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
        obs, _ = self.env.reset()  
        state = obs["observation"]
        episode_rewards = np.zeros(self.num_envs, dtype=float)
        pbar = tqdm(total=self.config["max_frames"], desc="Frames")
        
        while total_frames < self.config["max_frames"]:
            if self.agent.is_buffer_filled():
                actions = self.env.action_space.sample()
            else:
                actions = []
                for i in range(self.num_envs):
                    a_i = self.agent.select_action(state)
                    actions.append(a_i)
                    
                actions = np.array(actions, dtype=np.float32)
                
            next_obs_raw, rewards, terminateds, truncateds, _ = self.env.step(actions)
            dones = np.logical_or(terminateds, truncateds)
            
            for i in range(self.num_envs):
                obs = torch.as_tensor(state[i], dtype=torch.float32).to(self.device)
                next_obs = torch.as_tensor(next_obs_raw["observation"][i], dtype=torch.float32).to(self.device)
                
                self.agent.push(obs, actions[i], rewards[i], next_obs, dones[i])
                episode_rewards[i] += rewards[i]
                
                if dones[i]:
                    self.history["reward"].append(episode_rewards[i])
                    episode_rewards[i] = 0.0
                    
                total_frames += 1
                
                if self.agent.is_buffer_filled(): 
                    info = self.agent.update(total_frames)
                    if len(info) == 3: 
                        q1_loss, q2_loss, ac_loss = info
                        self.history["ac_loss"].append(ac_loss)
                    else: 
                        q1_loss, q2_loss = info
                    self.history["q1_loss"].append(q1_loss)
                    self.history["q2_loss"].append(q2_loss)
                
                if total_frames % self.save_freq == 0:
                    checkpoint_path = os.path.join(path, f"checkpoint")
                    self.agent.save_weights(checkpoint_path)
                    if self.verbose:
                        logger.info(f"Checkpoint saved at frame {total_frames}")
                        
            pbar.update(self.num_envs)
            
            if len(self.history["reward"]) >= self.config["window_size"]:
                recent_reward_avg = np.mean(self.history["reward"])
                if recent_reward_avg > self.best_reward:
                    self.best_reward = recent_reward_avg
                    self.agent.save_weights(os.path.join(path, "best_weights"))
                    self.test(os.path.join(path, "video"), num_episodes=1)

                    if self.verbose:
                        logger.info(f"New best model saved! Average reward: {recent_reward_avg:.2f}")

            state = next_obs_raw["observation"]

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
                    
    def test(self, path: str, num_episodes: int):
        os.makedirs(path, exist_ok=True)

        env = gym.make(self.env_id, render_mode="rgb_array")
        self.agent.actor.eval()

        state, _ = env.reset()
        frame = env.render()
        height, width, _ = frame.shape

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_path = os.path.join(path, "panda_manipulation.mp4")
        video = cv2.VideoWriter(video_path, fourcc, 60, (width, height))

        total_rewards = 0
        total_steps = 0

        for i in range(num_episodes):
            state, _ = env.reset()
            done = False
            rewards = 0
            steps = 0
            
            while not done:
                frame = env.render()

                action = self.agent.select_action(state["observation"], eval_action=True)
                video.write(frame)

                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                rewards += reward
                steps += 1

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
