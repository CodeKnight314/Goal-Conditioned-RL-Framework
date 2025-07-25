import torch 
import gymnasium as gym
import random
from src.agent import TD3Agent, SACAgent
from src.buffer import HERBuffer
import os
import logging
from tqdm import tqdm
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import cv2
import wandb
from src.utils import load_config, set_seed, load_her_config, RunningNormalizer, TerminateOnAchieve
import gymnasium_robotics

gym.register_envs(gymnasium_robotics)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class PandasEnv(): 
    def __init__(self, env_id: str, seed: int, config: str, num_envs: int, weights: str = None, verbose: bool = True, use_wandb: bool = True, agent_type: str = "TD3"):
        self.config = load_config(config)
        
        self.agent_config = self.config.agent
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
        
        if self.use_wandb:
            wandb.init(
                project="panda-manipulation-rl",
                name=f"{env_id}_{agent_type}_seed_{seed}",
                tags=[env_id, agent_type, "HER" if self.agent_config.buffer_type == "HER" else "Standard"]
            )
        
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
        
        if self.agent_type == "SAC":
            self.agent = SACAgent(
                config=self.agent_config,
                obs_dim=self.obs_dim + self.dg_dim,
                ac_dim=self.ac_dim,
                weights=weights,
                nenvs=self.num_envs,
            )
        else:
            self.agent = TD3Agent(
                config=self.agent_config,
                obs_dim=self.obs_dim + self.dg_dim,
                ac_dim=self.ac_dim,
                weights=weights,
                nenvs=self.num_envs,
            )
        
        if hasattr(self.agent.buffer, 'obs_normalizer'):
            self.agent.buffer.obs_normalizer = RunningNormalizer(size=self.obs_dim)
            if weights: 
                self.agent.buffer.obs_normalizer.load(os.path.join(weights, "obs.yaml"))
        
        self.history= {
            "reward": deque(maxlen=self.config.window_size),
            "q_value": deque(maxlen=self.config.window_size),
            "td_error": deque(maxlen=self.config.window_size),
            "success_rate": deque(maxlen=self.config.window_size),
            "q1_loss": deque(maxlen=self.config.window_size),
            "q2_loss": deque(maxlen=self.config.window_size),
            "ac_loss": deque(maxlen=self.config.window_size),
            "critic_1_grad": deque(maxlen=self.config.window_size),
            "critic_2_grad": deque(maxlen=self.config.window_size),
            "actor_grad": deque(maxlen=self.config.window_size),
            "alpha_loss": deque(maxlen=self.config.window_size),
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
            "alpha_loss_history": []
        }
        
        set_seed(self.seed, self.env)

    def log(self):
        logger.info("Initializing GameAgent with configuration:")
        logger.info(f"- Environment: {self.env_id}")
        logger.info(f"- Action Space: {self.ac_dim}")
        logger.info(f"- Update:Sample ratio: {(1/((self.num_envs)/self.gradient_step)):.4f}: {1}")
        logger.info(f"- Total Expected Gradient Steps: {round(self.config.max_frames/(self.num_envs)) * self.gradient_step}")
        logger.info(f"- Total Expected Samples Read: {round(self.config.max_frames/(self.num_envs)) * self.gradient_step * self.config.agent.batch_size}")
        logger.info(f"Environment initialized with seed: {self.seed}")
        
    def _process_step(self, state, actions, next_obs_raw, rewards, dones):
        self.agent.update_normalizers(
            [state["observation"], next_obs_raw["observation"]]
        )
        
        obs_np = self.agent.normalize_state_batch(state["observation"], state["desired_goal"])
        next_np = self.agent.normalize_state_batch(next_obs_raw["observation"], next_obs_raw["desired_goal"])
        
        obs_batch = torch.from_numpy(obs_np).float().to(self.device)
        next_obs_batch = torch.from_numpy(next_np).float().to(self.device)
        
        for i in range(self.num_envs):
            self._push_to_buffer(i, obs_batch, actions, rewards, next_obs_batch, dones, next_obs_raw)
    
    def _push_to_buffer(self, i, obs_batch, actions, rewards, next_obs_batch, dones, state):
        self.agent.push(
            obs_batch[i].detach(), 
            actions[i], 
            rewards[i], 
            next_obs_batch[i].detach(), 
            dones[i]
        )
        
    def warmup(self, warmup_steps: int = 10000):
        logger.info(f"Starting warmup process. Total samples: {warmup_steps}")
        
        state, _ = self.env.reset()
        iterations = warmup_steps // self.num_envs
        with tqdm(total=warmup_steps, desc="Warm Up (samples)") as pbar:
            for _ in range(iterations):
                actions = self.env.action_space.sample()
                next_obs_raw, rewards, terminateds, truncateds, _ = self.env.step(actions)
                dones = np.logical_or(terminateds, truncateds)
                self._process_step(state, actions, next_obs_raw, rewards, dones)
                state = next_obs_raw
                pbar.update(self.num_envs)
        
    def train(self, path: str):
        self.log()
        self.warmup()
        logger.info(f"Starting training process. Model will be saved to: {path}")
        os.makedirs(path, exist_ok=True)
        
        total_frames = 0
        state, _ = self.env.reset()  
        episode_rewards = np.zeros(self.num_envs, dtype=float)
        pbar = tqdm(total=self.config.max_frames, desc="Frames")
        
        while total_frames < self.config.max_frames:
            state_input = self.agent.normalize_state_batch(state["observation"], state["desired_goal"])
            actions = self.agent.select_action(state_input)
            actions = np.array(actions, dtype=np.float32)
                
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
                    self.agent.save_weights(os.path.join(path, "checkpoint"))
                    self.plot_history(os.path.join(path, "checkpoint"))
                    if hasattr(self.agent.buffer, 'obs_normalizer'):
                        self.agent.buffer.obs_normalizer.save(os.path.join(path, "checkpoint", "obs.yaml"))
                    if self.verbose:
                        logger.info(f"Checkpoint saved at frame {total_frames}")
                
                if total_frames % self.video_freq == 0: 
                    self.test(os.path.join(path, "checkpoint"), 10)
                    if self.verbose:
                        logger.info(f"Video saved at {os.path.join(path, 'checkpoint')}")
                        
            if self.agent.is_buffer_filled(): 
                for i in range(self.gradient_step):
                    info = self.agent.update(step=self.grad_counter)
                    if len(info) == 9:  # SAC with actor update (includes alpha_loss)
                        q1_loss, q2_loss, ac_loss, td_error, q_value, critic_1_grad, critic_2_grad, ac_grad, alpha_loss = info
                        self.history["ac_loss"].append(ac_loss)
                        self.history["actor_grad"].append(ac_grad)
                        self.history["alpha_loss"].append(alpha_loss)
                    elif len(info) == 8:  # TD3 with actor update
                        q1_loss, q2_loss, ac_loss, td_error, q_value, critic_1_grad, critic_2_grad, ac_grad = info
                        self.history["ac_loss"].append(ac_loss)
                        self.history["actor_grad"].append(ac_grad)
                        self.history["alpha_loss"].append(0.0)  # Default for TD3
                    else:  # Critic only update
                        q1_loss, q2_loss, td_error, q_value, critic_1_grad, critic_2_grad = info
                        self.history["alpha_loss"].append(0.0)  # Default when no actor update
                    self.history["q1_loss"].append(q1_loss)
                    self.history["q2_loss"].append(q2_loss)
                    self.history["td_error"].append(td_error)
                    self.history["q_value"].append(q_value)
                    self.history["critic_1_grad"].append(critic_1_grad)
                    self.history["critic_2_grad"].append(critic_2_grad)
                    self.grad_counter+=1
                        
            success_rate = np.mean(np.linalg.norm(next_obs_raw['achieved_goal'] - next_obs_raw['desired_goal'], axis=1) < 0.05)
            self.history["success_rate"].append(success_rate)
            pbar.update(self.num_envs)
            
            if len(self.history["reward"]) >= self.config.window_size:
                recent_reward_avg = np.mean(self.history["reward"])
                if recent_reward_avg > self.best_reward:
                    self.best_reward = recent_reward_avg
                    self.agent.save_weights(os.path.join(path, "best_weights"))
                    self.test(os.path.join(path, "video"), num_episodes=1)

                    if self.verbose:
                        logger.info(f"New best model saved! Average reward: {recent_reward_avg:.2f}")

            state = next_obs_raw

            self.history["reward_history"].append(np.mean(self.history['reward']) if len(self.history["reward"]) > 0 else 0.0)
            self.history["ac_loss_history"].append(np.mean(self.history['ac_loss']) if len(self.history["ac_loss"]) > 0 else 0.0)
            self.history["q1_loss_history"].append(np.mean(self.history["q1_loss"]) if len(self.history["q1_loss"]) > 0 else 0.0)
            self.history["q2_loss_history"].append(np.mean(self.history["q2_loss"]) if len(self.history["q2_loss"]) > 0 else 0.0)
            self.history["td_error_history"].append(np.mean(self.history["td_error"]) if len(self.history["td_error"]) > 0 else 0.0)
            self.history["q_value_history"].append(np.mean(self.history["q_value"]) if len(self.history["q_value"]) > 0 else 0.0)
            self.history["success_rate_history"].append(np.mean(self.history["success_rate"]) if len(self.history["success_rate"]) > 0 else 0.0)
            self.history["critic_1_grad_history"].append(np.mean(self.history["critic_1_grad"]) if len(self.history["critic_1_grad"]) > 0 else 0.0)
            self.history["critic_2_grad_history"].append(np.mean(self.history["critic_2_grad"]) if len(self.history["critic_2_grad"]) > 0 else 0.0)
            self.history["actor_grad_history"].append(np.mean(self.history["actor_grad"]) if len(self.history["actor_grad"]) > 0 else 0.0)
            self.history["alpha_loss_history"].append(np.mean(self.history["alpha_loss"]) if len(self.history["alpha_loss"]) > 0 else 0.0)
            
            if self.use_wandb and total_frames % 1000 == 0:
                best_reward_window = max(self.history["reward"]) if len(self.history["reward"]) > 0 else 0.0
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
                    "actor_grad": self.history["actor_grad_history"][-1]
                }
                
                if self.agent_type == "SAC":
                    log_dict["alpha_loss"] = self.history["alpha_loss_history"][-1]
                    log_dict["alpha"] = self.agent.alpha.item() if hasattr(self.agent, 'alpha') else 0.0
                
                wandb.log(log_dict, step=total_frames)
            
            pbar.set_postfix(
                reward=f"{self.history['reward_history'][-1]:.4f}", 
                ac_loss=f"{self.history['ac_loss_history'][-1]:.4f}", 
                success_rate=f"{success_rate}"
            )
        
        pbar.close()
        logger.info("Training completed. Saving final model weights...")
        self.agent.save_weights(os.path.join(path, "final"))
        self.plot_history(os.path.join(path, "final"))
        logger.info(f"Final model weights saved to: {os.path.join(path, 'final')}")
        
        # Return success rate instead of reward for optimization
        final_success_rate = self.history['success_rate_history'][-1] if self.history['success_rate_history'] else 0.0
        return final_success_rate
                    
    def test(self, path: str, num_episodes: int):
        os.makedirs(path, exist_ok=True)

        env = gym.make(self.env_id, render_mode="rgb_array")
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
            done = False
            rewards = 0
            steps = 0
            
            while not done:
                frame = env.render()

                normalized_state = self.agent.normalize_state_batch(
                    state["observation"].reshape(1, -1), 
                    state["desired_goal"].reshape(1, -1)
                )
                
                action = self.agent.select_action(normalized_state, eval_action=True).squeeze(0)

                video.write(frame)

                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
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
        plt.title("Critic Losses (Q1 & Q2)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(path, "critic_losses.png"))
        plt.close()

        if len(self.history["ac_loss_history"]) > 0:
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
            logger.info(f"Training plots saved to: {path}")
            
class PandasHEREnv(PandasEnv):
    def __init__(self, env_id, seed, config, num_envs, weights = None, verbose = True, use_wandb = True, agent_type = "TD3"):
        super().__init__(env_id, seed, config, num_envs, weights, verbose, use_wandb, agent_type)
        
        her_config = load_her_config(config)
        self.agent.buffer.compute_reward = gym.make(self.env_id).unwrapped.compute_reward

        self.config = her_config
        self.agent_config = self.config.agent
        self.max_episode = self.config.max_episode
        self.max_cycle = self.config.max_cycle
        self.max_epoch = self.config.max_epoch
        
        assert isinstance(self.agent.buffer, HERBuffer), "[ERROR] Agent Buffer is not of HERBuffer type. Change in configs to HER"
        
    def log(self):
        logger.info("Initializing HER GameAgent with configuration:")
        logger.info(f"- Environment: {self.env_id}")
        logger.info(f"- Action Space: {self.ac_dim}")
        logger.info(f"- Total Expected Gradient Steps: {round(self.max_epoch * self.max_cycle * self.gradient_step)}")
        logger.info(f"- Total Expected Episodes Read: {round(self.max_epoch * self.max_cycle * self.max_episode)}")
        logger.info(f"Environment initialized with seed: {self.seed}")
    
    def _push_to_buffer(self, i, obs_batch, actions, rewards, next_obs_batch, dones, state):
        self.agent.push_her(
            i, 
            obs_batch[i].detach(), 
            actions[i], 
            next_obs_batch[i].detach(), 
            rewards[i], 
            dones[i], 
            state["desired_goal"][i],
            state["achieved_goal"][i]
        )
     
    def train(self, path: str): 
        self.log()
        self.warmup()
        logger.info(f"Starting training process. Model will be saved to: {path}")
        os.makedirs(path, exist_ok=True)
        
        state, _ = self.env.reset()  
        episode_rewards = np.zeros(self.num_envs, dtype=float)
        total_frames = 0
        
        for epoch in tqdm(range(1, self.max_epoch+1), desc="Epoch", position=0):
            for cycle in tqdm(range(1, self.max_cycle + 1), desc="Cycle", position=1, leave=False):                
                episode_count = 0
                while episode_count < self.max_episode:
                    state_input = self.agent.normalize_state_batch(state["observation"], state["desired_goal"])
                    actions = self.agent.select_action(state_input)
                    actions = np.array(actions, dtype=np.float32)
                    
                    next_obs_raw, rewards, terminateds, truncateds, _ = self.env.step(actions)                                                        
                    dones = np.logical_or(terminateds, truncateds)                                                                                    
                    success_rate = np.mean(np.linalg.norm(next_obs_raw['achieved_goal'] - next_obs_raw['desired_goal'], axis=1) < 0.05)                                                                                    
                    self.history["success_rate"].append(success_rate)                                                                                 
                                                                                                                                                    
                    self._process_step(state, actions, next_obs_raw, rewards, terminateds)
                    
                    for i in range(self.num_envs):
                        episode_rewards[i] += rewards[i]
                        total_frames += 1
                                                
                        if dones[i]:
                            self.history["reward"].append(episode_rewards[i])
                            episode_rewards[i] = 0.0
                            episode_count += 1
                                                            
                    state = next_obs_raw
                
                for i in range(self.gradient_step):
                    info = self.agent.update(step=self.grad_counter)
                    if len(info) == 9:  # SAC with actor update (includes alpha_loss)
                        q1_loss, q2_loss, ac_loss, td_error, q_value, critic_1_grad, critic_2_grad, ac_grad, alpha_loss = info
                        self.history["ac_loss"].append(ac_loss)
                        self.history["actor_grad"].append(ac_grad)
                        self.history["alpha_loss"].append(alpha_loss)
                    elif len(info) == 8:  # TD3 with actor update
                        q1_loss, q2_loss, ac_loss, td_error, q_value, critic_1_grad, critic_2_grad, ac_grad = info
                        self.history["ac_loss"].append(ac_loss)
                        self.history["actor_grad"].append(ac_grad)
                        self.history["alpha_loss"].append(0.0)  # Default for TD3
                    else:  # Critic only update
                        q1_loss, q2_loss, td_error, q_value, critic_1_grad, critic_2_grad = info
                        self.history["alpha_loss"].append(0.0)  # Default when no actor update
                    self.history["q1_loss"].append(q1_loss)
                    self.history["q2_loss"].append(q2_loss)
                    self.history["td_error"].append(td_error)
                    self.history["q_value"].append(q_value)
                    self.history["critic_1_grad"].append(critic_1_grad)
                    self.history["critic_2_grad"].append(critic_2_grad)
                    self.grad_counter+=1     
                    
                self.history["reward_history"].append(np.mean(self.history['reward']) if len(self.history["reward"]) > 0 else 0.0)
                self.history["ac_loss_history"].append(np.mean(self.history['ac_loss']) if len(self.history["ac_loss"]) > 0 else 0.0)
                self.history["q1_loss_history"].append(np.mean(self.history["q1_loss"]) if len(self.history["q1_loss"]) > 0 else 0.0)
                self.history["q2_loss_history"].append(np.mean(self.history["q2_loss"]) if len(self.history["q2_loss"]) > 0 else 0.0)
                self.history["td_error_history"].append(np.mean(self.history["td_error"]) if len(self.history["td_error"]) > 0 else 0.0)
                self.history["q_value_history"].append(np.mean(self.history["q_value"]) if len(self.history["q_value"]) > 0 else 0.0)
                self.history["success_rate_history"].append(np.mean(self.history["success_rate"]) if len(self.history["success_rate"]) > 0 else 0.0)
                self.history["critic_1_grad_history"].append(np.mean(self.history["critic_1_grad"]) if len(self.history["critic_1_grad"]) > 0 else 0.0)
                self.history["critic_2_grad_history"].append(np.mean(self.history["critic_2_grad"]) if len(self.history["critic_2_grad"]) > 0 else 0.0)
                self.history["actor_grad_history"].append(np.mean(self.history["actor_grad"]) if len(self.history["actor_grad"]) > 0 else 0.0)
                self.history["alpha_loss_history"].append(np.mean(self.history["alpha_loss"]) if len(self.history["alpha_loss"]) > 0 else 0.0)
                
                if len(self.history["reward"]) >= self.config.window_size:
                    recent_reward_avg = np.mean(self.history["reward"])
                    if recent_reward_avg > self.best_reward:
                        self.best_reward = recent_reward_avg
                        self.agent.save_weights(os.path.join(path, "best_weights"))
                        self.test(os.path.join(path, "video"), num_episodes=1)
                        if self.verbose:
                            logger.info(f"New best model saved! Average reward: {recent_reward_avg:.2f}")
                
                if self.use_wandb:
                    best_reward_window = max(self.history["reward"]) if len(self.history["reward"]) > 0 else 0.0
                    log_dict = {
                        "reward": self.history["reward_history"][-1],
                        "q1_loss": self.history["q1_loss_history"][-1],
                        "q2_loss": self.history["q2_loss_history"][-1],
                        "actor_loss": self.history["ac_loss_history"][-1],
                        "td_error": self.history["td_error_history"][-1],
                        "q_value": self.history["q_value_history"][-1],
                        "success_rate": self.history["success_rate_history"][-1],
                        "best_reward": best_reward_window,
                        "epoch": epoch,
                        "cycle": cycle,
                        "critic_1_grad": self.history["critic_1_grad_history"][-1],
                        "critic_2_grad": self.history["critic_2_grad_history"][-1],
                        "actor_grad": self.history["actor_grad_history"][-1]
                    }
                    
                    if self.agent_type == "SAC":
                        log_dict["alpha_loss"] = self.history["alpha_loss_history"][-1]
                        log_dict["alpha"] = self.agent.alpha.item() if hasattr(self.agent, 'alpha') else 0.0
                    
                    wandb.log(log_dict, step=(epoch-1) * self.max_cycle + cycle)    
                            
            if epoch % self.save_freq == 0:
                self.agent.save_weights(os.path.join(path, "checkpoint"))
                self.plot_history(os.path.join(path, "checkpoint"))
                if hasattr(self.agent.buffer, 'obs_normalizer'):
                    self.agent.buffer.obs_normalizer.save(os.path.join(path, "checkpoint", "obs.yaml"))
                if self.verbose:
                    logger.info(f"Checkpoint saved at epoch {epoch}") 
            
            if epoch % self.video_freq == 0: 
                self.test(os.path.join(path, "checkpoint"), 10)
                if self.verbose:
                    logger.info(f"Video saved at {os.path.join(path, 'checkpoint')}")
                    
            if epoch % self.reset_freq == 0: 
                self.agent.reset()
                if self.verbose: 
                    logger.info(f"Resetting TD3 Network across both Actor and Critic Networks")
        
        logger.info("Training completed. Saving final model weights...")
        self.agent.save_weights(os.path.join(path, "final"))
        self.plot_history(os.path.join(path, "final"))
        logger.info(f"Final model weights saved to: {os.path.join(path, 'final')}")
        if hasattr(self.agent.buffer, 'obs_normalizer'):
            self.agent.buffer.obs_normalizer.save(os.path.join(path, "obs.yaml"))

        # Return success rate instead of reward for optimization
        final_success_rate = self.history['success_rate_history'][-1] if self.history['success_rate_history'] else 0.0
        return final_success_rate       