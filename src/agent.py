from src.model import Actor, Critic
from src.buffer import PERBuffer, ReplayBuffer, HERBuffer
import torch 
from torch.optim import AdamW
import os 
import numpy as np 

class PandaAgent():
    def __init__(self, 
                 obs_dim: int, 
                 ac_dim: int, 
                 config: dict,
                 weights: str): 
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config = config
        
        self.actor = Actor(obs_dim, self.config["hidden_dim"], ac_dim, self.config["layer_count"]).to(self.device)
        self.target_actor = Actor(obs_dim, self.config["hidden_dim"], ac_dim, self.config["layer_count"]).to(self.device)
        
        self.critic_1 = Critic(obs_dim + ac_dim, self.config["hidden_dim"], self.config["layer_count"]).to(self.device)
        self.critic_2 = Critic(obs_dim + ac_dim, self.config["hidden_dim"], self.config["layer_count"]).to(self.device)
        self.target_critic_1 = Critic(obs_dim + ac_dim, self.config["hidden_dim"], self.config["layer_count"]).to(self.device)
        self.target_critic_2 = Critic(obs_dim + ac_dim, self.config["hidden_dim"], self.config["layer_count"]).to(self.device)
        
        self.actor_opt = AdamW(self.actor.parameters(), self.config["actor_lr"])
        self.critic_1_opt = AdamW(self.critic_1.parameters(), self.config["critic_lr"])
        self.critic_2_opt = AdamW(self.critic_2.parameters(), self.config["critic_lr"])
        
        if self.config["buffer_type"] == "PER":
            self.buffer = PERBuffer(self.config["max_len"], self.config["alpha"])
        elif self.config["buffer_type"] == "REPLAY":
            self.buffer = ReplayBuffer(self.config["max_len"])
        else: 
            raise ValueError(f"[ERROR] Invalid Buffer type. Received {self.config['buffer_type']}.")
        
        self.noise_std = self.config["noise_std"]
        self.noise_clamp = self.config["noise_clamp"]
        self.policy_noise = self.config["policy_noise"]
        self.gamma = self.config["gamma"]
        self.batch_size = self.config["batch_size"]
        self.ac_update_freq = self.config["ac_update_freq"]
        
        if weights: 
            self.actor.load(os.path.join(weights, "actor.pth"))
            self.critic_1.load(os.path.join(weights, "critic_1.pth"))
            self.critic_2.load(os.path.join(weights, "critic_2.pth"))
            self.target_actor.load_state_dict(self.actor.state_dict())
            self.target_critic_1.load_state_dict(self.critic_1.state_dict())
            self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        
    def update_target_network(self, hard_update: bool = True, tau: float = 0.05):
        if hard_update: 
            self.target_actor.load_state_dict(self.actor.state_dict())
            self.target_critic_1.load_state_dict(self.critic_1.state_dict())
            self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        else: 
            for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                
            for target_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                
            for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                
    def actor_update(self, states: torch.Tensor):
        actions = self.actor(states)
        actor_loss = -self.critic_1(torch.cat([states, actions], dim=-1)).mean()
        
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        
        return actor_loss.item()
    
    def critic_update(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, next_state: torch.Tensor, done: torch.Tensor, weights: torch.Tensor=None):
        with torch.no_grad():
            noise = torch.clamp(torch.randn_like(action) * self.policy_noise, -self.noise_clamp, self.noise_clamp)
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
            critic_1_loss = torch.nn.functional.mse_loss(current_q1_value, target, reduction='none')
            critic_1_loss = (weights * critic_1_loss).mean()
        else: 
            critic_1_loss = torch.nn.functional.mse_loss(current_q1_value, target)
        critic_1_loss.backward()
        self.critic_1_opt.step()
        
        self.critic_2_opt.zero_grad()
        if weights is not None and weights.numel() > 0:
            critic_2_loss = torch.nn.functional.mse_loss(current_q2_value, target, reduction='none')
            critic_2_loss = (weights * critic_2_loss).mean()
        else: 
            critic_2_loss = torch.nn.functional.mse_loss(current_q2_value, target)
        critic_2_loss.backward()
        self.critic_2_opt.step()
        
        if weights is not None and weights.numel() > 0:
            td_error = 0.5 * (current_q1_value + current_q2_value - 2 * target).detach().cpu().numpy()
            return critic_1_loss.item(), critic_2_loss.item(), td_error
        else: 
            return critic_1_loss.item(), critic_2_loss.item()

    def select_action(self, obs_tensor: np.array, eval_action: bool=False):
        if not eval_action:
            obs_tensor = torch.as_tensor(obs_tensor, dtype=torch.float32).to(self.device)
            with torch.no_grad(): 
                action = self.actor(obs_tensor).detach().cpu().numpy()
            action += np.random.normal(0, self.noise_std, size=action.shape)
            action = np.clip(action, -1, 1)
            return action
        else: 
            obs_tensor = torch.as_tensor(obs_tensor, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                return self.actor(obs_tensor).cpu().numpy()
            
    def push(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)
        
    def update(self, step: int, beta: float = 0.7):
        ac_loss = None
        if isinstance(self.buffer, PERBuffer):
            states, actions, rewards, next_states, dones, weights, indices = self.buffer.sample(self.batch_size, beta)
            q1_loss, q2_loss, td_error = self.critic_update(states, actions, rewards, next_states, dones, weights) 
            self.buffer.update_priorities(indices=indices, priorities=td_error)
        else: 
            states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
            q1_loss, q2_loss = self.critic_update(states, actions, rewards, next_states, dones)
            
        if step % self.ac_update_freq == 0: 
            ac_loss = self.actor_update(states)
            
        self.update_target_network(hard_update=False)
        if step % self.ac_update_freq == 0:
            return q1_loss, q2_loss, ac_loss
        else: 
            return q1_loss, q2_loss
            
    def save_weights(self, path: str):
        self.actor.save(os.path.join(path, "actor.pth"))
        self.critic_1.save(os.path.join(path, "critic_1.pth"))
        self.critic_2.save(os.path.join(path, "critic_2.pth"))
        
    def is_buffer_filled(self):
        return len(self.buffer) >= self.batch_size