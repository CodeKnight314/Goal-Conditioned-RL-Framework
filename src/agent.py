from src.model import Actor, Critic
from src.buffer import PERBuffer, ReplayBuffer, HERBuffer, PERBufferSumTree
from src.utils import AgentConfig
import torch 
from torch.optim import AdamW, lr_scheduler
import os 
import numpy as np 

class PandaAgent():
    def __init__(self, 
                 obs_dim: int, 
                 ac_dim: int, 
                 config: AgentConfig,
                 weights: str,
                 nenvs: int): 
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config = config
        
        self.actor = Actor(obs_dim, self.config.hidden_dim, ac_dim, self.config.layer_count).to(self.device)
        self.target_actor = Actor(obs_dim, self.config.hidden_dim, ac_dim, self.config.layer_count).to(self.device)
        
        self.critic_1 = Critic(obs_dim + ac_dim, self.config.hidden_dim, self.config.layer_count).to(self.device)
        self.critic_2 = Critic(obs_dim + ac_dim, self.config.hidden_dim, self.config.layer_count).to(self.device)
        self.target_critic_1 = Critic(obs_dim + ac_dim, self.config.hidden_dim, self.config.layer_count).to(self.device)
        self.target_critic_2 = Critic(obs_dim + ac_dim, self.config.hidden_dim, self.config.layer_count).to(self.device)
        
        self.actor_opt = AdamW(self.actor.parameters(), self.config.actor_lr)
        self.critic_1_opt = AdamW(self.critic_1.parameters(), self.config.critic_lr)
        self.critic_2_opt = AdamW(self.critic_2.parameters(), self.config.critic_lr)
        
        self.actor_scheduler = lr_scheduler.CosineAnnealingLR(self.actor_opt, T_max=self.config.ac_scheduler_steps, eta_min=self.config.actor_lr_min)
        self.critic_1_scheduler = lr_scheduler.CosineAnnealingLR(self.critic_1_opt, T_max=self.config.cr_scheduler_steps, eta_min=self.config.critic_lr_min)
        self.critic_2_scheduler = lr_scheduler.CosineAnnealingLR(self.critic_2_opt, T_max=self.config.cr_scheduler_steps, eta_min=self.config.critic_lr_min)
        
        if self.config.buffer_type == "PER":
            self.buffer = PERBuffer(self.config.max_len, self.config.alpha)
        elif self.config.buffer_type == "PER_SUMTREE":
            self.buffer = PERBufferSumTree(self.config.max_len, self.config.alpha)
        elif self.config.buffer_type == "REPLAY":
            self.buffer = ReplayBuffer(self.config.max_len)
        elif self.config.buffer_type == "HER":
            self.buffer = HERBuffer(self.config.max_len, self.config.max_eps_len, nenvs, k_future=self.config.k_future)
        else: 
            raise ValueError(f"[ERROR] Invalid Buffer type. Received {self.config.buffer_type}.")
        
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
        self.ac_smoothing_coeff = self.config.smooth_coeff
        self.grad_clip = self.config.grad_clip
        self.tau = self.config.tau
        
        if weights: 
            self.actor.load(os.path.join(weights, "actor.pth"), self.device)
            self.critic_1.load(os.path.join(weights, "critic_1.pth"), self.device)
            self.critic_2.load(os.path.join(weights, "critic_2.pth"), self.device)
            self.target_actor.load_state_dict(self.actor.state_dict())
            self.target_critic_1.load_state_dict(self.critic_1.state_dict())
            self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        else:
            self.target_actor.load_state_dict(self.actor.state_dict())
            self.target_critic_1.load_state_dict(self.critic_1.state_dict())
            self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        
    def update_target_network(self, hard_update: bool = True, tau: float = 0.005):
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

    def beta_scheduler(self, step: int):
        ratio = step / self.beta_end
        self.beta = min(
            self.beta_max,
            self.beta_start + ratio * (self.beta_max - self.beta_start)
        )
        
    def get_gradient_norm(self, model: torch.nn.Module):
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
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
    
    def critic_update(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, next_state: torch.Tensor, done: torch.Tensor, weights: torch.Tensor=None):
        with torch.no_grad():
            noise = torch.clamp(torch.randn_like(action) * self.policy_noise, -self.noise_clamp, self.noise_clamp)
            next_action = torch.clamp(self.target_actor(next_state) + noise, -1, 1)
            
            target_critic_input = torch.concat([next_state, next_action], dim=-1)
            target_Q1_value = self.target_critic_1(target_critic_input)
            target_Q2_value = self.target_critic_2(target_critic_input)
            target_Q_value = torch.min(target_Q1_value, target_Q2_value)
            
            target = reward + self.gamma * (1 - done) * target_Q_value
            
            min_q_value = -1.0/(1.0-self.gamma)
            max_q_value = 0
            target = torch.clamp(target, min=min_q_value, max=max_q_value)
            
        current_critic_input = torch.concat([state, action], dim=-1)
        current_q1_value = self.critic_1(current_critic_input)
        current_q2_value = self.critic_2(current_critic_input)
        
        self.critic_1_opt.zero_grad()
        if weights is not None and weights.numel() > 0:
            critic_1_loss = torch.nn.functional.smooth_l1_loss(current_q1_value, target, reduction='none')
            critic_1_loss = (weights * critic_1_loss).mean()
        else: 
            critic_1_loss = torch.nn.functional.smooth_l1_loss(current_q1_value, target)
        critic_1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_1.parameters(), self.grad_clip)
        critic_1_grad_norm = self.get_gradient_norm(self.critic_1)
        self.critic_1_opt.step()
        
        self.critic_2_opt.zero_grad()
        if weights is not None and weights.numel() > 0:
            critic_2_loss = torch.nn.functional.smooth_l1_loss(current_q2_value, target, reduction='none')
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
        
        q_value = torch.cat([current_q1_value, current_q2_value], dim=-1).mean().detach().cpu().item()

        if weights is not None and weights.numel() > 0:
            td_error = torch.maximum(td_error_1, td_error_2).cpu().numpy()
            return critic_1_loss.item(), critic_2_loss.item(), td_error, q_value, critic_1_grad_norm, critic_2_grad_norm
        else:
            td_error = torch.mean(torch.maximum(td_error_1, td_error_2)).cpu().numpy()
            return critic_1_loss.item(), critic_2_loss.item(), td_error, q_value, critic_1_grad_norm, critic_2_grad_norm

    def select_action(self, obs_tensor: np.array, eval_action: bool=False):
        self.set_eval()
        if not eval_action:
            obs_tensor = torch.as_tensor(obs_tensor, dtype=torch.float32).to(self.device)
            with torch.no_grad(): 
                action = torch.tanh(self.actor(obs_tensor)).detach().cpu().numpy()
            noise = np.random.normal(0, self.noise_std, size=action.shape)
            action = np.clip(action + noise, -1, 1)
            return action
        else: 
            obs_tensor = torch.as_tensor(obs_tensor, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                return np.clip(torch.tanh(self.actor(obs_tensor)).cpu().numpy(), -1, 1)
            
    def push(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)
        
    def push_her(self, idx, state, action, next_state, reward, done, desired_goal, achieved_goal):
        self.buffer.push(idx, state, action, next_state, reward, done, desired_goal, achieved_goal)
        
    def update(self, step: int):
        self.set_train()
        if isinstance(self.buffer, PERBuffer) or isinstance(self.buffer, PERBufferSumTree):
            states, actions, rewards, next_states, dones, weights, indices = self.buffer.sample(self.batch_size, self.beta)
            q1_loss, q2_loss, td_error, q_value, critic_1_grad, critic_2_grad = self.critic_update(states, actions, rewards, next_states, dones, weights) 
            self.buffer.update_priorities(indices=indices, priorities=td_error)
        else: 
            states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
            q1_loss, q2_loss, td_error, q_value, critic_1_grad, critic_2_grad = self.critic_update(states, actions, rewards, next_states, dones)
            
        self.beta_scheduler(step)
        self.update_target_network(hard_update=False, tau=self.tau)
        if step % self.ac_update_freq == 0:
            ac_loss, ac_grad = self.actor_update(states)
            return q1_loss, q2_loss, ac_loss, td_error, q_value, critic_1_grad, critic_2_grad, ac_grad
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
    
    def update_normalizers(self, obs_list, dg_list):
        if obs_list:
            combined_obs = np.concatenate(obs_list, axis=0)
            self.buffer.obs_normalizer.update(combined_obs)
        if dg_list:
            combined_dg = np.concatenate(dg_list, axis=0)
            self.buffer.dg_normalizer.update(combined_dg)
    
    def normalize_state_batch(self, obs_batch, dg_batch):
        normalized_obs = self.buffer.obs_normalizer.normalize(obs_batch)
        normalized_dg = self.buffer.dg_normalizer.normalize(dg_batch)
        return np.concatenate([normalized_obs, normalized_dg], axis=-1)
    
    def reset(self):
        self.actor.reset()
        self.target_actor.reset()
        self.critic_1.reset()
        self.critic_2.reset() 
        self.target_critic_1.reset() 
        self.target_critic_2.reset()