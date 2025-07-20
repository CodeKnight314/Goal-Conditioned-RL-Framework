import torch 
import torch.nn as nn
import os
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int, ac_dim: int, layer_stack: int = 3):
        super().__init__()
        layers = []
        current_dim = obs_dim
        
        for i in range(layer_stack):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU() if i != layer_stack - 1 else nn.Tanh()
            ])
            current_dim = hidden_dim
        self.base_net = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_dim, ac_dim)
        self.apply(self._init_weights)
        
    def forward(self, x: torch.Tensor):
        x = self.base_net(x)
        preactivation = self.output_layer(x)
        return preactivation

    def load(self, weights: str, device: str="cuda"):
        self.load_state_dict(torch.load(weights, map_location=torch.device(device)))
        
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)
        
    def reset(self): 
        self.apply(self._init_weights)

class SACActorModel(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int, ac_dim: int, layer_stack: int = 3, 
                 log_std_min: float = -20.0, log_std_max: float = 2.0):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        layers = []
        current_dim = obs_dim
        
        for i in range(layer_stack):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            ])
            current_dim = hidden_dim
            
        self.base_net = nn.Sequential(*layers)
        self.mean_head = nn.Linear(hidden_dim, ac_dim)
        self.log_std_head = nn.Linear(hidden_dim, ac_dim)
        self.apply(self._init_weights)
    
    def forward(self, x: torch.Tensor):
        features = self.base_net(x)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std
    
    def sample(self, x: torch.Tensor, deterministic: bool = False):
        mean, log_std = self.forward(x)
        std = log_std.exp()
        
        if deterministic:
            action = torch.tanh(mean)
            log_prob = None
        else:
            # Reparameterization trick
            normal = torch.distributions.Normal(mean, std)
            x_t = normal.rsample()  # reparameterization trick (mean + std * N(0,1))
            action = torch.tanh(x_t)
            
            # Compute log probability with correction for tanh squashing
            log_prob = normal.log_prob(x_t)
            # Correction for tanh transformation
            log_prob -= torch.log(1 - action.pow(2) + 1e-8)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
            
        return action, log_prob
    
    def load(self, weights: str, device: str="cuda"):
        self.load_state_dict(torch.load(weights, map_location=torch.device(device)))
        
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)
        
    def reset(self): 
        self.apply(self._init_weights)
        
class Critic(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, layer_stack: int = 3):
        super().__init__()
        layers = []
        current_dim = input_dim
        
        for i in range(layer_stack-1):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            ])
            current_dim = hidden_dim
            
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)
        self.apply(self._init_weights)
        
    def forward(self, x: torch.Tensor):
        return self.net(x)

    def load(self, weights: str, device: str="cuda"):
        self.load_state_dict(torch.load(weights, map_location=torch.device(device)))
        
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)
        
    def reset(self): 
        self.apply(self._init_weights)