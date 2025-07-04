import torch 
import torch.nn as nn
import os

class Actor(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int, ac_dim: int, layer_stack: int = 4):
        super().__init__()
        
        self.net = [
            nn.Linear(obs_dim, hidden_dim), 
            nn.ReLU()
        ]
        
        for i in range(layer_stack):
            self.net.append(nn.Linear(hidden_dim, hidden_dim))
            self.net.append(nn.ReLU())
        
        self.net.append(nn.Linear(hidden_dim, ac_dim))
        self.net.append(nn.Tanh())
        
        self.net = nn.Sequential(*self.net)
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
        
class Critic(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, layer_stack: int = 4):
        super().__init__()
        
        self.net = [
            nn.Linear(input_dim, hidden_dim), 
            nn.ReLU()
        ]
        
        for i in range(layer_stack):
            self.net.append(nn.Linear(hidden_dim, hidden_dim))
            self.net.append(nn.ReLU())
        
        self.net.append(nn.Linear(hidden_dim, 1))
        
        self.net = nn.Sequential(*self.net)
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