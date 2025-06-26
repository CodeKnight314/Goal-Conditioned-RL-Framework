import torch 
import torch.nn as nn
from typing import List

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
        
    def forward(self, x: torch.Tensor):
        return self.net(x)

    def load(self, weights: str):
        self.load_state_dict(torch.load(weights))
        
    def save(self, path: str):
        torch.save(self.state_dict(), path)
        
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
        
    def forward(self, x: torch.Tensor):
        return self.net(x)

    def load(self, weights: str):
        self.load_state_dict(torch.load(weights))
        
    def save(self, path: str):
        torch.save(self.state_dict(), path)