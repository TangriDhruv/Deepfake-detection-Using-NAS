import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# PPO Controller for architecture search
class PPOController(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(PPOController, self).__init__()
        
        # Print dimensions for debugging
        print(f"Initializing PPO controller with state_dim={state_dim}, action_dim={action_dim}")
        
        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state):
        # Returns action probabilities and estimated value
        action_probs = F.softmax(self.actor(state), dim=-1)
        value = self.critic(state)
        return action_probs, value
    
    def act(self, state):
        action_probs, _ = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.detach(), log_prob.detach()