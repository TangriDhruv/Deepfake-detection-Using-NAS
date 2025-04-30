import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import random
from tqdm import tqdm

# PPO Training function
def train_ppo(controller, optimizer, memories, clip_ratio=0.2, epochs=10, entropy_coef=0.01):
    """Train the PPO controller on collected experiences"""
    # Unpack memories
    states = torch.cat([m['state'] for m in memories])
    actions = torch.cat([m['action'] for m in memories])
    old_log_probs = torch.cat([m['log_prob'] for m in memories])
    rewards = torch.cat([m['reward'] for m in memories])
    
    # Normalize rewards for stable training
    if rewards.std() > 0:
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    
    # Store metrics for logging
    metrics = {
        'actor_loss': 0,
        'critic_loss': 0,
        'entropy_loss': 0
    }
    
    # Train for multiple epochs
    for _ in range(epochs):
        # Evaluate current policy
        log_probs = []
        values = []
        entropy = []
        
        for i in range(len(states)):
            state_i = states[i:i+1]
            action_i = actions[i]
            
            # Get action probabilities and value
            action_probs, value = controller(state_i)
            
            # Create categorical distribution
            dist = Categorical(action_probs)
            
            # Get log probability and entropy
            log_prob = dist.log_prob(action_i)
            entropy_i = dist.entropy()
            
            log_probs.append(log_prob)
            values.append(value.squeeze())
            entropy.append(entropy_i)
        
        # Stack results
        log_probs = torch.stack(log_probs)
        values = torch.stack(values)
        entropy = torch.stack(entropy)
        
        # Compute ratio and surrogate loss
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * rewards
        surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * rewards
        
        # PPO losses
        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = F.mse_loss(values, rewards)
        entropy_loss = -entropy.mean()
        
        # Total loss
        loss = actor_loss + 0.5 * critic_loss - entropy_coef * entropy_loss
        
        # Update controller
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update metrics
        metrics['actor_loss'] += actor_loss.item() / epochs
        metrics['critic_loss'] += critic_loss.item() / epochs
        metrics['entropy_loss'] += entropy_loss.item() / epochs
    
    return metrics['actor_loss'], metrics['critic_loss'], metrics['entropy_loss']