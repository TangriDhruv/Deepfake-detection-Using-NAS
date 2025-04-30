import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import wandb
from tqdm import tqdm
from utils.evaluation import evaluate_architecture
from models.model import DeepfakeDetectionModel
from search.controller import PPOController
from search.darts import train_ppo
from utils.visualization import *

def search_architecture_hybrid(train_loader, val_loader, device, input_channels=60, num_cells=3, 
                              num_nodes=4, num_ops=10, epochs=30, ppo_updates=5, 
                              project_name="deepfake-nas-hybrid"):
    """Hybrid approach combining PPO for exploration with DARTS for optimization"""
    
    
    # Initialize wandb
    wandb.init(project=project_name, name=f"Hybrid_PPO_DARTS_cells{num_cells}_nodes{num_nodes}")
    
    # Log hyperparameters
    config = {
        "input_channels": input_channels,
        "num_cells": num_cells,
        "num_nodes": num_nodes,
        "num_ops": num_ops,
        "epochs": epochs,
        "w_lr": 0.001,        # Weight learning rate
        "alpha_lr": 0.0003,   # Architecture parameter learning rate
        "ppo_lr": 0.0005,     # PPO controller learning rate
        "ppo_updates": ppo_updates,
        "exploration_ratio": 0.3,  # Ratio of epochs to use PPO exploration
        "visualization_enabled": True  # Enable visualization
    }
    wandb.config.update(config)
    
    # Initialize model with expanded operation set
    model = DeepfakeDetectionModel(input_channels, num_cells, num_nodes, num_ops).to(device)
    
    # Calculate edges for PPO controller
    edges_per_cell = sum(range(1, num_nodes+1))
    total_edges = num_cells * edges_per_cell
    
    # Initialize PPO controller for exploration
    state_dim = 1  # Single value for validation performance
    action_dim = num_ops  # Number of operations per edge
    controller = PPOController(state_dim, action_dim).to(device)
    controller_optimizer = optim.Adam(controller.parameters(), lr=config["ppo_lr"])
    
    # Setup optimizers for DARTS
    w_optimizer = optim.Adam(model.weights(), lr=config["w_lr"], weight_decay=3e-4)
    w_scheduler = optim.lr_scheduler.CosineAnnealingLR(w_optimizer, epochs)
    alpha_optimizer = optim.Adam(model.alphas(), lr=config["alpha_lr"], betas=(0.5, 0.999), weight_decay=1e-3)
    
    # Metrics tracking
    best_val_eer = 1.0
    best_architecture = None
    best_mode = None
    
    with tqdm(total=epochs, desc="Hybrid Search Progress", position=0, leave=True) as epoch_pbar:
        for epoch in range(epochs):
            # Determine exploration mode for this epoch
            # More exploration in early stages, more exploitation later
            use_ppo = (random.random() < config["exploration_ratio"] * (1 - epoch/epochs))
            
            # PPO exploration phase
            if use_ppo:
                model.train()
                train_loss = 0.0
                batch_count = 0
                
                # For PPO
                memories = []
                current_architecture = []
                
                # Sample architecture using PPO
                for i in range(total_edges):
                    # Use current validation EER as state
                    state = torch.FloatTensor([min(best_val_eer, 0.5) * 2]).to(device)
                    
                    # Sample architecture weights for this edge
                    for j in range(num_ops):
                        action, log_prob = controller.act(state)
                        current_architecture.append(action.item())
                        
                        # Store experience for PPO
                        memories.append({
                            'state': state.clone(),
                            'action': action.unsqueeze(0),
                            'log_prob': log_prob.unsqueeze(0),
                            'reward': torch.zeros(1).to(device)  # Updated later
                        })
                
                # Convert architecture to tensor for PPO mode
                architecture_weights = torch.FloatTensor(current_architecture).to(device)
                
                # Train model with PPO-generated architecture
                for inputs, targets in train_loader:
                    batch_count += 1
                    if batch_count % 10 == 0:
                        print(f"\rPPO Training batch {batch_count}/{len(train_loader)}", end="")
                    
                    inputs, targets = inputs.to(device), targets.to(device)
                    if torch.isnan(inputs).any():
                        inputs = torch.nan_to_num(inputs, nan=0.0)
                    
                    # Update weights
                    w_optimizer.zero_grad()
                    outputs = model(inputs, architecture_weights)  # Use PPO architecture
                    loss = F.cross_entropy(outputs, targets)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.weights(), max_norm=1.0)
                    w_optimizer.step()
                    
                    train_loss += loss.item()
                
                print()  # Line break after training
                
                # Evaluate architecture from PPO
                val_eer = evaluate_architecture(model, val_loader, device, architecture_weights)
                
                # Update PPO controller based on performance
                reward = best_val_eer - val_eer if val_eer < best_val_eer else 0
                for memory in memories:
                    memory['reward'] = torch.FloatTensor([reward]).to(device)
                
                # Update best architecture if improved
                if val_eer < best_val_eer:
                    best_val_eer = val_eer
                    best_architecture = architecture_weights.clone()
                    best_mode = 'ppo'
                    print(f"\nNew best architecture found via PPO! EER: {best_val_eer:.4f}")
                
                # Update PPO controller
                if epoch % config["ppo_updates"] == 0 and memories:
                    actor_loss, critic_loss, entropy_loss = train_ppo(
                        controller, controller_optimizer, memories)
                    
                    # Log PPO metrics
                    wandb.log({
                        "actor_loss": actor_loss,
                        "critic_loss": critic_loss,
                        "entropy_loss": entropy_loss
                    })
            
            # DARTS optimization phase
            else:
                model.train()
                train_loss = 0.0
                batch_count = 0
                
                # Phase 1: Train model weights using DARTS approach
                for inputs, targets in train_loader:
                    batch_count += 1
                    if batch_count % 10 == 0:
                        print(f"\rDARTS Weight Training batch {batch_count}/{len(train_loader)}", end="")
                    
                    inputs, targets = inputs.to(device), targets.to(device)
                    if torch.isnan(inputs).any():
                        inputs = torch.nan_to_num(inputs, nan=0.0)
                    
                    # Update weights with internal alphas
                    w_optimizer.zero_grad()
                    outputs = model(inputs)  # Use alphas without external weights
                    loss = F.cross_entropy(outputs, targets)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.weights(), max_norm=1.0)
                    w_optimizer.step()
                    
                    train_loss += loss.item()
                
                print()  # Line break after training
                
                # Phase 2: Update architecture parameters on validation set
                model.train()  # Keep in train mode for alpha updates
                val_batch_count = 0
                
                for inputs, targets in val_loader:
                    # Use a subset of validation data
                    if random.random() > 0.2:  # Sample ~20% for alpha updates
                        continue
                        
                    val_batch_count += 1
                    if val_batch_count > 50:  # Limit validation batches for speed
                        break
                    
                    inputs, targets = inputs.to(device), targets.to(device)
                    if torch.isnan(inputs).any():
                        inputs = torch.nan_to_num(inputs, nan=0.0)
                    
                    # Update alphas
                    alpha_optimizer.zero_grad()
                    outputs = model(inputs)  # Use internal alphas
                    loss = F.cross_entropy(outputs, targets)
                    loss.backward()
                    alpha_optimizer.step()
                
                # Evaluate DARTS architecture
                val_eer = evaluate_architecture(model, val_loader, device, discrete=True)
                
                # Update best architecture if improved
                if val_eer < best_val_eer:
                    best_val_eer = val_eer
                    # Save alphas as best architecture
                    best_architecture = model._alphas.detach().clone()
                    best_mode = 'darts'
                    print(f"\nNew best architecture found via DARTS! EER: {best_val_eer:.4f}")
            
            # Update learning rate for weights
            w_scheduler.step()
            
            # Update epoch progress bar
            epoch_pbar.update(1)
            epoch_pbar.set_postfix({
                "Mode": "PPO" if use_ppo else "DARTS",
                "Val EER": f"{val_eer:.4f}",
                "Best EER": f"{best_val_eer:.4f}"
            })
            
            # Log metrics to wandb
            avg_train_loss = train_loss / max(batch_count, 1)
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_eer": val_eer,
                "best_val_eer": best_val_eer,
                "mode": "PPO" if use_ppo else "DARTS",
                "learning_rate": w_optimizer.param_groups[0]['lr']
            })
            
            # Save checkpoint for the best model
            if val_eer <= best_val_eer:
                checkpoint_path = 'best_hybrid_model.pth'
                if use_ppo:
                    # Save PPO-generated architecture
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'ppo_architecture': best_architecture,
                        'eer': best_val_eer,
                        'epoch': epoch + 1,
                        'mode': 'PPO'
                    }, checkpoint_path)
                    # Add visualization during training
                    if epoch % 5 == 0 and config["visualization_enabled"]:  
                        vis_path = visualize_ppo_architecture(
                        best_architecture, 
                        num_cells=num_cells,
                        num_nodes=num_nodes,
                        num_ops=num_ops,
                        save_path=f"ppo_arch_epoch_{epoch}.png",
                        save_to_wandb=True
                        )
                else:
                    # Save DARTS-generated architecture
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'darts_alphas': model._alphas,
                        'eer': best_val_eer,
                        'epoch': epoch + 1,
                        'mode': 'DARTS'
                    }, checkpoint_path)
                    # Add visualization during training
                    if epoch % 5 == 0 and config["visualization_enabled"]:
                        vis_path = visualize_darts_architecture(
                        model._alphas,
                        num_cells=num_cells,
                        num_nodes=num_nodes, 
                        num_ops=num_ops,
                        save_path=f"darts_arch_epoch_{epoch}.png",
                        save_to_wandb=True
                        )
                
                # Log best model to wandb
                wandb.save(checkpoint_path)
    
    # Finish wandb run
    wandb.finish()
    
    # Return the best architecture (either from PPO or DARTS)
    final_model = model
    final_architecture = best_architecture
    
    return final_model, final_architecture, best_val_eer
