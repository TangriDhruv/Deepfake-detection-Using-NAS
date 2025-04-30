import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from tqdm import tqdm

def evaluate_architecture(model, val_loader, device, architecture_weights=None, discrete=False):
    """Evaluate the performance of an architecture"""
    model.eval()
    all_targets = []
    all_scores = []
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            try:
                inputs, targets = inputs.to(device), targets.to(device)
                if torch.isnan(inputs).any():
                    inputs = torch.nan_to_num(inputs, nan=0.0)
                
                # Use the appropriate mode
                if architecture_weights is not None:
                    # PPO mode with external weights
                    outputs = model(inputs, architecture_weights)
                else:
                    # DARTS mode with internal alphas
                    outputs = model(inputs, discrete=discrete)
                
                scores = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                all_targets.extend(targets.cpu().numpy())
                all_scores.extend(scores)
                
            except Exception as e:
                print(f"Error in evaluation: {e}")
                continue
    
    # Calculate EER with robust error handling
    try:
        if len(all_targets) > 0 and len(all_scores) > 0:
            unique_targets = np.unique(all_targets)
            if len(unique_targets) >= 2:
                fpr, tpr, thresholds = roc_curve(all_targets, all_scores, pos_label=1)
                fnr = 1 - tpr
                idx = np.nanargmin(np.absolute(fnr - fpr))
                eer = (fpr[idx] + fnr[idx]) / 2
            else:
                eer = 0.5
        else:
            eer = 0.5
    except Exception as e:
        print(f"Error calculating EER: {e}")
        eer = 0.5
    
    return eer

def evaluate_model(model, architecture, test_loader, device, log_to_wandb=True):
    """Evaluate the model with the best architecture"""
    model.eval()
    all_targets = []
    all_scores = []
    test_loss = 0.0
    correct = 0
    total = 0
    
    # Determine if architecture is from PPO or DARTS
    is_ppo_arch = architecture.dim() == 1
    
    # Create a progress bar for evaluation
    eval_pbar = tqdm(test_loader, desc="Evaluating")
    
    with torch.no_grad():
        for inputs, targets in eval_pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Choose correct evaluation mode
            if is_ppo_arch:
                # PPO architecture
                outputs = model(inputs, architecture)
            else:
                # DARTS architecture - set model's alphas and use discrete mode
                model._alphas.data = architecture.data
                outputs = model(inputs, discrete=True)
            
            # Compute loss
            loss = F.cross_entropy(outputs, targets)
            test_loss += loss.item()
            
            # Compute accuracy
            _, predicted = outputs.max(1)