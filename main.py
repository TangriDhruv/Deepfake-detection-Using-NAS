import os
import time
import torch
import numpy as np
import random
import wandb
import json
import warnings
warnings.filterwarnings('ignore')

# Import local modules
from config import *
from data.dataset import ASVSpoofDataset
from models.model import DeepfakeDetectionModel
from utils.training import search_architecture_hybrid
from utils.evaluation import evaluate_model
#from utils.visualization import (
#    visualize_architecture, 
#    visualize_darts_architecture,
#    visualize_ppo_architecture,
#    analyze_architecture_statistics
#)
from utils.visualization import *

def setup_progress_monitoring():
    """Setup enhanced progress monitoring"""
    import threading
    import time
    import psutil
    import os
    
    def monitor_resources():
        process = psutil.Process(os.getpid())
        start_time = time.time()
        while True:
            try:
                # CPU usage
                cpu_percent = process.cpu_percent(interval=1)
                # Memory usage
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / (1024 * 1024)
                # GPU memory if available
                gpu_memory_mb = 0
                try:
                    if torch.cuda.is_available():
                        gpu_memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                except:
                    pass
                
                elapsed = time.time() - start_time
                hours, remainder = divmod(elapsed, 3600)
                minutes, seconds = divmod(remainder, 60)
                
                print(f"\r[{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}] "
                      f"CPU: {cpu_percent:.1f}% | RAM: {memory_mb:.1f} MB | "
                      f"GPU: {gpu_memory_mb:.1f} MB", end="", flush=True)
                
                time.sleep(5)  # Update every 5 seconds
            except:
                break
    
    # Start monitoring in a background thread
    monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
    monitor_thread.start()
    print("Resource monitoring started...")

# Set random seeds for reproducibility
def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

# Helper function for JSON serialization
def convert_numpy_types(obj):
    """Convert NumPy types to Python native types for JSON serialization"""
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def numpy_safe_json_dump(obj, f, indent=4):
    """JSON dump that handles NumPy types"""
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
    
    json.dump(obj, f, indent=indent, cls=NumpyEncoder)

def main():
    # Start the resource monitoring
    setup_progress_monitoring()
    
    # Set random seed
    set_seed()
    
    # Create unique experiment name based on timestamp
    experiment_name = f"ASVspoof2019_NAS_{int(time.time())}"
    
    print("Starting Deepfake Audio Detection with NAS")
    print("=" * 80)
    
    # Log dataset information
    print(f"Train data directory: {DATA_DIR_TRAIN}")
    print(f"Train protocol file: {TRAIN_PROTOCOL}")
    print(f"Dev data directory: {DATA_DIR_DEV}")
    print(f"Dev protocol file: {DEV_PROTOCOL}")
    print(f"Eval data directory: {DATA_DIR_EVAL}")
    print(f"Eval protocol file: {EVAL_PROTOCOL}")
    
    # Initialize wandb
    if WANDB_API_KEY:
        wandb.login(key=WANDB_API_KEY)
    
    wandb.init(
        project=WANDB_PROJECT,
        name=f"{experiment_name}_{SEARCH_METHOD}",
        config={
            "feature_type": FEATURE_TYPE,
            "max_sequence_length": MAX_SEQ_LEN,
            "batch_size_train": BATCH_SIZE_TRAIN,
            "batch_size_eval": BATCH_SIZE_EVAL,
            "num_workers": NUM_WORKERS,
            "dataset": "ASVspoof2019 LA",
            "search_method": SEARCH_METHOD
        }
    )
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    print("Creating datasets...")
    
    print("Loading training dataset...")
    train_dataset = ASVSpoofDataset(
        root_dir=DATA_DIR_TRAIN,
        protocol_file=TRAIN_PROTOCOL,
        feature_type=FEATURE_TYPE,
        max_len=MAX_SEQ_LEN,
        is_train=True
    )
    
    print("Loading validation dataset...")
    dev_dataset = ASVSpoofDataset(
        root_dir=DATA_DIR_DEV,
        protocol_file=DEV_PROTOCOL,
        feature_type=FEATURE_TYPE,
        max_len=MAX_SEQ_LEN,
        is_train=False
    )
    
    print("Loading evaluation dataset...")
    eval_dataset = ASVSpoofDataset(
        root_dir=DATA_DIR_EVAL,
        protocol_file=EVAL_PROTOCOL,
        feature_type=FEATURE_TYPE,
        max_len=MAX_SEQ_LEN,
        is_train=False
    )
    
    # Log dataset sizes
    print(f"Training dataset size: {len(train_dataset)} samples")
    print(f"Validation dataset size: {len(dev_dataset)} samples")
    print(f"Evaluation dataset size: {len(eval_dataset)} samples")
    wandb.log({
        "train_dataset_size": len(train_dataset),
        "val_dataset_size": len(dev_dataset),
        "eval_dataset_size": len(eval_dataset)
    })
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE_TRAIN, 
        shuffle=True, 
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    dev_loader = torch.utils.data.DataLoader(
        dev_dataset, 
        batch_size=BATCH_SIZE_EVAL, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, 
        batch_size=BATCH_SIZE_EVAL, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    # Architecture search parameters
    nas_config = {
        "input_channels": INPUT_CHANNELS,
        "num_cells": NUM_CELLS,
        "num_nodes": NUM_NODES,
        "num_ops": NUM_OPS,
        "epochs": NAS_EPOCHS,
        "ppo_updates": PPO_UPDATES,
        "project_name": WANDB_PROJECT
    }
    
    # Log NAS configuration
    print("\nNeural Architecture Search Configuration:")
    for key, value in nas_config.items():
        print(f"  {key}: {value}")
    
    # Create output directory for results
    output_dir = os.path.join(OUTPUT_DIR, f"{experiment_name}_{SEARCH_METHOD}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nResults will be saved to {output_dir}")
    
    # Perform architecture search using hybrid PPO-DARTS approach
    print(f"\nStarting Neural Architecture Search using {SEARCH_METHOD.upper()} method...")
    
    model, best_architecture, best_val_eer = search_architecture_hybrid(
        train_loader=train_loader,
        val_loader=dev_loader,
        device=device,
        input_channels=nas_config["input_channels"],
        num_cells=nas_config["num_cells"],
        num_nodes=nas_config["num_nodes"],
        num_ops=nas_config["num_ops"],
        epochs=nas_config["epochs"],
        ppo_updates=nas_config["ppo_updates"],
        project_name=nas_config["project_name"] + "-hybrid"
    )
    
    # Save best architecture
    torch.save(best_architecture, os.path.join(output_dir, "best_architecture.pt"))
    
    # Initialize a new wandb run for final evaluation
    wandb.finish()  # Finish the NAS run
    wandb.init(
        project=WANDB_PROJECT,
        name=f"{experiment_name}_{SEARCH_METHOD}_final_evaluation",
        config={
            "feature_type": FEATURE_TYPE,
            "best_val_eer": float(best_val_eer),  # Convert to Python float
            "search_method": SEARCH_METHOD
        }
    )
    
    # Evaluate on the evaluation set
    print("\nPerforming final evaluation on test set...")
    test_eer, eer_threshold = evaluate_model(model, best_architecture, eval_loader, device)
    
    # Log final metrics - convert NumPy types to Python native types
    wandb.log({
        "final_test_eer": float(test_eer),
        "eer_threshold": float(eer_threshold)
    })
    
    # Visualize the architecture with annotations
    print("\nVisualizing the best architecture...")
    # Check if it's a PPO or DARTS architecture
    is_ppo_arch = best_architecture.dim() == 1
    
    if is_ppo_arch:
        fig_path = visualize_ppo_architecture(
            best_architecture, 
            num_cells=nas_config["num_cells"], 
            num_nodes=nas_config["num_nodes"], 
            num_ops=nas_config["num_ops"],
            save_to_wandb=True
        )
    else:
        fig_path = visualize_darts_architecture(
            best_architecture, 
            num_cells=nas_config["num_cells"], 
            num_nodes=nas_config["num_nodes"], 
            num_ops=nas_config["num_ops"],
            save_to_wandb=True
        )
    
    # Save architecture visualization
    import shutil
    shutil.copy(fig_path, os.path.join(output_dir, "architecture_visualization.png"))
    
    # Create a summary report
    summary = {
        "experiment_name": experiment_name,
        "search_method": SEARCH_METHOD,
        "feature_type": FEATURE_TYPE,
        "best_validation_eer": float(best_val_eer),
        "test_eer": float(test_eer),
        "eer_threshold": float(eer_threshold),
        "model_architecture": {
            "num_cells": nas_config["num_cells"],
            "num_nodes": nas_config["num_nodes"],
            "num_operations": nas_config["num_ops"]
        }
    }
    
    # Save summary as JSON
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        numpy_safe_json_dump(summary, f)
    
    # Also save as text for easy reading
    with open(os.path.join(output_dir, "summary.txt"), "w") as f:
        f.write("ASVspoof 2019 Deepfake Detection Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Experiment name: {experiment_name}\n")
        f.write(f"Search method: {SEARCH_METHOD.upper()}\n")
        f.write(f"Feature type: {FEATURE_TYPE}\n\n")
        f.write("Performance metrics:\n")
        f.write(f"  Best validation EER: {float(best_val_eer):.4f}\n")
        f.write(f"  Test EER: {float(test_eer):.4f}\n")
        f.write(f"  EER threshold: {float(eer_threshold):.4f}\n\n")
        f.write("Model architecture:\n")
        f.write(f"  Number of cells: {nas_config['num_cells']}\n")
        f.write(f"  Number of nodes per cell: {nas_config['num_nodes']}\n")
        f.write(f"  Number of operations: {nas_config['num_ops']}\n")
    
    # Log summary to wandb
    wandb.save(os.path.join(output_dir, "summary.txt"))
    wandb.save(os.path.join(output_dir, "summary.json"))
    
    print("\nExperiment completed!")
    print(f"Final Test EER: {float(test_eer):.4f}, EER Threshold: {float(eer_threshold):.4f}")
    print(f"All results saved to {output_dir}")
    print("=" * 80)
    
    # Finish wandb
    wandb.finish()

if __name__ == "__main__":
    main()
