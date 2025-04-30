import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
from matplotlib.patches import FancyArrowPatch

def visualize_architecture(architecture, num_cells=3, num_nodes=4, num_ops=10, save_path='architecture_visualization.png', 
                           save_to_wandb=False, title="Hybrid PPO-DARTS Architecture"):
    """
    Visualize the architecture discovered by the hybrid PPO-DARTS approach.
    
    Args:
        architecture: Tensor containing the selected operations (for PPO) or operation weights (for DARTS)
        num_cells: Number of cells in the architecture
        num_nodes: Number of intermediate nodes in each cell
        num_ops: Number of possible operations for each edge
        save_path: Path to save the visualization
        save_to_wandb: Whether to save the visualization to wandb
        title: Title of the visualization
    
    Returns:
        Path to the saved visualization
    """
    # Define operation names for visualization
    operation_names = [
        'Conv 3x3', 'Conv 5x5', 'LSTM', 'Dilated Conv', 'Skip Connect',
        'Self Attention', 'Separable Conv', 'Squeeze-Excitation', 'Frequency-Aware', 'Gated Conv'
    ]
    
    # Define colors for different operations
    colors = list(mcolors.TABLEAU_COLORS.values())
    
    # Calculate edges per cell
    edges_per_cell = sum(range(1, num_nodes+1))
    total_edges = num_cells * edges_per_cell
    
    # Check if architecture is from PPO (1D tensor of indices) or DARTS (2D tensor of weights)
    is_ppo = architecture.dim() == 1
    
    # Create a figure with multiple subplots (one per cell)
    fig, axes = plt.subplots(1, num_cells, figsize=(6*num_cells, 6), constrained_layout=True)
    if num_cells == 1:
        axes = [axes]
    
    plt.suptitle(title, fontsize=20, y=1.05)
    
    # For each cell
    for cell_idx in range(num_cells):
        ax = axes[cell_idx]
        G = nx.DiGraph()
        
        # Label for cell type
        cell_type = "Normal Cell" if cell_idx % 2 == 0 else "Expand Cell"
        ax.set_title(f"Cell {cell_idx+1}: {cell_type}", fontsize=16)
        
        # Add nodes to the graph
        for i in range(num_nodes + 2):  # +2 for input and output nodes
            if i == 0 or i == 1:
                G.add_node(i, label=f"Input {i+1}")
            elif i == num_nodes + 1:
                G.add_node(i, label="Output")
            else:
                G.add_node(i, label=f"Node {i}")
        
        # Add edges to the graph based on the architecture
        edge_offset = cell_idx * edges_per_cell
        edge_count = 0
        
        for i in range(2, num_nodes + 2):  # For each intermediate node
            for j in range(i):  # For all previous nodes
                edge_idx = edge_offset + edge_count
                
                if is_ppo:
                    # For PPO, the architecture contains operation indices
                    if edge_idx < len(architecture):
                        op_idx = int(architecture[edge_idx].item())
                        op_name = operation_names[op_idx]
                        G.add_edge(j, i, label=op_name, color=colors[op_idx])
                else:
                    # For DARTS, the architecture contains operation weights
                    if edge_idx < architecture.size(0):
                        op_idx = torch.argmax(architecture[edge_idx]).item()
                        op_name = operation_names[op_idx]
                        G.add_edge(j, i, label=op_name, color=colors[op_idx])
                
                edge_count += 1
        
        # Position nodes in a hierarchical layout
        pos = {}
        pos[0] = np.array([-1, 0.5])
        pos[1] = np.array([-1, -0.5])
        
        # Position intermediate nodes in a line
        for i in range(2, num_nodes + 2):
            level = (i - 1) / (num_nodes + 1)
            pos[i] = np.array([level*2 - 1, 0])
        
        # Adjust output node position
        pos[num_nodes + 1] = np.array([1, 0])
        
        # Draw nodes
        for n in G.nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=[n], node_size=1200, 
                                  node_color='lightblue', alpha=0.8, ax=ax)
        
        # Draw node labels
        nx.draw_networkx_labels(G, pos, labels=nx.get_node_attributes(G, 'label'), 
                               font_size=10, font_family='sans-serif', ax=ax)
        
        # Draw edges with custom arrows
        for u, v, data in G.edges(data=True):
            color = data.get('color', 'gray')
            label = data.get('label', '')
            
            # Create a curved arrow
            arrow = FancyArrowPatch(pos[u], pos[v], connectionstyle="arc3,rad=0.2",
                                   arrowstyle="-|>", color=color, lw=1.5, alpha=0.8)
            ax.add_patch(arrow)
            
            # Add edge label (operation name)
            # Calculate label position (midpoint of the curved edge with slight offset)
            x = (pos[u][0] + pos[v][0]) / 2
            y = (pos[u][1] + pos[v][1]) / 2
            offset = 0.1 if pos[u][1] < pos[v][1] else -0.1
            ax.text(x, y + offset, label, fontsize=8, ha='center', va='center', 
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
        
        # Remove axis ticks and frame
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Log to wandb if requested
    if save_to_wandb:
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({"architecture_visualization": wandb.Image(save_path)})
        except ImportError:
            print("Warning: wandb not installed, skipping wandb logging")
    
    plt.close()
    return save_path

def analyze_architecture_statistics(best_architecture, is_ppo=True, num_cells=3, num_nodes=4, num_ops=10):
    """
    Analyze the statistics of the discovered architecture.
    
    Args:
        best_architecture: The architecture tensor (PPO indices or DARTS weights)
        is_ppo: Whether the architecture is from PPO (True) or DARTS (False)
        num_cells: Number of cells
        num_nodes: Number of nodes per cell
        num_ops: Number of operations
    
    Returns:
        Dictionary of statistics about the architecture
    """
    operation_names = [
        'Conv 3x3', 'Conv 5x5', 'LSTM', 'Dilated Conv', 'Skip Connect',
        'Self Attention', 'Separable Conv', 'Squeeze-Excitation', 'Frequency-Aware', 'Gated Conv'
    ]
    
    # Calculate edges per cell
    edges_per_cell = sum(range(1, num_nodes+1))
    total_edges = num_cells * edges_per_cell
    
    # Initialize operation counts
    op_counts = {op: 0 for op in operation_names}
    
    # Count operations by cell
    op_counts_by_cell = []
    for cell_idx in range(num_cells):
        cell_op_counts = {op: 0 for op in operation_names}
        edge_offset = cell_idx * edges_per_cell
        
        for edge_idx in range(edges_per_cell):
            global_edge_idx = edge_offset + edge_idx
            
            if is_ppo:
                # For PPO architecture (indices)
                if global_edge_idx < len(best_architecture):
                    op_idx = int(best_architecture[global_edge_idx].item())
                    op_name = operation_names[op_idx]
                    op_counts[op_name] += 1
                    cell_op_counts[op_name] += 1
            else:
                # For DARTS architecture (weights)
                if global_edge_idx < best_architecture.size(0):
                    op_idx = torch.argmax(best_architecture[global_edge_idx]).item()
                    op_name = operation_names[op_idx]
                    op_counts[op_name] += 1
                    cell_op_counts[op_name] += 1
        
        op_counts_by_cell.append(cell_op_counts)
    
    # Calculate percentages
    total_ops = sum(op_counts.values())
    op_percentages = {op: count/total_ops*100 for op, count in op_counts.items()}
    
    # Find most and least common operations
    most_common_op = max(op_counts.items(), key=lambda x: x[1])[0]
    least_common_op = min(op_counts.items(), key=lambda x: x[1])[0]
    
    # Analyze patterns
    patterns = {}
    patterns["most_common_op"] = most_common_op
    patterns["most_common_percentage"] = op_percentages[most_common_op]
    patterns["least_common_op"] = least_common_op
    patterns["least_common_percentage"] = op_percentages[least_common_op]
    
    # Return combined statistics
    statistics = {
        "operation_counts": op_counts,
        "operation_percentages": op_percentages,
        "patterns": patterns,
        "by_cell": op_counts_by_cell
    }
    
    return statistics