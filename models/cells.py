import torch
import torch.nn as nn
from models.operations import MixedOp

# Cell structure
class Cell(nn.Module):
    def __init__(self, channels, num_nodes=4):
        super(Cell, self).__init__()
        self.channels = channels
        self.num_nodes = num_nodes
        
        # For each node, create edges from all previous nodes
        self.edges = nn.ModuleList()
        for i in range(num_nodes):
            for j in range(i+1):  # connections from input and previous nodes
                self.edges.append(MixedOp(channels))
        
        # Output projection
        self.project = nn.Conv1d(channels * num_nodes, channels, 1)
    
    def forward(self, x, weights):
        """
        Forward pass through the cell
        Args:
            x: Input tensor [B, C, T]
            weights: List of weight tensors for each edge
        """
        states = [x]
        offset = 0
        
        # Process each node
        for i in range(self.num_nodes):
            # Gather inputs from previous nodes
            node_inputs = []
            for j in range(i+1):
                edge_output = self.edges[offset + j](states[j], weights[offset + j])
                node_inputs.append(edge_output)
            
            node_input = sum(node_inputs)
            offset += i+1
            states.append(node_input)
        
        # Concatenate all intermediate nodes
        cat_states = torch.cat(states[1:], dim=1)
        
        return self.project(cat_states)