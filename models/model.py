import torch
import torch.nn as nn
import torch.nn.functional as F
from models.cells import Cell

class DeepfakeDetectionModel(nn.Module):
    def __init__(self, input_channels, num_cells=3, num_nodes=4, num_ops=10):
        super(DeepfakeDetectionModel, self).__init__()
        self.input_channels = input_channels
        self.num_cells = num_cells
        self.num_nodes = num_nodes
        self.num_ops = num_ops
        
        # Initial projection
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        # Cells
        self.cells = nn.ModuleList()
        for i in range(num_cells):
            self.cells.append(Cell(64, num_nodes))
        
        # Classification head
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(64, 2)  # Binary classification
        
        # Initialize architectural parameters (alphas) for DARTS
        self._initialize_alphas()
        
        # Calculate total number of weights needed for PPO
        edges_per_cell = sum(range(1, num_nodes+1))
        self.total_weights = num_cells * edges_per_cell * num_ops
    
    def _initialize_alphas(self):
        """Initialize architectural parameters for DARTS"""
        edges_per_cell = sum(range(1, self.num_nodes+1))
        total_edges = self.num_cells * edges_per_cell
        # Create parameter tensor for alphas
        self._alphas = nn.Parameter(torch.zeros(total_edges, self.num_ops))
        # Initialize with small random values
        nn.init.normal_(self._alphas, mean=0, std=0.001)
    
    def alphas(self):
        """Return architectural parameters for optimizer"""
        return [self._alphas]  # Wrapped in list for optimizer compatibility
    
    def weights(self):
        """Return model weights excluding alphas"""
        return [p for n, p in self.named_parameters() if '_alphas' not in n]
    
    def forward(self, x, architecture_weights=None, discrete=False):
        """
        Forward pass with multiple modes:
        - PPO mode: Using external architecture_weights
        - DARTS mode: Using internal alphas with continuous relaxation
        - Evaluation mode: Using internal alphas with discrete operations
        """
        # Input shape handling
        if x.shape[1] == self.input_channels:
            # Input is already [B, C, T]
            pass
        else:
            # Input is [B, T, C], convert to [B, C, T]
            x = x.permute(0, 2, 1)
        
        # Process input
        x = self.stem(x)
        
        # Determine which weights to use
        edges_per_cell = sum(range(1, self.num_nodes+1))
        
        if architecture_weights is not None:
            # PPO mode: use external weights
            edge_weights = []
            for i in range(len(architecture_weights) // self.num_ops):
                start_idx = i * self.num_ops
                end_idx = start_idx + self.num_ops
                # Apply softmax to get probability distribution
                edge_weights.append(F.softmax(architecture_weights[start_idx:end_idx], dim=0))
        else:
            # DARTS mode: use internal alphas
            if discrete:
                # Convert to discrete (one-hot) for evaluation
                max_indices = torch.argmax(self._alphas, dim=1)
                edge_weights = []
                for j, idx in enumerate(max_indices):
                    weights = torch.zeros_like(self._alphas[j])
                    weights[idx] = 1.0
                    edge_weights.append(weights)
            else:
                # Use softmax for continuous relaxation
                edge_weights = [F.softmax(self._alphas[i], dim=0) for i in range(self._alphas.size(0))]
        
        # Process cells
        offset = 0
        for i, cell in enumerate(self.cells):
            cell_weights = edge_weights[offset:offset + edges_per_cell]
            offset += edges_per_cell
            x = cell(x, cell_weights)
        
        # Classification
        x = self.pool(x).squeeze(-1)
        x = self.classifier(x)
        
        return x