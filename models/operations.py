import torch
import torch.nn as nn
import torch.nn.functional as F

# Get global device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Base operation class
class Operation(nn.Module):
    """Base class for all operations in the search space"""
    def __init__(self, channels, stride=1):
        super(Operation, self).__init__()
        self.channels = channels
        self.stride = stride
    
    def forward(self, x):
        raise NotImplementedError

# Convolutional Block
class ConvBlock(Operation):
    def __init__(self, channels, kernel_size, stride=1):
        super(ConvBlock, self).__init__(channels, stride)
        self.conv = nn.Conv1d(channels, channels, kernel_size, stride=stride, padding=kernel_size//2)
        self.bn = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

# LSTM Block
class LSTM(Operation):
    def __init__(self, channels, stride=1):
        super(LSTM, self).__init__(channels, stride)
        self.lstm = nn.LSTM(channels, channels, batch_first=True)
        self.input_proj = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=1)
    
    def forward(self, x):
        # x shape: [B, C, T]
        batch_size, channels, seq_len = x.size()
        x = x.permute(0, 2, 1)  # [B, C, T] -> [B, T, C]
        
        # LSTM with batch_first=True
        x, _ = self.lstm(x)
        
        # Return to original dimension ordering
        x = x.permute(0, 2, 1)  # [B, T, C] -> [B, C, T]
        
        return x

# Dilated Convolution
class Dilated(Operation):
    def __init__(self, channels, stride=1):
        super(Dilated, self).__init__(channels, stride)
        self.conv = nn.Conv1d(channels, channels, 3, stride=stride, padding=2, dilation=2)
        self.bn = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

# Skip Connection
class SkipConnect(Operation):
    def __init__(self, channels, stride=1):
        super(SkipConnect, self).__init__(channels, stride)
    
    def forward(self, x):
        return x

# Self-Attention Block
class Attention(Operation):
    def __init__(self, channels, stride=1):
        super(Attention, self).__init__(channels, stride)
        self.query = nn.Conv1d(channels, channels, 1)
        self.key = nn.Conv1d(channels, channels, 1)
        self.value = nn.Conv1d(channels, channels, 1)
        self.scale = torch.sqrt(torch.FloatTensor([channels])).to(device)
    
    def forward(self, x):
        # x shape: [B, C, T]
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        # Reshape for attention
        batch_size, C, T = q.size()
        q = q.permute(0, 2, 1)  # [B, T, C]
        k = k.permute(0, 2, 1)  # [B, T, C]
        v = v.permute(0, 2, 1)  # [B, T, C]
        
        # Self-attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attention = F.softmax(scores, dim=-1)
        context = torch.matmul(attention, v)
        
        # Reshape back
        return context.permute(0, 2, 1)  # [B, C, T]

# Separable Convolution
class SeparableConv(Operation):
    def __init__(self, channels, stride=1):
        super(SeparableConv, self).__init__(channels, stride)
        self.depthwise = nn.Conv1d(channels, channels, 3, stride=stride, padding=1, groups=channels)
        self.pointwise = nn.Conv1d(channels, channels, 1)
        self.bn = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(self.bn(self.pointwise(self.depthwise(x))))

# Squeeze-and-Excitation Block
class SqueezeExcitation(Operation):
    def __init__(self, channels, stride=1, reduction=16):
        super(SqueezeExcitation, self).__init__(channels, stride)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Conv1d(channels, max(channels // reduction, 1), kernel_size=1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv1d(max(channels // reduction, 1), channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x shape: [B, C, T]
        scale = self.avg_pool(x)
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        scale = self.sigmoid(scale)
        return x * scale

# Frequency-Aware Convolution (Audio-specific)
class FrequencyAwareConv(Operation):
    def __init__(self, channels, stride=1, bands=4):
        super(FrequencyAwareConv, self).__init__(channels, stride)
        # Ensure band_size is at least 1
        self.band_size = max(channels // bands, 1)
        self.bands = min(bands, channels)
        
        # Create different kernel sizes for frequency bands
        self.convs = nn.ModuleList([
            nn.Conv1d(self.band_size, self.band_size, 3 + i*2, padding=(3+i*2)//2, stride=stride)
            for i in range(self.bands)
        ])
        
        self.bn = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Split along channel dimension into bands
        split_sizes = [self.band_size] * self.bands
        remaining = self.channels - (self.band_size * self.bands)
        if remaining > 0:
            split_sizes[-1] += remaining
            
        x_bands = torch.split(x, split_sizes, dim=1)
        
        # Process each band separately
        out_bands = []
        for i, band in enumerate(x_bands):
            if i < self.bands:
                out_bands.append(self.convs[i](band))
        
        # Concatenate results
        out = torch.cat(out_bands, dim=1)
        
        return self.relu(self.bn(out))

# Gated Convolution
class GatedConv(Operation):
    def __init__(self, channels, stride=1):
        super(GatedConv, self).__init__(channels, stride)
        self.conv_features = nn.Conv1d(channels, channels, 3, stride=stride, padding=1)
        self.conv_gate = nn.Conv1d(channels, channels, 3, stride=stride, padding=1)
        self.bn = nn.BatchNorm1d(channels)
    
    def forward(self, x):
        features = self.conv_features(x)
        gate = torch.sigmoid(self.conv_gate(x))
        return self.bn(features * gate)

# Mixed Operation (Weighted sum of operations)
class MixedOp(nn.Module):
    def __init__(self, channels, stride=1):
        super(MixedOp, self).__init__()
        self.ops = nn.ModuleList([
            # Original operations
            ConvBlock(channels, 3, stride),
            ConvBlock(channels, 5, stride),
            LSTM(channels, stride),
            Dilated(channels, stride),
            SkipConnect(channels, stride),
            Attention(channels, stride),
            # New operations
            SeparableConv(channels, stride),
            SqueezeExcitation(channels, stride),
            FrequencyAwareConv(channels, stride),
            GatedConv(channels, stride)
        ])
    
    def forward(self, x, weights):
        """Forward pass with operation weights"""
        return sum(w * op(x) for w, op in zip(weights, self.ops))