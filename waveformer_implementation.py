"""
WaveFormer: Transformer-based Denoising Method for Gravitational-Wave Data
Implementation based on the paper: https://arxiv.org/html/2212.14283v2

This implementation adapts the WaveFormer architecture for classification between
gravitational wave signals and glitches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer input."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]


class ConvolutionalEmbedding(nn.Module):
    """Convolutional embedding module for local feature extraction."""
    
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        self.conv1d = nn.Conv1d(input_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        self.activation = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_dim)
        x = x.transpose(1, 2)  # (batch_size, input_dim, seq_len)
        x = self.conv1d(x)
        x = self.activation(x)
        x = x.transpose(1, 2)  # (batch_size, seq_len, hidden_dim)
        return x


class TokenEmbedding(nn.Module):
    """Token embedding module."""
    
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        self.conv1d = nn.Conv1d(input_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_dim)
        x = x.transpose(1, 2)  # (batch_size, input_dim, seq_len)
        x = self.conv1d(x)
        x = x.transpose(1, 2)  # (batch_size, seq_len, hidden_dim)
        return x


class ResidualModule(nn.Module):
    """Residual module with 2D convolution for mid-level feature extraction."""
    
    def __init__(self, hidden_dim: int, kernel_size: int = 7, dropout: float = 0.1):
        super().__init__()
        self.conv2d = nn.Conv2d(1, 1, kernel_size, padding=kernel_size//2)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, hidden_dim)
        batch_size, seq_len, hidden_dim = x.shape
        
        # Reshape for 2D convolution - treat as single channel 2D image
        x_reshaped = x.unsqueeze(1)  # (batch_size, 1, seq_len, hidden_dim)
        
        residual = x_reshaped
        x_reshaped = self.conv2d(x_reshaped)
        x_reshaped = self.activation(x_reshaped)
        x_reshaped = x_reshaped + residual
        
        # Reshape back
        x = x_reshaped.squeeze(1)  # (batch_size, seq_len, hidden_dim)
        x = self.dropout(x)
        
        return x


class SwiGLU(nn.Module):
    """SwiGLU activation function as mentioned in the paper."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.w = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.w(x)) * self.v(x)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention module with modifications from the paper."""
    
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Remove bias as mentioned in the paper
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        
        # Linear projections
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        output = self.w_o(context)
        return output


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with SwiGLU activation."""
    
    def __init__(self, d_model: int, n_heads: int = 8, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # MLP with SwiGLU activation
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            SwiGLU(d_ff),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection
        attn_output = self.self_attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # MLP with residual connection
        mlp_output = self.mlp(x)
        x = self.norm2(x + self.dropout(mlp_output))
        
        return x


class WaveFormerEncoder(nn.Module):
    """WaveFormer encoder with multiple transformer layers."""
    
    def __init__(self, d_model: int, n_layers: int = 24, n_heads: int = 32, 
                 d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return x


class WaveFormerClassifier(nn.Module):
    """
    WaveFormer model adapted for classification between signals and glitches.
    
    Based on the paper architecture but modified for classification task.
    """
    
    def __init__(self, 
                 input_length: int = 16512,  # 8.0625s * 2048Hz
                 win_length: int = 256,      # 0.125s * 2048Hz
                 stride: int = 128,          # 50% overlap
                 d_model: int = 2048,
                 n_layers: int = 24,
                 n_heads: int = 32,
                 d_ff: int = 2048,
                 num_classes: int = 2,       # Signal vs Glitch
                 dropout: float = 0.1):
        super().__init__()
        
        self.input_length = input_length
        self.win_length = win_length
        self.stride = stride
        self.d_model = d_model
        
        # Calculate number of subsequences
        self.n_subsequences = (input_length - win_length) // stride + 1
        
        # Embedding modules
        self.token_embedding = TokenEmbedding(1, d_model // 3)  # Input is 1D signal
        self.conv_embedding = ConvolutionalEmbedding(1, d_model // 3)
        self.positional_encoding = PositionalEncoding(d_model // 3, self.n_subsequences)
        
        # Residual module
        self.residual_module = ResidualModule(d_model, dropout=dropout)
        
        # Transformer encoder
        self.encoder = WaveFormerEncoder(d_model, n_layers, n_heads, d_ff, dropout)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
    def segment_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Segment input into overlapping subsequences.
        
        Args:
            x: Input tensor of shape (batch_size, input_length)
            
        Returns:
            Segmented tensor of shape (batch_size, n_subsequences, win_length)
        """
        batch_size = x.shape[0]
        segments = []
        
        for i in range(0, self.input_length - self.win_length + 1, self.stride):
            segment = x[:, i:i + self.win_length]
            segments.append(segment)
        
        return torch.stack(segments, dim=1)  # (batch_size, n_subsequences, win_length)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of WaveFormer classifier.
        
        Args:
            x: Input tensor of shape (batch_size, input_length)
            mask: Optional attention mask
            
        Returns:
            Classification logits of shape (batch_size, num_classes)
        """
        batch_size = x.shape[0]
        
        # Segment input into subsequences
        x_segments = self.segment_input(x)  # (batch_size, n_subsequences, win_length)
        
        # Reshape for embedding
        x_segments = x_segments.unsqueeze(-1)  # (batch_size, n_subsequences, win_length, 1)
        
        # Process each subsequence
        embedded_segments = []
        for i in range(self.n_subsequences):
            segment = x_segments[:, i, :, :]  # (batch_size, win_length, 1)
            
            # Token embedding
            te = self.token_embedding(segment)  # (batch_size, win_length, d_model//3)
            
            # Convolutional embedding
            ce = self.conv_embedding(segment)  # (batch_size, win_length, d_model//3)
            
            # Positional embedding
            pe = self.positional_encoding(torch.zeros_like(te))  # (batch_size, win_length, d_model//3)
            
            # Combine embeddings
            dense_features = torch.cat([te, ce, pe], dim=-1)  # (batch_size, win_length, d_model)
            
            # Global average pooling to get sequence representation
            segment_repr = dense_features.mean(dim=1)  # (batch_size, d_model)
            embedded_segments.append(segment_repr)
        
        # Stack segment representations
        x = torch.stack(embedded_segments, dim=1)  # (batch_size, n_subsequences, d_model)
        
        # Apply residual module
        x = self.residual_module(x)
        
        # Apply transformer encoder
        x = self.encoder(x, mask)
        
        # Global average pooling across subsequences
        x = x.mean(dim=1)  # (batch_size, d_model)
        
        # Classification
        logits = self.classifier(x)  # (batch_size, num_classes)
        
        return logits


class MaskedLoss(nn.Module):
    """
    Masked loss function as described in the paper.
    Adapted for classification task.
    """
    
    def __init__(self, alpha: float = 1/6):
        super().__init__()
        self.alpha = alpha
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute masked loss.
        
        Args:
            predictions: Model predictions (batch_size, num_classes)
            targets: Ground truth labels (batch_size,)
            mask: Optional mask for weighted loss
            
        Returns:
            Masked loss value
        """
        loss = self.criterion(predictions, targets)
        
        if mask is not None:
            # Apply mask weighting
            masked_loss = loss * mask
            unmasked_loss = loss * (1 - mask)
            total_loss = masked_loss.mean() + self.alpha * unmasked_loss.mean()
        else:
            total_loss = loss.mean()
        
        return total_loss


def create_waveformer_model(config: dict = None) -> WaveFormerClassifier:
    """
    Create a WaveFormer model with default or custom configuration.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        WaveFormer model instance
    """
    default_config = {
        'input_length': 16512,  # 8.0625s * 2048Hz
        'win_length': 256,      # 0.125s * 2048Hz  
        'stride': 128,          # 50% overlap
        'd_model': 2048,
        'n_layers': 24,
        'n_heads': 32,
        'd_ff': 2048,
        'num_classes': 2,
        'dropout': 0.1
    }
    
    if config:
        default_config.update(config)
    
    model = WaveFormerClassifier(**default_config)
    return model


if __name__ == "__main__":
    # Test the model
    model = create_waveformer_model()
    
    # Create dummy input
    batch_size = 2
    input_length = 16512
    x = torch.randn(batch_size, input_length)
    
    # Forward pass
    with torch.no_grad():
        logits = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {logits.shape}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

