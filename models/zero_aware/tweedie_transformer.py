"""
Enhanced Tweedie Transformer for zero-inflated time series forecasting.

This implementation improves upon the original Tweedie Transformer with:
- Better numerical stability
- Improved attention mechanisms
- Zero-aware features
- More robust training procedures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any, Tuple
import math

from ..losses.tweedie_loss import TweedieLoss


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (seq_len, batch_size, d_model)
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class ZeroAwareMultiheadAttention(nn.Module):
    """
    Multi-head attention with zero-aware masking.
    
    This attention mechanism can optionally mask out zero values or give them
    reduced attention, which can be beneficial for zero-inflated data.
    """
    
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1,
                 zero_aware: bool = True, zero_attention_weight: float = 0.1):
        super().__init__()
        
        assert d_model % nhead == 0
        
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.zero_aware = zero_aware
        self.zero_attention_weight = zero_attention_weight
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                zero_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (batch_size, seq_len, d_model)
            key: (batch_size, seq_len, d_model)
            value: (batch_size, seq_len, d_model)
            mask: Optional attention mask
            zero_mask: Optional mask indicating zero positions
        """
        batch_size, seq_len = query.size(0), query.size(1)
        
        # Linear projections
        query_seq_len = query.size(1)
        key_seq_len = key.size(1)
        value_seq_len = value.size(1)
        
        Q = self.w_q(query).view(batch_size, query_seq_len, self.nhead, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, key_seq_len, self.nhead, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, value_seq_len, self.nhead, self.d_k).transpose(1, 2)
        
        # Debug: Check V tensor shape
        # print(f"V shape after projection: {V.shape}")
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply regular mask
        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)
        
        # Apply zero-aware attention (simplified to avoid dimension issues)
        if self.zero_aware and zero_mask is not None:
            # Simply reduce attention scores for zero positions
            # zero_mask shape: (batch_size, seq_len, input_dim) -> reduce to (batch_size, seq_len)
            if zero_mask.dim() == 3:
                zero_mask_2d = zero_mask.squeeze(-1)  # (batch_size, seq_len)
            else:
                zero_mask_2d = zero_mask
            
            # Apply zero penalty to attention scores
            # For cross-attention, we need to match the key sequence length
            if zero_mask_2d.size(1) == scores.size(-1):  # Key dimension
                zero_penalty = (1 - zero_mask_2d.unsqueeze(1).unsqueeze(2)) * (1 - self.zero_attention_weight)
                scores = scores * (1 - zero_penalty.expand_as(scores))
            # For now, skip zero-aware attention in cross-attention cases with mismatched dimensions
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)  # (batch_size, nhead, seq_len, d_k)
        
        # Concatenate heads
        attended = attended.transpose(1, 2)  # (batch_size, query_seq_len, nhead, d_k)
        attended = attended.contiguous().view(batch_size, query_seq_len, self.nhead * self.d_k)
        
        output = self.w_o(attended)
        
        return output, attention_weights.mean(dim=1)  # Average over heads


class EnhancedTweedieTransformer(nn.Module):
    """
    Enhanced Tweedie Transformer with improved architecture and zero-awareness.
    
    Improvements over the original:
    - Zero-aware attention mechanisms
    - Better numerical stability
    - Layer normalization and residual connections
    - Improved positional encoding
    - Configurable architecture
    """
    
    def __init__(self,
                 input_dim: int = 1,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 seq_len: int = 96,
                 pred_len: int = 24,
                 tweedie_power: float = 1.5,
                 activation: str = 'gelu',
                 zero_aware_attention: bool = True,
                 layer_norm_eps: float = 1e-5):
        """
        Initialize Enhanced Tweedie Transformer.
        
        Args:
            input_dim: Input feature dimension
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward dimension
            dropout: Dropout rate
            seq_len: Input sequence length
            pred_len: Prediction sequence length  
            tweedie_power: Tweedie power parameter (1 < power < 2)
            activation: Activation function ('relu', 'gelu')
            zero_aware_attention: Whether to use zero-aware attention
            layer_norm_eps: Layer normalization epsilon
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.tweedie_power = tweedie_power
        self.zero_aware_attention = zero_aware_attention
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len=seq_len + pred_len, dropout=dropout)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                zero_aware_attention=zero_aware_attention,
                layer_norm_eps=layer_norm_eps
            )
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, input_dim),
            nn.Softplus()  # Ensure positive outputs for Tweedie
        )
        
        # Tweedie loss
        self.criterion = TweedieLoss(power=tweedie_power)
        
        # Initialize parameters
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize model parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def create_zero_mask(self, x: torch.Tensor, threshold: float = 1e-6) -> torch.Tensor:
        """Create mask for zero positions."""
        return (x.abs() > threshold).float()
    
    def forward(self, src: torch.Tensor, 
                tgt: Optional[torch.Tensor] = None,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            src: Source sequence (batch_size, seq_len, input_dim)
            tgt: Target sequence for teacher forcing (batch_size, pred_len, input_dim)
            src_mask: Source attention mask
            tgt_mask: Target attention mask
            
        Returns:
            Predictions (batch_size, pred_len, input_dim)
        """
        batch_size = src.size(0)
        
        # Create zero masks if using zero-aware attention
        src_zero_mask = self.create_zero_mask(src) if self.zero_aware_attention else None
        
        # Encode source sequence
        src_embedded = self.input_projection(src)
        src_embedded = src_embedded.transpose(0, 1)  # (seq_len, batch_size, d_model)
        src_embedded = self.pos_encoding(src_embedded)
        
        # Apply transformer layers to source
        encoded = src_embedded
        attention_weights = []
        
        for layer in self.layers:
            encoded, attn_weights = layer(
                encoded, 
                encoded, 
                encoded,
                mask=src_mask,
                zero_mask=src_zero_mask.transpose(0, 1) if src_zero_mask is not None else None
            )
            attention_weights.append(attn_weights)
        
        if self.training and tgt is not None:
            # Teacher forcing during training
            tgt_zero_mask = self.create_zero_mask(tgt) if self.zero_aware_attention else None
            
            tgt_embedded = self.input_projection(tgt)
            tgt_embedded = tgt_embedded.transpose(0, 1)
            tgt_embedded = self.pos_encoding(tgt_embedded)
            
            # Create causal mask for target
            if tgt_mask is None:
                tgt_mask = self._generate_square_subsequent_mask(self.pred_len).to(src.device)
            
            # Decode
            decoded = tgt_embedded
            for layer in self.layers:
                decoded, _ = layer(
                    decoded,
                    encoded,
                    encoded,
                    mask=None,  # Don't use mask for cross-attention
                    zero_mask=tgt_zero_mask.transpose(0, 1) if tgt_zero_mask is not None else None
                )
            
            output = decoded.transpose(0, 1)  # Back to (batch_size, seq_len, d_model)
            
        else:
            # Autoregressive generation during inference
            output = self._autoregressive_generation(encoded, batch_size, src.device)
        
        # Project to output dimension
        predictions = self.output_projection(output)
        
        return predictions
    
    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate causal mask for decoder."""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def _autoregressive_generation(self, memory: torch.Tensor, 
                                  batch_size: int, device: torch.device) -> torch.Tensor:
        """Generate predictions autoregressively."""
        # Initialize with last encoded state
        decoder_input = memory[-1:, :, :]  # (1, batch_size, d_model)
        outputs = []
        
        for i in range(self.pred_len):
            # Apply transformer layers
            current_input = decoder_input
            for layer in self.layers:
                current_input, _ = layer(current_input, memory, memory)
            
            # Take the last (and only) time step
            current_output = current_input[-1:, :, :]  # (1, batch_size, d_model)
            outputs.append(current_output)
            
            # Update decoder input for next step
            decoder_input = torch.cat([decoder_input, current_output], dim=0)
            if decoder_input.size(0) > self.pred_len:
                decoder_input = decoder_input[1:]  # Keep only last pred_len steps
        
        # Concatenate all outputs
        output = torch.cat(outputs, dim=0).transpose(0, 1)  # (batch_size, pred_len, d_model)
        
        return output
    
    def compute_loss(self, predictions: torch.Tensor, 
                    targets: torch.Tensor) -> torch.Tensor:
        """Compute Tweedie loss."""
        predictions = torch.clamp(predictions, min=1e-8)
        return self.criterion(predictions, targets)
    
    def predict_with_uncertainty(self, src: torch.Tensor,
                                n_samples: int = 100) -> Dict[str, torch.Tensor]:
        """
        Make predictions with uncertainty estimation using Monte Carlo dropout.
        
        Args:
            src: Source sequence
            n_samples: Number of MC samples
            
        Returns:
            Dictionary with predictions, uncertainties, and quantiles
        """
        self.train()  # Enable dropout
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(src)
                predictions.append(pred)
        
        self.eval()  # Disable dropout
        
        predictions = torch.stack(predictions, dim=0)  # (n_samples, batch_size, pred_len, input_dim)
        
        mean_pred = torch.mean(predictions, dim=0)
        std_pred = torch.std(predictions, dim=0)
        
        # Compute quantiles
        quantiles = torch.quantile(predictions, torch.tensor([0.05, 0.25, 0.75, 0.95]), dim=0)
        
        return {
            'predictions': mean_pred,
            'uncertainty': std_pred,
            'quantiles': {
                'q05': quantiles[0],
                'q25': quantiles[1], 
                'q75': quantiles[2],
                'q95': quantiles[3]
            },
            'samples': predictions
        }


class TransformerLayer(nn.Module):
    """Single transformer layer with zero-aware attention."""
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int,
                 dropout: float = 0.1, activation: str = 'gelu',
                 zero_aware_attention: bool = True, layer_norm_eps: float = 1e-5):
        super().__init__()
        
        # Multi-head attention
        if zero_aware_attention:
            self.self_attn = ZeroAwareMultiheadAttention(
                d_model, nhead, dropout=dropout
            )
        else:
            self.self_attn = nn.MultiheadAttention(
                d_model, nhead, dropout=dropout, batch_first=True
            )
        
        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Activation
        self.activation = nn.GELU() if activation == 'gelu' else nn.ReLU()
        self.zero_aware_attention = zero_aware_attention
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                zero_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (seq_len, batch_size, d_model)
            key: (seq_len, batch_size, d_model)
            value: (seq_len, batch_size, d_model)
        """
        # Convert to batch_first for attention
        query_bf = query.transpose(0, 1)
        key_bf = key.transpose(0, 1)
        value_bf = value.transpose(0, 1)
        
        # Self-attention
        if self.zero_aware_attention:
            zero_mask_bf = zero_mask.transpose(0, 1) if zero_mask is not None else None
            attn_output, attn_weights = self.self_attn(
                query_bf, key_bf, value_bf, mask=mask, zero_mask=zero_mask_bf
            )
        else:
            attn_output, attn_weights = self.self_attn(
                query_bf, key_bf, value_bf, attn_mask=mask
            )
        
        # Convert back to seq_first
        attn_output = attn_output.transpose(0, 1)
        
        # Add & norm
        query = self.norm1(query + self.dropout1(attn_output))
        
        # Feedforward
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(query))))
        
        # Add & norm
        output = self.norm2(query + self.dropout2(ff_output))
        
        return output, attn_weights