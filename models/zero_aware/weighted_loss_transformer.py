"""
Transformer with Weighted Loss for zero-inflated time series.

This model uses adaptive weighted loss functions to handle zero-inflation
by dynamically adjusting the importance of zero vs non-zero predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any, Tuple
import math

from ..losses.zero_aware_losses import AdaptiveWeightedLoss, compute_class_weights


class WeightedLossTransformer(nn.Module):
    """
    Transformer model with adaptive weighted loss for zero-inflated data.
    
    This model automatically adjusts loss weights based on the distribution
    of zeros vs non-zeros in the data, providing better handling of
    class imbalance inherent in zero-inflated time series.
    """
    
    def __init__(self,
                 input_dim: int = 1,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 seq_len: int = 96,
                 pred_len: int = 24,
                 activation: str = 'gelu',
                 loss_type: str = 'adaptive',
                 weight_adjustment: str = 'dynamic'):
        """
        Initialize Weighted Loss Transformer.
        
        Args:
            input_dim: Input feature dimension
            d_model: Model dimension
            nhead: Number of attention heads
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            dim_feedforward: Feedforward dimension
            dropout: Dropout rate
            seq_len: Input sequence length
            pred_len: Prediction sequence length
            activation: Activation function
            loss_type: Type of weighted loss ('adaptive', 'fixed_weighted')
            weight_adjustment: How to adjust weights ('dynamic', 'batch', 'epoch')
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.loss_type = loss_type
        self.weight_adjustment = weight_adjustment
        
        # Input embedding
        self.input_embedding = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, input_dim)
        )
        
        # Loss functions
        if loss_type == 'adaptive':
            self.criterion = AdaptiveWeightedLoss(base_loss='mse')
        else:
            # Will be set based on data statistics
            self.criterion = None
            self.zero_weight = 1.0
            self.nonzero_weight = 1.0
        
        # Statistics tracking
        self.register_buffer('batch_count', torch.tensor(0))
        self.register_buffer('running_zero_ratio', torch.tensor(0.5))
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def update_loss_weights(self, targets: torch.Tensor):
        """Update loss weights based on current batch statistics."""
        if self.loss_type == 'fixed_weighted' and self.weight_adjustment == 'dynamic':
            zero_weight, nonzero_weight = compute_class_weights(targets, method='balanced')
            self.zero_weight = zero_weight
            self.nonzero_weight = nonzero_weight
            
            # Update criterion
            from ..losses.zero_aware_losses import WeightedMSELoss
            self.criterion = WeightedMSELoss(
                zero_weight=zero_weight,
                nonzero_weight=nonzero_weight
            )
        
        # Update running statistics
        current_zero_ratio = torch.mean((targets == 0).float())
        if self.batch_count == 0:
            self.running_zero_ratio = current_zero_ratio
        else:
            momentum = 0.9
            self.running_zero_ratio = momentum * self.running_zero_ratio + (1 - momentum) * current_zero_ratio
        
        self.batch_count += 1
    
    def forward(self, src: torch.Tensor,
                tgt: Optional[torch.Tensor] = None,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            src: Source sequence (batch_size, seq_len, input_dim)
            tgt: Target sequence (batch_size, pred_len, input_dim)
            src_mask: Source attention mask
            tgt_mask: Target attention mask
            
        Returns:
            Predictions (batch_size, pred_len, input_dim)
        """
        batch_size = src.size(0)
        
        # Embed inputs
        src_embedded = self.input_embedding(src)
        src_embedded = self.pos_encoding(src_embedded)
        
        if self.training and tgt is not None:
            # Teacher forcing
            tgt_embedded = self.input_embedding(tgt)
            tgt_embedded = self.pos_encoding(tgt_embedded)
            
            # Create causal mask if not provided
            if tgt_mask is None:
                tgt_mask = self._generate_square_subsequent_mask(self.pred_len).to(src.device)
            
            # Transformer forward pass
            output = self.transformer(
                src=src_embedded,
                tgt=tgt_embedded,
                src_mask=src_mask,
                tgt_mask=tgt_mask
            )
        else:
            # Inference mode - autoregressive generation
            output = self._autoregressive_generation(src_embedded, batch_size)
        
        # Project to output dimension
        predictions = self.output_projection(output)
        
        return predictions
    
    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate causal mask."""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def _autoregressive_generation(self, src_embedded: torch.Tensor, 
                                  batch_size: int) -> torch.Tensor:
        """Generate predictions autoregressively."""
        device = src_embedded.device
        
        # Initialize decoder input
        decoder_input = torch.zeros(batch_size, 1, self.d_model, device=device)
        outputs = []
        
        for i in range(self.pred_len):
            # Create causal mask
            tgt_mask = self._generate_square_subsequent_mask(i + 1).to(device)
            
            # Transformer forward pass
            output = self.transformer(
                src=src_embedded,
                tgt=decoder_input,
                tgt_mask=tgt_mask
            )
            
            # Get the last time step
            current_output = output[:, -1:, :]
            outputs.append(current_output)
            
            # Update decoder input
            decoder_input = torch.cat([decoder_input, current_output], dim=1)
        
        return torch.cat(outputs, dim=1)
    
    def compute_loss(self, predictions: torch.Tensor, 
                    targets: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Weighted loss
        """
        # Update weights if needed
        if self.training:
            self.update_loss_weights(targets)
        
        # Compute loss
        if self.criterion is None:
            # Fallback to MSE if no criterion is set
            return F.mse_loss(predictions, targets)
        
        return self.criterion(predictions, targets)
    
    def predict_with_confidence(self, src: torch.Tensor,
                               return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        Make predictions with confidence estimation.
        
        Args:
            src: Source sequence
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary with predictions and optional attention weights
        """
        self.eval()
        
        with torch.no_grad():
            predictions = self.forward(src)
            
            # Compute prediction confidence based on attention entropy
            # (This is a simple heuristic - more sophisticated methods exist)
            confidence = torch.ones_like(predictions)
            
            result = {
                'predictions': predictions,
                'confidence': confidence
            }
            
            # TODO: Add attention weight extraction if needed
            if return_attention:
                result['attention_weights'] = None
            
            return result
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': 'WeightedLossTransformer',
            'input_dim': self.input_dim,
            'd_model': self.d_model,
            'seq_len': self.seq_len,
            'pred_len': self.pred_len,
            'loss_type': self.loss_type,
            'weight_adjustment': self.weight_adjustment,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'running_zero_ratio': self.running_zero_ratio.item(),
            'batch_count': self.batch_count.item()
        }


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class AdaptiveWeightScheduler:
    """
    Scheduler for adaptive weight adjustment during training.
    
    This class manages how weights are updated over the course of training,
    allowing for different strategies like gradual adaptation or periodic updates.
    """
    
    def __init__(self, model: WeightedLossTransformer, 
                 strategy: str = 'exponential',
                 update_frequency: int = 100):
        """
        Initialize weight scheduler.
        
        Args:
            model: The transformer model
            strategy: Update strategy ('exponential', 'linear', 'step')
            update_frequency: How often to update weights (in batches)
        """
        self.model = model
        self.strategy = strategy
        self.update_frequency = update_frequency
        self.step_count = 0
        
    def step(self, targets: torch.Tensor):
        """Update weights based on current batch."""
        self.step_count += 1
        
        if self.step_count % self.update_frequency == 0:
            if self.strategy == 'exponential':
                # Exponentially weighted moving average
                self.model.update_loss_weights(targets)
            elif self.strategy == 'linear':
                # Linear interpolation towards balanced weights
                zero_weight, nonzero_weight = compute_class_weights(targets)
                alpha = min(self.step_count / 10000.0, 1.0)  # Gradually increase influence
                
                current_zero = getattr(self.model, 'zero_weight', 1.0)
                current_nonzero = getattr(self.model, 'nonzero_weight', 1.0)
                
                new_zero = (1 - alpha) * current_zero + alpha * zero_weight
                new_nonzero = (1 - alpha) * current_nonzero + alpha * nonzero_weight
                
                self.model.zero_weight = new_zero
                self.model.nonzero_weight = new_nonzero
            
            elif self.strategy == 'step':
                # Step-wise updates at fixed intervals
                if self.step_count in [1000, 5000, 10000]:
                    self.model.update_loss_weights(targets)