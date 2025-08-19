"""
Dual-Branch Network for zero-inflated time series forecasting.

This architecture separates the modeling of zero and non-zero outcomes:
1. Binary branch: Predicts zero vs non-zero
2. Magnitude branch: Predicts the magnitude of non-zero values

This approach is inspired by hurdle models but implemented as a neural network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any, Tuple
import math


class DualBranchNetwork(nn.Module):
    """
    Dual-branch neural network for zero-inflated time series.
    
    The network consists of:
    1. Shared encoder: Extracts common features from input
    2. Binary branch: Predicts probability of non-zero outcome
    3. Magnitude branch: Predicts magnitude given non-zero outcome
    
    The final prediction combines both branches.
    """
    
    def __init__(self,
                 input_dim: int = 1,
                 hidden_dim: int = 256,
                 num_layers: int = 3,
                 seq_len: int = 96,
                 pred_len: int = 24,
                 dropout: float = 0.1,
                 activation: str = 'gelu',
                 architecture: str = 'lstm',
                 binary_threshold: float = 0.5,
                 magnitude_activation: str = 'softplus'):
        """
        Initialize Dual-Branch Network.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of layers in each branch
            seq_len: Input sequence length
            pred_len: Prediction sequence length
            dropout: Dropout rate
            activation: Activation function
            architecture: Base architecture ('lstm', 'gru', 'transformer')
            binary_threshold: Threshold for binary classification
            magnitude_activation: Activation for magnitude output
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.binary_threshold = binary_threshold
        self.architecture = architecture
        
        # Shared encoder
        if architecture == 'lstm':
            self.shared_encoder = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
                bidirectional=False
            )
        elif architecture == 'gru':
            self.shared_encoder = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
                bidirectional=False
            )
        elif architecture == 'transformer':
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation=activation,
                batch_first=True
            )
            self.input_projection = nn.Linear(input_dim, hidden_dim)
            self.pos_encoding = PositionalEncoding(hidden_dim, dropout=dropout)
            self.shared_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        # Binary branch (zero vs non-zero)
        self.binary_branch = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, pred_len * input_dim)
        )
        
        # Magnitude branch (for non-zero values)
        magnitude_layers = [
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, pred_len * input_dim)
        ]
        
        # Add appropriate output activation for magnitude
        if magnitude_activation == 'softplus':
            magnitude_layers.append(nn.Softplus())
        elif magnitude_activation == 'relu':
            magnitude_layers.append(nn.ReLU())
        elif magnitude_activation == 'exp':
            magnitude_layers.append(lambda x: torch.exp(x))
        
        self.magnitude_branch = nn.Sequential(*magnitude_layers)
        
        # Attention mechanism for branch fusion (optional)
        self.use_attention_fusion = True
        if self.use_attention_fusion:
            self.attention_fusion = BranchAttentionFusion(hidden_dim, dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.LSTM, nn.GRU)):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.zeros_(param.data)
    
    def forward(self, x: torch.Tensor, 
                return_components: bool = False) -> torch.Tensor:
        """
        Forward pass through dual-branch network.
        
        Args:
            x: Input sequence (batch_size, seq_len, input_dim)
            return_components: Whether to return binary and magnitude components
            
        Returns:
            Predictions or dictionary with components
        """
        batch_size = x.size(0)
        
        # Shared encoding
        if self.architecture in ['lstm', 'gru']:
            encoded, _ = self.shared_encoder(x)
            # Use the last hidden state for prediction
            shared_features = encoded[:, -1, :]  # (batch_size, hidden_dim)
            
        elif self.architecture == 'transformer':
            # Project input and add positional encoding
            x_proj = self.input_projection(x)
            x_pos = self.pos_encoding(x_proj)
            
            # Transformer encoding
            encoded = self.shared_encoder(x_pos)
            # Use mean pooling over sequence dimension
            shared_features = torch.mean(encoded, dim=1)  # (batch_size, hidden_dim)
        
        # Binary branch prediction (logits for zero/non-zero)
        binary_logits = self.binary_branch(shared_features)
        binary_logits = binary_logits.view(batch_size, self.pred_len, self.input_dim)
        binary_probs = torch.sigmoid(binary_logits)
        
        # Magnitude branch prediction (for non-zero values)
        magnitude_pred = self.magnitude_branch(shared_features)
        magnitude_pred = magnitude_pred.view(batch_size, self.pred_len, self.input_dim)
        
        # Combine branches with attention fusion if enabled
        if self.use_attention_fusion:
            combined_features = torch.cat([
                binary_logits.view(batch_size, -1),
                magnitude_pred.view(batch_size, -1)
            ], dim=-1)
            
            attention_weights = self.attention_fusion(shared_features, combined_features)
            binary_weight = attention_weights[:, 0:1]
            magnitude_weight = attention_weights[:, 1:2]
        else:
            binary_weight = magnitude_weight = torch.tensor(1.0)
        
        # Final prediction: P(non-zero) * magnitude
        final_predictions = binary_probs * magnitude_pred
        
        if return_components:
            return {
                'predictions': final_predictions,
                'binary_probs': binary_probs,
                'magnitude_pred': magnitude_pred,
                'binary_logits': binary_logits,
                'attention_weights': (binary_weight, magnitude_weight) if self.use_attention_fusion else None
            }
        
        return final_predictions
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor,
                    binary_probs: Optional[torch.Tensor] = None,
                    magnitude_pred: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute dual-branch loss combining binary and magnitude losses.
        
        Args:
            predictions: Final predictions
            targets: Ground truth targets
            binary_probs: Binary branch probabilities
            magnitude_pred: Magnitude branch predictions
            
        Returns:
            Dictionary of losses
        """
        # If components not provided, get them from forward pass
        if binary_probs is None or magnitude_pred is None:
            components = self.forward(predictions.detach(), return_components=True)
            binary_probs = components['binary_probs']
            magnitude_pred = components['magnitude_pred']
        
        # Binary classification loss
        binary_targets = (targets > 1e-6).float()
        binary_loss = F.binary_cross_entropy(binary_probs, binary_targets)
        
        # Magnitude regression loss (only on non-zero targets)
        nonzero_mask = binary_targets == 1
        if torch.any(nonzero_mask):
            magnitude_loss = F.mse_loss(
                magnitude_pred[nonzero_mask], 
                targets[nonzero_mask]
            )
        else:
            magnitude_loss = torch.tensor(0.0, device=predictions.device)
        
        # Combined prediction loss
        combined_loss = F.mse_loss(predictions, targets)
        
        # Total loss (weighted combination)
        total_loss = 0.3 * binary_loss + 0.3 * magnitude_loss + 0.4 * combined_loss
        
        return {
            'total_loss': total_loss,
            'binary_loss': binary_loss,
            'magnitude_loss': magnitude_loss,
            'combined_loss': combined_loss
        }
    
    def predict_with_interpretation(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Make predictions with interpretation of binary and magnitude components.
        
        Args:
            x: Input sequence
            
        Returns:
            Dictionary with predictions and interpretations
        """
        self.eval()
        with torch.no_grad():
            components = self.forward(x, return_components=True)
            
            # Compute interpretation metrics
            zero_probability = 1 - components['binary_probs']
            expected_magnitude = components['magnitude_pred']
            prediction_confidence = components['binary_probs'] * expected_magnitude
            
            return {
                'predictions': components['predictions'],
                'zero_probability': zero_probability,
                'expected_magnitude': expected_magnitude,
                'prediction_confidence': prediction_confidence,
                'binary_logits': components['binary_logits']
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': 'DualBranchNetwork',
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'seq_len': self.seq_len,
            'pred_len': self.pred_len,
            'architecture': self.architecture,
            'binary_threshold': self.binary_threshold,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        }


class BranchAttentionFusion(nn.Module):
    """
    Attention mechanism for fusing binary and magnitude branch outputs.
    
    This module learns to weight the importance of binary vs magnitude
    predictions based on the shared features.
    """
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        
        # We'll calculate the correct input dimension in forward pass
        self.hidden_dim = hidden_dim
        self.dropout_layer = nn.Dropout(dropout)
        self.attention_linear1 = None  # Will be initialized dynamically
        self.attention_linear2 = None  # Will be initialized dynamically
    
    def forward(self, shared_features: torch.Tensor, 
                combined_features: torch.Tensor) -> torch.Tensor:
        """
        Compute attention weights for branch fusion.
        
        Args:
            shared_features: Shared encoder features
            combined_features: Concatenated branch outputs
            
        Returns:
            Attention weights for each branch
        """
        # Concatenate features
        fusion_input = torch.cat([shared_features, combined_features], dim=-1)
        
        # Initialize layers if not already done
        if self.attention_linear1 is None:
            input_dim = fusion_input.size(-1)
            self.attention_linear1 = nn.Linear(input_dim, self.hidden_dim).to(fusion_input.device)
            self.attention_linear2 = nn.Linear(self.hidden_dim, 2).to(fusion_input.device)
        
        # Compute attention weights
        x = self.attention_linear1(fusion_input)
        x = torch.tanh(x)
        x = self.dropout_layer(x)
        attention_weights = F.softmax(self.attention_linear2(x), dim=-1)
        
        return attention_weights


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer architecture."""
    
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


# Utility functions for dual-branch training
def compute_branch_importance(binary_probs: torch.Tensor, 
                            targets: torch.Tensor) -> Tuple[float, float]:
    """
    Compute the relative importance of binary vs magnitude branches
    based on prediction accuracy.
    
    Args:
        binary_probs: Predicted probabilities for non-zero outcomes
        targets: Ground truth targets
        
    Returns:
        Tuple of (binary_importance, magnitude_importance)
    """
    # Binary accuracy
    binary_targets = (targets > 1e-6).float()
    binary_predictions = (binary_probs > 0.5).float()
    binary_accuracy = torch.mean((binary_predictions == binary_targets).float())
    
    # Magnitude accuracy (only on non-zero values)
    nonzero_mask = binary_targets == 1
    if torch.any(nonzero_mask):
        # Use relative error as inverse of accuracy
        relative_errors = torch.abs(binary_probs[nonzero_mask] - targets[nonzero_mask]) / (targets[nonzero_mask] + 1e-8)
        magnitude_accuracy = 1.0 / (torch.mean(relative_errors) + 1e-8)
    else:
        magnitude_accuracy = torch.tensor(1.0)
    
    # Normalize importances
    total_importance = binary_accuracy + magnitude_accuracy
    binary_importance = binary_accuracy / total_importance
    magnitude_importance = magnitude_accuracy / total_importance
    
    return binary_importance.item(), magnitude_importance.item()


def adaptive_loss_weighting(epoch: int, max_epochs: int, 
                          initial_weights: Tuple[float, float, float] = (0.3, 0.3, 0.4),
                          final_weights: Tuple[float, float, float] = (0.2, 0.2, 0.6)) -> Tuple[float, float, float]:
    """
    Compute adaptive loss weights that change over training.
    
    Args:
        epoch: Current epoch
        max_epochs: Total training epochs
        initial_weights: Initial weights for (binary, magnitude, combined) losses
        final_weights: Final weights
        
    Returns:
        Current weights for the three loss components
    """
    progress = min(epoch / max_epochs, 1.0)
    
    weights = []
    for initial, final in zip(initial_weights, final_weights):
        current_weight = initial + (final - initial) * progress
        weights.append(current_weight)
    
    return tuple(weights)