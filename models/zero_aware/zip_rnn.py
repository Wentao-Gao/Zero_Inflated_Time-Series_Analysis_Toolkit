"""
Zero-Inflated Poisson RNN (ZIP-RNN) for time series forecasting.

This model combines recurrent neural networks with the Zero-Inflated Poisson
distribution, explicitly modeling both the zero-inflation process and the
count generation process using neural networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any, Tuple
from scipy.special import loggamma
import math


class ZIPRNN(nn.Module):
    """
    Zero-Inflated Poisson RNN model.
    
    This model uses RNNs to model both components of the ZIP distribution:
    1. Binary RNN: Models the zero-inflation probability π
    2. Count RNN: Models the Poisson rate parameter λ
    
    The final prediction combines both components according to the ZIP distribution.
    """
    
    def __init__(self,
                 input_dim: int = 1,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 seq_len: int = 96,
                 pred_len: int = 24,
                 rnn_type: str = 'LSTM',
                 dropout: float = 0.1,
                 bidirectional: bool = False,
                 shared_encoder: bool = False):
        """
        Initialize ZIP-RNN model.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension for RNNs
            num_layers: Number of RNN layers
            seq_len: Input sequence length
            pred_len: Prediction sequence length
            rnn_type: Type of RNN ('LSTM', 'GRU')
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional RNNs
            shared_encoder: Whether to share encoder between branches
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.shared_encoder = shared_encoder
        
        # Determine actual hidden dimension considering bidirectionality
        actual_hidden_dim = hidden_dim * (2 if bidirectional else 1)
        
        if shared_encoder:
            # Shared encoder for both branches
            if rnn_type == 'LSTM':
                self.shared_rnn = nn.LSTM(
                    input_size=input_dim,
                    hidden_size=hidden_dim,
                    num_layers=num_layers,
                    dropout=dropout if num_layers > 1 else 0,
                    batch_first=True,
                    bidirectional=bidirectional
                )
            elif rnn_type == 'GRU':
                self.shared_rnn = nn.GRU(
                    input_size=input_dim,
                    hidden_size=hidden_dim,
                    num_layers=num_layers,
                    dropout=dropout if num_layers > 1 else 0,
                    batch_first=True,
                    bidirectional=bidirectional
                )
            
            # Branch-specific heads
            self.pi_head = nn.Sequential(
                nn.Linear(actual_hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, pred_len * input_dim),
                nn.Sigmoid()  # Probability output
            )
            
            self.lambda_head = nn.Sequential(
                nn.Linear(actual_hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, pred_len * input_dim),
                nn.Softplus()  # Positive rate parameter
            )
            
        else:
            # Separate RNNs for each branch
            if rnn_type == 'LSTM':
                self.pi_rnn = nn.LSTM(
                    input_size=input_dim,
                    hidden_size=hidden_dim,
                    num_layers=num_layers,
                    dropout=dropout if num_layers > 1 else 0,
                    batch_first=True,
                    bidirectional=bidirectional
                )
                
                self.lambda_rnn = nn.LSTM(
                    input_size=input_dim,
                    hidden_size=hidden_dim,
                    num_layers=num_layers,
                    dropout=dropout if num_layers > 1 else 0,
                    batch_first=True,
                    bidirectional=bidirectional
                )
            elif rnn_type == 'GRU':
                self.pi_rnn = nn.GRU(
                    input_size=input_dim,
                    hidden_size=hidden_dim,
                    num_layers=num_layers,
                    dropout=dropout if num_layers > 1 else 0,
                    batch_first=True,
                    bidirectional=bidirectional
                )
                
                self.lambda_rnn = nn.GRU(
                    input_size=input_dim,
                    hidden_size=hidden_dim,
                    num_layers=num_layers,
                    dropout=dropout if num_layers > 1 else 0,
                    batch_first=True,
                    bidirectional=bidirectional
                )
            
            # Output layers
            self.pi_output = nn.Sequential(
                nn.Linear(actual_hidden_dim, pred_len * input_dim),
                nn.Sigmoid()
            )
            
            self.lambda_output = nn.Sequential(
                nn.Linear(actual_hidden_dim, pred_len * input_dim),
                nn.Softplus()
            )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)
            elif 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param.data)
    
    def forward(self, x: torch.Tensor, 
                return_components: bool = False) -> torch.Tensor:
        """
        Forward pass through ZIP-RNN.
        
        Args:
            x: Input sequence (batch_size, seq_len, input_dim)
            return_components: Whether to return π and λ components
            
        Returns:
            Predictions or dictionary with components
        """
        batch_size = x.size(0)
        
        if self.shared_encoder:
            # Shared encoder
            encoded, _ = self.shared_rnn(x)
            
            # Use last hidden state for prediction
            if self.bidirectional:
                # Concatenate forward and backward final states
                final_state = encoded[:, -1, :]
            else:
                final_state = encoded[:, -1, :]
            
            # Branch-specific predictions
            pi = self.pi_head(final_state).view(batch_size, self.pred_len, self.input_dim)
            lambda_param = self.lambda_head(final_state).view(batch_size, self.pred_len, self.input_dim)
            
        else:
            # Separate encoders
            pi_encoded, _ = self.pi_rnn(x)
            lambda_encoded, _ = self.lambda_rnn(x)
            
            # Extract final states
            pi_final = pi_encoded[:, -1, :]
            lambda_final = lambda_encoded[:, -1, :]
            
            # Predictions
            pi = self.pi_output(pi_final).view(batch_size, self.pred_len, self.input_dim)
            lambda_param = self.lambda_output(lambda_final).view(batch_size, self.pred_len, self.input_dim)
        
        # Ensure parameters are in valid ranges
        pi = torch.clamp(pi, 1e-6, 1 - 1e-6)
        lambda_param = torch.clamp(lambda_param, 1e-6, 50.0)  # Reasonable upper bound
        
        # Expected value of ZIP distribution: E[Y] = (1 - π) * λ
        expected_counts = (1 - pi) * lambda_param
        
        if return_components:
            return {
                'predictions': expected_counts,
                'pi': pi,
                'lambda': lambda_param,
                'zero_prob': pi + (1 - pi) * torch.exp(-lambda_param)
            }
        
        return expected_counts
    
    def compute_zip_loss(self, predictions: torch.Tensor, targets: torch.Tensor,
                        pi: Optional[torch.Tensor] = None,
                        lambda_param: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute ZIP negative log-likelihood loss.
        
        Args:
            predictions: Model predictions (not used directly)
            targets: Ground truth targets (should be non-negative integers)
            pi: Zero-inflation probabilities
            lambda_param: Poisson rate parameters
            
        Returns:
            ZIP negative log-likelihood loss
        """
        # Get components if not provided
        if pi is None or lambda_param is None:
            components = self.forward(predictions.detach(), return_components=True)
            pi = components['pi']
            lambda_param = components['lambda']
        
        # Ensure targets are integers (for proper ZIP likelihood)
        targets = torch.round(targets).clamp(min=0)
        
        # Compute ZIP log-likelihood
        log_likelihood = torch.zeros_like(targets)
        
        # For zero observations
        zero_mask = (targets == 0)
        if torch.any(zero_mask):
            zero_log_prob = torch.log(pi[zero_mask] + (1 - pi[zero_mask]) * torch.exp(-lambda_param[zero_mask]))
            log_likelihood[zero_mask] = zero_log_prob
        
        # For non-zero observations
        nonzero_mask = (targets > 0)
        if torch.any(nonzero_mask):
            y_nonzero = targets[nonzero_mask]
            pi_nonzero = pi[nonzero_mask]
            lambda_nonzero = lambda_param[nonzero_mask]
            
            # Poisson log-probability: log(P(Y=y)) = y*log(λ) - λ - log(y!)
            # Using loggamma(y+1) = log(y!)
            poisson_log_prob = (y_nonzero * torch.log(lambda_nonzero) - 
                               lambda_nonzero - torch.lgamma(y_nonzero + 1))
            
            nonzero_log_prob = torch.log(1 - pi_nonzero) + poisson_log_prob
            log_likelihood[nonzero_mask] = nonzero_log_prob
        
        # Return negative log-likelihood
        return -torch.mean(log_likelihood)
    
    def sample(self, x: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """
        Sample from the fitted ZIP distribution.
        
        Args:
            x: Input sequence
            n_samples: Number of samples to draw
            
        Returns:
            Samples from ZIP distribution
        """
        self.eval()
        with torch.no_grad():
            components = self.forward(x, return_components=True)
            pi = components['pi']
            lambda_param = components['lambda']
            
            batch_size, pred_len, input_dim = pi.shape
            samples = torch.zeros(batch_size, pred_len, input_dim, n_samples)
            
            for i in range(batch_size):
                for j in range(pred_len):
                    for k in range(input_dim):
                        pi_ijk = pi[i, j, k].item()
                        lambda_ijk = lambda_param[i, j, k].item()
                        
                        for s in range(n_samples):
                            # Sample from ZIP
                            if np.random.random() < pi_ijk:
                                # Excess zero
                                samples[i, j, k, s] = 0
                            else:
                                # Sample from Poisson
                                samples[i, j, k, s] = np.random.poisson(lambda_ijk)
        
        return samples
    
    def predict_with_uncertainty(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Make predictions with uncertainty quantification.
        
        Args:
            x: Input sequence
            
        Returns:
            Dictionary with predictions and uncertainties
        """
        self.eval()
        with torch.no_grad():
            components = self.forward(x, return_components=True)
            
            pi = components['pi']
            lambda_param = components['lambda']
            predictions = components['predictions']
            
            # Compute theoretical variance of ZIP distribution
            # Var[Y] = (1-π)*λ + (1-π)*π*λ²
            zip_variance = ((1 - pi) * lambda_param + 
                           (1 - pi) * pi * lambda_param**2)
            zip_std = torch.sqrt(zip_variance)
            
            # Confidence intervals (assuming approximate normality for large λ)
            lower_bound = torch.clamp(predictions - 1.96 * zip_std, min=0)
            upper_bound = predictions + 1.96 * zip_std
            
            return {
                'predictions': predictions,
                'uncertainty': zip_std,
                'confidence_interval': (lower_bound, upper_bound),
                'pi': pi,
                'lambda': lambda_param,
                'zero_probability': components['zero_prob']
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': 'ZIPRNN',
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'seq_len': self.seq_len,
            'pred_len': self.pred_len,
            'rnn_type': self.rnn_type,
            'bidirectional': self.bidirectional,
            'shared_encoder': self.shared_encoder,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        }


class ZIPLoss(nn.Module):
    """
    Standalone ZIP loss function.
    
    This can be used with any model that outputs π and λ parameters.
    """
    
    def __init__(self, reduction: str = 'mean', eps: float = 1e-8):
        super().__init__()
        self.reduction = reduction
        self.eps = eps
    
    def forward(self, pi: torch.Tensor, lambda_param: torch.Tensor, 
                targets: torch.Tensor) -> torch.Tensor:
        """
        Compute ZIP loss.
        
        Args:
            pi: Zero-inflation probabilities
            lambda_param: Poisson rate parameters
            targets: Ground truth targets
            
        Returns:
            ZIP loss
        """
        # Clamp parameters to avoid numerical issues
        pi = torch.clamp(pi, self.eps, 1 - self.eps)
        lambda_param = torch.clamp(lambda_param, self.eps, 50.0)
        
        # Round targets to integers
        targets = torch.round(targets).clamp(min=0)
        
        # Compute log-likelihood
        log_likelihood = torch.zeros_like(targets)
        
        # Zero observations
        zero_mask = (targets == 0)
        if torch.any(zero_mask):
            zero_log_prob = torch.log(pi[zero_mask] + (1 - pi[zero_mask]) * torch.exp(-lambda_param[zero_mask]))
            log_likelihood[zero_mask] = zero_log_prob
        
        # Non-zero observations
        nonzero_mask = (targets > 0)
        if torch.any(nonzero_mask):
            y_nonzero = targets[nonzero_mask]
            pi_nonzero = pi[nonzero_mask]
            lambda_nonzero = lambda_param[nonzero_mask]
            
            poisson_log_prob = (y_nonzero * torch.log(lambda_nonzero) - 
                               lambda_nonzero - torch.lgamma(y_nonzero + 1))
            
            nonzero_log_prob = torch.log(1 - pi_nonzero) + poisson_log_prob
            log_likelihood[nonzero_mask] = nonzero_log_prob
        
        # Apply reduction
        if self.reduction == 'none':
            return -log_likelihood
        elif self.reduction == 'mean':
            return -torch.mean(log_likelihood)
        elif self.reduction == 'sum':
            return -torch.sum(log_likelihood)
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")


# Utility functions for ZIP models
def zip_log_prob(y: torch.Tensor, pi: torch.Tensor, lambda_param: torch.Tensor) -> torch.Tensor:
    """
    Compute ZIP log-probability.
    
    Args:
        y: Observed values
        pi: Zero-inflation probability
        lambda_param: Poisson rate parameter
        
    Returns:
        Log-probability under ZIP distribution
    """
    log_prob = torch.zeros_like(y)
    
    # Zero case
    zero_mask = (y == 0)
    if torch.any(zero_mask):
        log_prob[zero_mask] = torch.log(pi[zero_mask] + (1 - pi[zero_mask]) * torch.exp(-lambda_param[zero_mask]))
    
    # Non-zero case
    nonzero_mask = (y > 0)
    if torch.any(nonzero_mask):
        y_nz = y[nonzero_mask]
        pi_nz = pi[nonzero_mask]
        lambda_nz = lambda_param[nonzero_mask]
        
        poisson_logprob = y_nz * torch.log(lambda_nz) - lambda_nz - torch.lgamma(y_nz + 1)
        log_prob[nonzero_mask] = torch.log(1 - pi_nz) + poisson_logprob
    
    return log_prob


def estimate_zip_parameters(data: torch.Tensor) -> Tuple[float, float]:
    """
    Estimate ZIP parameters using method of moments.
    
    Args:
        data: Observed data
        
    Returns:
        Tuple of (pi, lambda) estimates
    """
    data_np = data.detach().cpu().numpy().flatten()
    
    # Empirical moments
    mean_data = np.mean(data_np)
    var_data = np.var(data_np)
    zero_ratio = np.mean(data_np == 0)
    
    # Method of moments estimates
    if var_data <= mean_data:
        # Under-dispersed, may not be ZIP
        pi_est = 0.0
        lambda_est = mean_data
    else:
        # Over-dispersed, likely ZIP
        lambda_est = mean_data / (1 - zero_ratio + 1e-8)
        pi_est = zero_ratio - np.exp(-lambda_est)
        pi_est = max(0, min(pi_est, 0.999))  # Clamp to valid range
    
    return float(pi_est), float(lambda_est)