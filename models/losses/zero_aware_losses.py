"""
Zero-aware loss functions for time series models.

These loss functions are specifically designed to handle zero-inflated data
by applying different weights or treatments to zero and non-zero values.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union, Callable


class WeightedMSELoss(nn.Module):
    """
    Weighted Mean Squared Error loss that applies different weights to zero and non-zero values.
    
    This loss function helps address class imbalance in zero-inflated data by applying
    higher weights to the minority class (usually non-zero values).
    
    Parameters:
    -----------
    zero_weight : float, default=1.0
        Weight for zero values
    nonzero_weight : float, default=1.0  
        Weight for non-zero values
    reduction : str, default='mean'
        Reduction method ('none', 'mean', 'sum')
    """
    
    def __init__(self, 
                 zero_weight: float = 1.0,
                 nonzero_weight: float = 1.0,
                 reduction: str = 'mean'):
        super().__init__()
        self.zero_weight = zero_weight
        self.nonzero_weight = nonzero_weight
        self.reduction = reduction
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute weighted MSE loss."""
        mse = (predictions - targets) ** 2
        
        # Create weight mask
        zero_mask = (targets == 0).float()
        nonzero_mask = 1 - zero_mask
        
        weights = zero_mask * self.zero_weight + nonzero_mask * self.nonzero_weight
        weighted_mse = mse * weights
        
        if self.reduction == 'none':
            return weighted_mse
        elif self.reduction == 'mean':
            return torch.mean(weighted_mse)
        elif self.reduction == 'sum':
            return torch.sum(weighted_mse)
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")


class WeightedMAELoss(nn.Module):
    """
    Weighted Mean Absolute Error loss for zero-inflated data.
    
    Parameters:
    -----------
    zero_weight : float, default=1.0
        Weight for zero values
    nonzero_weight : float, default=1.0
        Weight for non-zero values  
    reduction : str, default='mean'
        Reduction method
    """
    
    def __init__(self,
                 zero_weight: float = 1.0,
                 nonzero_weight: float = 1.0,
                 reduction: str = 'mean'):
        super().__init__()
        self.zero_weight = zero_weight
        self.nonzero_weight = nonzero_weight
        self.reduction = reduction
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute weighted MAE loss."""
        mae = torch.abs(predictions - targets)
        
        # Create weight mask
        zero_mask = (targets == 0).float()
        nonzero_mask = 1 - zero_mask
        
        weights = zero_mask * self.zero_weight + nonzero_mask * self.nonzero_weight
        weighted_mae = mae * weights
        
        if self.reduction == 'none':
            return weighted_mae
        elif self.reduction == 'mean':
            return torch.mean(weighted_mae)
        elif self.reduction == 'sum':
            return torch.sum(weighted_mae)
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")


class ZeroInflatedLoss(nn.Module):
    """
    Composite loss function that combines zero classification and magnitude regression.
    
    This loss treats zero-inflation as a two-stage problem:
    1. Binary classification: zero vs non-zero
    2. Regression: magnitude of non-zero values
    
    Parameters:
    -----------
    classification_weight : float, default=1.0
        Weight for the binary classification loss
    regression_weight : float, default=1.0
        Weight for the regression loss
    regression_loss : str, default='mse'
        Type of regression loss ('mse', 'mae', 'huber')
    threshold : float, default=1e-6
        Threshold for considering a value as zero
    """
    
    def __init__(self,
                 classification_weight: float = 1.0,
                 regression_weight: float = 1.0,
                 regression_loss: str = 'mse',
                 threshold: float = 1e-6):
        super().__init__()
        self.classification_weight = classification_weight
        self.regression_weight = regression_weight
        self.regression_loss = regression_loss
        self.threshold = threshold
        
        # Binary classification loss
        self.bce_loss = nn.BCEWithLogitsLoss()
        
        # Regression loss
        if regression_loss == 'mse':
            self.reg_loss = nn.MSELoss()
        elif regression_loss == 'mae':
            self.reg_loss = nn.L1Loss()
        elif regression_loss == 'huber':
            self.reg_loss = nn.HuberLoss()
        else:
            raise ValueError(f"Unknown regression loss: {regression_loss}")
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                binary_logits: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute zero-inflated loss.
        
        Parameters:
        -----------
        predictions : torch.Tensor
            Predicted values (for regression part)
        targets : torch.Tensor
            Target values
        binary_logits : torch.Tensor, optional
            Logits for zero/non-zero classification
            If None, derived from predictions using threshold
            
        Returns:
        --------
        loss : torch.Tensor
            Combined loss
        """
        # Create binary targets (0 if zero, 1 if non-zero)
        binary_targets = (targets > self.threshold).float()
        
        # Binary classification loss
        if binary_logits is None:
            # Derive logits from predictions
            # High prediction -> high probability of non-zero
            binary_logits = torch.log(predictions + 1e-8) - torch.log(torch.ones_like(predictions))
        
        classification_loss = self.bce_loss(binary_logits, binary_targets)
        
        # Regression loss (only on non-zero values)
        nonzero_mask = binary_targets == 1
        if torch.any(nonzero_mask):
            pred_nonzero = predictions[nonzero_mask]
            target_nonzero = targets[nonzero_mask]
            regression_loss = self.reg_loss(pred_nonzero, target_nonzero)
        else:
            regression_loss = torch.tensor(0.0, device=predictions.device)
        
        # Combined loss
        total_loss = (self.classification_weight * classification_loss + 
                     self.regression_weight * regression_loss)
        
        return total_loss


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing extreme class imbalance in zero-inflated data.
    
    Focal Loss down-weights easy examples and focuses on hard examples.
    This is particularly useful when there's extreme imbalance between
    zero and non-zero values.
    
    Parameters:
    -----------
    alpha : float, default=1.0
        Weighting factor for rare class
    gamma : float, default=2.0
        Focusing parameter. Higher gamma puts more focus on hard examples
    reduction : str, default='mean'
        Reduction method
    """
    
    def __init__(self,
                 alpha: float = 1.0,
                 gamma: float = 2.0,
                 reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        This version treats the problem as regression with focal weighting
        based on the magnitude of the error.
        """
        # Compute base loss (MSE)
        mse = (predictions - targets) ** 2
        
        # Compute focal weight
        # Higher error gets more focus
        error_magnitude = torch.abs(predictions - targets)
        max_error = torch.max(error_magnitude)
        normalized_error = error_magnitude / (max_error + 1e-8)
        
        focal_weight = self.alpha * (normalized_error ** self.gamma)
        focal_loss = focal_weight * mse
        
        if self.reduction == 'none':
            return focal_loss
        elif self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")


class HuberLoss(nn.Module):
    """
    Huber Loss (Smooth L1 Loss) with zero-awareness.
    
    Huber loss is less sensitive to outliers than MSE, which can be beneficial
    for zero-inflated data where there might be extreme non-zero values.
    
    Parameters:
    -----------
    delta : float, default=1.0
        Threshold for switching between L1 and L2 loss
    zero_weight : float, default=1.0
        Weight for zero values
    nonzero_weight : float, default=1.0
        Weight for non-zero values
    reduction : str, default='mean'
        Reduction method
    """
    
    def __init__(self,
                 delta: float = 1.0,
                 zero_weight: float = 1.0,
                 nonzero_weight: float = 1.0,
                 reduction: str = 'mean'):
        super().__init__()
        self.delta = delta
        self.zero_weight = zero_weight
        self.nonzero_weight = nonzero_weight
        self.reduction = reduction
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Huber loss with zero-awareness."""
        diff = predictions - targets
        abs_diff = torch.abs(diff)
        
        # Huber loss
        quadratic = torch.min(abs_diff, torch.tensor(self.delta, device=abs_diff.device))
        linear = abs_diff - quadratic
        huber = 0.5 * quadratic**2 + self.delta * linear
        
        # Apply zero-aware weighting
        zero_mask = (targets == 0).float()
        nonzero_mask = 1 - zero_mask
        
        weights = zero_mask * self.zero_weight + nonzero_mask * self.nonzero_weight
        weighted_huber = huber * weights
        
        if self.reduction == 'none':
            return weighted_huber
        elif self.reduction == 'mean':
            return torch.mean(weighted_huber)
        elif self.reduction == 'sum':
            return torch.sum(weighted_huber)
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")


class AdaptiveWeightedLoss(nn.Module):
    """
    Adaptive weighted loss that automatically adjusts weights based on data distribution.
    
    This loss function dynamically computes weights for zero and non-zero values
    based on their frequency in the current batch, helping to maintain balance.
    
    Parameters:
    -----------
    base_loss : str, default='mse'
        Base loss function ('mse', 'mae', 'huber')
    smoothing_factor : float, default=0.1
        Smoothing factor for weight updates (0 = no smoothing, 1 = no update)
    """
    
    def __init__(self,
                 base_loss: str = 'mse',
                 smoothing_factor: float = 0.1):
        super().__init__()
        self.base_loss = base_loss
        self.smoothing_factor = smoothing_factor
        
        # Initialize weights
        self.register_buffer('zero_weight', torch.tensor(1.0))
        self.register_buffer('nonzero_weight', torch.tensor(1.0))
        
        # Base loss function
        if base_loss == 'mse':
            self.loss_fn = nn.MSELoss(reduction='none')
        elif base_loss == 'mae':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif base_loss == 'huber':
            self.loss_fn = nn.HuberLoss(reduction='none')
        else:
            raise ValueError(f"Unknown base loss: {base_loss}")
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute adaptive weighted loss."""
        # Compute base loss
        base_loss = self.loss_fn(predictions, targets)
        
        # Compute current batch statistics
        zero_mask = (targets == 0).float()
        nonzero_mask = 1 - zero_mask
        
        zero_ratio = torch.mean(zero_mask)
        nonzero_ratio = torch.mean(nonzero_mask)
        
        # Compute inverse frequency weights
        batch_zero_weight = 1.0 / (zero_ratio + 1e-8)
        batch_nonzero_weight = 1.0 / (nonzero_ratio + 1e-8)
        
        # Normalize weights
        total_weight = batch_zero_weight + batch_nonzero_weight
        batch_zero_weight /= total_weight
        batch_nonzero_weight /= total_weight
        
        # Update running weights with smoothing
        if self.training:
            self.zero_weight = (self.smoothing_factor * self.zero_weight + 
                               (1 - self.smoothing_factor) * batch_zero_weight)
            self.nonzero_weight = (self.smoothing_factor * self.nonzero_weight + 
                                  (1 - self.smoothing_factor) * batch_nonzero_weight)
        
        # Apply weights
        weights = zero_mask * self.zero_weight + nonzero_mask * self.nonzero_weight
        weighted_loss = base_loss * weights
        
        return torch.mean(weighted_loss)


def compute_class_weights(targets: torch.Tensor, 
                         method: str = 'inverse_frequency') -> tuple:
    """
    Compute class weights for zero and non-zero values.
    
    Parameters:
    -----------
    targets : torch.Tensor
        Target values
    method : str, default='inverse_frequency'
        Method for computing weights ('inverse_frequency', 'balanced')
        
    Returns:
    --------
    zero_weight, nonzero_weight : tuple
        Computed weights for zero and non-zero classes
    """
    zero_mask = (targets == 0).float()
    zero_ratio = torch.mean(zero_mask)
    nonzero_ratio = 1 - zero_ratio
    
    if method == 'inverse_frequency':
        zero_weight = 1.0 / (zero_ratio + 1e-8)
        nonzero_weight = 1.0 / (nonzero_ratio + 1e-8)
    elif method == 'balanced':
        total_samples = len(targets)
        n_zeros = torch.sum(zero_mask)
        n_nonzeros = total_samples - n_zeros
        
        zero_weight = total_samples / (2.0 * n_zeros + 1e-8)
        nonzero_weight = total_samples / (2.0 * n_nonzeros + 1e-8)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Normalize weights
    total_weight = zero_weight + nonzero_weight
    zero_weight /= total_weight
    nonzero_weight /= total_weight
    
    return zero_weight.item(), nonzero_weight.item()


# Utility functions for common loss configurations
def create_zero_aware_loss(loss_type: str, **kwargs) -> nn.Module:
    """
    Factory function for creating zero-aware loss functions.
    
    Parameters:
    -----------
    loss_type : str
        Type of loss ('weighted_mse', 'weighted_mae', 'zero_inflated', 
                     'focal', 'huber', 'adaptive')
    **kwargs
        Additional arguments for the loss function
        
    Returns:
    --------
    loss_fn : nn.Module
        Configured loss function
    """
    if loss_type == 'weighted_mse':
        return WeightedMSELoss(**kwargs)
    elif loss_type == 'weighted_mae':
        return WeightedMAELoss(**kwargs)
    elif loss_type == 'zero_inflated':
        return ZeroInflatedLoss(**kwargs)
    elif loss_type == 'focal':
        return FocalLoss(**kwargs)
    elif loss_type == 'huber':
        return HuberLoss(**kwargs)
    elif loss_type == 'adaptive':
        return AdaptiveWeightedLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")