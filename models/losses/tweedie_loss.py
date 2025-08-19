"""
Tweedie loss function for zero-inflated data.

The Tweedie distribution is particularly suitable for zero-inflated data
as it naturally models a mixture of point mass at zero and continuous
positive distribution for 1 < power < 2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union


class TweedieLoss(nn.Module):
    """
    Tweedie loss function for zero-inflated regression.
    
    The Tweedie distribution belongs to the exponential dispersion family
    and naturally handles zero-inflation when 1 < power < 2.
    
    Mathematical formulation:
    Loss = 2 * [y * μ^(1-p) / (1-p) - μ^(2-p) / (2-p)]
    
    where p is the power parameter and μ is the predicted mean.
    
    Parameters:
    -----------
    power : float, default=1.5
        Tweedie power parameter. Must be in (1, 2) for zero-inflation.
        - p = 1.5: Compound Poisson-Gamma distribution
        - p → 1: Approaches Poisson distribution
        - p → 2: Approaches Gamma distribution
    
    reduction : str, default='mean'
        Specifies the reduction to apply to the output
        - 'none': no reduction will be applied
        - 'mean': the sum of the output will be divided by the number of elements
        - 'sum': the output will be summed
    
    eps : float, default=1e-8
        Small constant to avoid numerical instability
    """
    
    def __init__(self, 
                 power: float = 1.5,
                 reduction: str = 'mean',
                 eps: float = 1e-8):
        super().__init__()
        
        if not (1 < power < 2):
            raise ValueError(f"Power parameter must be in (1, 2) for zero-inflation, got {power}")
        
        self.power = power
        self.reduction = reduction
        self.eps = eps
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Tweedie loss.
        
        Parameters:
        -----------
        predictions : torch.Tensor
            Predicted values (must be positive)
        targets : torch.Tensor
            Target values (non-negative)
            
        Returns:
        --------
        loss : torch.Tensor
            Tweedie loss
        """
        # Ensure predictions are positive
        predictions = torch.clamp(predictions, min=self.eps)
        
        # Ensure targets are non-negative
        targets = torch.clamp(targets, min=0)
        
        # Compute Tweedie deviance
        # D = 2 * [y * μ^(1-p) / (1-p) - μ^(2-p) / (2-p)]
        
        p = self.power
        
        # For numerical stability, handle the case where targets = 0 separately
        zero_mask = (targets == 0)
        nonzero_mask = ~zero_mask
        
        loss = torch.zeros_like(predictions)
        
        # For zero targets: D = 2 * μ^(2-p) / (2-p)
        if torch.any(zero_mask):
            mu_zero = predictions[zero_mask]
            if p != 2:
                loss[zero_mask] = 2 * torch.pow(mu_zero, 2 - p) / (2 - p)
            else:
                # Limiting case as p → 2: Gamma distribution
                loss[zero_mask] = 2 * torch.log(mu_zero)
        
        # For non-zero targets: D = 2 * [y * μ^(1-p) / (1-p) - μ^(2-p) / (2-p)]
        if torch.any(nonzero_mask):
            y_nonzero = targets[nonzero_mask]
            mu_nonzero = predictions[nonzero_mask]
            
            if p != 1 and p != 2:
                term1 = y_nonzero * torch.pow(mu_nonzero, 1 - p) / (1 - p)
                term2 = torch.pow(mu_nonzero, 2 - p) / (2 - p)
                loss[nonzero_mask] = 2 * (term1 - term2)
            elif p == 1:
                # Limiting case as p → 1: Poisson distribution
                # D = 2 * [y * log(μ) - μ]
                loss[nonzero_mask] = 2 * (y_nonzero * torch.log(mu_nonzero) - mu_nonzero)
            else:  # p == 2
                # Limiting case as p → 2: Gamma distribution
                # D = 2 * [log(y/μ) + y/μ - 1]
                ratio = y_nonzero / mu_nonzero
                loss[nonzero_mask] = 2 * (torch.log(ratio) + ratio - 1)
        
        # Apply reduction
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")
    
    def extra_repr(self) -> str:
        """String representation of the module."""
        return f"power={self.power}, reduction={self.reduction}, eps={self.eps}"


class TweedieNLLLoss(nn.Module):
    """
    Negative log-likelihood loss for Tweedie distribution.
    
    This is an alternative formulation that directly optimizes the
    negative log-likelihood of the Tweedie distribution.
    
    Parameters:
    -----------
    power : float, default=1.5
        Tweedie power parameter
    dispersion : float, default=1.0
        Dispersion parameter φ
    reduction : str, default='mean'
        Reduction method
    """
    
    def __init__(self,
                 power: float = 1.5,
                 dispersion: float = 1.0,
                 reduction: str = 'mean',
                 eps: float = 1e-8):
        super().__init__()
        
        if not (1 < power < 2):
            raise ValueError(f"Power parameter must be in (1, 2), got {power}")
            
        self.power = power
        self.dispersion = dispersion
        self.reduction = reduction
        self.eps = eps
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute negative log-likelihood.
        
        Note: This is an approximation since the exact Tweedie density
        involves infinite sums that are computationally intensive.
        """
        predictions = torch.clamp(predictions, min=self.eps)
        targets = torch.clamp(targets, min=0)
        
        p = self.power
        phi = self.dispersion
        
        # Approximate negative log-likelihood
        # -log L ≈ (deviance) / (2φ) + log(φ) / 2
        
        # Compute deviance using TweedieLoss
        tweedie_loss = TweedieLoss(power=p, reduction='none', eps=self.eps)
        deviance = tweedie_loss(predictions, targets)
        
        # Negative log-likelihood approximation
        nll = deviance / (2 * phi) + torch.log(torch.tensor(phi)) / 2
        
        # Apply reduction
        if self.reduction == 'none':
            return nll
        elif self.reduction == 'mean':
            return torch.mean(nll)
        elif self.reduction == 'sum':
            return torch.sum(nll)
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")


def tweedie_loss_function(predictions: torch.Tensor, 
                         targets: torch.Tensor,
                         power: float = 1.5,
                         reduction: str = 'mean',
                         eps: float = 1e-8) -> torch.Tensor:
    """
    Functional interface for Tweedie loss.
    
    Parameters:
    -----------
    predictions : torch.Tensor
        Predicted values
    targets : torch.Tensor  
        Target values
    power : float, default=1.5
        Tweedie power parameter
    reduction : str, default='mean'
        Reduction method
    eps : float, default=1e-8
        Small constant for numerical stability
        
    Returns:
    --------
    loss : torch.Tensor
        Tweedie loss
    """
    loss_fn = TweedieLoss(power=power, reduction=reduction, eps=eps)
    return loss_fn(predictions, targets)


# Convenience functions for common power parameters
def poisson_tweedie_loss(predictions: torch.Tensor, targets: torch.Tensor,
                        reduction: str = 'mean') -> torch.Tensor:
    """Tweedie loss with power=1.1 (close to Poisson)."""
    return tweedie_loss_function(predictions, targets, power=1.1, reduction=reduction)


def compound_poisson_gamma_loss(predictions: torch.Tensor, targets: torch.Tensor,
                               reduction: str = 'mean') -> torch.Tensor:
    """Tweedie loss with power=1.5 (compound Poisson-Gamma)."""
    return tweedie_loss_function(predictions, targets, power=1.5, reduction=reduction)


def gamma_tweedie_loss(predictions: torch.Tensor, targets: torch.Tensor,
                      reduction: str = 'mean') -> torch.Tensor:
    """Tweedie loss with power=1.9 (close to Gamma)."""
    return tweedie_loss_function(predictions, targets, power=1.9, reduction=reduction)