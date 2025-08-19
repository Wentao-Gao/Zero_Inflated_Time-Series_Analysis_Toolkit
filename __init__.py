"""
Zero-Inflated Time Series Analysis Toolkit
===========================================

A comprehensive toolkit for zero-inflated time series analysis, providing complete solutions 
from data preprocessing to model training and evaluation. This toolkit is specifically designed 
for handling time series data with excessive zeros, which is common in many real-world applications.

Key Features:
- Multi-format data support (NumPy, Pandas, CSV)
- Statistical models: ZIP, ZINB, Tweedie GLM, Hurdle
- Deep learning models: ZIP-RNN, Dual Branch Network, Weighted Loss Transformer
- Specialized evaluation metrics for zero-inflated data
- Comprehensive benchmarking suite

Main modules:
- data: Data loading, validation, and preprocessing utilities
- generation: Zero-inflation mechanisms and data generation strategies
- models: Statistical and deep learning models for zero-inflated time series
- evaluation: Specialized metrics, evaluation, and benchmarking tools
- experiments: Experimental framework and configurations

Example usage:
    >>> from data.loaders import ZeroInflatedDataLoader
    >>> from models.baseline.zip_model import ZeroInflatedPoisson
    >>> from evaluation.metrics import ZeroInflatedMetrics
    >>> 
    >>> # Load and prepare data
    >>> loader = ZeroInflatedDataLoader()
    >>> data = loader.load_and_prepare(your_data)
    >>> 
    >>> # Train model
    >>> model = ZeroInflatedPoisson()
    >>> model.fit(X_train, y_train)
    >>> 
    >>> # Evaluate
    >>> evaluator = ZeroInflatedMetrics()
    >>> metrics = evaluator(predictions, targets)
"""

__version__ = "1.0.0"
__title__ = "zero-inflated-timeseries"
__description__ = "A comprehensive toolkit for zero-inflated time series analysis"
__author__ = "Zero-Inflated Research Team"
__author_email__ = "contact@zero-inflated-toolkit.org"
__license__ = "MIT"
__url__ = "https://github.com/your-username/zero-inflated-comprehensive"

# Import main modules
from . import data
from . import generation
from . import models
from . import evaluation

# Make key classes easily accessible
from .data.loaders import ZeroInflatedDataLoader
from .data.formatters import validate_zero_inflated_data, convert_to_standard_format
from .evaluation.metrics import ZeroInflatedMetrics

__all__ = [
    # Modules
    'data',
    'generation',
    'models', 
    'evaluation',
    
    # Key classes
    'ZeroInflatedDataLoader',
    'validate_zero_inflated_data',
    'convert_to_standard_format',
    'ZeroInflatedMetrics',
    
    # Package metadata
    '__version__',
    '__title__',
    '__description__',
    '__author__',
    '__author_email__',
    '__license__',
    '__url__',
]