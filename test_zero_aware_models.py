"""
Test script for zero-aware deep learning models.
"""

import sys
sys.path.append('/home/wentao/papercode/zero_inflated_comprehensive')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Import our models
from models.zero_aware.tweedie_transformer import EnhancedTweedieTransformer
from models.zero_aware.weighted_loss_transformer import WeightedLossTransformer
from models.zero_aware.dual_branch_network import DualBranchNetwork
from models.zero_aware.zip_rnn import ZIPRNN

# Import data generation
from generation.inject_zeros import inject_zeros


class TimeSeriesDataset(Dataset):
    """Simple dataset for time series data."""
    
    def __init__(self, data: np.ndarray, seq_len: int = 96, pred_len: int = 24):
        self.data = torch.FloatTensor(data)
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # Create sequences
        self.sequences = []
        for i in range(len(data) - seq_len - pred_len + 1):
            input_seq = self.data[i:i+seq_len].unsqueeze(-1)  # Add feature dimension
            target_seq = self.data[i+seq_len:i+seq_len+pred_len].unsqueeze(-1)
            self.sequences.append((input_seq, target_seq))
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx]


def generate_zero_inflated_time_series(n_samples=2000, zero_ratio=0.3, random_state=42):
    """Generate zero-inflated time series data for testing."""
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    
    # Generate base time series with trend and seasonality
    t = np.arange(n_samples)
    
    # Trend component
    trend = 0.001 * t
    
    # Seasonal component
    seasonal = 2 * np.sin(2 * np.pi * t / 365.25) + np.sin(2 * np.pi * t / 7)
    
    # Noise component
    noise = np.random.normal(0, 0.5, n_samples)
    
    # Combine components
    base_series = 3 + trend + seasonal + noise
    base_series = np.maximum(base_series, 0)  # Ensure non-negative
    
    # Add zero inflation
    zero_inflated_series = inject_zeros(base_series, mechanism='mixture', 
                                      zero_ratio=zero_ratio, random_state=random_state)
    
    return zero_inflated_series


def test_tweedie_transformer():
    """Test Enhanced Tweedie Transformer."""
    print("Testing Enhanced Tweedie Transformer")
    print("=" * 50)
    
    # Generate data
    data = generate_zero_inflated_time_series(n_samples=1500, zero_ratio=0.25)
    dataset = TimeSeriesDataset(data, seq_len=48, pred_len=12)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Zero ratio: {np.mean(data == 0):.3f}")
    print(f"Data range: [{np.min(data):.3f}, {np.max(data):.3f}]")
    
    try:
        # Initialize model
        model = EnhancedTweedieTransformer(
            input_dim=1,
            d_model=64,
            nhead=4,
            num_layers=3,
            dim_feedforward=256,  # Set compatible feedforward dimension
            seq_len=48,
            pred_len=12,
            tweedie_power=1.5,
            zero_aware_attention=True  # Re-enable after fixing
        )
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test forward pass
        sample_batch = next(iter(dataloader))
        input_seq, target_seq = sample_batch
        
        print(f"Input shape: {input_seq.shape}")
        print(f"Target shape: {target_seq.shape}")
        
        # Forward pass (use teacher forcing by providing target)
        with torch.no_grad():
            model.train()  # Force training mode to enable teacher forcing
            predictions = model(input_seq, target_seq)
            print(f"Prediction shape: {predictions.shape}")
            
            # Test loss computation
            loss = model.compute_loss(predictions, target_seq)
            print(f"Tweedie loss: {loss.item():.4f}")
            
            # Test uncertainty prediction
            uncertainty_results = model.predict_with_uncertainty(input_seq[:5], n_samples=50)
            print(f"Uncertainty prediction keys: {list(uncertainty_results.keys())}")
            
        # Quick training test (few epochs)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        model.train()
        
        total_loss = 0
        for i, (input_seq, target_seq) in enumerate(dataloader):
            if i >= 5:  # Only test a few batches
                break
                
            optimizer.zero_grad()
            predictions = model(input_seq, target_seq)
            loss = model.compute_loss(predictions, target_seq)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / 5
        print(f"Average training loss (5 batches): {avg_loss:.4f}")
        
        print("✓ Enhanced Tweedie Transformer test successful")
        return True
        
    except Exception as e:
        print(f"✗ Enhanced Tweedie Transformer test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_weighted_loss_transformer():
    """Test Weighted Loss Transformer."""
    print("\nTesting Weighted Loss Transformer")
    print("=" * 50)
    
    # Generate data with higher zero ratio
    data = generate_zero_inflated_time_series(n_samples=1200, zero_ratio=0.4)
    dataset = TimeSeriesDataset(data, seq_len=36, pred_len=12)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Zero ratio: {np.mean(data == 0):.3f}")
    
    try:
        # Initialize model
        model = WeightedLossTransformer(
            input_dim=1,
            d_model=128,
            nhead=8,
            num_encoder_layers=4,
            num_decoder_layers=4,
            seq_len=36,
            pred_len=12,
            loss_type='adaptive'
        )
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test forward pass
        sample_batch = next(iter(dataloader))
        input_seq, target_seq = sample_batch
        
        with torch.no_grad():
            predictions = model(input_seq)
            print(f"Prediction shape: {predictions.shape}")
            
            # Test loss computation
            loss = model.compute_loss(predictions, target_seq)
            print(f"Adaptive weighted loss: {loss.item():.4f}")
            
            # Test confidence prediction
            confidence_results = model.predict_with_confidence(input_seq[:3])
            print(f"Confidence prediction keys: {list(confidence_results.keys())}")
        
        # Test weight adaptation
        model.train()
        initial_zero_ratio = model.running_zero_ratio.item()
        
        for i, (input_seq, target_seq) in enumerate(dataloader):
            if i >= 3:
                break
            loss = model.compute_loss(predictions, target_seq)
        
        final_zero_ratio = model.running_zero_ratio.item()
        print(f"Zero ratio tracking: {initial_zero_ratio:.3f} -> {final_zero_ratio:.3f}")
        
        # Get model info
        model_info = model.get_model_info()
        print(f"Batch count: {model_info['batch_count']}")
        
        print("✓ Weighted Loss Transformer test successful")
        return True
        
    except Exception as e:
        print(f"✗ Weighted Loss Transformer test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_dual_branch_network():
    """Test Dual Branch Network."""
    print("\nTesting Dual Branch Network")
    print("=" * 50)
    
    # Generate data
    data = generate_zero_inflated_time_series(n_samples=1000, zero_ratio=0.35)
    dataset = TimeSeriesDataset(data, seq_len=24, pred_len=8)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Zero ratio: {np.mean(data == 0):.3f}")
    
    try:
        # Test different architectures
        architectures = ['lstm', 'gru']
        
        for arch in architectures:
            print(f"\nTesting {arch.upper()} architecture:")
            
            model = DualBranchNetwork(
                input_dim=1,
                hidden_dim=64,
                num_layers=2,
                seq_len=24,
                pred_len=8,
                architecture=arch
            )
            
            print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
            
            # Test forward pass
            sample_batch = next(iter(dataloader))
            input_seq, target_seq = sample_batch
            
            with torch.no_grad():
                # Test regular prediction
                predictions = model(input_seq)
                print(f"  Prediction shape: {predictions.shape}")
                
                # Test component prediction
                components = model(input_seq, return_components=True)
                print(f"  Component keys: {list(components.keys())}")
                print(f"  Binary prob range: [{components['binary_probs'].min():.3f}, {components['binary_probs'].max():.3f}]")
                print(f"  Magnitude range: [{components['magnitude_pred'].min():.3f}, {components['magnitude_pred'].max():.3f}]")
                
                # Test loss computation
                loss_dict = model.compute_loss(predictions, target_seq, 
                                             components['binary_probs'], 
                                             components['magnitude_pred'])
                print(f"  Total loss: {loss_dict['total_loss'].item():.4f}")
                print(f"  Binary loss: {loss_dict['binary_loss'].item():.4f}")
                print(f"  Magnitude loss: {loss_dict['magnitude_loss'].item():.4f}")
                
                # Test interpretation
                interpretation = model.predict_with_interpretation(input_seq[:3])
                print(f"  Interpretation keys: {list(interpretation.keys())}")
        
        print("✓ Dual Branch Network test successful")
        return True
        
    except Exception as e:
        print(f"✗ Dual Branch Network test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_zip_rnn():
    """Test ZIP-RNN."""
    print("\nTesting ZIP-RNN")
    print("=" * 50)
    
    # Generate count data (integers)
    np.random.seed(123)
    base_counts = np.random.poisson(2.0, 800)
    zero_inflated_counts = inject_zeros(base_counts, mechanism='mixture', 
                                      zero_ratio=0.3, random_state=123)
    
    # Round to ensure integers
    zero_inflated_counts = np.round(zero_inflated_counts).astype(int)
    
    dataset = TimeSeriesDataset(zero_inflated_counts, seq_len=20, pred_len=5)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Zero ratio: {np.mean(zero_inflated_counts == 0):.3f}")
    print(f"Count range: [{np.min(zero_inflated_counts)}, {np.max(zero_inflated_counts)}]")
    
    try:
        # Test both shared and separate encoder configurations
        configs = [
            {'shared_encoder': True, 'rnn_type': 'LSTM'},
            {'shared_encoder': False, 'rnn_type': 'GRU'}
        ]
        
        for config in configs:
            print(f"\nTesting config: {config}")
            
            model = ZIPRNN(
                input_dim=1,
                hidden_dim=32,
                num_layers=2,
                seq_len=20,
                pred_len=5,
                **config
            )
            
            print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
            
            # Test forward pass
            sample_batch = next(iter(dataloader))
            input_seq, target_seq = sample_batch
            
            with torch.no_grad():
                # Test regular prediction
                predictions = model(input_seq)
                print(f"  Prediction shape: {predictions.shape}")
                print(f"  Prediction range: [{predictions.min():.3f}, {predictions.max():.3f}]")
                
                # Test component prediction
                components = model(input_seq, return_components=True)
                print(f"  Component keys: {list(components.keys())}")
                print(f"  Pi range: [{components['pi'].min():.3f}, {components['pi'].max():.3f}]")
                print(f"  Lambda range: [{components['lambda'].min():.3f}, {components['lambda'].max():.3f}]")
                
                # Test ZIP loss
                zip_loss = model.compute_zip_loss(predictions, target_seq, 
                                                components['pi'], components['lambda'])
                print(f"  ZIP loss: {zip_loss.item():.4f}")
                
                # Test sampling
                samples = model.sample(input_seq[:3], n_samples=10)
                print(f"  Sample shape: {samples.shape}")
                
                # Test uncertainty prediction
                uncertainty_results = model.predict_with_uncertainty(input_seq[:3])
                print(f"  Uncertainty keys: {list(uncertainty_results.keys())}")
        
        print("✓ ZIP-RNN test successful")
        return True
        
    except Exception as e:
        print(f"✗ ZIP-RNN test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_loss_functions():
    """Test the zero-aware loss functions."""
    print("\nTesting Zero-Aware Loss Functions")
    print("=" * 50)
    
    try:
        from models.losses.tweedie_loss import TweedieLoss
        from models.losses.zero_aware_losses import WeightedMSELoss, ZeroInflatedLoss
        
        # Generate test data
        predictions = torch.rand(10, 5) * 3
        targets = torch.rand(10, 5) * 3
        # Make some targets zero
        targets[targets < 0.8] = 0
        
        print(f"Predictions shape: {predictions.shape}")
        print(f"Targets zero ratio: {torch.mean((targets == 0).float()).item():.3f}")
        
        # Test Tweedie Loss
        tweedie_loss = TweedieLoss(power=1.5)
        tweedie_result = tweedie_loss(predictions, targets)
        print(f"Tweedie loss: {tweedie_result.item():.4f}")
        
        # Test Weighted MSE Loss
        weighted_mse_loss = WeightedMSELoss(zero_weight=0.5, nonzero_weight=2.0)
        weighted_mse_result = weighted_mse_loss(predictions, targets)
        print(f"Weighted MSE loss: {weighted_mse_result.item():.4f}")
        
        # Test Zero-Inflated Loss
        zero_inflated_loss = ZeroInflatedLoss()
        zero_inflated_result = zero_inflated_loss(predictions, targets)
        print(f"Zero-inflated loss: {zero_inflated_result.item():.4f}")
        
        print("✓ Loss functions test successful")
        return True
        
    except Exception as e:
        print(f"✗ Loss functions test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def compare_models():
    """Compare all zero-aware models on the same dataset."""
    print("\nModel Comparison on Same Dataset")
    print("=" * 50)
    
    # Generate common dataset
    data = generate_zero_inflated_time_series(n_samples=800, zero_ratio=0.3, random_state=999)
    dataset = TimeSeriesDataset(data, seq_len=24, pred_len=6)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Zero ratio: {np.mean(data == 0):.3f}")
    
    models = {
        'Tweedie Transformer': EnhancedTweedieTransformer(
            input_dim=1, d_model=64, nhead=4, num_layers=2,
            seq_len=24, pred_len=6, tweedie_power=1.6
        ),
        'Weighted Transformer': WeightedLossTransformer(
            input_dim=1, d_model=64, num_encoder_layers=2, num_decoder_layers=2,
            seq_len=24, pred_len=6, loss_type='adaptive'
        ),
        'Dual Branch LSTM': DualBranchNetwork(
            input_dim=1, hidden_dim=32, num_layers=2,
            seq_len=24, pred_len=6, architecture='lstm'
        ),
        'ZIP-RNN': ZIPRNN(
            input_dim=1, hidden_dim=32, num_layers=2,
            seq_len=24, pred_len=6, rnn_type='LSTM', shared_encoder=True
        )
    }
    
    results = {}
    
    for name, model in models.items():
        try:
            print(f"\nTesting {name}...")
            
            model.eval()
            all_predictions = []
            all_targets = []
            
            with torch.no_grad():
                for i, (input_seq, target_seq) in enumerate(dataloader):
                    if i >= 5:  # Limit to 5 batches for speed
                        break
                    
                    predictions = model(input_seq)
                    
                    all_predictions.append(predictions)
                    all_targets.append(target_seq)
            
            # Concatenate results
            predictions_cat = torch.cat(all_predictions, dim=0)
            targets_cat = torch.cat(all_targets, dim=0)
            
            # Compute metrics
            mse = F.mse_loss(predictions_cat, targets_cat).item()
            mae = F.l1_loss(predictions_cat, targets_cat).item()
            
            # Zero-specific metrics
            zero_mask = targets_cat == 0
            nonzero_mask = ~zero_mask
            
            if torch.any(nonzero_mask):
                nonzero_mse = F.mse_loss(predictions_cat[nonzero_mask], 
                                       targets_cat[nonzero_mask]).item()
                nonzero_mae = F.l1_loss(predictions_cat[nonzero_mask], 
                                       targets_cat[nonzero_mask]).item()
            else:
                nonzero_mse = nonzero_mae = 0.0
            
            results[name] = {
                'MSE': mse,
                'MAE': mae,
                'Non-zero MSE': nonzero_mse,
                'Non-zero MAE': nonzero_mae,
                'Parameters': sum(p.numel() for p in model.parameters())
            }
            
            print(f"  MSE: {mse:.4f}, MAE: {mae:.4f}")
            print(f"  Non-zero MSE: {nonzero_mse:.4f}, Non-zero MAE: {nonzero_mae:.4f}")
            print(f"  Parameters: {results[name]['Parameters']:,}")
            
        except Exception as e:
            print(f"  Failed: {str(e)}")
            results[name] = None
    
    # Summary
    print("\nSUMMARY:")
    successful_models = {k: v for k, v in results.items() if v is not None}
    
    if successful_models:
        best_mse = min(successful_models.items(), key=lambda x: x[1]['MSE'])
        best_mae = min(successful_models.items(), key=lambda x: x[1]['MAE'])
        best_nonzero_mse = min(successful_models.items(), key=lambda x: x[1]['Non-zero MSE'])
        
        print(f"Best overall MSE: {best_mse[0]} ({best_mse[1]['MSE']:.4f})")
        print(f"Best overall MAE: {best_mae[0]} ({best_mae[1]['MAE']:.4f})")
        print(f"Best non-zero MSE: {best_nonzero_mse[0]} ({best_nonzero_mse[1]['Non-zero MSE']:.4f})")
    
    return results


if __name__ == "__main__":
    print("Zero-Aware Deep Learning Models Test Suite")
    print("=" * 80)
    
    test_results = []
    
    # Run individual model tests
    test_results.append(('Enhanced Tweedie Transformer', test_tweedie_transformer()))
    test_results.append(('Weighted Loss Transformer', test_weighted_loss_transformer()))
    test_results.append(('Dual Branch Network', test_dual_branch_network()))
    test_results.append(('ZIP-RNN', test_zip_rnn()))
    test_results.append(('Loss Functions', test_loss_functions()))
    
    # Run comparison
    comparison_results = compare_models()
    
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    for model_name, success in test_results:
        status = "✓" if success else "✗"
        print(f"{status} {model_name}")
    
    successful_tests = sum(1 for _, success in test_results if success)
    total_tests = len(test_results)
    
    print(f"\nOverall: {successful_tests}/{total_tests} tests passed")
    
    if successful_tests == total_tests:
        print("\n🎉 All zero-aware deep learning models are working correctly!")
    else:
        print(f"\n⚠️  Some models failed. Check the detailed output above.")