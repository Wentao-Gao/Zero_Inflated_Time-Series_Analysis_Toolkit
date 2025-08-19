# API参考文档

本文档提供了零膨胀时间序列分析工具包的完整API参考。

## 📋 目录

- [数据模块 (data)](#数据模块-data)
- [模型模块 (models)](#模型模块-models)
- [评估模块 (evaluation)](#评估模块-evaluation)
- [生成模块 (generation)](#生成模块-generation)

## 数据模块 (data)

### data.formatters

#### `StandardTimeSeriesFormat`

标准时间序列数据格式类。

```python
class StandardTimeSeriesFormat:
    def __init__(self,
                 values: np.ndarray,
                 timestamps: Optional[np.ndarray] = None,
                 features: Optional[np.ndarray] = None,
                 metadata: Optional[Dict[str, Any]] = None,
                 zero_threshold: float = 1e-6,
                 frequency: Optional[str] = None)
```

**参数**:
- `values`: 时间序列值（1D或2D数组）
- `timestamps`: 可选的时间戳数组
- `features`: 可选的额外特征数组
- `metadata`: 可选的元数据字典
- `zero_threshold`: 零值阈值
- `frequency`: 时间序列频率

**方法**:

##### `get_zero_ratio() -> float`
计算零值比例。

##### `get_summary_stats() -> Dict[str, Any]`
获取时间序列的摘要统计信息。

**返回**: 包含长度、形状、零值比例等信息的字典。

#### `DataFormatValidator`

数据格式验证器类。

```python
class DataFormatValidator:
    def __init__(self,
                 zero_threshold: float = 1e-6,
                 allow_negative: bool = False)
```

**方法**:

##### `validate_numpy_array(data: np.ndarray) -> Tuple[bool, List[str]]`
验证NumPy数组。

**参数**:
- `data`: 要验证的NumPy数组

**返回**: (是否有效, 问题列表)

##### `validate_pandas_dataframe(data: pd.DataFrame, value_column: str) -> Tuple[bool, List[str]]`
验证Pandas DataFrame。

#### 便捷函数

##### `validate_zero_inflated_data()`

```python
def validate_zero_inflated_data(data: Union[np.ndarray, pd.Series, pd.DataFrame, str],
                               value_column: str = 'value',
                               zero_threshold: float = 1e-6,
                               allow_negative: bool = False) -> Tuple[bool, List[str]]
```

验证零膨胀时间序列数据。

##### `convert_to_standard_format()`

```python
def convert_to_standard_format(data: Union[np.ndarray, pd.Series, pd.DataFrame, str],
                              value_column: str = 'value',
                              timestamp_column: Optional[str] = None,
                              feature_columns: Optional[List[str]] = None,
                              zero_threshold: float = 1e-6) -> StandardTimeSeriesFormat
```

将各种数据格式转换为标准时间序列格式。

### data.loaders

#### `TimeSeriesDataset`

PyTorch数据集类。

```python
class TimeSeriesDataset(Dataset):
    def __init__(self, 
                 data: Union[np.ndarray, StandardTimeSeriesFormat],
                 sequence_length: int = 96,
                 prediction_horizon: int = 24,
                 stride: int = 1,
                 include_features: bool = True,
                 normalize: bool = False)
```

**参数**:
- `data`: 时间序列数据
- `sequence_length`: 输入序列长度
- `prediction_horizon`: 预测时间跨度
- `stride`: 序列间步长
- `include_features`: 是否包含额外特征
- `normalize`: 是否标准化

**方法**:

##### `get_feature_dimension() -> int`
获取特征维度。

##### `denormalize_values(normalized_values: np.ndarray) -> np.ndarray`
反标准化数值。

#### `ZeroInflatedDataLoader`

零膨胀数据加载器类。

```python
class ZeroInflatedDataLoader:
    def __init__(self, zero_threshold: float = 1e-6)
```

**方法**:

##### `load_and_prepare()`

```python
def load_and_prepare(self,
                    data: Union[str, np.ndarray, pd.Series, pd.DataFrame],
                    sequence_length: int = 96,
                    prediction_horizon: int = 24,
                    test_split: float = 0.2,
                    validation_split: float = 0.1,
                    batch_size: int = 32,
                    value_column: str = 'value',
                    timestamp_column: Optional[str] = None,
                    feature_columns: Optional[List[str]] = None,
                    normalize: bool = True,
                    **kwargs) -> Dict[str, Any]
```

加载和准备训练数据。

**返回**: 包含训练/验证/测试数据加载器和元信息的字典。

#### 便捷函数

##### `load_csv_data()`

```python
def load_csv_data(file_path: str,
                  value_column: str = 'value',
                  timestamp_column: Optional[str] = None,
                  feature_columns: Optional[List[str]] = None,
                  **kwargs) -> StandardTimeSeriesFormat
```

从CSV文件加载数据。

### data.preprocessors

#### `ZeroInflatedPreprocessor`

零膨胀数据预处理器。

```python
class ZeroInflatedPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self,
                 method: str = 'log_plus_one',
                 handle_outliers: bool = True,
                 outlier_threshold: float = 3.0,
                 zero_threshold: float = 1e-6)
```

**参数**:
- `method`: 变换方法 ('log_plus_one', 'sqrt', 'none')
- `handle_outliers`: 是否处理异常值
- `outlier_threshold`: 异常值阈值
- `zero_threshold`: 零值阈值

**方法**:

##### `fit(X: np.ndarray, y=None) -> self`
拟合预处理器。

##### `transform(X: np.ndarray) -> np.ndarray`
变换数据。

##### `inverse_transform(X: np.ndarray) -> np.ndarray`
逆变换数据。

#### `SequenceGenerator`

序列生成器。

```python
class SequenceGenerator:
    def __init__(self,
                 sequence_length: int = 96,
                 prediction_horizon: int = 24,
                 stride: int = 1)
```

**方法**:

##### `generate_sequences()`

```python
def generate_sequences(self, data: np.ndarray,
                      features: Optional[np.ndarray] = None
                      ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]
```

从时间序列数据生成序列。

**返回**: (输入序列, 目标序列, 特征序列)

## 模型模块 (models)

### models.baseline

#### `ZeroInflatedPoisson`

零膨胀泊松回归模型。

```python
class ZeroInflatedPoisson(BaseEstimator, RegressorMixin):
    def __init__(self,
                 alpha: float = 1.0,
                 max_iter: int = 100,
                 tol: float = 1e-6,
                 random_state: Optional[int] = None)
```

**参数**:
- `alpha`: 正则化参数
- `max_iter`: 最大迭代次数
- `tol`: 收敛容忍度
- `random_state`: 随机种子

**方法**:

##### `fit(X: np.ndarray, y: np.ndarray) -> self`
训练模型。

##### `predict(X: np.ndarray) -> np.ndarray`
预测。

##### `predict_proba(X: np.ndarray) -> Dict[str, np.ndarray]`
预测概率参数。

**返回**: 包含'pi'（零膨胀概率）和'lambda'（泊松参数）的字典。

#### `ZeroInflatedNegativeBinomial`

零膨胀负二项回归模型。

```python
class ZeroInflatedNegativeBinomial(BaseEstimator, RegressorMixin):
    def __init__(self,
                 alpha: float = 1.0,
                 max_iter: int = 100,
                 tol: float = 1e-6,
                 random_state: Optional[int] = None)
```

与ZIP模型类似，但处理过度分散的数据。

#### `TweedieGLM`

Tweedie广义线性模型。

```python
class TweedieGLM(BaseEstimator, RegressorMixin):
    def __init__(self,
                 power: float = 1.5,
                 alpha: float = 1.0,
                 max_iter: int = 100,
                 tol: float = 1e-6)
```

**参数**:
- `power`: Tweedie幂参数 (1 < power < 2)

### models.zero_aware

#### `ZIPRNN`

零膨胀泊松循环神经网络。

```python
class ZIPRNN(nn.Module):
    def __init__(self,
                 input_dim: int = 1,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 seq_len: int = 96,
                 pred_len: int = 24,
                 rnn_type: str = 'LSTM',
                 dropout: float = 0.1,
                 bidirectional: bool = False,
                 shared_encoder: bool = False)
```

**参数**:
- `input_dim`: 输入特征维度
- `hidden_dim`: 隐藏层维度
- `num_layers`: RNN层数
- `seq_len`: 输入序列长度
- `pred_len`: 预测序列长度
- `rnn_type`: RNN类型 ('LSTM' 或 'GRU')
- `dropout`: Dropout率
- `bidirectional`: 是否双向
- `shared_encoder`: 是否共享编码器

**方法**:

##### `forward(x: torch.Tensor, return_components: bool = False) -> torch.Tensor`
前向传播。

**参数**:
- `x`: 输入序列 (batch_size, seq_len, input_dim)
- `return_components`: 是否返回组件

**返回**: 预测结果或组件字典

##### `compute_zip_loss(predictions, targets, pi, lambda_param) -> torch.Tensor`
计算ZIP损失。

##### `sample(x: torch.Tensor, n_samples: int = 1) -> torch.Tensor`
从拟合的ZIP分布采样。

##### `predict_with_uncertainty(x: torch.Tensor) -> Dict[str, torch.Tensor]`
带不确定性的预测。

#### `DualBranchNetwork`

双分支网络。

```python
class DualBranchNetwork(nn.Module):
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
                 magnitude_activation: str = 'softplus')
```

**参数**:
- `architecture`: 基础架构 ('lstm', 'gru', 'transformer')
- `binary_threshold`: 二分类阈值
- `magnitude_activation`: 幅度输出激活函数

**方法**:

##### `compute_loss(predictions, targets, binary_probs, magnitude_pred) -> Dict[str, torch.Tensor]`
计算双分支损失。

**返回**: 包含各种损失的字典

##### `predict_with_interpretation(x: torch.Tensor) -> Dict[str, torch.Tensor]`
带解释的预测。

#### `WeightedLossTransformer`

带加权损失的Transformer。

```python
class WeightedLossTransformer(nn.Module):
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
                 weight_adjustment: str = 'dynamic')
```

**参数**:
- `loss_type`: 损失类型 ('adaptive', 'fixed_weighted')
- `weight_adjustment`: 权重调整方式 ('dynamic', 'batch', 'epoch')

**方法**:

##### `compute_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor`
计算加权损失。

##### `predict_with_confidence(src: torch.Tensor) -> Dict[str, torch.Tensor]`
带置信度的预测。

### models.losses

#### `TweedieLoss`

Tweedie损失函数。

```python
class TweedieLoss(nn.Module):
    def __init__(self, power: float = 1.5, eps: float = 1e-8)
```

**参数**:
- `power`: Tweedie幂参数
- `eps`: 数值稳定性参数

#### `WeightedMSELoss`

加权均方误差损失。

```python
class WeightedMSELoss(nn.Module):
    def __init__(self, zero_weight: float = 1.0, nonzero_weight: float = 1.0)
```

#### `ZeroInflatedLoss`

零膨胀损失函数。

```python
class ZeroInflatedLoss(nn.Module):
    def __init__(self, alpha: float = 0.5)
```

**参数**:
- `alpha`: 零/非零损失权重平衡参数

## 评估模块 (evaluation)

### evaluation.metrics

#### `ZeroInflatedMetrics`

零膨胀指标计算器。

```python
class ZeroInflatedMetrics:
    def __init__(self, zero_threshold: float = 1e-6)
```

**方法**:

##### `__call__(predictions: np.ndarray, targets: np.ndarray, return_components: bool = False) -> Dict[str, float]`
计算所有零膨胀指标。

**返回**: 指标字典

##### `compute_forecasting_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]`
计算标准预测指标。

**返回**: 包含MSE, MAE, RMSE, R²等指标的字典

##### `compute_zero_inflation_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]`
计算零膨胀特定指标。

**返回**: 包含零值分类准确率、F1得分等指标的字典

##### `compute_distribution_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]`
计算分布指标。

**返回**: 包含偏度、峰度误差等指标的字典

#### 便捷函数

##### `compute_torch_metrics()`

```python
def compute_torch_metrics(predictions: torch.Tensor, targets: torch.Tensor,
                         zero_threshold: float = 1e-6) -> Dict[str, float]
```

为PyTorch张量计算指标。

##### `format_metrics_report()`

```python
def format_metrics_report(metrics: Dict[str, float], title: str = "Metrics Report") -> str
```

格式化指标报告。

##### `compare_models_metrics()`

```python
def compare_models_metrics(model_metrics: Dict[str, Dict[str, float]], 
                          metric_names: Optional[list] = None) -> str
```

比较多个模型的指标。

### evaluation.evaluator

#### `ModelEvaluator`

模型评估器。

```python
class ModelEvaluator:
    def __init__(self, metrics_calculator: Optional[ZeroInflatedMetrics] = None,
                 device: str = 'cpu')
```

**方法**:

##### `evaluate_model()`

```python
def evaluate_model(self, model: Any, X: np.ndarray, y: np.ndarray,
                  model_name: str = "Unknown",
                  prediction_method: str = "predict",
                  **kwargs) -> EvaluationResult
```

评估单个模型。

##### `evaluate_pytorch_model()`

```python
def evaluate_pytorch_model(self, model: nn.Module, dataloader,
                          model_name: str = "PyTorch Model",
                          device: Optional[str] = None) -> EvaluationResult
```

评估PyTorch模型。

##### `compare_models()`

```python
def compare_models(self, results: List[EvaluationResult],
                  key_metrics: Optional[List[str]] = None) -> Dict[str, Any]
```

比较多个模型的评估结果。

#### `CrossValidationEvaluator`

交叉验证评估器。

```python
class CrossValidationEvaluator:
    def __init__(self, cv_strategy: str = 'time_series', n_splits: int = 5,
                 metrics_calculator: Optional[ZeroInflatedMetrics] = None)
```

**参数**:
- `cv_strategy`: 交叉验证策略 ('time_series' 或 'kfold')
- `n_splits`: 分割数

**方法**:

##### `cross_validate()`

```python
def cross_validate(self, model_factory: Callable, X: np.ndarray, y: np.ndarray,
                  model_name: str = "Model",
                  fit_method: str = "fit",
                  predict_method: str = "predict",
                  **model_kwargs) -> Dict[str, Any]
```

执行交叉验证。

#### `ComprehensiveEvaluation`

综合评估类。

```python
class ComprehensiveEvaluation:
    def __init__(self)
```

**方法**:

##### `full_evaluation()`

```python
def full_evaluation(self, models: Dict[str, Any], X: np.ndarray, y: np.ndarray,
                   test_split: float = 0.2, 
                   perform_cv: bool = True) -> Dict[str, Any]
```

执行全面评估。

##### `generate_report()`

```python
def generate_report(self, results: Optional[Dict[str, Any]] = None,
                   save_path: Optional[str] = None) -> str
```

生成综合评估报告。

### evaluation.benchmarks

#### `BenchmarkSuite`

基准测试套件。

```python
class BenchmarkSuite:
    def __init__(self, random_state: int = 42)
```

**方法**:

##### `create_synthetic_datasets() -> Dict[str, BenchmarkDataset]`
创建合成基准数据集。

##### `run_benchmark()`

```python
def run_benchmark(self, models: Dict[str, Any], 
                 dataset_names: Optional[List[str]] = None,
                 test_split: float = 0.3,
                 sequence_length: int = 48,
                 prediction_horizon: int = 12) -> Dict[str, Any]
```

运行基准测试。

##### `generate_benchmark_report()`

```python
def generate_benchmark_report(self, results: Optional[Dict[str, Any]] = None,
                             save_path: Optional[str] = None) -> str
```

生成基准测试报告。

#### `StandardBenchmarks`

标准基准测试。

**静态方法**:

##### `quick_benchmark(models: Dict[str, Any]) -> Dict[str, Any]`
快速基准测试。

##### `comprehensive_benchmark(models: Dict[str, Any]) -> Dict[str, Any]`
全面基准测试。

##### `zero_inflation_focused_benchmark(models: Dict[str, Any]) -> Dict[str, Any]`
零膨胀聚焦基准测试。

## 生成模块 (generation)

### generation.zero_mechanisms

#### `ThresholdZeroInflation`

基于阈值的零膨胀机制。

```python
class ThresholdZeroInflation(BaseZeroInflationMechanism):
    def __init__(self, threshold_value: float = 1.0, threshold_prob: float = 0.8)
```

#### `MixtureZeroInflation`

混合分布零膨胀机制。

```python
class MixtureZeroInflation(BaseZeroInflationMechanism):
    def __init__(self, zero_prob: float = 0.3)
```

#### `TweedieZeroInflation`

Tweedie分布零膨胀机制。

```python
class TweedieZeroInflation(BaseZeroInflationMechanism):
    def __init__(self, power: float = 1.5, phi: float = 1.0, zero_prob: float = 0.2)
```

#### `HurdleZeroInflation`

跨栏模型零膨胀机制。

```python
class HurdleZeroInflation(BaseZeroInflationMechanism):
    def __init__(self, zero_prob: float = 0.3, truncate_at: float = 0.0)
```

### generation.inject_zeros

#### 便捷函数

##### `inject_zeros()`

```python
def inject_zeros(data: np.ndarray, 
                mechanism: str = 'mixture',
                zero_ratio: float = 0.3,
                random_state: Optional[int] = None,
                **kwargs) -> np.ndarray
```

向时间序列数据注入零膨胀。

**参数**:
- `data`: 原始时间序列数据
- `mechanism`: 零膨胀机制 ('threshold', 'mixture', 'tweedie', 'hurdle')
- `zero_ratio`: 目标零值比例
- `random_state`: 随机种子

**返回**: 零膨胀后的时间序列数据

##### `auto_inject_zeros()`

```python
def auto_inject_zeros(data: np.ndarray,
                     target_zero_ratio: float = 0.3,
                     random_state: Optional[int] = None) -> Tuple[np.ndarray, str]
```

自动选择最佳零膨胀机制。

**返回**: (零膨胀数据, 选择的机制名称)

##### `compare_zero_inflation_methods()`

```python
def compare_zero_inflation_methods(data: np.ndarray,
                                  zero_ratio: float = 0.3,
                                  random_state: Optional[int] = None) -> Dict[str, np.ndarray]
```

比较不同零膨胀方法。

**返回**: 不同方法生成的零膨胀数据字典

## 使用示例

### 基本工作流程

```python
# 1. 数据准备
from data.loaders import ZeroInflatedDataLoader
from data.formatters import validate_zero_inflated_data

# 验证数据
is_valid, issues = validate_zero_inflated_data(your_data)

# 准备训练数据
loader = ZeroInflatedDataLoader()
prepared_data = loader.load_and_prepare(
    data=your_data,
    sequence_length=24,
    prediction_horizon=6
)

# 2. 模型训练
from models.zero_aware.zip_rnn import ZIPRNN

model = ZIPRNN(
    input_dim=prepared_data['data_info']['feature_dimension'],
    seq_len=24,
    pred_len=6
)

# 3. 评估
from evaluation.metrics import ZeroInflatedMetrics

evaluator = ZeroInflatedMetrics()
metrics = evaluator(predictions, targets)

# 4. 基准测试
from evaluation.benchmarks import StandardBenchmarks

results = StandardBenchmarks.quick_benchmark({'My_Model': model})
```

这个API参考文档提供了工具包中所有主要类和函数的详细信息。更多使用示例请参考用户指南和代码示例。