# Step 03: Layer normalization

<div class="note">
    Learn to implement layer normalization for stabilizing neural network training.
</div>

## What is layer normalization?

In this section you will create the `LayerNorm` class. This normalizes activations across the feature dimension to stabilize training. Unlike batch normalization, [layer normalization](https://arxiv.org/abs/1607.06450) works independently for each example, making it ideal for transformers.

The process:
1. Compute mean and variance across features
2. Normalize by subtracting mean and dividing by standard deviation (plus a small epsilon)
3. Scale and shift using learned weight and bias parameters

GPT-2 applies layer normalization before the attention and MLP blocks in each transformer layer.

## Why use layer normalization?

**1. Training Stability**: Layer normalization reduces internal covariate shift, stabilizing the distribution of layer inputs during training. This allows for higher learning rates and faster convergence.

**2. Position-Independent**: Unlike batch normalization, layer norm doesn't depend on batch size or statistics, making it ideal for:
   - Variable-length sequences
   - Small batch sizes
   - Recurrent and transformer architectures

**3. Inference Consistency**: Layer norm computes statistics per example, so there's no train-test discrepancy (batch norm requires tracking running statistics for inference).

**4. Transformer Standard**: Layer normalization has become the de facto normalization technique in transformer architectures, including GPT-2, BERT, and their variants. GPT-2 uses layer norm before the attention and MLP blocks in each transformer layer.

### Key concepts

**Layer Normalization Mechanics**:
- Normalizes across the feature/embedding dimension (last dimension)
- Computes mean and variance independently for each example
- Learns two parameters per feature: weight (�/gamma) and bias (�/beta)
- Small epsilon (typically 1e-5) prevents division by zero

**Learnable Parameters**:
- `weight` (gamma): Learned scaling parameter, initialized to ones
- `bias` (beta): Learned shift parameter, initialized to zeros
- Both have shape `[dim]` where dim is the feature dimension

**MAX Tensor Initialization**:
- [`Tensor.ones()`](https://docs.modular.com/max/api/python/experimental/tensor#max.experimental.tensor.Tensor.ones): Creates tensor filled with 1.0 values
- [`Tensor.zeros()`](https://docs.modular.com/max/api/python/experimental/tensor#max.experimental.tensor.Tensor.zeros): Creates tensor filled with 0.0 values
- Both methods take a shape argument as a list: `[dim]`

**MAX Layer Normalization**:
- [`F.layer_norm()`](https://docs.modular.com/max/api/python/experimental/functional#max.experimental.functional.layer_norm): Applies layer normalization
- Parameters:
  - `input`: Tensor to normalize
  - `gamma`: Weight/scale parameter (our `self.weight`)
  - `beta`: Bias/shift parameter (our `self.bias`)
  - `epsilon`: Small constant for numerical stability

### Implementation tasks (`step_03.py`)

1. **Import Required Modules** (Lines 1-6):
   - `functional as F` from `max.experimental` - provides F.layer_norm()
   - `Tensor` from `max.experimental.tensor` - needed for Tensor.ones() and Tensor.zeros()

2. **Initialize Weight Parameter** (Lines 24-27):
   - Use `Tensor.ones([dim])` to create weight parameter
   - Initialized to ones so initial normalization is identity (before training)
   - This is the gamma (�) parameter that scales the normalized values

3. **Initialize Bias Parameter** (Lines 29-32):
   - Use `Tensor.zeros([dim])` to create bias parameter
   - Initialized to zeros so initial normalization has no shift (before training)
   - This is the beta (�) parameter that shifts the normalized values

4. **Apply Layer Normalization** (Lines 43-47):
   - Use `F.layer_norm(x, gamma=self.weight, beta=self.bias, epsilon=self.eps)`
   - Returns the normalized tensor
   - The epsilon value (1e-5) is already set in `__init__`

**Implementation**:
```python
# Import required modules from MAX
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.graph import DimLike
from max.nn.module_v3 import Module


class LayerNorm(Module):
    """Layer normalization module."""

    def __init__(self, dim: DimLike, *, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

        # Initialize learnable weight parameter (gamma)
        self.weight = Tensor.ones([dim])

        # Initialize learnable bias parameter (beta)
        self.bias = Tensor.zeros([dim])

    def __call__(self, x: Tensor) -> Tensor:
        """Apply layer normalization."""
        # Apply layer normalization with learned parameters
        return F.layer_norm(x, gamma=self.weight, beta=self.bias, epsilon=self.eps)
```

### Validation
Run `pixi run s03`

A failed test will show:
```bash
Running tests for Step 03: Implement Layer Normalization...

Results:
❌ Error importing step_03 module: name 'Tensor' is not defined
```

A successful test will show:
```bash
Running tests for Step 03: Implement Layer Normalization...

Results:
✅ functional module is correctly imported as F from max.experimental
✅ Tensor is correctly imported from max.experimental.tensor
✅ LayerNorm class exists
✅ Tensor.ones is used for weight initialization
✅ Tensor.zeros is used for bias initialization
✅ F.layer_norm is used
✅ F.layer_norm is used with correct parameters (gamma, beta, epsilon)
✅ All placeholder 'None' values have been replaced
✅ LayerNorm class can be instantiated
✅ LayerNorm.weight is initialized
✅ LayerNorm.bias is initialized
✅ LayerNorm.eps has correct default value: 1e-05
```

**Reference**: `solutions/solution_03.py`