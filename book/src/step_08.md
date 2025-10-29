# Step 08: Attention mechanism with causal masking

<div class="note">
    Learn to implement the core attention mechanism using scaled dot-product attention with causal masking.
</div>

## What is the attention mechanism?

In this section you will implement the attention mechanism. Given query, key, and value tensors (from Step 07), attention computes how much each position should "pay attention to" other positions, then creates weighted combinations of values.

The process:
1. Compute similarity scores: Q @ K^T (dot product of queries and keys)
2. Scale by sqrt(d_k) to prevent large values
3. Apply causal mask to block future tokens
4. Apply softmax to convert scores to probabilities
5. Multiply probabilities by values to get the output

This creates context-aware representations where each token incorporates information from relevant previous tokens.

## Why use causal masking?

**1. Autoregressive Generation**: Language models generate text one token at a time, left-to-right. During generation, token N cannot see tokens N+1, N+2, etc. because they don't exist yet. Causal masking enforces this constraint during training, ensuring the model learns to generate each token using only previous context. Without causal masking, the model would "cheat" during training by looking at future tokens, then fail at generation time when those tokens aren't available.

**2. Training-Inference Consistency**: Training with causal masking ensures the model sees the same information during training as it will during generation. If the model trained with access to future tokens, it would learn to rely on that information. At generation time, this information isn't available, causing a train-test mismatch that degrades performance. Causal masking eliminates this mismatch.

**3. Parallel Training**: Without masking, you'd need to process each token sequentially to prevent information leakage from future tokens. Causal masking allows parallel processing of the entire sequence during training—all positions are computed simultaneously, with the mask preventing future information flow. This dramatically speeds up training while maintaining autoregressive properties.

**4. Mathematical Foundation**: The mask works by adding -∞ to attention scores for future positions. When softmax is applied, e^(-∞) = 0, so future positions get zero attention weight. This is mathematically clean and differentiable, allowing gradients to flow properly during backpropagation. The model learns which past tokens are relevant without ever seeing the future.

### Key concepts

**Scaled Dot-Product Attention**:
- Compute similarity: `query @ key.transpose(-1, -2)` produces shape `[..., seq_length, seq_length]`
- Scale by `sqrt(d_k)` to prevent large values that saturate softmax
- Without scaling, large d_k leads to large dot products → extreme softmax values → vanishing gradients
- Mathematical formula:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

where $Q$ is the query matrix, $K$ is the key matrix, $V$ is the value matrix, and $d_k$ is the dimension of the keys.

**Causal Mask Implementation**:
- Uses [`F.band_part(mask, num_lower=None, num_upper=0, exclude=True)`](https://docs.modular.com/max/api/python/experimental/functional#max.experimental.functional.band_part)
- `num_lower=None` keeps all lower diagonal elements (past tokens)
- `num_upper=0` with `exclude=True` masks upper diagonal elements (future tokens)
- Creates upper-triangular matrix of -∞ values
- Shape: `[seq_length, seq_length]`

**Mask Application**:
- Add mask to attention scores: `attn_weights + mask`
- Masked positions (future) have score = original_score + (-∞) = -∞
- Unmasked positions (past/present) have score = original_score + 0 = original_score
- After softmax: masked positions contribute 0 to the weighted sum

**Softmax Normalization**:
- [`F.softmax(attn_weights)`](https://docs.modular.com/max/api/python/experimental/functional#max.experimental.functional.softmax) converts scores to probabilities
- Probabilities sum to 1 across each query position
- Applied to the last dimension (across all keys for each query)
- Produces attention weights in range [0, 1]

**Attention Output**:
- Weighted sum: `attn_weights @ value`
- Each position's output is a weighted combination of all value vectors
- High attention weights mean more influence from those positions
- Shape preserved: `[batch, seq_length, d_v]`

### Implementation tasks (`step_08.py`)

1. **Import Required Modules** (Lines 13-19):
   - Import `math` for `sqrt` function
   - Import `functional as F` from `max.experimental`
   - Import `Tensor` from `max.experimental.tensor`
   - Import `Device` from `max.driver`
   - Import `DType` from `max.dtype`
   - Import `Dim, DimLike` from `max.graph`

2. **Implement causal_mask Function** (Lines 46-51):
   - Use `@F.functional` decorator
   - Calculate total length: `n = Dim(sequence_length) + num_tokens`
   - Create -∞ constant: `Tensor.constant(float("-inf"), dtype=dtype, device=device)`
   - Broadcast to shape: `F.broadcast_to(mask, shape=(sequence_length, n))`
   - Return upper triangle as -∞: `F.band_part(mask, num_lower=None, num_upper=0, exclude=True)`

3. **Compute Attention Scores** (Lines 68-69):
   - Multiply query and transposed key: `query @ key.transpose(-1, -2)`
   - This gives similarity scores between all query-key pairs
   - Shape: `[..., seq_length, seq_length]`

4. **Scale Scores** (Lines 72-74):
   - Calculate scale factor: `math.sqrt(int(value.shape[-1]))`
   - Divide attention weights: `attn_weights / scale_factor`
   - Prevents softmax saturation with large embedding dimensions

5. **Apply Causal Mask** (Lines 77-80):
   - Get sequence length: `seq_len = query.shape[-2]`
   - Create mask: `causal_mask(seq_len, 0, dtype=query.dtype, device=query.device)`
   - Add mask to scores: `attn_weights + mask` (-∞ for future positions)

6. **Apply Softmax and Compute Output** (Lines 83-89):
   - Normalize to probabilities: `F.softmax(attn_weights)`
   - Weighted sum of values: `attn_weights @ value`
   - Return the attention output

**Implementation**:
```python
import math

from max.driver import Device
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.graph import Dim, DimLike


@F.functional
def causal_mask(
    sequence_length: DimLike,
    num_tokens: DimLike,
    *,
    dtype: DType,
    device: Device,
):
    """Create a causal attention mask."""
    n = Dim(sequence_length) + num_tokens
    mask = Tensor.constant(float("-inf"), dtype=dtype, device=device)
    mask = F.broadcast_to(mask, shape=(sequence_length, n))
    return F.band_part(mask, num_lower=None, num_upper=0, exclude=True)


def compute_attention(query, key, value):
    """Compute scaled dot-product attention with causal masking."""
    # Step 1: Compute attention scores
    attn_weights = query @ key.transpose(-1, -2)

    # Step 2: Scale by sqrt(d_k)
    scale_factor = math.sqrt(int(value.shape[-1]))
    attn_weights = attn_weights / scale_factor

    # Step 3: Apply causal mask
    seq_len = query.shape[-2]
    mask = causal_mask(seq_len, 0, dtype=query.dtype, device=query.device)
    attn_weights = attn_weights + mask

    # Step 4: Softmax
    attn_weights = F.softmax(attn_weights)

    # Step 5: Weighted sum of values
    attn_output = attn_weights @ value

    return attn_output
```

### Validation
Run `pixi run s08`

A failed test will show:
```bash
Running tests for Step 08: Attention Mechanism with Causal Masking...

Results:
❌ math is not imported
   Hint: Add 'import math' at the top
❌ functional is not imported from max.experimental
   Hint: Add 'from max.experimental import functional as F'
❌ Tensor is not imported from max.experimental.tensor
   Hint: Add 'from max.experimental.tensor import Tensor'
❌ causal_mask function not found in step_08 module
   Hint: Define causal_mask function with @F.functional decorator
❌ compute_attention function not found in step_08 module
   Hint: Define compute_attention function
❌ Attention scores should be computed with query @ key.transpose(-1, -2)
   Hint: attn_weights = query @ key.transpose(-1, -2)
❌ Found placeholder 'None' values that need to be replaced:
   return None
   attn_weights = None
   return None
   Hint: Replace all 'None' values with the actual implementation

============================================================
⚠️ Some checks failed. Review the hints above and try again.
============================================================
```

A successful test will show:
```bash
Running tests for Step 08: Attention Mechanism with Causal Masking...

Results:
✅ math is correctly imported
✅ functional is correctly imported from max.experimental
✅ Tensor is correctly imported from max.experimental.tensor
✅ Device is correctly imported from max.driver
✅ DType is correctly imported from max.dtype
✅ Dim is correctly imported from max.graph
✅ DimLike is correctly imported from max.graph
✅ causal_mask function exists
✅ compute_attention function exists
✅ causal_mask uses @F.functional decorator
✅ causal_mask creates -inf constant correctly
✅ causal_mask uses F.broadcast_to
✅ causal_mask uses F.band_part
✅ Attention scores computed with Q @ K^T
✅ Scaling uses math.sqrt
✅ Causal mask is applied to attention weights
✅ Softmax is applied to attention weights
✅ Weighted sum computed with attention @ value
✅ All placeholder 'None' values have been replaced
✅ causal_mask executes without errors
✅ causal_mask shape is correct: (4, 4)
✅ causal_mask has correct values (0 for past/present, -inf for future)
✅ compute_attention executes without errors
✅ compute_attention output shape is correct: (2, 4, 64)
✅ Output contains non-zero values
✅ Output is different from input value (attention is applied)

============================================================
🎉 All checks passed! Your implementation is complete.
============================================================
```

**Reference**: `solutions/solution_08.py`

---

**Next**: In [Step 09](./step_09.md), you'll extend this single-head attention to multi-head attention, allowing the model to attend to different representation subspaces simultaneously.
