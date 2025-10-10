# Step 02: Implement Causal Masking (`step_02.py`)

**Purpose**: Create attention masks to prevent the model from "seeing" future tokens.

## What is Causal Masking?

[Causal masking](https://docs.modular.com/glossary/ai/attention-mask/) is a fundamental technique in autoregressive language models that enforces temporal causality - ensuring that predictions for a given position can only depend on known outputs at earlier positions, not future positions.

In practice, this is implemented by creating a mask matrix that sets attention scores to negative infinity (`-inf`) for future token positions. When the softmax function is applied during attention computation, these `-inf` values become zero probabilities, effectively blocking information flow from future tokens.

This creates a lower triangular attention pattern where each token can only "see" itself and previous tokens, enabling true autoregressive generation.

## Why Use Causal Masking?

**1. Autoregressive Generation**: GPT-2 generates text one token at a time, left-to-right. During training, causal masking prevents the model from "cheating" by looking ahead at tokens it should be predicting, forcing it to learn genuine next-token prediction.

**2. Training-Inference Consistency**: Without causal masking during training, the model would have access to information it won't have during actual text generation, creating a train-test mismatch that degrades performance.

**3. Parallel Training**: Causal masking enables efficient parallel training. Instead of computing predictions sequentially, we can process an entire sequence at once, with the mask ensuring each position only uses appropriate context.

**4. KV Cache Compatibility**: The causal structure allows for key-value caching during generation - we can cache past token representations and only compute new ones, significantly speeding up inference.

### Key Concepts:

**Causal Masking Mechanics**:
- Sets attention scores to `-inf` for future positions before softmax
- After softmax, `-inf` values become 0 probability (no attention)
- Creates lower triangular attention pattern: position `i` can attend to positions `0` through `i`
- Essential for autoregressive language modeling

**MAX Functional Programming**:
- [`@F.functional`](https://docs.modular.com/max/api/python/experimental/functional/#max.experimental.functional.functional) decorator converts functions to graph operations
- Enables type flexibility and optimization across different tensor types
- Required for functions that will be traced and compiled by MAX

**MAX Tensor Operations**:
- [`Tensor.constant()`](https://docs.modular.com/max/api/python/experimental/tensor#max.experimental.tensor.Tensor.constant): Creates a scalar constant tensor with specified dtype and device
- [`F.broadcast_to()`](https://docs.modular.com/max/api/python/experimental/functional#max.experimental.functional.broadcast_to): Expands tensor dimensions to match target shape
- [`F.band_part()`](https://docs.modular.com/max/api/python/experimental/functional#max.experimental.functional.band_part): Extracts band matrix (keeps central diagonal band, zeros out rest)

**Mask Shape and Dimensions**:
- Input: `sequence_length` (current sequence) and `num_tokens` (additional context)
- Output shape: `(sequence_length, sequence_length + num_tokens)`
- Allows attending to both current sequence and previous context (KV cache)

### Implementation Tasks (`step_02.py`):

1. **Import Required Modules** (Lines 1-10):
   - `Device` from `max.driver` - specifies hardware device (CPU/GPU)
   - `DType` from `max.dtype` - data type specification
   - `functional as F` from `max.experimental` - functional operations library
   - `Tensor` from `max.experimental.tensor` - tensor operations
   - `Dim`, `DimLike` from `max.graph` - dimension handling

2. **Add @F.functional Decorator** (Line 13):
   - Converts the function to a MAX graph operation
   - Required for compilation and optimization

3. **Calculate Total Sequence Length** (Line 34):
   - Combine `sequence_length` and `num_tokens` using `Dim()`
   - This determines the width of the attention mask

4. **Create Constant Tensor** (Lines 36-40):
   - Use `Tensor.constant(float("-inf"), dtype=dtype, device=device)`
   - Creates a scalar tensor that will be broadcast
   - `-inf` values will become 0 after softmax, blocking attention

5. **Broadcast to Target Shape** (Lines 42-46):
   - Use `F.broadcast_to(mask, shape=(sequence_length, n))`
   - Expands the scalar to a 2D matrix
   - Shape: `[sequence_length, sequence_length + num_tokens]`

6. **Apply Band Part** (Lines 48-52):
   - Use `F.band_part(mask, num_lower=None, num_upper=0, exclude=True)`
   - `num_lower=None`: Keep all diagonals below main diagonal (past tokens)
   - `num_upper=0`: Keep main diagonal (current token)
   - `exclude=True`: Set band to 0, outside band to original values (inverted behavior)
   - Creates lower triangular pattern with 0s on/below diagonal, `-inf` above

**Implementation**:
```python
# Import required modules from MAX
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
    """Create a causal mask for autoregressive attention."""
    # Calculate total sequence length
    n = Dim(sequence_length) + num_tokens

    # Create constant tensor filled with -inf
    mask = Tensor.constant(float("-inf"), dtype=dtype, device=device)

    # Broadcast to target shape
    mask = F.broadcast_to(mask, shape=(sequence_length, n))

    # Apply band_part to create lower triangular structure
    return F.band_part(mask, num_lower=None, num_upper=0, exclude=True)
```

### Validation:
Run `pixi run s02`

A failed test will show:
```bash
Running tests for Step 02: Implement Causal Masking...

Results:
❌ functional module is not imported from max.experimental
❌ Tensor is not imported from max.experimental.tensor
❌ causal_mask function does not have @F.functional decorator
❌ causal_mask does not create Tensor.constant correctly
❌ causal_mask does not use F.broadcast_to correctly
❌ causal_mask does not use F.band_part correctly
```

A successful test will show:
```bash
Running tests for Step 02: Implement Causal Masking...

Results:
✅ functional module is correctly imported as F
✅ Tensor is correctly imported
✅ causal_mask has @F.functional decorator
✅ Mask shape is correct: (sequence_length, sequence_length + num_tokens)
✅ Mask creates proper causal (lower triangular) pattern
✅ All implementation steps completed correctly
```

**Reference**: `solutions/solution_02.py`