# Step 09: Multi-head attention

<div class="note">
    Learn to extend single-head attention to multi-head attention, enabling the model to attend to different representation subspaces.
</div>

## What is multi-head attention?

In this section you will extend single-head attention to multi-head attention. Instead of one attention operation, you'll run 12 attention operations in parallel, each learning to focus on different patterns.

GPT-2 uses 12 heads with 768-dimensional embeddings. Each head operates on 768 ÷ 12 = 64 dimensions. The Q, K, V tensors are reshaped to split the embedding dimension across heads, then attention is computed for all heads in parallel. The outputs are concatenated back together.

Each head can specialize—one might attend to adjacent tokens, another to syntactic relationships, another to semantic similarity. This gives the model multiple simultaneous views of the input.

## Why use multiple heads?

**1. Diverse Attention Patterns**: Different heads learn to attend to different types of relationships. Research on attention visualization shows that individual heads specialize—some attend to adjacent tokens, others to tokens at fixed distances, others to specific syntactic roles. A single head cannot capture all these patterns simultaneously. Multiple heads let the model learn complementary attention strategies.

**2. Representation Subspaces**: Each head projects inputs into a different learned subspace before computing attention. This means head 1 might learn a subspace where semantically similar words are close, while head 2 learns a subspace optimized for positional relationships. These different subspaces enable the model to simultaneously consider multiple aspects of the input that might not be compatible in a single representation.

**3. Increased Model Capacity**: Multiple heads increase the model's expressiveness without dramatically increasing computation. Instead of one large attention operation, we perform multiple smaller ones in parallel. The total number of parameters is similar (the projection matrices sum to the same size), but the model gains flexibility to learn distinct attention patterns for each head.

**4. Gradient Flow and Learning**: Multiple heads provide multiple paths for gradients during training. If one head gets stuck in a poor local optimum, other heads can still learn useful patterns. This redundancy makes training more robust and helps the model learn diverse features that complement each other.

### Key concepts

**Head Splitting**:
- Transform `[batch, seq_length, n_embd]` → `[batch, num_heads, seq_length, head_dim]`
- First reshape to add head dimension: `[batch, seq_length, num_heads, head_dim]`
- Then transpose to move heads: `[batch, num_heads, seq_length, head_dim]`
- GPT-2: 768 dims split into 12 heads × 64 dims/head
- Each head operates independently on its 64-dimensional subspace

**Head Merging**:
- Reverse of splitting: `[batch, num_heads, seq_length, head_dim]` → `[batch, seq_length, n_embd]`
- First transpose: `[batch, seq_length, num_heads, head_dim]`
- Then reshape to flatten heads: `[batch, seq_length, n_embd]`
- Concatenates all head outputs back into original dimension

**Parallel Attention**:
- Same attention mechanism as Step 08, but applied to all heads simultaneously
- Shape `[batch, num_heads, seq_length, head_dim]` allows broadcasting
- Q @ K^T operates on last two dimensions: `[seq_length, head_dim] @ [head_dim, seq_length]`
- All heads computed in single operation—highly efficient

**Output Projection (c_proj)**:
- After merging heads, apply learned linear transformation
- Maps `[batch, seq_length, n_embd]` → `[batch, seq_length, n_embd]`
- Allows model to mix information across heads
- Essential for combining the different head perspectives

**HuggingFace Architecture**:
- `c_attn`: Combined Q/K/V projection to `3 * n_embd`
- `c_proj`: Output projection after merging heads
- This naming matches original GPT-2 implementation
- Required for loading pretrained weights

### Implementation tasks (`step_09.py`)

1. **Import Required Modules** (Lines 13-15):
   - Copy imports from `solution_08.py` (math, F, Tensor, Device, DType, Dim, DimLike)
   - Add imports for `Linear` and `Module` from `max.nn.module_v3`
   - Copy the `causal_mask` function from Step 08

2. **Create Projection Layers** (Lines 38-43):
   - Create `c_attn`: `Linear(self.embed_dim, 3 * self.embed_dim, bias=True)`
   - Create `c_proj`: `Linear(self.embed_dim, self.embed_dim, bias=True)`
   - Store `self.num_heads` and `self.head_dim` from config

3. **Implement _split_heads** (Lines 58-64):
   - Calculate new shape: `tensor.shape[:-1] + [num_heads, attn_head_size]`
   - Reshape to add head dimension: `tensor.reshape(new_shape)`
   - Transpose to move heads to position 1: `tensor.transpose(-3, -2)`
   - Returns shape `[batch, num_heads, seq_length, head_size]`

4. **Implement _merge_heads** (Lines 79-85):
   - Transpose heads back: `tensor.transpose(-3, -2)`
   - Calculate flattened shape: `tensor.shape[:-2] + [num_heads * attn_head_size]`
   - Reshape to remove head dimension: `tensor.reshape(new_shape)`
   - Returns shape `[batch, seq_length, n_embd]`

5. **Implement _attn** (Lines 100-103):
   - Copy the attention computation from Step 08's `compute_attention` function
   - Same 5-step process: compute scores, scale, mask, softmax, weighted sum
   - Works identically but operates on all heads in parallel

6. **Implement Forward Pass** (Lines 117-138):
   - Project input: `qkv = self.c_attn(hidden_states)`
   - Split Q/K/V: `F.split(qkv, [self.split_size, self.split_size, self.split_size], axis=-1)`
   - Split heads for each: `self._split_heads(query, self.num_heads, self.head_dim)` (and for key, value)
   - Compute attention: `self._attn(query, key, value)`
   - Merge heads: `self._merge_heads(attn_output, self.num_heads, self.head_dim)`
   - Output projection: `self.c_proj(attn_output)`

**Implementation**:
```python
import math

from max.driver import Device
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.graph import Dim, DimLike
from max.nn.module_v3 import Linear, Module

from solutions.solution_01 import GPT2Config


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


class GPT2MultiHeadAttention(Module):
    """Multi-head attention for GPT-2, matching HuggingFace structure."""

    def __init__(self, config: GPT2Config):
        super().__init__()

        self.embed_dim = config.n_embd
        self.num_heads = config.n_head
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim

        # Combined Q/K/V projection
        self.c_attn = Linear(self.embed_dim, 3 * self.embed_dim, bias=True)
        # Output projection
        self.c_proj = Linear(self.embed_dim, self.embed_dim, bias=True)

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """Split the last dimension into (num_heads, head_size)."""
        new_shape = tensor.shape[:-1] + [num_heads, attn_head_size]
        tensor = tensor.reshape(new_shape)
        return tensor.transpose(-3, -2)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """Merge attention heads back to original shape."""
        tensor = tensor.transpose(-3, -2)
        new_shape = tensor.shape[:-2] + [num_heads * attn_head_size]
        return tensor.reshape(new_shape)

    def _attn(self, query, key, value):
        """Compute attention for all heads in parallel."""
        attn_weights = query @ key.transpose(-1, -2)
        attn_weights = attn_weights / math.sqrt(int(value.shape[-1]))

        seq_len = query.shape[-2]
        mask = causal_mask(seq_len, 0, dtype=query.dtype, device=query.device)
        attn_weights = attn_weights + mask

        attn_weights = F.softmax(attn_weights)
        attn_output = attn_weights @ value

        return attn_output

    def __call__(self, hidden_states):
        """Apply multi-head attention."""
        # Project to Q, K, V
        qkv = self.c_attn(hidden_states)
        query, key, value = F.split(
            qkv, [self.split_size, self.split_size, self.split_size], axis=-1
        )

        # Split into multiple heads
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        # Apply attention
        attn_output = self._attn(query, key, value)

        # Merge heads back
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)

        # Output projection
        attn_output = self.c_proj(attn_output)

        return attn_output
```

### Validation
Run `pixi run s09`

A failed test will show:
```bash
Running tests for Step 09: Multi-head Attention...

Results:
❌ math is not imported
   Hint: Add 'import math'
❌ Linear is not imported from max.nn.module_v3
   Hint: Add 'from max.nn.module_v3 import Linear, Module'
❌ causal_mask function not found
   Hint: Copy causal_mask from solution_08.py
❌ GPT2MultiHeadAttention class not found in step_09 module
   Hint: Create class GPT2MultiHeadAttention(Module)
❌ self.c_attn linear layer is not created correctly
   Hint: Use Linear(self.embed_dim, 3 * self.embed_dim, bias=True)
❌ Found placeholder 'None' values that need to be replaced:
   self.c_attn = None
   self.c_proj = None
   return None
   Hint: Replace all 'None' values with the actual implementation

============================================================
⚠️ Some checks failed. Review the hints above and try again.
============================================================
```

A successful test will show:
```bash
Running tests for Step 09: Multi-head Attention...

Results:
✅ math is correctly imported
✅ Linear is correctly imported from max.nn.module_v3
✅ Module is correctly imported from max.nn.module_v3
✅ functional is correctly imported from max.experimental
✅ GPT2MultiHeadAttention class exists
✅ causal_mask function exists
✅ GPT2MultiHeadAttention inherits from Module
✅ self.c_attn linear layer is created correctly
✅ self.c_proj linear layer is created correctly
✅ _split_heads uses tensor.reshape
✅ _split_heads uses transpose(-3, -2)
✅ _merge_heads uses transpose and reshape
✅ _attn computes Q @ K^T
✅ _attn uses F.softmax
✅ Forward pass projects with c_attn
✅ Forward pass splits Q, K, V
✅ Forward pass calls _split_heads
✅ Forward pass calls _attn
✅ Forward pass calls _merge_heads
✅ Forward pass uses c_proj for output projection
✅ All placeholder 'None' values have been replaced
✅ GPT2MultiHeadAttention class can be instantiated
✅ GPT2MultiHeadAttention.c_attn is initialized
✅ GPT2MultiHeadAttention.c_proj is initialized
✅ GPT2MultiHeadAttention forward pass executes without errors
✅ Output shape is correct: (2, 8, 768)
✅ Output contains non-zero values
✅ _split_heads output shape is correct: (2, 12, 8, 64)
✅ _merge_heads output shape is correct: (2, 8, 768)

============================================================
🎉 All checks passed! Your implementation is complete.
============================================================
```

**Reference**: `solutions/solution_09.py`

---

**Next**: In [Step 10](./step_10.md), you'll implement residual connections and layer normalization, which stabilize training and enable stacking many transformer layers.
