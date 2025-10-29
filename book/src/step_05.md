# Step 05: Implement Multi-Head Attention

**Purpose**: Build the multi-head self-attention mechanism, the core component that enables GPT-2 to learn relationships between tokens in a sequence.

## What is Multi-Head Attention?

Multi-Head Attention is the fundamental mechanism that allows transformer models to process and relate different positions in a sequence. Introduced in the seminal paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762), it computes attention scores between all pairs of tokens, allowing each token to "attend to" and aggregate information from other tokens in the sequence.

The "multi-head" aspect means the model performs multiple parallel attention operations (heads), each potentially learning to focus on different types of relationships or features in the data. These parallel attention heads are then concatenated and projected to produce the final output.

The attention mechanism works by:

1. **Projecting** the input into three representations: Query (Q), Key (K), and Value (V)
2. **Computing** attention scores between queries and keys
3. **Splitting** into multiple heads to capture different relationship patterns
4. **Applying** causal masking to prevent attending to future tokens (for autoregressive generation)
5. **Merging** the heads back together and projecting to the output dimension

Mathematically, for each head:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

where $d_k$ is the dimension of the key vectors (head_dim).

## Why Use Multi-Head Attention?

**1. Capturing Different Relationships**: Multiple attention heads allow the model to simultaneously attend to different types of information at different positions. One head might focus on syntactic relationships while another captures semantic similarities or positional patterns.

**2. Parallel Information Processing**: Unlike sequential processing in RNNs, attention computes relationships between all token pairs in parallel, making it highly efficient on modern hardware and enabling long-range dependencies without the vanishing gradient problem.

**3. Dynamic Context Aggregation**: Attention weights are computed dynamically based on the input, allowing the model to adaptively determine which tokens are relevant for processing each position. This is more flexible than fixed convolutional windows.

**4. Interpretability**: Attention weights can be visualized to understand what relationships the model has learned, providing insights into model behavior that are difficult to obtain from fully-connected or convolutional layers.

**5. Foundation of Modern NLP**: Multi-head attention is the core innovation that enables transformers to achieve state-of-the-art performance across language tasks. Every major language model (GPT, BERT, T5, etc.) uses this mechanism.

### Key Concepts:

**Query, Key, Value (Q, K, V)**:
- The input is projected into three different representations using a single linear layer (`c_attn`)
- **Query**: "What am I looking for?" - used to compute attention scores
- **Key**: "What do I have?" - matched against queries to compute attention
- **Value**: "What information do I carry?" - weighted and aggregated based on attention scores
- In self-attention, all three come from the same input sequence

**Attention Score Computation**:
- Scores computed by matrix multiplication: `Q @ K^T`
- Higher scores indicate stronger relationships between tokens
- Scaled by `1/d_k` to prevent extremely large values that make softmax saturate
- Causal mask added to prevent attending to future positions
- Softmax converts scores to probabilities that sum to 1

**Multi-Head Mechanism**:
- Embedding dimension (768) split into multiple heads (12 in GPT-2)
- Each head has dimension `head_dim = embed_dim / num_heads = 64`
- Heads process independently in parallel
- After attention, heads are concatenated back to full embedding dimension
- Final projection layer mixes information across heads

**Causal Masking**:
- Prevents tokens from attending to future positions
- Essential for autoregressive generation (predicting next token)
- Implemented by adding `-inf` to future positions before softmax
- After softmax, `-inf` becomes 0 probability

**Shape Transformations**:
- Input: `[batch, seq_len, embed_dim]`
- After Q/K/V projection: `[batch, seq_len, 3 * embed_dim]`
- Split heads: `[batch, num_heads, seq_len, head_dim]`
- After attention: `[batch, num_heads, seq_len, head_dim]`
- Merge heads: `[batch, seq_len, embed_dim]`
- Final output: `[batch, seq_len, embed_dim]`

**MAX Operations**:
- [`Linear(in_features, out_features, bias=True)`](https://docs.modular.com/max/api/python/nn/module_v3#max.nn.module_v3.Linear): Linear transformation layer
- [`F.split(tensor, split_sizes, axis)`](https://docs.modular.com/max/api/python/experimental/functional#max.experimental.functional.split): Split tensor into multiple chunks
- [`tensor.reshape(new_shape)`](https://docs.modular.com/max/api/python/experimental/tensor#max.experimental.tensor.Tensor.reshape): Reshape tensor to new dimensions
- [`tensor.transpose(dim0, dim1)`](https://docs.modular.com/max/api/python/experimental/tensor#max.experimental.tensor.Tensor.transpose): Swap two dimensions
- [`F.softmax(tensor)`](https://docs.modular.com/max/api/python/experimental/functional#max.experimental.functional.softmax): Apply softmax activation
- [`math.sqrt(x)`](https://docs.python.org/3/library/math.html#math.sqrt): Compute square root for scaling

**Layer Naming Convention**:
- `c_attn`: Combined Q/K/V projection (3x embedding dimension output)
- `c_proj`: Output projection after merging heads
- Names match HuggingFace GPT-2 for weight loading compatibility

### Implementation Tasks (`step_05.py`):

1. **Import Required Modules** (Lines 1-10):
   - `math` for `math.sqrt()` in attention scaling (already imported)
   - `functional as F` from `max.experimental` - provides F.split(), F.softmax()
   - `Tensor` from `max.experimental.tensor` - tensor operations
   - `Linear` and `Module` from `max.nn.module_v3` - linear layers and base class

2. **Create Attention Projection Layers** (Lines 24-33):
   - Create `self.c_attn = Linear(self.embed_dim, 3 * self.embed_dim, bias=True)`
   - This projects input to Q, K, V simultaneously (3x the dimension)
   - Create `self.c_proj = Linear(self.embed_dim, self.embed_dim, bias=True)`
   - This projects concatenated attention output back to embedding dimension

3. **Implement _split_heads Method** (Lines 48-61):
   - Calculate `new_shape = tensor.shape[:-1] + [num_heads, attn_head_size]`
   - This adds head dimensions: `[batch, seq, embed] � [batch, seq, heads, head_dim]`
   - Reshape tensor: `tensor = tensor.reshape(new_shape)`
   - Transpose to move heads before sequence: `return tensor.transpose(-3, -2)`
   - Final shape: `[batch, num_heads, seq_len, head_dim]`

4. **Implement _merge_heads Method** (Lines 76-89):
   - Transpose back: `tensor = tensor.transpose(-3, -2)`
   - This reverses split_heads transpose: `[batch, heads, seq, head_dim] � [batch, seq, heads, head_dim]`
   - Calculate `new_shape = tensor.shape[:-2] + [num_heads * attn_head_size]`
   - Reshape to merge heads: `return tensor.reshape(new_shape)`
   - Final shape: `[batch, seq_len, embed_dim]`

5. **Implement _attn Method - Compute Attention Scores** (Lines 102-111):
   - Compute attention: `attn_weights = query @ key.transpose(-1, -2)`
   - This computes similarity between all query-key pairs
   - Scale weights: `attn_weights = attn_weights / math.sqrt(int(value.shape[-1]))`
   - Scaling prevents dot products from growing too large

6. **Implement _attn Method - Apply Causal Mask** (Lines 113-125):
   - Extract sequence length: `seq_len = query.shape[-2]`
   - Create mask: `mask = causal_mask(seq_len, 0, dtype=query.dtype, device=query.device)`
   - Add mask to weights: `attn_weights = attn_weights + mask`
   - Mask contains `-inf` for future positions

7. **Implement _attn Method - Compute Output** (Lines 127-137):
   - Apply softmax: `attn_weights = F.softmax(attn_weights)`
   - Converts scores to probabilities summing to 1
   - Compute weighted sum: `attn_output = attn_weights @ value`
   - Return `attn_output`

8. **Implement __call__ Method - Project and Split** (Lines 149-167):
   - Project to Q/K/V: `query, key, value = F.split(self.c_attn(hidden_states), [self.split_size, self.split_size, self.split_size], axis=2)`
   - Split heads for query: `query = self._split_heads(query, self.num_heads, self.head_dim)`
   - Split heads for key: `key = self._split_heads(key, self.num_heads, self.head_dim)`
   - Split heads for value: `value = self._split_heads(value, self.num_heads, self.head_dim)`

9. **Implement __call__ Method - Compute Attention and Project** (Lines 169-181):
   - Compute attention: `attn_output = self._attn(query, key, value)`
   - Merge heads: `attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)`
   - Final projection: `return self.c_proj(attn_output)`

**Implementation**:
```python
# Import required modules
import math
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.nn.module_v3 import Linear, Module

from solutions.solution_01 import GPT2Config
from solutions.solution_02 import causal_mask

class GPT2Attention(Module):
    """Multi-head self-attention matching HuggingFace GPT-2 structure."""

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.embed_dim = config.n_embd
        self.num_heads = config.n_head
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim

        # Create projection layers
        self.c_attn = Linear(self.embed_dim, 3 * self.embed_dim, bias=True)
        self.c_proj = Linear(self.embed_dim, self.embed_dim, bias=True)

    def _split_heads(
        self, tensor: Tensor, num_heads: int, attn_head_size: int
    ) -> Tensor:
        """Split the last dimension into (num_heads, head_size)."""
        new_shape = tensor.shape[:-1] + [num_heads, attn_head_size]
        tensor = tensor.reshape(new_shape)
        return tensor.transpose(-3, -2)

    def _merge_heads(
        self, tensor: Tensor, num_heads: int, attn_head_size: int
    ) -> Tensor:
        """Merge attention heads back."""
        tensor = tensor.transpose(-3, -2)
        new_shape = tensor.shape[:-2] + [num_heads * attn_head_size]
        return tensor.reshape(new_shape)

    def _attn(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        """Compute attention scores and apply to values."""
        attn_weights = query @ key.transpose(-1, -2)

        # Scale attention weights
        attn_weights = attn_weights / math.sqrt(int(value.shape[-1]))

        # Apply causal mask
        seq_len = query.shape[-2]
        mask = causal_mask(seq_len, 0, dtype=query.dtype, device=query.device)
        attn_weights = attn_weights + mask

        attn_weights = F.softmax(attn_weights)
        attn_output = attn_weights @ value

        return attn_output

    def __call__(self, hidden_states: Tensor) -> Tensor:
        """Apply multi-head self-attention."""
        query, key, value = F.split(
            self.c_attn(hidden_states),
            [self.split_size, self.split_size, self.split_size],
            axis=2,
        )

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        attn_output = self._attn(query, key, value)
        attn_output = self._merge_heads(
            attn_output, self.num_heads, self.head_dim
        )
        attn_output = self.c_proj(attn_output)

        return attn_output
```

### Validation:
Run `pixi run s05`

A failed test will show:
```bash
Running tests for Step 05: Implement Multi-Head Attention...

Results:
❌ functional module is not imported from max.experimental
   Hint: Add 'from max.experimental import functional as F'
❌ Tensor is not imported from max.experimental.tensor
   Hint: Add 'from max.experimental.tensor import Tensor'
❌ Linear and Module are not imported from max.nn.module_v3
   Hint: Add 'from max.nn.module_v3 import Linear, Module'
❌ self.c_attn Linear layer is not created correctly
   Hint: Use Linear(self.embed_dim, 3 * self.embed_dim, bias=True)
❌ self.c_proj Linear layer is not created correctly
   Hint: Use Linear(self.embed_dim, self.embed_dim, bias=True)
❌ _split_heads: new_shape calculation is not correct
   Hint: new_shape = tensor.shape[:-1] + [num_heads, attn_head_size]
❌ _split_heads: tensor.reshape is not used
   Hint: Use tensor.reshape(new_shape)
❌ _split_heads: tensor.transpose(-3, -2) is not used
   Hint: Use tensor.transpose(-3, -2) to move heads before sequence length
❌ _merge_heads: new_shape calculation is not correct
   Hint: new_shape = tensor.shape[:-2] + [num_heads * attn_head_size]
❌ _attn: attention scores not computed correctly
   Hint: Use query @ key.transpose(-1, -2)
❌ _attn: attention weights are not scaled correctly
   Hint: Scale by math.sqrt(int(value.shape[-1]))
❌ _attn: sequence length not extracted correctly
   Hint: seq_len = query.shape[-2]
❌ _attn: causal_mask function is not called
   Hint: Use causal_mask(seq_len, 0, dtype=query.dtype, device=query.device)
❌ _attn: F.softmax is not applied to attention weights
   Hint: Use F.softmax(attn_weights)
❌ _attn: attention output not computed correctly
   Hint: Use attn_weights @ value
❌ __call__: F.split is not used correctly
   Hint: Use F.split(self.c_attn(hidden_states), ...)
❌ __call__: F.split does not use correct split sizes
   Hint: Split into [self.split_size, self.split_size, self.split_size]
❌ __call__: self._split_heads is not called for all query, key, and value
   Hint: Call self._split_heads for query, key, and value
❌ __call__: self._attn is not called
   Hint: Call self._attn(query, key, value)
❌ __call__: self._merge_heads is not called
   Hint: Call self._merge_heads(attn_output, self.num_heads, self.head_dim)
❌ __call__: self.c_proj is not called
   Hint: Call self.c_proj(attn_output)
❌ Found placeholder 'None' values that need to be replaced:
   self.c_attn = None
   self.c_proj = None
   new_shape = None
   tensor = None
   ... and 8 more
   Hint: Replace all 'None' values with the actual implementation

============================================================
⚠️ Some checks failed. Review the hints above and try again.
============================================================
```

A successful test will show:
```bash
Running tests for Step 05: Implement Multi-Head Attention...

Results:
✅ functional module is correctly imported as F from max.experimental
✅ Tensor is correctly imported from max.experimental.tensor
✅ Linear and Module are imported from max.nn.module_v3
✅ GPT2Attention class exists
✅ self.c_attn Linear layer is created correctly
✅ self.c_proj Linear layer is created correctly
✅ _split_heads: new_shape calculation is correct
✅ _split_heads: tensor.reshape is used
✅ _split_heads: tensor.transpose(-3, -2) is used
✅ _merge_heads: new_shape calculation is correct
✅ _attn: attention scores computed with query @ key.transpose(-1, -2)
✅ _attn: attention weights are scaled
✅ _attn: sequence length extracted from query.shape[-2]
✅ _attn: causal_mask function is called
✅ _attn: F.softmax is applied to attention weights
✅ _attn: attention output computed with attn_weights @ value
✅ __call__: F.split is used on self.c_attn(hidden_states)
✅ __call__: F.split uses correct split sizes
✅ __call__: self._split_heads is called for query, key, and value
✅ __call__: self._attn is called
✅ __call__: self._merge_heads is called
✅ __call__: self.c_proj is called
✅ All placeholder 'None' values have been replaced
✅ GPT2Attention class can be instantiated
✅ GPT2Attention.c_attn is initialized
✅ GPT2Attention.c_proj is initialized
✅ GPT2Attention.embed_dim is correct: 768
✅ GPT2Attention.num_heads is correct: 12
✅ GPT2Attention.head_dim is correct: 64
✅ GPT2Attention forward pass executes without errors
✅ Output shape is correct: (1, 4, 768)

============================================================
🎉 All checks passed! Your implementation matches the solution.
============================================================
```

**Reference**: `solutions/solution_05.py`
