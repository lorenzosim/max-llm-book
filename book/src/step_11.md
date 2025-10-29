# Step 11: Transformer block

<div class="note">
    Learn to combine attention, MLP, layer normalization, and residual connections into a complete transformer block.
</div>

## What is a transformer block?

In this section you will build the `Block` class that combines all previous components into a complete transformer block. This is the fundamental repeating unit that makes up GPT-2.

Each block has two sub-layers:
1. Multi-head attention (with layer norm and residual)
2. Feed-forward MLP (with layer norm and residual)

The pattern for each sub-layer:
```
x = x + sublayer(layer_norm(x))
```

GPT-2 stacks 12 identical transformer blocks (for the base model). Each block independently processes the sequence, gradually refining the representations. Early blocks might capture low-level patterns like word associations, while later blocks capture more abstract relationships and long-range dependencies.

## Why this architecture?

**1. Pre-norm Stability**: The pre-norm pattern (normalize before sublayer) is more stable than post-norm (normalize after sublayer + residual). In pre-norm, layer normalization sees the full range of activations before they're processed, preventing extreme values from destabilizing training. This is crucial for training deep networks with 12+ layers. Research shows pre-norm allows training deeper transformers without careful initialization or learning rate warmup.

**2. Attention + MLP Complementarity**: Attention and MLP serve different purposes. Attention performs context-dependent mixing—it routes information between positions based on learned relevance patterns. The MLP processes each position independently with the same weights, performing context-independent transformations. Together, they provide both positional mixing (attention) and per-position computation (MLP). This combination is essential for the model's expressiveness.

**3. Residual Paths Preserve Information**: With two residual connections per block and 12 blocks, there are 24 addition operations in the forward pass. This creates exponentially many paths through the network (2^24 possible combinations of which sublayers to traverse). During training, gradients can flow directly through these skip connections, enabling efficient learning in very deep networks. The residual paths ensure that information from early layers (like positional embeddings) can directly influence later layers.

**4. Modular Composition**: Each transformer block is identical in structure, differing only in learned parameters. This modularity simplifies implementation, enables easy scaling (just change the number of blocks), and allows interesting properties like progressive training (training shallow networks then adding blocks) or block-wise analysis (measuring each block's contribution).

### Key concepts

**Pre-norm Architecture**:
- Pattern: `output = input + sublayer(layer_norm(input))`
- Normalization happens first, before attention or MLP
- More stable than post-norm for deep networks
- Used in GPT-2, GPT-3, and most modern transformers

**Two Sub-layers**:
- **Attention sub-layer**: Multi-head self-attention from Step 09
- **MLP sub-layer**: Position-wise feed-forward from Step 04
- Each has its own layer norm and residual connection
- Process sequentially: attention first, then MLP

**Layer Dimensions**:
- `hidden_size = n_embd = 768` (embedding dimension)
- `inner_dim = 4 * hidden_size = 3072` (MLP inner dimension)
- `layer_norm_epsilon = 1e-5` (numerical stability)
- All sublayers maintain the 768-dimensional representation

**Information Flow**:
- Input: `[batch, seq_length, n_embd]`
- After attention block: same shape (residual preserves dimensions)
- After MLP block: same shape (residual preserves dimensions)
- Output: `[batch, seq_length, n_embd]`
- Dimension preservation allows stacking arbitrary numbers of blocks

**HuggingFace Naming**:
- `ln_1`: First layer norm (before attention)
- `attn`: Multi-head attention
- `ln_2`: Second layer norm (before MLP)
- `mlp`: Feed-forward network
- Matches original GPT-2 implementation for weight loading

### Implementation tasks (`step_11.py`)

1. **Import Required Modules** (Lines 13-18):
   - Import `Module` from `max.nn.module_v3`
   - Import `GPT2Config` from `solutions.solution_01`
   - Import `GPT2MLP` from `solutions.solution_04`
   - Import `GPT2MultiHeadAttention` from `solutions.solution_09`
   - Import `LayerNorm` from `solutions.solution_10`

2. **Create Sub-layers** (Lines 40-53):
   - Create `ln_1`: `LayerNorm(hidden_size, eps=config.layer_norm_epsilon)`
   - Create `attn`: `GPT2MultiHeadAttention(config)`
   - Create `ln_2`: `LayerNorm(hidden_size, eps=config.layer_norm_epsilon)`
   - Create `mlp`: `GPT2MLP(inner_dim, config)`

3. **Implement Attention Block** (Lines 67-71):
   - Store residual: `residual = hidden_states`
   - Normalize: `hidden_states = self.ln_1(hidden_states)`
   - Apply attention: `attn_output = self.attn(hidden_states)`
   - Add residual: `hidden_states = attn_output + residual`

4. **Implement MLP Block** (Lines 74-78):
   - Store residual: `residual = hidden_states`
   - Normalize: `hidden_states = self.ln_2(hidden_states)`
   - Apply MLP: `feed_forward_hidden_states = self.mlp(hidden_states)`
   - Add residual: `hidden_states = residual + feed_forward_hidden_states`

5. **Return Output** (Line 81):
   - Return the final `hidden_states`

**Implementation**:
```python
from max.nn.module_v3 import Module

from solutions.solution_01 import GPT2Config
from solutions.solution_04 import GPT2MLP
from solutions.solution_09 import GPT2MultiHeadAttention
from solutions.solution_10 import LayerNorm


class GPT2Block(Module):
    """Complete GPT-2 transformer block matching HuggingFace structure."""

    def __init__(self, config: GPT2Config):
        super().__init__()

        hidden_size = config.n_embd
        inner_dim = (
            config.n_inner
            if hasattr(config, "n_inner") and config.n_inner is not None
            else 4 * hidden_size
        )

        self.ln_1 = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2MultiHeadAttention(config)
        self.ln_2 = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(inner_dim, config)

    def __call__(self, hidden_states):
        # Attention block with residual connection
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output = self.attn(hidden_states)
        hidden_states = attn_output + residual

        # MLP block with residual connection
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = residual + feed_forward_hidden_states

        return hidden_states
```

### Validation
Run `pixi run s11`

A failed test will show:
```bash
Running tests for Step 11: Transformer Block...

Results:
❌ Module is not imported from max.nn.module_v3
   Hint: Add 'from max.nn.module_v3 import Module'
❌ GPT2Config is not imported
   Hint: Add 'from solutions.solution_01 import GPT2Config'
❌ GPT2MLP is not imported
   Hint: Add 'from solutions.solution_04 import GPT2MLP'
❌ GPT2MultiHeadAttention is not imported
   Hint: Add 'from solutions.solution_09 import GPT2MultiHeadAttention'
❌ LayerNorm is not imported
   Hint: Add 'from solutions.solution_10 import LayerNorm'
❌ Found placeholder 'None' values that need to be replaced

============================================================
⚠️ Some checks failed. Review the hints above and try again.
============================================================
```

A successful test will show:
```bash
Running tests for Step 11: Transformer Block...

Results:
✅ Module is correctly imported from max.nn.module_v3
✅ GPT2Config is correctly imported from solutions.solution_01
✅ GPT2MLP is correctly imported from solutions.solution_04
✅ GPT2MultiHeadAttention is correctly imported from solutions.solution_09
✅ LayerNorm is correctly imported from solutions.solution_10
✅ GPT2Block class exists
✅ GPT2Block inherits from Module
✅ self.ln_1 is created correctly
✅ self.attn is created correctly
✅ self.ln_2 is created correctly
✅ self.mlp is created correctly
✅ Forward pass calls ln_1
✅ Forward pass calls attn
✅ Forward pass calls ln_2
✅ Forward pass calls mlp
✅ Forward pass uses residual connections (at least 2)
✅ All placeholder 'None' values have been replaced
✅ GPT2Block class can be instantiated
✅ GPT2Block.ln_1 is initialized
✅ GPT2Block.attn is initialized
✅ GPT2Block.ln_2 is initialized
✅ GPT2Block.mlp is initialized
✅ GPT2Block forward pass executes without errors
✅ Output shape is correct: (2, 8, 768)
✅ Output contains non-zero values

============================================================
🎉 All checks passed! Your implementation is complete.
============================================================
```

**Reference**: `solutions/solution_11.py`

---

**Next**: In [Step 12](./step_12.md), you'll stack 12 transformer blocks together to create the complete GPT-2 model architecture.
