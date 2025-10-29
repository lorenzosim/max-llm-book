# Step 12: Stacking transformer blocks

<div class="note">
    Learn to stack 12 transformer blocks with embeddings and final normalization to create the complete GPT-2 model.
</div>

## What is model stacking?

In this section you will create the `GPT2Model` class that stacks 12 transformer blocks with embeddings to form the complete GPT-2 architecture.

The flow:
1. Token IDs → token embeddings (lookup table)
2. Add position embeddings
3. Pass through 12 transformer blocks in sequence
4. Apply final layer normalization
5. Output: contextualized representations for each token

Each block processes the output of the previous block, progressively refining the representations from simple embeddings to rich contextual understanding.

This stacking creates a hierarchy of representations. Early blocks might learn simple patterns like word associations, middle blocks capture syntactic structure, and later blocks encode high-level semantic and pragmatic information. The depth (12 layers) is crucial—shallow networks cannot capture the complexity of natural language.

## Why stack 12 blocks?

**1. Hierarchical Representation Learning**: Each transformer block learns features at a different level of abstraction. Research analyzing GPT models shows that early layers capture surface-level patterns (e.g., "the" is often followed by a noun), middle layers learn syntactic structures (e.g., subject-verb agreement across long distances), and deep layers encode semantic relationships and world knowledge. This hierarchy emerges from training, not explicit design—the model learns to use depth for abstraction.

**2. Increased Model Capacity**: More layers mean more parameters to learn. GPT-2 base with 12 layers has ~117M parameters. Doubling layers roughly doubles parameters, allowing the model to memorize more patterns and generalize better. However, depth provides capacity more efficiently than width—a 12-layer model with 768 dimensions outperforms a 6-layer model with 1536 dimensions despite similar parameter counts, because depth enables hierarchical feature learning.

**3. Long-Range Dependencies**: Each attention layer has a finite receptive field determined by sequence length, but stacking layers increases the effective receptive field. Information can propagate across the entire sequence through multiple layers. A token at position 0 can influence position 100 through a chain of attention connections across 12 layers. This multi-hop reasoning is essential for understanding long documents.

**4. Residual Gradient Flow**: With residual connections in each block, gradients can flow through many paths—some skipping blocks, others passing through them. This creates an implicit ensemble of networks of varying depths. During training, the model can learn to use different depth paths for different patterns, improving robustness and trainability.

### Key concepts

**Sequential Composition**:
- [`Sequential(*modules)`](https://docs.modular.com/max/api/python/nn/module_v3#max.nn.module_v3.Sequential) chains modules
- Applies each module in order: `output = module_n(...module_2(module_1(input)))`
- For GPT-2: `Sequential(block1, block2, ..., block12)`
- Convenient for stacking identical structures

**Embedding Combination**:
- Token embeddings: `wte(input_ids)` maps each token ID to 768-dim vector
- Position embeddings: `wpe(positions)` maps each position to 768-dim vector
- Combined: `tok_embeds + pos_embeds` (element-wise addition)
- Both contribute to initial representation

**Position Encoding**:
- Create positions: `Tensor.arange(seq_length, dtype=input_ids.dtype, device=input_ids.device)`
- Must match input dtype and device for compatibility
- Positions are [0, 1, 2, ..., seq_length-1]
- Same positions used for all examples in batch (broadcast)

**Final Layer Normalization**:
- Applied after all transformer blocks
- Stabilizes the output distribution
- Called `ln_f` (layer norm final) in HuggingFace
- Essential for consistent output scale

**Model Hyperparameters**:
- `config.n_layer = 12`: Number of transformer blocks
- `config.n_embd = 768`: Embedding/hidden dimension
- `config.vocab_size = 50257`: Vocabulary size
- `config.n_positions = 1024`: Maximum sequence length

### Implementation tasks (`step_12.py`)

1. **Import Required Modules** (Lines 13-18):
   - Import `Tensor` from `max.experimental.tensor`
   - Import `Embedding, Module, Sequential` from `max.nn.module_v3`
   - Import `GPT2Config` from `solutions.solution_01`
   - Import `LayerNorm` from `solutions.solution_10`
   - Import `GPT2Block` from `solutions.solution_11`

2. **Create Embeddings** (Lines 34-39):
   - Token embeddings: `Embedding(config.vocab_size, dim=config.n_embd)`
   - Position embeddings: `Embedding(config.n_positions, dim=config.n_embd)`
   - Store as `self.wte` and `self.wpe`

3. **Stack Transformer Blocks** (Lines 42-44):
   - Use `Sequential(*(GPT2Block(config) for _ in range(config.n_layer)))`
   - Generator creates 12 identical blocks
   - Sequential chains them together

4. **Create Final Layer Norm** (Lines 47-48):
   - `LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)`
   - Store as `self.ln_f`

5. **Implement Forward Pass** (Lines 61-87):
   - Extract shape: `batch_size, seq_length = input_ids.shape`
   - Token embeddings: `tok_embeds = self.wte(input_ids)`
   - Position indices: `Tensor.arange(seq_length, dtype=input_ids.dtype, device=input_ids.device)`
   - Position embeddings: `pos_embeds = self.wpe(position_indices)`
   - Combine: `x = tok_embeds + pos_embeds`
   - Apply blocks: `x = self.h(x)`
   - Final norm: `x = self.ln_f(x)`
   - Return `x`

**Implementation**:
```python
from max.experimental.tensor import Tensor
from max.nn.module_v3 import Embedding, Module, Sequential

from solutions.solution_01 import GPT2Config
from solutions.solution_10 import LayerNorm
from solutions.solution_11 import GPT2Block


class GPT2Model(Module):
    """Complete GPT-2 transformer model matching HuggingFace structure."""

    def __init__(self, config: GPT2Config):
        super().__init__()

        self.wte = Embedding(config.vocab_size, dim=config.n_embd)
        self.wpe = Embedding(config.n_positions, dim=config.n_embd)
        self.h = Sequential(*(GPT2Block(config) for _ in range(config.n_layer)))
        self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def __call__(self, input_ids):
        batch_size, seq_length = input_ids.shape

        tok_embeds = self.wte(input_ids)
        pos_embeds = self.wpe(
            Tensor.arange(seq_length, dtype=input_ids.dtype, device=input_ids.device)
        )

        x = tok_embeds + pos_embeds
        x = self.h(x)
        x = self.ln_f(x)

        return x
```

### Validation
Run `pixi run s12`

**Reference**: `solutions/solution_12.py`

---

**Next**: In [Step 13](./step_13.md), you'll add the language modeling head that projects hidden states back to vocabulary logits for text generation.
