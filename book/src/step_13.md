# Step 13: Language model head

<div class="note">
    Learn to add the final linear projection layer that converts hidden states to vocabulary logits for next-token prediction.
</div>

## What is the language model head?

In this section you will add the language model head to create the complete `GPT2LMHeadModel`. This final linear layer projects transformer outputs (768-dimensional) to vocabulary logits (50,257-dimensional).

For each position in the sequence, the model outputs a score for every possible next token. These logits are converted to probabilities using softmax, enabling next-token prediction.

The LM head is a linear layer without bias (`bias=False`), matching Hugging Face's implementation. At 768 × 50,257 = 38.6M parameters, it represents about 33% of GPT-2's 117M total parameters.

## Why is this the final step?

**1. Connects Hidden Space to Vocabulary**: The transformer operates in a continuous 768-dimensional space, processing semantic relationships through attention and transformations. But to generate text, we need discrete tokens from a vocabulary. The LM head bridges this gap—it projects continuous representations back to discrete vocabulary predictions. Without it, the transformer would output meaningless vectors.

**2. Enables Autoregressive Generation**: With the LM head, the model can generate text one token at a time. Generate logits for the next token, sample from the distribution, append to the sequence, and repeat. This autoregressive property—using the model's own outputs as inputs for subsequent predictions—is what makes language models generative rather than just discriminative.

**3. Shared Training Objective**: During training, the model learns to minimize cross-entropy loss between predicted logits and actual next tokens. The LM head provides the predicted logits, and the loss signal backpropagates through it to train the entire model. Every layer, from embeddings through all transformer blocks, learns to produce hidden states that the LM head can convert into accurate next-token predictions.

**4. No Additional Complexity**: The LM head is just matrix multiplication—the simplest possible design. This simplicity is intentional. All the model's complexity lives in the transformer blocks (attention, MLPs, etc.). The head should just expose what the model knows, not add new computation.

### Key concepts

**Linear Projection**:
- Maps `[batch, seq_length, n_embd]` → `[batch, seq_length, vocab_size]`
- Uses [`Linear(n_embd, vocab_size, bias=False)`](https://docs.modular.com/max/api/python/nn/module_v3#max.nn.module_v3.Linear)
- Each position gets independent logits over vocabulary
- No activation function—raw logits are used

**Logits vs Probabilities**:
- **Logits**: Raw scores before softmax, can be any real number
- **Probabilities**: After softmax, sum to 1, range [0, 1]
- For generation, we often work with logits directly (temperature, top-k, etc.)
- Softmax applied during generation, not in the model forward pass

**Vocabulary Projection**:
- Output dimension = `vocab_size = 50,257`
- Each position outputs 50,257 scores (one per vocabulary token)
- Higher score = model thinks that token is more likely next
- Shape: `[batch, seq_length, vocab_size]`

**No Bias Term**:
- `bias=False` means only weights, no bias vector
- Reduces parameters: saves 50,257 values
- Common in language models (GPT-2, GPT-3, etc.)
- Bias provides little benefit for vocabulary prediction

**Complete Model Architecture**:
1. Input: Token IDs `[batch, seq_length]`
2. Embeddings: Token + position `[batch, seq_length, 768]`
3. Transformer blocks (×12): `[batch, seq_length, 768]` → `[batch, seq_length, 768]`
4. Final layer norm: `[batch, seq_length, 768]`
5. LM head: `[batch, seq_length, 768]` → `[batch, seq_length, 50257]`
6. Output: Logits `[batch, seq_length, 50257]`

### Implementation tasks (`step_13.py`)

1. **Import Required Modules** (Lines 13-16):
   - Import `Linear` and `Module` from `max.nn.module_v3`
   - Import `GPT2Config` from `solutions.solution_01`
   - Import `GPT2Model` from `solutions.solution_12`

2. **Create Transformer** (Lines 32-33):
   - Initialize `self.transformer = GPT2Model(config)`
   - This is the complete transformer from Step 12

3. **Create LM Head** (Lines 36-38):
   - Initialize `self.lm_head = Linear(config.n_embd, config.vocab_size, bias=False)`
   - Projects hidden states to vocabulary logits
   - Note `bias=False`

4. **Implement Forward Pass** (Lines 51-59):
   - Get hidden states: `hidden_states = self.transformer(input_ids)`
   - Project to logits: `logits = self.lm_head(hidden_states)`
   - Return `logits`

**Implementation**:
```python
from max.nn.module_v3 import Linear, Module

from solutions.solution_01 import GPT2Config
from solutions.solution_12 import GPT2Model


class MaxGPT2LMHeadModel(Module):
    """Complete GPT-2 model with language modeling head."""

    def __init__(self, config: GPT2Config):
        super().__init__()

        self.config = config
        self.transformer = GPT2Model(config)
        self.lm_head = Linear(config.n_embd, config.vocab_size, bias=False)

    def __call__(self, input_ids):
        hidden_states = self.transformer(input_ids)
        logits = self.lm_head(hidden_states)
        return logits
```

### Validation
Run `pixi run s13`

**Reference**: `solutions/solution_13.py`

---

**Next**: In [Step 14](./step_14.md), you'll implement text generation—sampling from the model's predictions to generate coherent text autoregressively.
