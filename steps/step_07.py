"""
Step 07: Query/Key/Value Projections (Single Head)

Implement Q/K/V projection layers that transform embeddings for attention.

Tasks:
1. Import Linear and Module from max.nn.module_v3
2. Import F (functional) from max.experimental
3. Create c_attn linear layer projecting to 3 * n_embd
4. Implement forward pass that projects and splits into Q, K, V

Run: pixi run s07
"""

# TODO: Import required modules from MAX
# Hint: You'll need Linear and Module from max.nn.module_v3
# Hint: You'll need functional as F from max.experimental

from solutions.solution_01 import GPT2Config


class GPT2SingleHeadQKV(Module):
    """Single-head Q/K/V projections for GPT-2."""

    def __init__(self, config: GPT2Config):
        super().__init__()

        # TODO: Create combined Q/K/V projection layer
        # Hint: Use Linear(config.n_embd, 3 * config.n_embd, bias=True)
        # This projects from embedding dimension to 3x that size (for Q, K, V)
        self.c_attn = None  # Line 27-30

        # Store config for splitting
        self.n_embd = config.n_embd

    def __call__(self, x):
        """Project input to Q, K, V.

        Args:
            x: Input tensor, shape [batch, seq_length, n_embd]

        Returns:
            Tuple of (query, key, value), each with shape [batch, seq_length, n_embd]
        """
        # TODO: Project input to concatenated Q/K/V
        # Hint: Call self.c_attn(x) to get shape [batch, seq_length, 3 * n_embd]
        qkv = None  # Line 45-46

        # TODO: Split into separate Q, K, V tensors
        # Hint: Use F.split(qkv, [self.n_embd, self.n_embd, self.n_embd], axis=-1)
        # This splits the last dimension into three equal parts
        query, key, value = None, None, None  # Line 49-51

        return query, key, value
