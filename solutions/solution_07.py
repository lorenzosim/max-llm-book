"""
Solution for Step 07: Query/Key/Value Projections (Single Head)

This module implements the Q/K/V projection layers that transform input embeddings
into query, key, and value representations for attention computation.
"""

from max.nn.module_v3 import Linear, Module

from solutions.solution_01 import GPT2Config


class GPT2SingleHeadQKV(Module):
    """Single-head Q/K/V projections for GPT-2, matching HuggingFace structure.

    This is a simplified version that computes Q/K/V for a single attention head.
    In Step 09, we'll extend this to multi-head attention.
    """

    def __init__(self, config: GPT2Config):
        """Initialize Q/K/V projection layers.

        Args:
            config: GPT2Config containing n_embd
        """
        super().__init__()

        # Single combined projection for Q, K, V (HuggingFace style)
        # Projects from n_embd to 3 * n_embd (concatenated Q, K, V)
        self.c_attn = Linear(config.n_embd, 3 * config.n_embd, bias=True)

        # Store config for splitting
        self.n_embd = config.n_embd

    def __call__(self, x):
        """Project input to Q, K, V.

        Args:
            x: Input tensor, shape [batch, seq_length, n_embd]

        Returns:
            Tuple of (query, key, value), each with shape [batch, seq_length, n_embd]
        """
        # Single projection produces concatenated Q, K, V
        # Shape: [batch, seq_length, 3 * n_embd]
        qkv = self.c_attn(x)

        # Split into separate Q, K, V tensors
        # Each has shape: [batch, seq_length, n_embd]
        from max.experimental import functional as F

        query, key, value = F.split(
            qkv, [self.n_embd, self.n_embd, self.n_embd], axis=-1
        )

        return query, key, value
