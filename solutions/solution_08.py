"""
Solution for Step 08: Attention Mechanism with Causal Masking

This module implements the core attention mechanism that computes relevance-based
weighted combinations of values using scaled dot-product attention with causal masking.
"""

import math

from max.driver import Device
from max.dtype import DType
from max.experimental import functional as F
from max.graph import Dim, DimLike
from max.experimental.tensor import Tensor


@F.functional
def causal_mask(
    sequence_length: DimLike,
    num_tokens: DimLike,
    *,
    dtype: DType,
    device: Device,
):
    """Create a causal attention mask.

    This mask prevents tokens from attending to future positions.
    The mask contains 0 for allowed positions and -inf for masked positions.

    Args:
        sequence_length: Length of the sequence
        num_tokens: Number of new tokens (usually 0 for full sequence)
        dtype: Data type of the mask
        device: Device to create the mask on

    Returns:
        Causal mask tensor of shape [sequence_length, sequence_length + num_tokens]
    """
    n = Dim(sequence_length) + num_tokens
    mask = Tensor.constant(float("-inf"), dtype=dtype, device=device)
    mask = F.broadcast_to(mask, shape=(sequence_length, n))
    # band_part with exclude=True gives upper triangle (excluding diagonal) as -inf
    return F.band_part(mask, num_lower=None, num_upper=0, exclude=True)


def compute_attention(query, key, value):
    """Compute scaled dot-product attention with causal masking.

    This implements the core attention mechanism:
    1. Compute attention scores (Q @ K^T)
    2. Scale by sqrt(d_k)
    3. Apply causal mask (prevent attending to future)
    4. Softmax to get attention probabilities
    5. Weighted sum of values

    Args:
        query: Query tensor, shape [..., seq_length, d_k]
        key: Key tensor, shape [..., seq_length, d_k]
        value: Value tensor, shape [..., seq_length, d_v]

    Returns:
        Attention output, shape [..., seq_length, d_v]
    """
    # Step 1: Compute attention scores
    # Shape: [..., seq_length, seq_length]
    attn_weights = query @ key.transpose(-1, -2)

    # Step 2: Scale by sqrt(d_k) to prevent softmax saturation
    # d_k is the last dimension of the key (or value)
    scale_factor = math.sqrt(int(value.shape[-1]))
    attn_weights = attn_weights / scale_factor

    # Step 3: Apply causal mask to prevent attending to future positions
    seq_len = query.shape[-2]
    mask = causal_mask(seq_len, 0, dtype=query.dtype, device=query.device)
    attn_weights = attn_weights + mask  # -inf for future positions

    # Step 4: Softmax converts scores to probabilities
    attn_weights = F.softmax(attn_weights)

    # Step 5: Weighted sum of values using attention probabilities
    attn_output = attn_weights @ value

    return attn_output
