"""
Step 08: Attention Mechanism with Causal Masking

Implement the core attention mechanism with scaled dot-product attention.

Tasks:
1. Import math (for sqrt), F (functional), Tensor, Device, DType, Dim, DimLike
2. Implement causal_mask function using F.band_part
3. Compute attention scores (Q @ K^T)
4. Scale scores by sqrt(d_k)
5. Apply causal mask and softmax
6. Compute weighted sum of values

Run: pixi run s08
"""

# TODO: Import required modules
# Hint: You'll need math for sqrt
# Hint: You'll need F (functional) from max.experimental
# Hint: You'll need Tensor from max.experimental.tensor
# Hint: You'll need Device, DType from max.driver and max.dtype
# Hint: You'll need Dim, DimLike from max.graph


# TODO: Implement causal_mask function
# Hint: Use @F.functional decorator
# Hint: Parameters: sequence_length, num_tokens, *, dtype, device
# Hint: Create -inf constant, broadcast, and use F.band_part
@F.functional
def causal_mask(
    sequence_length: DimLike,
    num_tokens: DimLike,
    *,
    dtype: DType,
    device: Device,
):
    """Create a causal attention mask.

    Args:
        sequence_length: Length of the sequence
        num_tokens: Number of new tokens (usually 0)
        dtype: Data type of the mask
        device: Device to create the mask on

    Returns:
        Causal mask with -inf for future positions
    """
    # TODO: Create the mask
    # Hint: n = Dim(sequence_length) + num_tokens
    # Hint: mask = Tensor.constant(float("-inf"), dtype=dtype, device=device)
    # Hint: mask = F.broadcast_to(mask, shape=(sequence_length, n))
    # Hint: return F.band_part(mask, num_lower=None, num_upper=0, exclude=True)
    return None  # Line 46-51


def compute_attention(query, key, value):
    """Compute scaled dot-product attention with causal masking.

    Args:
        query: Query tensor, shape [..., seq_length, d_k]
        key: Key tensor, shape [..., seq_length, d_k]
        value: Value tensor, shape [..., seq_length, d_v]

    Returns:
        Attention output, shape [..., seq_length, d_v]
    """
    # TODO: Step 1 - Compute attention scores (Q @ K^T)
    # Hint: Use query @ key.transpose(-1, -2)
    attn_weights = None  # Line 68-69

    # TODO: Step 2 - Scale by sqrt(d_k)
    # Hint: scale_factor = math.sqrt(int(value.shape[-1]))
    # Hint: attn_weights = attn_weights / scale_factor
    pass  # Line 72-74

    # TODO: Step 3 - Apply causal mask
    # Hint: seq_len = query.shape[-2]
    # Hint: mask = causal_mask(seq_len, 0, dtype=query.dtype, device=query.device)
    # Hint: attn_weights = attn_weights + mask
    pass  # Line 77-80

    # TODO: Step 4 - Apply softmax
    # Hint: attn_weights = F.softmax(attn_weights)
    pass  # Line 83-84

    # TODO: Step 5 - Weighted sum of values
    # Hint: attn_output = attn_weights @ value
    # Hint: return attn_output
    return None  # Line 87-89
