"""
Step 12: Stacking Transformer Blocks

Stack multiple transformer blocks with embeddings to create
the complete GPT-2 model architecture.

Tasks:
1. Import Tensor, Embedding, Module, Sequential, and previous components
2. Create token and position embeddings
3. Stack n_layer transformer blocks using Sequential
4. Create final layer normalization
5. Implement forward pass: embeddings -> blocks -> layer norm

Run: pixi run s12
"""

# TODO: Import required modules
# Hint: You'll need Tensor from max.experimental.tensor
# Hint: You'll need Embedding, Module, Sequential from max.nn.module_v3
# Hint: Import GPT2Config from solutions.solution_01
# Hint: Import LayerNorm from solutions.solution_10
# Hint: Import GPT2Block from solutions.solution_11


class GPT2Model(Module):
    """Complete GPT-2 transformer model."""

    def __init__(self, config: GPT2Config):
        """Initialize GPT-2 model.

        Args:
            config: GPT2Config containing model hyperparameters
        """
        super().__init__()

        # TODO: Create token embeddings
        # Hint: Use Embedding(config.vocab_size, dim=config.n_embd)
        self.wte = None  # Line 34-35

        # TODO: Create position embeddings
        # Hint: Use Embedding(config.n_positions, dim=config.n_embd)
        self.wpe = None  # Line 38-39

        # TODO: Stack transformer blocks
        # Hint: Use Sequential(*(GPT2Block(config) for _ in range(config.n_layer)))
        # This creates config.n_layer blocks (12 for GPT-2 base)
        self.h = None  # Line 42-44

        # TODO: Create final layer normalization
        # Hint: Use LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.ln_f = None  # Line 47-48

    def __call__(self, input_ids):
        """Forward pass through the transformer.

        Args:
            input_ids: Token IDs, shape [batch, seq_length]

        Returns:
            Hidden states, shape [batch, seq_length, n_embd]
        """
        # TODO: Get batch size and sequence length
        # Hint: batch_size, seq_length = input_ids.shape
        pass  # Line 61-62

        # TODO: Get token embeddings
        # Hint: tok_embeds = self.wte(input_ids)
        pass  # Line 65-66

        # TODO: Get position embeddings
        # Hint: Create position indices with Tensor.arange(seq_length, dtype=input_ids.dtype, device=input_ids.device)
        # Hint: pos_embeds = self.wpe(position_indices)
        pass  # Line 69-72

        # TODO: Combine embeddings
        # Hint: x = tok_embeds + pos_embeds
        pass  # Line 75-76

        # TODO: Apply transformer blocks
        # Hint: x = self.h(x)
        pass  # Line 79-80

        # TODO: Apply final layer norm
        # Hint: x = self.ln_f(x)
        pass  # Line 83-84

        # TODO: Return the output
        return None  # Line 87
