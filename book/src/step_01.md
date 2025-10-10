# Step 01: Create Model Configuration (`step_01.py`)

**Purpose**: Define the GPT-2 model architecture parameters.

## What is Model Configuration?

Model configuration is the foundational step in building or loading a neural network. It defines the architecture's "blueprint" - specifying dimensions, layer counts, and other structural parameters that determine how the model processes information. Think of it as the architectural specification before construction begins.

In this step, we're creating a configuration class that stores GPT-2's hyperparameters. These parameters aren't learned during training; rather, they define the model's structure: how many layers it has, how wide those layers are, how many attention heads to use, and so on.

## Why Create This Configuration?

**1. Compatibility with Pretrained Models**: By matching Hugging Face's GPT-2 configuration exactly, we ensure our implementation can load their pretrained weights. This is crucial - it means we can leverage a model trained on billions of tokens without needing to retrain from scratch.

**2. Reproducibility**: Having configuration separate from implementation makes experiments reproducible. You can easily save, share, and recreate exact model architectures.

**3. Code Organization**: Centralizing all hyperparameters in one place makes the codebase cleaner and easier to modify. Want to experiment with a different model size? Just change the config values.

**4. Foundation for Implementation**: This configuration serves as the contract for all subsequent steps. Every component we build (embeddings, attention, MLP layers) will reference these parameters to ensure dimensional consistency.

### Key Concepts:
**Dataclasses in Python**:
- Python's [`@dataclass`](https://docs.python.org/3/library/dataclasses.html) decorator reduces boilerplate code when creating configuration objects and provides clean syntax for defining class attributes with type hints and default values

**Model Configuration**:
- Configuration objects centralize hyperparameters and architecture settings in one place
- Makes it easy to experiment with different model sizes and settings
- Essential for reproducibility and model initialization

**Matching Hugging Face Models**:
- Using the same configuration values as [Hugging Face's pretrained GPT-2](https://huggingface.co/openai-community/gpt2) ensures weight compatibility
- Allows loading and using pretrained weights for inference without retraining
- Configuration values are accessed with the [transformers library](https://pypi.org/project/transformers/) in `hugging-face-model.py`

**GPT-2 Architecture Parameters**:
- `vocab_size`: Size of the token vocabulary (number of unique tokens the model can process)
- `n_positions`: Maximum sequence length (context window)
- `n_embd`: Embedding dimension (size of hidden states)
- `n_layer`: Number of transformer blocks stacked vertically
- `n_head`: Number of attention heads per layer
- `n_inner`: Dimension of the MLP intermediate layer (typically 4x n_embd)
- `layer_norm_epsilon`: Small constant for numerical stability in layer normalization

### Implementation Tasks (`step_01.py`):
1. Import dataclass from the dataclasses module 
2. Add the Python @dataclass decorator to the GPT2Config class
3. Get the correct values for the model parameters
    - Option 1: Run `pixi run huggingface` to access these parameters from the Hugging Face `transformers` library
    - Option 2: Read the values from [GPT3 model card](https://huggingface.co/openai-community/gpt2/blob/main/config.json) on huggingface.co
4. Replace the None of the GPT2Config properties with the correct values

**Implementation**:
```python
from dataclasses import dataclass

@dataclass
class GPT2Config:
   # run `pixi run hugging-face-model` to get the correct values
    vocab_size: int = ?
    n_positions: int = ?
    n_embd: int = ?
    n_layer: int = ?
    n_head: int = ?
    n_inner: int = ?
    layer_norm_epsilon: float = ?
```

### Validation:
run `pixi run s01`

A failed test will show
```bash
Running tests for Step 01: Create Model Configuration...

Results:
❌ dataclass is not imported from dataclasses
❌ GPT2Config does not have the @dataclass decorator
❌ vocab_size is incorrect: expected match with Hugging Face model configuration, got None
❌ n_positions is incorrect: expected match with Hugging Face model configuration, got None
❌ n_embd is incorrect: expected match with Hugging Face model configuration, got None
❌ n_layer is incorrect: expected match with Hugging Face model configuration, got None
❌ n_head is incorrect: expected match with Hugging Face model configuration, got None
❌ n_inner is incorrect: expected match with Hugging Face model configuration, got None
❌ layer_norm_epsilon is incorrect: expected match with Hugging Face model configuration, got None
```

A sucessful test will show
```bash
Running tests for Step 01: Create Model Configuration...

Results:
✅ dataclass is correctly imported from dataclasses
✅ GPT2Config has the @dataclass decorator
✅ vocab_size is correct
✅ n_positions is correct
✅ n_embd is correct
✅ n_layer is correct
✅ n_head is correct
✅ n_inner is correct
✅ layer_norm_epsilon is correct
```

**Reference**: `steps/step_01.py`