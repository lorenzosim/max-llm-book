# Step 14: Text generation

<div class="note">
    Learn to implement autoregressive text generation with sampling and temperature control.
</div>

## What is autoregressive generation?

In this section you will implement text generation. The model generates text one token at a time, using each prediction as input for the next.

The process:
1. Start with prompt: `[15496, 995]` ("Hello world")
2. Predict next token: `[15496, 995, 318]` ("Hello world is")
3. Predict next token: `[15496, 995, 318, 257]` ("Hello world is a")
4. Repeat until reaching desired length

You'll implement temperature control (adjusting randomness) and sampling (choosing from the probability distribution) to control generation quality and creativity.

## Why temperature and sampling?

**Temperature** controls randomness in generation. The model outputs logits (raw scores) for each vocabulary token. These logits can vary in magnitude—some tokens might have much higher scores than others. Temperature scaling divides logits by a temperature value before applying softmax:

$$P(x_i) = \frac{e^{z_i/T}}{\sum_j e^{z_j/T}}$$

where $z_i$ are the logits, $T$ is the temperature, and $P(x_i)$ is the probability of token $i$.

- **Temperature = 1.0**: No scaling, use original distribution
- **Temperature < 1.0** (e.g., 0.7): Sharpens distribution, makes high-probability tokens more likely (more focused, less random)
- **Temperature > 1.0** (e.g., 1.2): Flattens distribution, makes low-probability tokens more likely (more diverse, more random)
- **Temperature → 0**: Approaches greedy decoding (always pick highest probability)
- **Temperature → ∞**: Approaches uniform distribution (random token selection)

**Sampling vs. Greedy**: Greedy decoding always selects the highest-probability token (`argmax`). This produces deterministic, repetitive text. Sampling randomly selects tokens according to the probability distribution, producing diverse, creative text. Most practical generation uses sampling with temperature to balance coherence and creativity.

## Why these design choices?

**1. Autoregressive for Tractability**: Generating all tokens simultaneously would require modeling the joint distribution `P(token_1, token_2, ..., token_n)`, which is exponentially complex. The autoregressive factorization `P(token_i | token_1, ..., token_{i-1})` breaks this into n simple conditional distributions that the model can learn. This makes training feasible while maintaining expressiveness.

**2. Temperature for Control**: Without temperature control, you're stuck with the model's trained distribution—which might be too peaky (repetitive) or too flat (incoherent). Temperature gives users a simple knob to adjust creativity. News articles might use temperature=0.7 for factual consistency, while creative writing uses temperature=1.2 for diversity. One model serves many use cases.

**3. Sampling for Diversity**: Humans don't write by always choosing the most likely next word—that would produce boring, formulaic text. Sampling introduces controlled randomness that mirrors human creativity. The same prompt can generate many different completions, each valid and coherent. This diversity is essential for applications like story generation, brainstorming, or creative writing.

**4. One Token at a Time**: While parallel generation techniques exist, autoregressive generation is simple, reliable, and provides fine-grained control. You can stop generation at any point, apply constraints per-token (e.g., banning certain words), or adjust temperature mid-generation. The sequential nature also matches human intuition about how language unfolds.

### Key concepts

**Autoregressive Loop**:
- Repeat: predict next token, append to sequence
- Input grows: `[batch, seq_length]` → `[batch, seq_length + 1]` → ...
- Stops after `max_new_tokens` or when end-of-sequence token generated
- Each iteration requires a full forward pass through the model

**Logit Extraction**:
- Model outputs: `[batch, seq_length, vocab_size]`
- For next token, use last position: `logits[0, -1, :]`
- Shape: `[vocab_size]` (one score per vocabulary token)
- These are raw logits, not probabilities yet

**Temperature Scaling**:
- Formula: `scaled_logits = logits / temperature`
- Implemented as: `logits / Tensor.constant(temperature, ...)`
- Must match dtype and device of logits
- Applied before softmax

**Sampling with NumPy**:
- Convert to probabilities: `F.softmax(logits)`
- Transfer to CPU: `probs.to(CPU())`
- Convert to NumPy: `np.from_dlpack(probs)`
- Sample: `np.random.choice(len(probs), p=probs)`
- NumPy used because MAX doesn't have built-in sampling yet

**Greedy Decoding**:
- Select highest probability token: [`F.argmax(logits)`](https://docs.modular.com/max/api/python/experimental/functional#max.experimental.functional.argmax)
- Deterministic: same input always produces same output
- Faster than sampling (no softmax or random choice needed)
- Often produces repetitive text

**Concatenation**:
- Append token to sequence: [`F.concat([seq, new_token], axis=1)`](https://docs.modular.com/max/api/python/experimental/functional#max.experimental.functional.concat)
- New token must be 2D: `reshape([1, -1])` or `reshape([1, 1])`
- Axis=1 concatenates along sequence dimension
- Result has one more token than input

### Implementation tasks (`step_14.py`)

1. **Import Required Modules** (Lines 13-17):
   - Import `numpy as np`
   - Import `CPU` from `max.driver`
   - Import `DType` from `max.dtype`
   - Import `functional as F` from `max.experimental`
   - Import `Tensor` from `max.experimental.tensor`

2. **Get Model Logits** (Lines 32-37):
   - Call `logits = model(input_ids)`
   - Extract last position: `next_token_logits = logits[0, -1, :]`

3. **Apply Temperature and Sample** (Lines 42-54):
   - Create temperature tensor: `Tensor.constant(temperature, dtype=..., device=...)`
   - Scale logits: `next_token_logits / temp_tensor`
   - Get probabilities: `F.softmax(next_token_logits)`
   - Convert to NumPy: `np.from_dlpack(probs.to(CPU()))`
   - Sample: `np.random.choice(len(probs_np), p=probs_np)`
   - Convert back: `Tensor.constant(next_token_id, dtype=DType.int64, device=...)`

4. **Implement Greedy Decoding** (Lines 57-58):
   - If not sampling: `next_token_tensor = F.argmax(next_token_logits)`

5. **Implement Generation Loop** (Lines 77-94):
   - Initialize: `generated_tokens = input_ids`
   - Loop `max_new_tokens` times
   - Generate next token: `generate_next_token(model, generated_tokens, ...)`
   - Reshape: `next_token.reshape([1, -1])`
   - Concatenate: `F.concat([generated_tokens, next_token_2d], axis=1)`

**Implementation**:
```python
import numpy as np

from max.driver import CPU
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor


def generate_next_token(model, input_ids, temperature=1.0, do_sample=True):
    logits = model(input_ids)
    next_token_logits = logits[0, -1, :]

    if do_sample and temperature > 0:
        temp_tensor = Tensor.constant(
            temperature, dtype=next_token_logits.dtype, device=next_token_logits.device
        )
        next_token_logits = next_token_logits / temp_tensor
        probs = F.softmax(next_token_logits)

        probs_np = np.from_dlpack(probs.to(CPU()))
        next_token_id = np.random.choice(len(probs_np), p=probs_np)
        next_token_tensor = Tensor.constant(
            next_token_id, dtype=DType.int64, device=input_ids.device
        )
    else:
        next_token_tensor = F.argmax(next_token_logits)

    return next_token_tensor


def generate_tokens(
    model, input_ids, max_new_tokens=10, temperature=1.0, do_sample=True
):
    generated_tokens = input_ids

    for _ in range(max_new_tokens):
        next_token = generate_next_token(
            model, generated_tokens, temperature=temperature, do_sample=do_sample
        )
        next_token_2d = next_token.reshape([1, -1])
        generated_tokens = F.concat([generated_tokens, next_token_2d], axis=1)

    return generated_tokens
```

### Validation
Run `pixi run s14`

**Reference**: `solutions/solution_14.py`

---

**Congratulations!** You've completed all 14 steps and built a complete GPT-2 model from scratch using Modular's MAX! You now understand:

- Token and position embeddings (Steps 05-06)
- Query/Key/Value projections and attention (Steps 07-08)
- Multi-head attention (Step 09)
- Layer normalization and residual connections (Step 10)
- Transformer blocks and model stacking (Steps 11-12)
- Language modeling and text generation (Steps 13-14)

Your model can now generate text! To use it with pretrained weights, see [WEIGHT_LOADING.md](../../WEIGHT_LOADING.md) in the repository root.
