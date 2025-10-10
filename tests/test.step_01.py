"""Tests for GPT2Config in puzzles/config.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from steps import step_01 as config_module
import inspect


def test_gpt2_config():
    """Test that GPT2Config has the correct hyperparameter values."""
    config = config_module.GPT2Config()

    # Test if dataclass is imported from dataclasses
    if 'dataclasses' in config_module.__dict__.get('__annotations__', {}) or hasattr(config_module, 'dataclass'):
        # Check if dataclass is actually imported
        source = inspect.getsource(config_module)
        if 'from dataclasses import dataclass' in source:
            print("✅ dataclass is correctly imported from dataclasses")
        else:
            print("❌ dataclass is not imported from dataclasses")
    else:
        source = inspect.getsource(config_module)
        if 'from dataclasses import dataclass' in source:
            print("✅ dataclass is correctly imported from dataclasses")
        else:
            print("❌ dataclass is not imported from dataclasses")

    # Test if GPT2Config has the @dataclass decorator
    if hasattr(config_module.GPT2Config(), '__dataclass_fields__'):
        print("✅ GPT2Config has the @dataclass decorator")
    else:
        print("❌ GPT2Config does not have the @dataclass decorator")

    # Test vocab_size
    if(config.vocab_size == 50257):
        print("✅ vocab_size is correct: 50257")
    else:
        print(f"❌ vocab_size is incorrect: expected match with Hugging Face model configuration, got {config.vocab_size}")

    # Test n_positions
    if(config.n_positions == 1024):
        print("✅ n_positions is correct: 1024")
    else:
        print(f"❌ n_positions is incorrect: expected match with Hugging Face model configuration, got {config.n_positions}")

    # Test n_embd
    if(config.n_embd == 768):
        print("✅ n_embd is correct: 768")
    else:
        print(f"❌ n_embd is incorrect: expected match with Hugging Face model configuration, got {config.n_embd}")

    # Test n_layer
    if(config.n_layer == 12):
        print("✅ n_layer is correct: 12")
    else:
        print(f"❌ n_layer is incorrect: expected match with Hugging Face model configuration, got {config.n_layer}")

    # Test n_head
    if(config.n_head == 12):
        print("✅ n_head is correct: 12")
    else:
        print(f"❌ n_head is incorrect: expected match with Hugging Face model configuration, got {config.n_head}")

    # Test n_inner
    if(config.n_inner == 3072):
        print("✅ n_inner is correct: 3072")
    else:
        print(f"❌ n_inner is incorrect: expected match with Hugging Face model configuration, got {config.n_inner}")

    # Test layer_norm_epsilon
    if(config.layer_norm_epsilon == 1e-05):
        print("✅ layer_norm_epsilon is correct: 1e-05")
    else:
        print(f"❌ layer_norm_epsilon is incorrect: expected match with Hugging Face model configuration, got {config.layer_norm_epsilon}")

if __name__ == "__main__":
    test_gpt2_config()

