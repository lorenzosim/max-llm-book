"""Tests for Step 07: Query/Key/Value Projections"""

import ast
import inspect
from pathlib import Path


def test_step_07():
    """Comprehensive validation for Step 07 implementation."""

    results = []
    step_file = Path("steps/step_07.py")

    # Read source
    if not step_file.exists():
        print(f"❌ File not found: {step_file}")
        return

    source = step_file.read_text()
    tree = ast.parse(source)

    # Phase 1: Import checks
    has_linear = False
    has_module = False
    has_functional = False

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module == "max.nn.module_v3":
                for alias in node.names:
                    if alias.name == "Linear":
                        has_linear = True
                    if alias.name == "Module":
                        has_module = True
            if node.module == "max.experimental":
                for alias in node.names:
                    if alias.name == "functional" or (
                        alias.asname and alias.asname == "F"
                    ):
                        has_functional = True

    if has_linear:
        results.append("✅ Linear is correctly imported from max.nn.module_v3")
    else:
        results.append("❌ Linear is not imported from max.nn.module_v3")
        results.append("   Hint: Add 'from max.nn.module_v3 import Linear, Module'")

    if has_module:
        results.append("✅ Module is correctly imported from max.nn.module_v3")
    else:
        results.append("❌ Module is not imported from max.nn.module_v3")
        results.append("   Hint: Add 'from max.nn.module_v3 import Linear, Module'")

    if has_functional:
        results.append("✅ functional is correctly imported from max.experimental")
    else:
        results.append("❌ functional is not imported from max.experimental")
        results.append("   Hint: Add 'from max.experimental import functional as F'")

    # Phase 2: Structure checks
    try:
        from steps.step_07 import GPT2SingleHeadQKV

        results.append("✅ GPT2SingleHeadQKV class exists")
    except ImportError:
        results.append("❌ GPT2SingleHeadQKV class not found in step_07 module")
        results.append("   Hint: Create class GPT2SingleHeadQKV(Module)")
        print("\n".join(results))
        return

    # Check inheritance
    from max.nn.module_v3 import Module

    if issubclass(GPT2SingleHeadQKV, Module):
        results.append("✅ GPT2SingleHeadQKV inherits from Module")
    else:
        results.append("❌ GPT2SingleHeadQKV must inherit from Module")

    # Phase 3: Implementation checks
    if "self.c_attn = Linear" in source or (
        "self.c_attn =" in source
        and "None" not in source.split("self.c_attn =")[1].split("\n")[0]
    ):
        results.append("✅ self.c_attn linear layer is created correctly")
    else:
        results.append("❌ self.c_attn linear layer is not created correctly")
        results.append(
            "   Hint: Use Linear(config.n_embd, 3 * config.n_embd, bias=True)"
        )

    # Check if 3 * n_embd is used
    if "3 * config.n_embd" in source or "config.n_embd * 3" in source:
        results.append("✅ Output dimension is correctly set to 3 * n_embd")
    else:
        results.append("❌ Output dimension should be 3 * config.n_embd")
        results.append("   Hint: Second parameter should be 3 * config.n_embd")

    # Check if bias=True
    if "bias=True" in source:
        results.append("✅ bias=True is set correctly")
    else:
        results.append("❌ bias parameter not found or incorrect")
        results.append("   Hint: Add bias=True to Linear layer")

    # Check forward pass
    if "self.c_attn(x)" in source.replace(" ", "") or "self.c_attn(x)" in source:
        results.append("✅ self.c_attn is called with input x")
    else:
        results.append("❌ self.c_attn is not called with x")
        results.append("   Hint: Call self.c_attn(x) to project input")

    # Check F.split usage
    if "F.split" in source:
        results.append("✅ F.split is used to separate Q, K, V")
    else:
        results.append("❌ F.split is not used")
        results.append(
            "   Hint: Use F.split(qkv, [self.n_embd, self.n_embd, self.n_embd], axis=-1)"
        )

    # Check if split sizes are correct
    if "[self.n_embd, self.n_embd, self.n_embd]" in source.replace(" ", ""):
        results.append("✅ Split sizes are correctly set to [n_embd, n_embd, n_embd]")
    else:
        results.append("❌ Split sizes may be incorrect")
        results.append("   Hint: Use [self.n_embd, self.n_embd, self.n_embd]")

    # Check if axis=-1
    if "axis=-1" in source or "axis=2" in source:
        results.append("✅ Split axis is correctly set")
    else:
        results.append("❌ Split axis should be -1 or 2")
        results.append("   Hint: Add axis=-1 to F.split")

    # Phase 4: Placeholder detection
    none_lines = [
        line.strip()
        for line in source.split("\n")
        if "= None" in line
        and not line.strip().startswith("#")
        and "def " not in line
        and ":" not in line.split("=")[0]
    ]
    if none_lines:
        results.append("❌ Found placeholder 'None' values that need to be replaced:")
        for line in none_lines[:3]:
            results.append(f"   {line}")
        results.append(
            "   Hint: Replace all 'None' values with the actual implementation"
        )
    else:
        results.append("✅ All placeholder 'None' values have been replaced")

    # Phase 5: Functional tests
    try:
        from max.driver import CPU
        from max.dtype import DType
        from max.experimental.tensor import Tensor
        from solutions.solution_01 import GPT2Config

        config = GPT2Config()
        qkv_layer = GPT2SingleHeadQKV(config)
        results.append("✅ GPT2SingleHeadQKV class can be instantiated")

        # Check c_attn attribute exists
        if hasattr(qkv_layer, "c_attn"):
            results.append("✅ GPT2SingleHeadQKV.c_attn is initialized")
        else:
            results.append("❌ GPT2SingleHeadQKV.c_attn attribute not found")

        # Test forward pass with sample input
        batch_size = 2
        seq_length = 8
        test_input = Tensor.randn(
            batch_size, seq_length, config.n_embd, dtype=DType.float32, device=CPU()
        )

        query, key, value = qkv_layer(test_input)
        results.append("✅ GPT2SingleHeadQKV forward pass executes without errors")

        # Check output shapes
        expected_shape = (batch_size, seq_length, config.n_embd)
        if query.shape == expected_shape:
            results.append(f"✅ Query shape is correct: {expected_shape}")
        else:
            results.append(
                f"❌ Query shape is incorrect: expected {expected_shape}, got {query.shape}"
            )

        if key.shape == expected_shape:
            results.append(f"✅ Key shape is correct: {expected_shape}")
        else:
            results.append(
                f"❌ Key shape is incorrect: expected {expected_shape}, got {key.shape}"
            )

        if value.shape == expected_shape:
            results.append(f"✅ Value shape is correct: {expected_shape}")
        else:
            results.append(
                f"❌ Value shape is incorrect: expected {expected_shape}, got {value.shape}"
            )

        # Verify outputs are not all zeros
        import numpy as np

        query_np = np.from_dlpack(query.to(CPU()))
        key_np = np.from_dlpack(key.to(CPU()))
        value_np = np.from_dlpack(value.to(CPU()))

        if not np.allclose(query_np, 0):
            results.append("✅ Query contains non-zero values")
        else:
            results.append("❌ Query is all zeros - projection may not be initialized")

        if not np.allclose(key_np, 0):
            results.append("✅ Key contains non-zero values")
        else:
            results.append("❌ Key is all zeros - projection may not be initialized")

        if not np.allclose(value_np, 0):
            results.append("✅ Value contains non-zero values")
        else:
            results.append("❌ Value is all zeros - projection may not be initialized")

        # Test that Q, K, V are different
        if not np.allclose(query_np, key_np):
            results.append("✅ Query and Key are different (as expected)")
        else:
            results.append("⚠️ Warning: Query and Key are identical")

    except Exception as e:
        results.append(f"❌ Functional test failed: {e}")
        import traceback

        results.append(f"   {traceback.format_exc().split('Error:')[-1].strip()}")

    # Print all results
    print("Running tests for Step 07: Query/Key/Value Projections...\n")
    print("Results:")
    print("\n".join(results))

    # Summary
    failed = any(r.startswith("❌") for r in results)
    if not failed:
        print("\n" + "=" * 60)
        print("🎉 All checks passed! Your implementation is complete.")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("⚠️ Some checks failed. Review the hints above and try again.")
        print("=" * 60)


if __name__ == "__main__":
    test_step_07()
