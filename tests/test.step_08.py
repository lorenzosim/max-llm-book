"""Tests for Step 08: Attention Mechanism with Causal Masking"""

import ast
from pathlib import Path


def test_step_08():
    """Comprehensive validation for Step 08 implementation."""

    results = []
    step_file = Path("steps/step_08.py")

    # Read source
    if not step_file.exists():
        print(f"❌ File not found: {step_file}")
        return

    source = step_file.read_text()
    tree = ast.parse(source)

    # Phase 1: Import checks
    has_math = False
    has_functional = False
    has_tensor = False
    has_device = False
    has_dtype = False
    has_dim = False
    has_dimlike = False

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "math":
                    has_math = True
        if isinstance(node, ast.ImportFrom):
            if node.module == "max.experimental":
                for alias in node.names:
                    if alias.name == "functional" or (
                        alias.asname and alias.asname == "F"
                    ):
                        has_functional = True
            if node.module == "max.experimental.tensor":
                for alias in node.names:
                    if alias.name == "Tensor":
                        has_tensor = True
            if node.module == "max.driver":
                for alias in node.names:
                    if alias.name == "Device":
                        has_device = True
            if node.module == "max.dtype":
                for alias in node.names:
                    if alias.name == "DType":
                        has_dtype = True
            if node.module == "max.graph":
                for alias in node.names:
                    if alias.name == "Dim":
                        has_dim = True
                    if alias.name == "DimLike":
                        has_dimlike = True

    if has_math:
        results.append("✅ math is correctly imported")
    else:
        results.append("❌ math is not imported")
        results.append("   Hint: Add 'import math' at the top")

    if has_functional:
        results.append("✅ functional is correctly imported from max.experimental")
    else:
        results.append("❌ functional is not imported from max.experimental")
        results.append("   Hint: Add 'from max.experimental import functional as F'")

    if has_tensor:
        results.append("✅ Tensor is correctly imported from max.experimental.tensor")
    else:
        results.append("❌ Tensor is not imported from max.experimental.tensor")
        results.append("   Hint: Add 'from max.experimental.tensor import Tensor'")

    if has_device:
        results.append("✅ Device is correctly imported from max.driver")
    else:
        results.append("❌ Device is not imported from max.driver")
        results.append("   Hint: Add 'from max.driver import Device'")

    if has_dtype:
        results.append("✅ DType is correctly imported from max.dtype")
    else:
        results.append("❌ DType is not imported from max.dtype")
        results.append("   Hint: Add 'from max.dtype import DType'")

    if has_dim:
        results.append("✅ Dim is correctly imported from max.graph")
    else:
        results.append("❌ Dim is not imported from max.graph")
        results.append("   Hint: Add 'from max.graph import Dim, DimLike'")

    if has_dimlike:
        results.append("✅ DimLike is correctly imported from max.graph")
    else:
        results.append("❌ DimLike is not imported from max.graph")
        results.append("   Hint: Add 'from max.graph import Dim, DimLike'")

    # Phase 2: Structure checks
    try:
        from steps.step_08 import causal_mask, compute_attention

        results.append("✅ causal_mask function exists")
        results.append("✅ compute_attention function exists")
    except ImportError as e:
        if "causal_mask" in str(e):
            results.append("❌ causal_mask function not found in step_08 module")
            results.append(
                "   Hint: Define causal_mask function with @F.functional decorator"
            )
        if "compute_attention" in str(e):
            results.append("❌ compute_attention function not found in step_08 module")
            results.append("   Hint: Define compute_attention function")
        print("\n".join(results))
        return

    # Phase 3: Implementation checks
    # Check causal_mask implementation
    if "@F.functional" in source or "@functional" in source:
        results.append("✅ causal_mask uses @F.functional decorator")
    else:
        results.append("❌ causal_mask missing @F.functional decorator")
        results.append("   Hint: Add @F.functional before causal_mask definition")

    if "Tensor.constant" in source and 'float("-inf")' in source:
        results.append("✅ causal_mask creates -inf constant correctly")
    else:
        results.append("❌ causal_mask should create Tensor.constant with -inf")
        results.append(
            '   Hint: mask = Tensor.constant(float("-inf"), dtype=dtype, device=device)'
        )

    if "F.broadcast_to" in source:
        results.append("✅ causal_mask uses F.broadcast_to")
    else:
        results.append("❌ causal_mask should use F.broadcast_to")
        results.append(
            "   Hint: mask = F.broadcast_to(mask, shape=(sequence_length, n))"
        )

    if "F.band_part" in source:
        results.append("✅ causal_mask uses F.band_part")
    else:
        results.append("❌ causal_mask should use F.band_part")
        results.append(
            "   Hint: return F.band_part(mask, num_lower=None, num_upper=0, exclude=True)"
        )

    # Check compute_attention implementation
    if "query @ key.transpose(-1, -2)" in source.replace(
        " ", ""
    ) or "query@key.transpose(-1,-2)" in source.replace(" ", ""):
        results.append("✅ Attention scores computed with Q @ K^T")
    else:
        results.append(
            "❌ Attention scores should be computed with query @ key.transpose(-1, -2)"
        )
        results.append("   Hint: attn_weights = query @ key.transpose(-1, -2)")

    if "math.sqrt" in source:
        results.append("✅ Scaling uses math.sqrt")
    else:
        results.append("❌ Attention scores should be scaled by sqrt(d_k)")
        results.append("   Hint: scale_factor = math.sqrt(int(value.shape[-1]))")

    if (
        "causal_mask(" in source
        and "attn_weights +" in source
        or "attn_weights+" in source
    ):
        results.append("✅ Causal mask is applied to attention weights")
    else:
        results.append("❌ Causal mask should be applied to attention weights")
        results.append(
            "   Hint: mask = causal_mask(...) then attn_weights = attn_weights + mask"
        )

    if "F.softmax" in source:
        results.append("✅ Softmax is applied to attention weights")
    else:
        results.append("❌ F.softmax should be applied to attention weights")
        results.append("   Hint: attn_weights = F.softmax(attn_weights)")

    if (
        source.count("attn_weights @ value") > 0
        or source.count("attn_weights@value") > 0
    ):
        results.append("✅ Weighted sum computed with attention @ value")
    else:
        results.append("❌ Final output should be attn_weights @ value")
        results.append("   Hint: attn_output = attn_weights @ value")

    # Phase 4: Placeholder detection
    none_lines = [
        line.strip()
        for line in source.split("\n")
        if ("= None" in line or "return None" in line)
        and not line.strip().startswith("#")
        and "def " not in line
        and "Optional" not in line
    ]
    if none_lines:
        results.append("❌ Found placeholder 'None' values that need to be replaced:")
        for line in none_lines[:5]:
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
        import numpy as np

        # Test causal_mask
        seq_len = 4
        mask = causal_mask(seq_len, 0, dtype=DType.float32, device=CPU())
        results.append("✅ causal_mask executes without errors")

        # Check mask shape
        expected_shape = (seq_len, seq_len)
        if mask.shape == expected_shape:
            results.append(f"✅ causal_mask shape is correct: {expected_shape}")
        else:
            results.append(
                f"❌ causal_mask shape is incorrect: expected {expected_shape}, got {mask.shape}"
            )

        # Check mask values (should be -inf in upper triangle, 0 elsewhere)
        mask_np = np.from_dlpack(mask.to(CPU()))
        # Lower triangle and diagonal should be 0
        lower_triangle_ok = np.all(np.tril(mask_np) == 0)
        # Upper triangle (excluding diagonal) should be -inf
        upper_triangle = np.triu(mask_np, k=1)
        upper_triangle_ok = np.all(np.isinf(upper_triangle)) and np.all(
            upper_triangle < 0
        )

        if lower_triangle_ok and upper_triangle_ok:
            results.append(
                "✅ causal_mask has correct values (0 for past/present, -inf for future)"
            )
        else:
            results.append("❌ causal_mask values are incorrect")
            results.append(f"   Lower triangle all zeros: {lower_triangle_ok}")
            results.append(f"   Upper triangle all -inf: {upper_triangle_ok}")

        # Test compute_attention
        batch_size = 2
        seq_len = 4
        d_k = 64
        d_v = 64

        query = Tensor.randn(
            batch_size, seq_len, d_k, dtype=DType.float32, device=CPU()
        )
        key = Tensor.randn(batch_size, seq_len, d_k, dtype=DType.float32, device=CPU())
        value = Tensor.randn(
            batch_size, seq_len, d_v, dtype=DType.float32, device=CPU()
        )

        output = compute_attention(query, key, value)
        results.append("✅ compute_attention executes without errors")

        # Check output shape
        expected_output_shape = (batch_size, seq_len, d_v)
        if output.shape == expected_output_shape:
            results.append(
                f"✅ compute_attention output shape is correct: {expected_output_shape}"
            )
        else:
            results.append(
                f"❌ Output shape is incorrect: expected {expected_output_shape}, got {output.shape}"
            )

        # Check output contains non-zero values
        output_np = np.from_dlpack(output.to(CPU()))
        if not np.allclose(output_np, 0):
            results.append("✅ Output contains non-zero values")
        else:
            results.append("❌ Output is all zeros")

        # Check that output is different from input value
        value_np = np.from_dlpack(value.to(CPU()))
        if not np.allclose(output_np, value_np):
            results.append(
                "✅ Output is different from input value (attention is applied)"
            )
        else:
            results.append("⚠️ Warning: Output is identical to input value")

    except Exception as e:
        results.append(f"❌ Functional test failed: {e}")
        import traceback

        tb = traceback.format_exc()
        # Get last meaningful error line
        error_lines = [line for line in tb.split("\n") if line.strip()]
        if error_lines:
            results.append(f"   {error_lines[-1]}")

    # Print all results
    print("Running tests for Step 08: Attention Mechanism with Causal Masking...\n")
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
    test_step_08()
