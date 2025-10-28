#!/usr/bin/env python3
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""
Example: Type-safe operations with JaggedTensor

This example demonstrates how to use fvdb's type-safe functions that work
with both regular torch.Tensor and fvdb.JaggedTensor, with proper type inference.
"""

import torch

import fvdb


def example_basic_operations():
    """Basic operations with proper typing."""
    print("=" * 60)
    print("Example 1: Basic Operations")
    print("=" * 60)

    # Create a JaggedTensor
    jt = fvdb.JaggedTensor([torch.randn(100, 3), torch.randn(150, 3), torch.randn(120, 3)])
    print(f"Created JaggedTensor with {jt.num_tensors} sub-tensors")

    # These are all type-safe and return JaggedTensor
    activated = fvdb.relu(jt)
    print(f"  relu:    {type(activated).__name__}")

    normalized = fvdb.sigmoid(jt)
    print(f"  sigmoid: {type(normalized).__name__}")

    scaled = fvdb.mul(jt, 2.0)
    print(f"  mul:     {type(scaled).__name__}")

    # Access JaggedTensor-specific attributes (type-safe!)
    print(f"\nShape info: {activated.lshape}")
    print()


def example_comparison_torch_vs_fvdb():
    """Compare torch.* vs fvdb.* function calls."""
    print("=" * 60)
    print("Example 2: torch.* vs fvdb.* Comparison")
    print("=" * 60)

    jt = fvdb.JaggedTensor([torch.randn(50, 4) for _ in range(3)])

    # Both work at runtime, but only fvdb.* is type-safe
    out1 = torch.relu(jt)  # type: ignore  # [!] Type checker sees this as Tensor
    out2 = fvdb.relu(jt)  # [*] Type checker correctly sees JaggedTensor

    print(f"torch.relu(jt): {type(out1).__name__}")
    print(f"fvdb.relu(jt):  {type(out2).__name__}")

    # Both produce the same result
    assert torch.allclose(out1.jdata, out2.jdata)  # type: ignore
    print("\n[*] Both approaches produce identical results at runtime")
    print()


def example_works_with_regular_tensors():
    """fvdb functions also work with regular Tensors."""
    print("=" * 60)
    print("Example 3: Works with Regular Tensors Too")
    print("=" * 60)

    # Regular torch.Tensor
    t = torch.randn(10, 3)

    # fvdb functions work seamlessly with regular Tensors
    out = fvdb.relu(t)
    print(f"fvdb.relu(Tensor) -> {type(out).__name__}")
    print(f"Output shape: {out.shape}")

    # Compare with torch - identical results
    torch_out = torch.relu(t)
    assert torch.allclose(out, torch_out)
    print("\n[*] fvdb.relu produces identical results to torch.relu")
    print()


def example_reduction_operations():
    """Reduction operations that preserve jagged structure."""
    print("=" * 60)
    print("Example 4: Reduction Operations")
    print("=" * 60)

    jt = fvdb.JaggedTensor([torch.randn(100, 10, 3), torch.randn(150, 10, 3)])
    print(f"Original: {jt.jdata.shape}, num_tensors={jt.num_tensors}")

    # Reduce over non-primary dimensions (preserves jagged structure)
    summed = fvdb.sum(jt, dim=-1)
    print(f"  sum(dim=-1):  {summed.jdata.shape}")

    mean_val = fvdb.mean(jt, dim=1)
    print(f"  mean(dim=1):  {mean_val.jdata.shape}")

    max_val = fvdb.amax(jt, dim=-1)
    print(f"  amax(dim=-1): {max_val.jdata.shape}")

    print(f"\n[*] All reductions preserve jagged structure (num_tensors={summed.num_tensors})")
    print()


def example_chaining_operations():
    """Chain multiple operations together."""
    print("=" * 60)
    print("Example 5: Chaining Operations")
    print("=" * 60)

    jt = fvdb.JaggedTensor([torch.randn(100, 3) for _ in range(5)])

    # Chain operations - all type-safe!
    result = fvdb.sigmoid(fvdb.add(fvdb.relu(jt), 1.0))
    print(f"Chained: sigmoid(add(relu(jt), 1.0)) -> {type(result).__name__}")

    # Or in a more readable format:
    activated = fvdb.relu(jt)
    shifted = fvdb.add(activated, 1.0)
    normalized = fvdb.sigmoid(shifted)

    assert torch.allclose(result.jdata, normalized.jdata)
    print("[*] Chained operations work correctly")
    print()


def example_binary_operations():
    """Binary operations between JaggedTensors."""
    print("=" * 60)
    print("Example 6: Binary Operations")
    print("=" * 60)

    jt1 = fvdb.JaggedTensor([torch.randn(100, 3) for _ in range(3)])
    jt2 = fvdb.JaggedTensor([torch.randn(100, 3) for _ in range(3)])

    # All type-safe binary operations
    added = fvdb.add(jt1, jt2)
    multiplied = fvdb.mul(jt1, jt2)
    max_vals = fvdb.maximum(jt1, jt2)

    print(f"add(jt1, jt2):     {type(added).__name__}")
    print(f"mul(jt1, jt2):     {type(multiplied).__name__}")
    print(f"maximum(jt1, jt2): {type(max_vals).__name__}")

    # Comparisons
    mask = fvdb.gt(jt1, jt2)
    print(f"gt(jt1, jt2):      {type(mask).__name__} (dtype={mask.dtype})")

    # Conditional selection
    selected = fvdb.where(mask, jt1, jt2)
    print(f"where(mask, ...):  {type(selected).__name__}")
    print()


if __name__ == "__main__":
    example_basic_operations()
    example_comparison_torch_vs_fvdb()
    example_works_with_regular_tensors()
    example_reduction_operations()
    example_chaining_operations()
    example_binary_operations()

    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("  1. Use fvdb.* functions for type-safe operations")
    print("  2. They work with both JaggedTensor and regular Tensor")
    print("  3. Identical runtime behavior to torch.* functions")
    print("  4. Proper type inference in static type checkers")
