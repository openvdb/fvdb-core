# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import unittest
from typing import Any

import numpy as np
import torch
from fvdb.types import (  # Type guard functions; Type definitions; Conversion functions; Helper function
    DeviceIdentifier,
    NumericMaxRank1,
    NumericMaxRank2,
    NumericScalar,
    NumericScalarNative,
    ValueConstraint,
    is_NumericMaxRank1,
    is_NumericMaxRank2,
    is_NumericScalar,
    is_NumericScalarNative,
    resolve_device,
    to_FloatingScalar,
    to_FloatingTensorBroadcastableRank1,
    to_FloatingTensorBroadcastableRank2,
    to_IntegerScalar,
    to_IntegerTensorBroadcastableRank1,
    to_IntegerTensorBroadcastableRank2,
    to_Vec3f,
    to_Vec3fBatch,
    to_Vec3i,
    to_Vec3iBatch,
)
from parameterized import parameterized

all_devices = ["cpu", "cuda"]
all_dtypes = [torch.float32, torch.float64, torch.int32, torch.int64]


class TestTypesRedux(unittest.TestCase):
    def setUp(self):
        torch.random.manual_seed(42)
        np.random.seed(42)

    def assertSameDevice(self, a: Any, b: Any):
        if isinstance(a, torch.Tensor):
            a_device = a.device
        else:
            a_device = torch.device("cpu")

        if isinstance(b, torch.Tensor):
            b_device = b.device
        else:
            b_device = torch.device("cpu")
        self.assertEqual(a_device, b_device)

    # ========== Type Guard Tests ==========

    def test_is_NumericScalarNative(self):
        """Test is_NumericScalarNative type guard"""
        # Valid cases
        self.assertTrue(is_NumericScalarNative(1))
        self.assertTrue(is_NumericScalarNative(1.0))
        self.assertTrue(is_NumericScalarNative(np.int32(1)))
        self.assertTrue(is_NumericScalarNative(np.float64(1.0)))

        # Invalid cases
        self.assertFalse(is_NumericScalarNative([1]))
        self.assertFalse(is_NumericScalarNative(torch.tensor(1)))
        self.assertFalse(is_NumericScalarNative("1"))
        self.assertFalse(is_NumericScalarNative(None))

    def test_is_NumericScalar(self):
        """Test is_NumericScalar type guard"""
        # Valid cases - native scalars
        self.assertTrue(is_NumericScalar(1))
        self.assertTrue(is_NumericScalar(1.0))
        self.assertTrue(is_NumericScalar(np.int32(1)))

        # Valid cases - tensor scalars
        self.assertTrue(is_NumericScalar(torch.tensor(1)))
        self.assertTrue(is_NumericScalar(torch.tensor(1.0)))
        self.assertTrue(is_NumericScalar(np.array(1)))

        # Invalid cases
        self.assertFalse(is_NumericScalar([1]))
        self.assertFalse(is_NumericScalar(torch.tensor([1])))
        self.assertFalse(is_NumericScalar(torch.tensor([[1]])))

    def test_is_NumericMaxRank1(self):
        """Test is_NumericMaxRank1 type guard"""
        # Valid cases - scalars
        self.assertTrue(is_NumericMaxRank1(1))
        self.assertTrue(is_NumericMaxRank1(torch.tensor(1)))

        # Valid cases - sequences
        self.assertTrue(is_NumericMaxRank1([1, 2, 3]))
        self.assertTrue(is_NumericMaxRank1(torch.Size([1, 2, 3])))

        # Valid cases - rank 1 tensors
        self.assertTrue(is_NumericMaxRank1(torch.tensor([1, 2, 3])))
        self.assertTrue(is_NumericMaxRank1(np.array([1, 2, 3])))

        # Invalid cases
        self.assertFalse(is_NumericMaxRank1([[1, 2], [3, 4]]))
        self.assertFalse(is_NumericMaxRank1(torch.tensor([[1, 2], [3, 4]])))

    def test_is_NumericMaxRank2(self):
        """Test is_NumericMaxRank2 type guard"""
        # Valid cases - everything from rank 1
        self.assertTrue(is_NumericMaxRank2(1))
        self.assertTrue(is_NumericMaxRank2([1, 2, 3]))

        # Valid cases - rank 2 specific
        self.assertTrue(is_NumericMaxRank2([[1, 2], [3, 4]]))
        self.assertTrue(is_NumericMaxRank2(torch.tensor([[1, 2], [3, 4]])))
        self.assertTrue(is_NumericMaxRank2(np.array([[1, 2], [3, 4]])))

        # Invalid cases
        self.assertFalse(is_NumericMaxRank2([[[1]]]))
        self.assertFalse(is_NumericMaxRank2(torch.tensor([[[1]]])))

    # ========== Type Guard Functional Tests ==========

    def test_type_guard_narrowing_functionality(self):
        """Test that type guards actually enable type narrowing in practice"""

        def process_numeric_scalar(x: Any) -> float:
            """Function that demonstrates type guard narrowing"""
            if is_NumericScalar(x):
                # After type guard, x should be treated as NumericScalar
                # This would fail static type checking without the type guard
                if isinstance(x, (torch.Tensor, np.ndarray)):
                    return float(x.item())
                else:
                    return float(x)
            else:
                raise TypeError(f"Expected NumericScalar, got {type(x)}")

        # Test that the function works with various inputs
        self.assertEqual(process_numeric_scalar(1), 1.0)
        self.assertEqual(process_numeric_scalar(1.5), 1.5)
        self.assertEqual(process_numeric_scalar(torch.tensor(2.5)), 2.5)
        self.assertEqual(process_numeric_scalar(np.array(3.5)), 3.5)

        # Test that it fails appropriately
        with self.assertRaises(TypeError):
            process_numeric_scalar([1, 2, 3])

    def test_type_guard_conversion_chain(self):
        """Test using type guards in a conversion chain"""

        def smart_to_vec3i(x: Any) -> torch.Tensor:
            """Function that uses type guards to handle different input types"""
            if is_NumericScalar(x):
                # Convert scalar to Vec3i by broadcasting
                return to_Vec3i(x)
            elif is_NumericMaxRank1(x):
                # Convert rank 1 input to Vec3i
                return to_Vec3i(x)
            else:
                raise TypeError(f"Cannot convert {type(x)} to Vec3i")

        # Test scalar input
        result = smart_to_vec3i(5)
        self.assertTrue(torch.equal(result, torch.tensor([5, 5, 5])))

        # Test list input
        result = smart_to_vec3i([1, 2, 3])
        self.assertTrue(torch.equal(result, torch.tensor([1, 2, 3])))

        # Test tensor input
        result = smart_to_vec3i(torch.tensor([4, 5, 6]))
        self.assertTrue(torch.equal(result, torch.tensor([4, 5, 6])))

        # Test failure case
        with self.assertRaises(TypeError):
            smart_to_vec3i([[1, 2], [3, 4]])

    def test_type_guard_with_union_types(self):
        """Test type guards with Union types"""

        def process_mixed_input(x: int | list | torch.Tensor) -> str:
            """Function that processes different union types"""
            if is_NumericScalarNative(x):
                return f"native_scalar: {x}"
            elif is_NumericMaxRank1(x):
                return f"rank1: {x}"
            else:
                return f"unknown: {x}"

        # Test different union type inputs
        self.assertEqual(process_mixed_input(42), "native_scalar: 42")
        self.assertEqual(process_mixed_input([1, 2, 3]), "rank1: [1, 2, 3]")
        self.assertTrue(process_mixed_input(torch.tensor([1, 2])).startswith("rank1:"))

    def test_type_guard_edge_cases(self):
        """Test type guards with edge cases and boundary conditions"""

        # Empty containers
        self.assertTrue(is_NumericMaxRank1([]))
        self.assertTrue(is_NumericMaxRank2([]))

        # Single element containers
        self.assertTrue(is_NumericMaxRank1([1]))
        self.assertTrue(is_NumericMaxRank2([[1]]))

        # Zero-dimensional arrays
        self.assertTrue(is_NumericScalar(np.array(1)))
        self.assertTrue(is_NumericScalar(torch.tensor(1)))

        # Different numpy dtypes
        self.assertTrue(is_NumericScalarNative(np.int8(1)))
        self.assertTrue(is_NumericScalarNative(np.uint64(1)))
        self.assertTrue(is_NumericScalarNative(np.float16(1.0)))

    # ========== Device Resolution Tests ==========

    def test_resolve_device_explicit_device(self):
        """Test resolve_device when device_id is explicitly provided"""
        # Test explicit CPU device
        result = resolve_device("cpu")
        self.assertEqual(result, torch.device("cpu"))

        # Test explicit CUDA device (should normalize to current device)
        result = resolve_device("cuda")
        expected = torch.device("cuda", torch.cuda.current_device())
        self.assertEqual(result, expected)

        # Test explicit CUDA device with index
        result = resolve_device("cuda:0")
        self.assertEqual(result, torch.device("cuda", 0))

        # Test torch.device objects (should normalize CUDA)
        result = resolve_device(torch.device("cpu"))
        self.assertEqual(result, torch.device("cpu"))

        result = resolve_device(torch.device("cuda"))
        expected = torch.device("cuda", torch.cuda.current_device())
        self.assertEqual(result, expected)

    def test_resolve_device_inheritance(self):
        """Test resolve_device inheritance behavior when device_id is None"""
        # Test inheriting from torch tensor
        tensor_cpu = torch.tensor([1, 2, 3], device="cpu")
        result = resolve_device(None, inherit_from=tensor_cpu)
        self.assertEqual(result, torch.device("cpu"))

        # Test inheriting from torch tensor on CUDA (if available)
        if torch.cuda.is_available():
            tensor_cuda = torch.tensor([1, 2, 3], device="cuda:0")
            result = resolve_device(None, inherit_from=tensor_cuda)
            self.assertEqual(result, torch.device("cuda", 0))

        # Test inheriting from Python objects (should default to CPU)
        result = resolve_device(None, inherit_from=[1, 2, 3])
        self.assertEqual(result, torch.device("cpu"))

        result = resolve_device(None, inherit_from=42)
        self.assertEqual(result, torch.device("cpu"))

        # Test inheriting from numpy arrays (should default to CPU)
        np_array = np.array([1, 2, 3])
        result = resolve_device(None, inherit_from=np_array)
        self.assertEqual(result, torch.device("cpu"))

        # Test with None inherit_from (should default to CPU)
        result = resolve_device(None, inherit_from=None)
        self.assertEqual(result, torch.device("cpu"))

        # Test with no inherit_from argument (should default to CPU)
        result = resolve_device(None)
        self.assertEqual(result, torch.device("cpu"))

    def test_resolve_device_precedence(self):
        """Test that device_id takes precedence over inherit_from"""
        # Create a tensor on CPU
        tensor_cpu = torch.tensor([1, 2, 3], device="cpu")

        # Even though tensor is on CPU, explicit device_id should take precedence
        result = resolve_device("cpu", inherit_from=tensor_cpu)
        self.assertEqual(result, torch.device("cpu"))

        # Test with CUDA if available
        if torch.cuda.is_available():
            # Tensor on CUDA, but explicit CPU device should take precedence
            tensor_cuda = torch.tensor([1, 2, 3], device="cuda:0")
            result = resolve_device("cpu", inherit_from=tensor_cuda)
            self.assertEqual(result, torch.device("cpu"))

            # CPU tensor, but explicit CUDA device should take precedence
            result = resolve_device("cuda:0", inherit_from=tensor_cpu)
            self.assertEqual(result, torch.device("cuda", 0))

        # Test with non-torch objects (device_id should still take precedence)
        result = resolve_device("cpu", inherit_from=[1, 2, 3])
        self.assertEqual(result, torch.device("cpu"))

    def test_resolve_device_normalization(self):
        """Test that resolve_device properly normalizes device specifications"""
        # Test CUDA normalization (should add explicit index)
        result = resolve_device("cuda")
        expected = torch.device("cuda", torch.cuda.current_device())
        self.assertEqual(result, expected)

        # Test that CPU doesn't get an index
        result = resolve_device("cpu")
        self.assertEqual(result, torch.device("cpu"))
        self.assertIsNone(result.index)

        # Test that explicit CUDA index is preserved
        result = resolve_device("cuda:0")
        self.assertEqual(result, torch.device("cuda", 0))

        # Test that torch.device objects are normalized
        result = resolve_device(torch.device("cuda"))
        expected = torch.device("cuda", torch.cuda.current_device())
        self.assertEqual(result, expected)

    def test_resolve_device_tensor_device_preservation(self):
        """Test that tensor devices are preserved without normalization when inherited"""
        # Create a tensor and verify its device is preserved as-is when inherited
        tensor_cpu = torch.tensor([1, 2, 3], device="cpu")
        result = resolve_device(None, inherit_from=tensor_cpu)
        # Should preserve the exact device from the tensor
        self.assertEqual(result, tensor_cpu.device)

        if torch.cuda.is_available():
            tensor_cuda = torch.tensor([1, 2, 3], device="cuda:0")
            result = resolve_device(None, inherit_from=tensor_cuda)
            # Should preserve the exact device from the tensor
            self.assertEqual(result, tensor_cuda.device)

            # Test that when we inherit, we get the exact device without normalization
            # Create a tensor with a specific device
            specific_device = torch.device("cuda", 0)
            tensor_specific = torch.tensor([1, 2, 3], device=specific_device)
            result = resolve_device(None, inherit_from=tensor_specific)
            self.assertEqual(result, specific_device)

    def test_resolve_device_error_cases(self):
        """Test resolve_device error handling"""
        # Test invalid device_id types
        with self.assertRaises(TypeError):
            resolve_device(123)  # type: ignore

        with self.assertRaises(TypeError):
            resolve_device([1, 2, 3])  # type: ignore

    # ========== Scalar Conversion Tests ==========

    @parameterized.expand(all_devices)
    def test_to_IntegerScalar(self, device):
        """Test to_IntegerScalar conversion"""
        # Test valid integer input types
        valid_inputs = [
            1,
            np.int32(1),
            torch.tensor(1, dtype=torch.int32, device=device),
            torch.tensor(1, dtype=torch.int64, device=device),
        ]

        for inp in valid_inputs:
            result = to_IntegerScalar(inp)
            self.assertSameDevice(result, inp)
            self.assertEqual(result.shape, torch.Size([]))
            self.assertTrue(result.dtype in [torch.int32, torch.int64])
            self.assertEqual(result.item(), 1)

        # Test that floating point inputs fail
        invalid_inputs = [1.0, np.float32(1.0), np.array(1.0), torch.tensor(1.0, device=device)]
        for inp in invalid_inputs:
            with self.assertRaises(Exception):
                result = to_IntegerScalar(inp)

    @parameterized.expand(all_devices)
    def test_to_FloatingScalar(self, device):
        """Test to_FloatingScalar conversion"""
        # Test various input types
        inputs = [1, 1.5, np.float32(1.5), torch.tensor(1.5, device=device)]

        # Integers and floats both upcast to floats.
        for inp in inputs:
            result = to_FloatingScalar(inp)
            self.assertSameDevice(result, inp)
            self.assertEqual(result.shape, torch.Size([]))
            if inp is not 1:
                self.assertAlmostEqual(result.item(), 1.5, places=5)

    def test_scalar_conversion_failures(self):
        """Test scalar conversion failure cases"""
        # Non-scalar tensors should fail
        with self.assertRaises(Exception):
            to_IntegerScalar(torch.tensor([1, 2]))

        with self.assertRaises(Exception):
            to_FloatingScalar(np.array([1, 2]))

        # Invalid dtypes
        with self.assertRaises(ValueError):
            to_IntegerScalar(1, dtype=torch.float32)

        # Invalid input types
        with self.assertRaises(TypeError):
            to_IntegerScalar("not a number")  # type: ignore

    # ========== Tensor Broadcasting Tests ==========

    def test_to_IntegerTensorBroadcastableRank1(self):
        """Test rank 1 integer tensor broadcasting"""
        # Test scalar to broadcast
        result = to_IntegerTensorBroadcastableRank1(5, (3,))
        self.assertEqual(result.shape, torch.Size([]))

        # Test list input
        result = to_IntegerTensorBroadcastableRank1([1, 2, 3], (3,))
        self.assertEqual(result.shape, torch.Size([3]))
        self.assertTrue(torch.equal(result, torch.tensor([1, 2, 3], dtype=result.dtype)))

        # Test broadcasting failure
        with self.assertRaises(ValueError):
            to_IntegerTensorBroadcastableRank1([1, 2], (3,))

    def test_to_FloatingTensorBroadcastableRank2(self):
        """Test rank 2 floating tensor broadcasting"""
        # Test scalar input
        result = to_FloatingTensorBroadcastableRank2(1.5, (2, 3))
        self.assertEqual(result.shape, torch.Size([]))

        # Test 2D list input
        input_data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        result = to_FloatingTensorBroadcastableRank2(input_data, (2, 3))
        self.assertEqual(result.shape, torch.Size([2, 3]))

    def test_tensor_broadcasting_failures(self):
        """Test tensor broadcasting failure cases"""
        # Wrong rank for test_shape
        with self.assertRaises(ValueError):
            to_IntegerTensorBroadcastableRank1(1, (2, 3))  # type: ignore

        # Incompatible shapes
        with self.assertRaises(ValueError):
            to_IntegerTensorBroadcastableRank1([1, 2], (3,))

        # Invalid dtypes
        with self.assertRaises(ValueError):
            to_IntegerTensorBroadcastableRank1(1, (3,), dtype=torch.float32)

    # ========== Vec3 Conversion Tests ==========

    def test_to_Vec3i(self):
        """Test Vec3i conversion"""
        # Test scalar input (should broadcast)
        result = to_Vec3i(5)
        self.assertEqual(result.shape, torch.Size([3]))
        self.assertTrue(torch.equal(result, torch.tensor([5, 5, 5], dtype=result.dtype)))

        # Test list input
        result = to_Vec3i([1, 2, 3])
        self.assertEqual(result.shape, torch.Size([3]))
        self.assertTrue(torch.equal(result, torch.tensor([1, 2, 3], dtype=result.dtype)))

        # Test failure case
        with self.assertRaises(ValueError):
            to_Vec3i([1, 2])

    def test_to_Vec3f(self):
        """Test Vec3f conversion"""
        # Test scalar input
        result = to_Vec3f(1.5)
        self.assertEqual(result.shape, torch.Size([3]))
        expected = torch.tensor([1.5, 1.5, 1.5], dtype=result.dtype)
        self.assertTrue(torch.allclose(result, expected))

        # Test list input
        result = to_Vec3f([1.0, 2.5, 3.0])
        self.assertEqual(result.shape, torch.Size([3]))
        expected = torch.tensor([1.0, 2.5, 3.0], dtype=result.dtype)
        self.assertTrue(torch.allclose(result, expected))

    def test_to_Vec3iBatch(self):
        """Test Vec3iBatch conversion"""
        # Test scalar input (should create batch of size 1)
        result = to_Vec3iBatch(5)
        self.assertEqual(result.shape, torch.Size([1, 3]))
        expected = torch.tensor([[5, 5, 5]], dtype=result.dtype)
        self.assertTrue(torch.equal(result, expected))

        # Test 2D list input
        input_data = [[1, 2, 3], [4, 5, 6]]
        result = to_Vec3iBatch(input_data)
        self.assertEqual(result.shape, torch.Size([2, 3]))
        expected = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=result.dtype)
        self.assertTrue(torch.equal(result, expected))

    def test_to_Vec3fBatch(self):
        """Test Vec3fBatch conversion"""
        # Test scalar input
        result = to_Vec3fBatch(1.5)
        self.assertEqual(result.shape, torch.Size([1, 3]))
        expected = torch.tensor([[1.5, 1.5, 1.5]], dtype=result.dtype)
        self.assertTrue(torch.allclose(result, expected))

        # Test 2D list input
        input_data = [[1.0, 2.0, 3.0], [4.5, 5.5, 6.5]]
        result = to_Vec3fBatch(input_data)
        self.assertEqual(result.shape, torch.Size([2, 3]))

    def test_vec3_conversion_failures(self):
        """Test Vec3 conversion failure cases"""
        # Wrong size for Vec3
        with self.assertRaises(ValueError):
            to_Vec3i([1, 2, 3, 4])

        with self.assertRaises(ValueError):
            to_Vec3f([1, 2])

        # Invalid rank for Vec3iBatch
        with self.assertRaises(TypeError):
            to_Vec3iBatch([[[1, 2, 3]]])  # type: ignore

    # ========== Positive Vec3 Conversion Tests ==========

    def test_to_PositiveVec3i(self):
        """Test PositiveVec3i conversion with valid positive values"""
        # Test scalar input (should broadcast)
        result = to_Vec3i(5, value_constraint=ValueConstraint.POSITIVE)
        self.assertEqual(result.shape, torch.Size([3]))
        self.assertTrue(torch.equal(result, torch.tensor([5, 5, 5], dtype=result.dtype)))

        # Test list input with positive values
        result = to_Vec3i([1, 2, 3], value_constraint=ValueConstraint.POSITIVE)
        self.assertEqual(result.shape, torch.Size([3]))
        self.assertTrue(torch.equal(result, torch.tensor([1, 2, 3], dtype=result.dtype)))

        # Test tensor input with positive values
        input_tensor = torch.tensor([10, 20, 30], dtype=torch.int32)
        result = to_Vec3i(input_tensor, value_constraint=ValueConstraint.POSITIVE)
        self.assertEqual(result.shape, torch.Size([3]))
        self.assertTrue(torch.equal(result, torch.tensor([10, 20, 30], dtype=result.dtype)))

    def test_to_PositiveVec3f(self):
        """Test PositiveVec3f conversion with valid positive values"""
        # Test scalar input
        result = to_Vec3f(1.5, value_constraint=ValueConstraint.POSITIVE)
        self.assertEqual(result.shape, torch.Size([3]))
        expected = torch.tensor([1.5, 1.5, 1.5], dtype=result.dtype)
        self.assertTrue(torch.allclose(result, expected))

        # Test list input with positive values
        result = to_Vec3f([1.0, 2.5, 3.0], value_constraint=ValueConstraint.POSITIVE)
        self.assertEqual(result.shape, torch.Size([3]))
        expected = torch.tensor([1.0, 2.5, 3.0], dtype=result.dtype)
        self.assertTrue(torch.allclose(result, expected))

        # Test very small positive values
        result = to_Vec3f([0.001, 0.002, 0.003], value_constraint=ValueConstraint.POSITIVE)
        self.assertEqual(result.shape, torch.Size([3]))
        expected = torch.tensor([0.001, 0.002, 0.003], dtype=result.dtype)
        self.assertTrue(torch.allclose(result, expected))

    def test_to_PositiveVec3iBatch(self):
        """Test PositiveVec3iBatch conversion with valid positive values"""
        # Test scalar input (should create batch of size 1)
        result = to_Vec3iBatch(5, value_constraint=ValueConstraint.POSITIVE)
        self.assertEqual(result.shape, torch.Size([1, 3]))
        expected = torch.tensor([[5, 5, 5]], dtype=result.dtype)
        self.assertTrue(torch.equal(result, expected))

        # Test 2D list input with positive values
        input_data = [[1, 2, 3], [4, 5, 6]]
        result = to_Vec3iBatch(input_data, value_constraint=ValueConstraint.POSITIVE)
        self.assertEqual(result.shape, torch.Size([2, 3]))
        expected = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=result.dtype)
        self.assertTrue(torch.equal(result, expected))

        # Test with large positive values
        input_data = [[100, 200, 300], [400, 500, 600]]
        result = to_Vec3iBatch(input_data, value_constraint=ValueConstraint.POSITIVE)
        self.assertEqual(result.shape, torch.Size([2, 3]))
        expected = torch.tensor([[100, 200, 300], [400, 500, 600]], dtype=result.dtype)
        self.assertTrue(torch.equal(result, expected))

    def test_to_PositiveVec3fBatch(self):
        """Test PositiveVec3fBatch conversion with valid positive values"""
        # Test scalar input
        result = to_Vec3fBatch(1.5, value_constraint=ValueConstraint.POSITIVE)
        self.assertEqual(result.shape, torch.Size([1, 3]))
        expected = torch.tensor([[1.5, 1.5, 1.5]], dtype=result.dtype)
        self.assertTrue(torch.allclose(result, expected))

        # Test 2D list input with positive values
        input_data = [[1.0, 2.0, 3.0], [4.5, 5.5, 6.5]]
        result = to_Vec3fBatch(input_data, value_constraint=ValueConstraint.POSITIVE)
        self.assertEqual(result.shape, torch.Size([2, 3]))
        expected = torch.tensor([[1.0, 2.0, 3.0], [4.5, 5.5, 6.5]], dtype=result.dtype)
        self.assertTrue(torch.allclose(result, expected))

        # Test with very small positive values
        input_data = [[0.1, 0.2, 0.3], [0.001, 0.002, 0.003]]
        result = to_Vec3fBatch(input_data, value_constraint=ValueConstraint.POSITIVE)
        self.assertEqual(result.shape, torch.Size([2, 3]))
        expected = torch.tensor([[0.1, 0.2, 0.3], [0.001, 0.002, 0.003]], dtype=result.dtype)
        self.assertTrue(torch.allclose(result, expected))

    def test_positive_vec3_conversion_failures(self):
        """Test Positive Vec3 conversion failure cases"""
        # Test zero values - should fail
        with self.assertRaises(ValueError):
            to_Vec3i([0, 1, 2], value_constraint=ValueConstraint.POSITIVE)

        with self.assertRaises(ValueError):
            to_Vec3f([1.0, 0.0, 2.0], value_constraint=ValueConstraint.POSITIVE)

        # Test negative values - should fail
        with self.assertRaises(ValueError):
            to_Vec3i([-1, 2, 3], value_constraint=ValueConstraint.POSITIVE)

        with self.assertRaises(ValueError):
            to_Vec3f([1.0, -1.0, 2.0], value_constraint=ValueConstraint.POSITIVE)

        # Test scalar zero - should fail
        with self.assertRaises(ValueError):
            to_Vec3i(0, value_constraint=ValueConstraint.POSITIVE)

        with self.assertRaises(ValueError):
            to_Vec3f(0.0, value_constraint=ValueConstraint.POSITIVE)

        # Test scalar negative - should fail
        with self.assertRaises(ValueError):
            to_Vec3i(-5, value_constraint=ValueConstraint.POSITIVE)

        with self.assertRaises(ValueError):
            to_Vec3f(-1.5, value_constraint=ValueConstraint.POSITIVE)

        # Test batch with zero values - should fail
        with self.assertRaises(ValueError):
            to_Vec3iBatch([[1, 2, 3], [0, 5, 6]], value_constraint=ValueConstraint.POSITIVE)

        with self.assertRaises(ValueError):
            to_Vec3fBatch([[1.0, 2.0, 3.0], [4.0, 0.0, 6.0]], value_constraint=ValueConstraint.POSITIVE)

        # Test batch with negative values - should fail
        with self.assertRaises(ValueError):
            to_Vec3iBatch([[1, 2, 3], [-1, 5, 6]], value_constraint=ValueConstraint.POSITIVE)

        with self.assertRaises(ValueError):
            to_Vec3fBatch([[1.0, 2.0, 3.0], [4.0, -1.0, 6.0]], value_constraint=ValueConstraint.POSITIVE)

        # Test mixed positive and non-positive values - should fail
        with self.assertRaises(ValueError):
            to_Vec3i([1, 0, 3], value_constraint=ValueConstraint.POSITIVE)

        with self.assertRaises(ValueError):
            to_Vec3f([1.0, 2.0, -0.1], value_constraint=ValueConstraint.POSITIVE)

    def test_positive_vec3_edge_cases(self):
        """Test Positive Vec3 conversion edge cases"""
        # Test very small positive values (should succeed)
        result = to_Vec3f([1e-6, 1e-5, 1e-4], value_constraint=ValueConstraint.POSITIVE)
        self.assertEqual(result.shape, torch.Size([3]))
        self.assertTrue(torch.all(result > 0))

        # Test tensor inputs with positive values
        input_tensor = torch.tensor([1, 2, 3], dtype=torch.int64)
        result = to_Vec3i(input_tensor, value_constraint=ValueConstraint.POSITIVE)
        self.assertEqual(result.shape, torch.Size([3]))
        self.assertTrue(torch.all(result > 0))

        # Test numpy inputs with positive values
        np_input = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = to_Vec3f(np_input, value_constraint=ValueConstraint.POSITIVE)
        self.assertEqual(result.shape, torch.Size([3]))
        self.assertTrue(torch.all(result > 0))

        # Test that error messages include the problematic tensor
        with self.assertRaises(ValueError) as context:
            to_Vec3i([-1, 2, 3], value_constraint=ValueConstraint.POSITIVE)
        self.assertIn("Expected positive values", str(context.exception))

    def test_positive_vec3_dtype_and_device_handling(self):
        """Test that positive Vec3 functions handle dtype and device correctly"""
        # Test custom dtype
        result = to_Vec3i([1, 2, 3], dtype=torch.int32, value_constraint=ValueConstraint.POSITIVE)
        self.assertEqual(result.dtype, torch.int32)

        result = to_Vec3f([1.0, 2.0, 3.0], dtype=torch.float64, value_constraint=ValueConstraint.POSITIVE)
        self.assertEqual(result.dtype, torch.float64)

        # Test device specification
        result = to_Vec3i([1, 2, 3], value_constraint=ValueConstraint.POSITIVE)
        self.assertEqual(result.device.type, "cpu")

        result = to_Vec3f([1.0, 2.0, 3.0], value_constraint=ValueConstraint.POSITIVE)
        self.assertEqual(result.device.type, "cpu")

        # Test batch versions
        result = to_Vec3iBatch([[1, 2, 3]], dtype=torch.int32, value_constraint=ValueConstraint.POSITIVE)
        self.assertEqual(result.dtype, torch.int32)

        result = to_Vec3fBatch([[1.0, 2.0, 3.0]], dtype=torch.float64, value_constraint=ValueConstraint.POSITIVE)
        self.assertEqual(result.dtype, torch.float64)

    # ========== NonNegative Vec3 Conversion Tests ==========

    def test_to_NonNegativeVec3i(self):
        """Test NonNegativeVec3i conversion - allows zero, rejects negative"""
        # Test positive values (should work)
        result = to_Vec3i([1, 2, 3], value_constraint=ValueConstraint.NON_NEGATIVE)
        self.assertTrue(torch.equal(result, torch.tensor([1, 2, 3], dtype=result.dtype)))

        # Test zero values (should work - key difference from Positive)
        result = to_Vec3i([0, 1, 2], value_constraint=ValueConstraint.NON_NEGATIVE)
        self.assertTrue(torch.equal(result, torch.tensor([0, 1, 2], dtype=result.dtype)))

        # Test scalar zero (should work)
        result = to_Vec3i(0, value_constraint=ValueConstraint.NON_NEGATIVE)
        self.assertTrue(torch.equal(result, torch.tensor([0, 0, 0], dtype=result.dtype)))

        # Test negative values (should fail)
        with self.assertRaises(ValueError):
            to_Vec3i([-1, 2, 3], value_constraint=ValueConstraint.NON_NEGATIVE)

    def test_to_NonNegativeVec3f(self):
        """Test NonNegativeVec3f conversion - allows zero, rejects negative"""
        # Test positive values
        result = to_Vec3f([1.0, 2.0, 3.0], value_constraint=ValueConstraint.NON_NEGATIVE)
        expected = torch.tensor([1.0, 2.0, 3.0], dtype=result.dtype)
        self.assertTrue(torch.allclose(result, expected))

        # Test zero values (should work)
        result = to_Vec3f([0.0, 1.0, 2.0], value_constraint=ValueConstraint.NON_NEGATIVE)
        expected = torch.tensor([0.0, 1.0, 2.0], dtype=result.dtype)
        self.assertTrue(torch.allclose(result, expected))

        # Test scalar zero (should work)
        result = to_Vec3f(0.0, value_constraint=ValueConstraint.NON_NEGATIVE)
        expected = torch.tensor([0.0, 0.0, 0.0], dtype=result.dtype)
        self.assertTrue(torch.allclose(result, expected))

        # Test negative values (should fail)
        with self.assertRaises(ValueError):
            to_Vec3f([1.0, -1.0, 2.0], value_constraint=ValueConstraint.NON_NEGATIVE)

    def test_to_NonNegativeVec3iBatch(self):
        """Test NonNegativeVec3iBatch conversion - allows zero, rejects negative"""
        # Test positive values
        input_data = [[1, 2, 3], [4, 5, 6]]
        result = to_Vec3iBatch(input_data, value_constraint=ValueConstraint.NON_NEGATIVE)
        expected = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=result.dtype)
        self.assertTrue(torch.equal(result, expected))

        # Test with zero values (should work)
        input_data = [[0, 1, 2], [3, 0, 5]]
        result = to_Vec3iBatch(input_data, value_constraint=ValueConstraint.NON_NEGATIVE)
        expected = torch.tensor([[0, 1, 2], [3, 0, 5]], dtype=result.dtype)
        self.assertTrue(torch.equal(result, expected))

        # Test negative values (should fail)
        with self.assertRaises(ValueError):
            to_Vec3iBatch([[1, 2, 3], [-1, 5, 6]], value_constraint=ValueConstraint.NON_NEGATIVE)

    def test_to_NonNegativeVec3fBatch(self):
        """Test NonNegativeVec3fBatch conversion - allows zero, rejects negative"""
        # Test positive values
        input_data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        result = to_Vec3fBatch(input_data, value_constraint=ValueConstraint.NON_NEGATIVE)
        expected = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=result.dtype)
        self.assertTrue(torch.allclose(result, expected))

        # Test with zero values (should work)
        input_data = [[0.0, 1.0, 2.0], [3.0, 0.0, 5.0]]
        result = to_Vec3fBatch(input_data, value_constraint=ValueConstraint.NON_NEGATIVE)
        expected = torch.tensor([[0.0, 1.0, 2.0], [3.0, 0.0, 5.0]], dtype=result.dtype)
        self.assertTrue(torch.allclose(result, expected))

        # Test negative values (should fail)
        with self.assertRaises(ValueError):
            to_Vec3fBatch([[1.0, 2.0, 3.0], [4.0, -1.0, 6.0]], value_constraint=ValueConstraint.NON_NEGATIVE)

    def test_nonnegative_vs_positive_comparison(self):
        """Test key differences between NonNegative and Positive variants"""
        # Zero should work for NonNegative but fail for Positive
        to_Vec3i([0, 1, 2], value_constraint=ValueConstraint.NON_NEGATIVE)  # Should work
        with self.assertRaises(ValueError):
            to_Vec3i([0, 1, 2], value_constraint=ValueConstraint.POSITIVE)  # Should fail

        to_Vec3f([0.0, 1.0, 2.0], value_constraint=ValueConstraint.NON_NEGATIVE)  # Should work
        with self.assertRaises(ValueError):
            to_Vec3f([0.0, 1.0, 2.0], value_constraint=ValueConstraint.POSITIVE)  # Should fail

        # Negative should fail for both
        with self.assertRaises(ValueError):
            to_Vec3i([-1, 2, 3], value_constraint=ValueConstraint.NON_NEGATIVE)
        with self.assertRaises(ValueError):
            to_Vec3i([-1, 2, 3], value_constraint=ValueConstraint.POSITIVE)

    # ========== Dtype and Device Handling Tests ==========

    def test_dtype_inheritance(self):
        """Test that input tensor dtypes are properly handled"""
        # Integer tensor should work with integer conversion
        int_tensor = torch.tensor([1, 2, 3], dtype=torch.int32)
        result = to_IntegerTensorBroadcastableRank1(int_tensor, (3,))
        self.assertTrue(result.dtype in [torch.int32, torch.int64])

        # Float tensor should work with float conversion
        float_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        result = to_FloatingTensorBroadcastableRank1(float_tensor, (3,))
        self.assertTrue(result.dtype in [torch.float16, torch.float32, torch.float64])

    @parameterized.expand(all_devices)
    def test_device_inheritance(self, device):
        """Test device inheritance from input tensors"""
        # Test tensor should inherit device when device=None
        input_tensor = torch.tensor([1, 2, 3], device=device)
        result = to_IntegerTensorBroadcastableRank1(input_tensor, (3,))
        self.assertSameDevice(result, input_tensor)

    def test_numpy_compatibility(self):
        """Test numpy array compatibility"""
        # Numpy arrays should be converted properly
        np_array = np.array([1, 2, 3], dtype=np.int32)
        result = to_IntegerTensorBroadcastableRank1(np_array, (3,))
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, torch.Size([3]))
        self.assertEqual(result.device.type, "cpu")

        # 2D numpy arrays
        np_array_2d = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        result = to_FloatingTensorBroadcastableRank2(np_array_2d, (2, 3))
        self.assertEqual(result.shape, torch.Size([2, 3]))
        self.assertEqual(result.device.type, "cpu")

    def test_torch_size_compatibility(self):
        """Test torch.Size compatibility"""
        size = torch.Size([1, 2, 3])
        result = to_IntegerTensorBroadcastableRank1(size, (3,))
        self.assertEqual(result.shape, torch.Size([3]))
        self.assertEqual(result.device.type, "cpu")
        self.assertTrue(torch.equal(result, torch.tensor([1, 2, 3], dtype=result.dtype)))


if __name__ == "__main__":
    unittest.main()
