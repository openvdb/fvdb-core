# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""
Test the PyTorch ground truth for transposed convolutions (conv_transpose3d) in 3D.

These tests validate that PyTorch's dense conv_transpose3d behaves as we expect,
establishing baseline understanding of:
  - Transposed convolution semantics (adjoint of cross-correlation)
  - Impulse response behavior (forward and backward)
  - Strided transposed convolution coordinate mapping (upsampling)
  - Gradient computation for inputs and weights

This file does NOT test fVDB sparse transposed convolution - it validates our
understanding of PyTorch semantics that we will use as ground truth.

=============================================================================
THEORY: Transposed Convolution vs Regular Convolution
=============================================================================

Regular conv3d (cross-correlation):
    output[o] = sum_k input[o + k - K//2] * kernel[k]

    With an impulse at input coordinate c:
    - output[c - (k - K//2)] = kernel[k]
    - The output is the FLIPPED kernel centered at c

Transposed conv3d (adjoint of cross-correlation):
    output[o] = sum_k input[(o - k + K//2) / S] * kernel[k]  (when (o - k + K//2) % S == 0)

    With an impulse at input coordinate c (stride=1):
    - output[c + (k - K//2)] = kernel[k]
    - The output is the ORIGINAL kernel (not flipped) centered at c

Key insight: conv_transpose3d is the gradient of conv3d with respect to input.
If we run conv3d forward and then backward, the backward pass w.r.t. input
uses conv_transpose3d with the flipped kernel. Since cross-correlation already
uses the flipped kernel implicitly, the conv_transpose produces the original
kernel pattern.

=============================================================================
STRIDED TRANSPOSED CONVOLUTION
=============================================================================

For strided conv_transpose3d with stride=S:
    - Input at coordinate c produces outputs at coordinates:
      c*S + (k - K//2) for each kernel position k

    - This is "upsampling" - each input voxel contributes to S^3 output regions

    - The output size is larger than input: output_size = (input_size - 1) * S + K

Compare to strided conv3d (downsampling):
    - Output at coordinate o receives input from: o*S + (k - K//2)
    - This is "downsampling" - multiple input voxels contribute to each output

The relationship: strided conv_transpose is the adjoint of strided conv3d.
"""

import unittest

import torch
from fvdb.types import DeviceIdentifier, resolve_device
from fvdb.utils.tests import (
    fourier_anti_symmetric_kernel,
    generate_hermit_impulses_dense,
    has_any_symmetry,
)
from fvdb.utils.tests.convolution_utils import (
    ALL_DEVICE_DTYPE_COMBOS,
    REDUCED_DEVICE_DTYPE_COMBOS,
    disable_tf32,
    get_tolerances,
)
from parameterized import parameterized


def _validate_impulse_conv_transpose(
    impulse_coord: torch.Tensor,
    kernel: torch.Tensor,
    output: torch.Tensor,
    kernel_size: tuple[int, int, int],
    test_case: unittest.TestCase,
    check_bounds: bool,
    stride: tuple[int, int, int] = (1, 1, 1),
) -> None:
    """
    Validate that transposed convolution of an impulse produces the expected kernel pattern.

    Unlike regular conv3d (cross-correlation), conv_transpose3d produces the
    ORIGINAL kernel (not flipped) centered at the impulse location (scaled by stride).

    For stride > 1, the output region is:
        center = impulse_coord * stride
        region = [center - K//2, center + K//2]

    Args:
        impulse_coord: Location of the impulse in the input volume (unscaled)
        kernel: Original kernel used for transposed convolution
        output: Result of conv_transpose3d operation
        kernel_size: Kernel dimensions (k0, k1, k2)
        test_case: TestCase for assertions
        check_bounds: If True, verify the non-zero region exactly matches expected bounds
        stride: Stride values for each dimension
    """
    kernel_half = tuple(k // 2 for k in kernel_size)

    # For strided conv_transpose, output center is at impulse_coord * stride
    output_center = tuple(int(impulse_coord[i].item()) * stride[i] for i in range(3))

    # Region where kernel response should appear
    start_coords = tuple(max(0, output_center[i] - kernel_half[i]) for i in range(3))
    end_coords = tuple(min(output.shape[i + 1], output_center[i] + kernel_half[i] + 1) for i in range(3))

    if check_bounds:
        non_zero_mask = output[0] != 0
        non_zero_coords = torch.nonzero(non_zero_mask)

        if non_zero_coords.shape[0] > 0:
            actual_start = tuple(non_zero_coords[:, dim].min().item() for dim in range(3))
            actual_end = tuple(non_zero_coords[:, dim].max().item() + 1 for dim in range(3))
            test_case.assertEqual(actual_start, start_coords)
            test_case.assertEqual(actual_end, end_coords)

    output_region = output[
        0, start_coords[0] : end_coords[0], start_coords[1] : end_coords[1], start_coords[2] : end_coords[2]
    ]
    test_case.assertEqual(output_region.shape, kernel.shape)

    # KEY DIFFERENCE FROM REGULAR CONV:
    # conv_transpose3d produces the ORIGINAL kernel (not flipped)
    # because it's the adjoint of cross-correlation
    tols = get_tolerances(kernel.dtype)
    torch.testing.assert_close(output_region, kernel, rtol=tols["forward"][0], atol=tols["forward"][1])


class TestConvTransposeGroundTruth(unittest.TestCase):
    """
    Validate PyTorch conv_transpose3d behavior that we rely on as ground truth.

    These tests verify our understanding of:
    - Transposed convolution semantics (impulse -> original kernel, not flipped)
    - Backward pass gradient computation
    - Strided transposed convolution coordinate mapping (upsampling)
    """

    VOLUME_SHAPE = (71, 34, 58)
    KERNEL_SIZE = (3, 5, 7)
    SINGLE_VOLUME_SHAPE = (5, 7, 9)
    SINGLE_COORD = (2, 3, 4)
    NUM_CANDIDATES = 1000

    def setUp(self):
        torch.random.manual_seed(2024)

    # =========================================================================
    # Forward Pass Tests
    # =========================================================================

    @parameterized.expand(ALL_DEVICE_DTYPE_COMBOS)
    def test_single_impulse(self, device: DeviceIdentifier, dtype: torch.dtype):
        """
        Test that conv_transpose3d of an impulse produces the original kernel.

        This is the KEY difference from regular conv3d:
        - conv3d (cross-correlation): impulse -> FLIPPED kernel
        - conv_transpose3d: impulse -> ORIGINAL kernel

        This happens because conv_transpose3d is the adjoint of conv3d.
        """
        device = resolve_device(device)

        expected_volume_shape = tuple(a + 2 for a in self.KERNEL_SIZE)
        expected_impulse_coord = tuple(1 + a // 2 for a in self.KERNEL_SIZE)

        self.assertEqual(expected_volume_shape, self.SINGLE_VOLUME_SHAPE)
        self.assertEqual(expected_impulse_coord, self.SINGLE_COORD)

        coord = torch.tensor(self.SINGLE_COORD, device=device, dtype=torch.int32)
        impulse_field = torch.zeros((1,) + self.SINGLE_VOLUME_SHAPE, device=device, dtype=dtype)

        impulse_field[0, coord[0], coord[1], coord[2]] = 1

        self.assertEqual(impulse_field.sum().item(), 1)

        kernel = fourier_anti_symmetric_kernel(self.KERNEL_SIZE, dtype=dtype, device=device)
        self.assertFalse(has_any_symmetry(kernel))

        kernel_with_channels = kernel.reshape(1, 1, *self.KERNEL_SIZE)

        # Do a transposed convolution with same padding
        with disable_tf32():
            output = torch.nn.functional.conv_transpose3d(
                input=impulse_field, weight=kernel_with_channels, padding=tuple(k // 2 for k in self.KERNEL_SIZE)
            )
        self.assertEqual(impulse_field.shape, output.shape)

        # Validate using the helper function
        _validate_impulse_conv_transpose(
            impulse_coord=coord,
            kernel=kernel,
            output=output,
            kernel_size=self.KERNEL_SIZE,
            test_case=self,
            check_bounds=True,
        )

    @parameterized.expand(REDUCED_DEVICE_DTYPE_COMBOS)
    def test_single_impulse_activation_and_weights(self, device: DeviceIdentifier, dtype: torch.dtype):
        """
        Test the output coordinate for each kernel weight position in conv_transpose3d.

        For conv_transpose3d with an impulse at coordinate c and kernel impulse at (k0, k1, k2):
            output_coord = c + (kernel_idx - kernel_half)

        This is OPPOSITE to regular conv3d where:
            output_coord = c - (kernel_idx - kernel_half)

        This relationship is key to understanding why conv_transpose "upsamples".
        """
        device = resolve_device(device)

        coord = torch.tensor(self.SINGLE_COORD, device=device, dtype=torch.int32)
        impulse_field = torch.zeros((1,) + self.SINGLE_VOLUME_SHAPE, device=device, dtype=dtype)

        impulse_field[0, coord[0], coord[1], coord[2]] = 1

        self.assertEqual(impulse_field.sum().item(), 1)

        kernel_half_width = tuple(k // 2 for k in self.KERNEL_SIZE)

        for k0 in range(self.KERNEL_SIZE[0]):
            for k1 in range(self.KERNEL_SIZE[1]):
                for k2 in range(self.KERNEL_SIZE[2]):
                    # Create a kernel with a single impulse at the current location
                    weights = torch.zeros((1, 1, *self.KERNEL_SIZE), device=device, dtype=dtype)
                    weights[0, 0, k0, k1, k2] = 1
                    self.assertEqual(weights.sum().item(), 1)

                    output = torch.nn.functional.conv_transpose3d(
                        input=impulse_field,
                        weight=weights,
                        stride=1,
                        padding=kernel_half_width,
                    )
                    self.assertEqual(impulse_field.shape, output.shape)
                    self.assertEqual(output.sum().item(), 1)

                    # Find the output coordinate
                    nonzero_coords = torch.nonzero(output[0])
                    self.assertEqual(nonzero_coords.shape[0], 1)

                    # KEY DIFFERENCE: For conv_transpose, output_coord = c + centered_kernel_coord
                    # (not c - centered_kernel_coord as in regular conv)
                    centered_kernel_coord = (
                        k0 - kernel_half_width[0],
                        k1 - kernel_half_width[1],
                        k2 - kernel_half_width[2],
                    )
                    expected_output_coord = (
                        self.SINGLE_COORD[0] + centered_kernel_coord[0],
                        self.SINGLE_COORD[1] + centered_kernel_coord[1],
                        self.SINGLE_COORD[2] + centered_kernel_coord[2],
                    )
                    got_output_coord = tuple(nonzero_coords[0].tolist())
                    self.assertEqual(got_output_coord, expected_output_coord)

    @parameterized.expand(REDUCED_DEVICE_DTYPE_COMBOS)
    def test_multiple_impulses(self, device: DeviceIdentifier, dtype: torch.dtype):
        """Test that non-overlapping impulses each produce independent kernel responses."""
        device = resolve_device(device)

        impulse_coords, impulse_field = generate_hermit_impulses_dense(
            num_candidates=self.NUM_CANDIDATES,
            volume_shape=self.VOLUME_SHAPE,
            kernel_size=self.KERNEL_SIZE,
            impulse_value=1,
            dtype=dtype,
            device=device,
        )

        num_impulses = len(impulse_coords)

        self.assertEqual(impulse_field.shape, self.VOLUME_SHAPE)
        self.assertEqual(round(torch.sum(impulse_field).item()), num_impulses)

        kernel = fourier_anti_symmetric_kernel(self.KERNEL_SIZE, dtype=dtype, device=device)
        self.assertFalse(has_any_symmetry(kernel))

        impulse_field_with_channel = impulse_field.reshape(1, *self.VOLUME_SHAPE)
        kernel_with_channels = kernel.reshape(1, 1, *self.KERNEL_SIZE)

        with disable_tf32():
            output = torch.nn.functional.conv_transpose3d(
                input=impulse_field_with_channel,
                weight=kernel_with_channels,
                padding=tuple(k // 2 for k in self.KERNEL_SIZE),
            )
        self.assertEqual(impulse_field_with_channel.shape, output.shape)

        for i in range(num_impulses):
            impulse_coord = impulse_coords[i]
            _validate_impulse_conv_transpose(
                impulse_coord=impulse_coord,
                kernel=kernel,
                output=output,
                kernel_size=self.KERNEL_SIZE,
                test_case=self,
                check_bounds=False,
            )

    @parameterized.expand(ALL_DEVICE_DTYPE_COMBOS)
    def test_conv_transpose_is_adjoint_of_conv(self, device: DeviceIdentifier, dtype: torch.dtype):
        """
        Verify that conv_transpose3d is the adjoint of conv3d.

        For operators A (conv) and A^T (conv_transpose), the adjoint property is:
            <Ax, y> = <x, A^T y>

        where <,> is the inner product. This means:
            sum(conv3d(x, k) * y) = sum(x * conv_transpose3d(y, k))

        This is a fundamental property that defines transposed convolution.
        """
        device = resolve_device(device)

        # Create random input and output tensors
        x = torch.randn((1, 1) + self.SINGLE_VOLUME_SHAPE, device=device, dtype=dtype)
        y = torch.randn((1, 1) + self.SINGLE_VOLUME_SHAPE, device=device, dtype=dtype)

        kernel = fourier_anti_symmetric_kernel(self.KERNEL_SIZE, dtype=dtype, device=device)
        kernel_5d = kernel.reshape(1, 1, *self.KERNEL_SIZE)
        padding = tuple(k // 2 for k in self.KERNEL_SIZE)

        with disable_tf32():
            # Compute <conv(x), y>
            conv_x = torch.nn.functional.conv3d(input=x, weight=kernel_5d, padding=padding)
            inner_conv = (conv_x * y).sum()

            # Compute <x, conv_transpose(y)>
            conv_t_y = torch.nn.functional.conv_transpose3d(input=y, weight=kernel_5d, padding=padding)
            inner_conv_t = (x * conv_t_y).sum()

        tols = get_tolerances(dtype)
        torch.testing.assert_close(inner_conv, inner_conv_t, rtol=tols["forward"][0], atol=tols["forward"][1])

    # =========================================================================
    # Backward Pass Tests
    # =========================================================================
    #
    # The backward of conv_transpose3d computes:
    #   - d/d(input): conv3d(dy, weights) with same padding = dy * flip(weights)
    #   - d/d(weights): correlation of dy with input
    #
    # Since conv_transpose is the adjoint of conv, its backward w.r.t. input
    # is the forward conv! This creates a nice symmetry.

    @parameterized.expand(ALL_DEVICE_DTYPE_COMBOS)
    def test_single_impulse_backward_input_grad(self, device: DeviceIdentifier, dtype: torch.dtype):
        """
        Test that backward pass w.r.t. input produces expected gradients.

        Since conv_transpose is the adjoint of conv, the backward of conv_transpose
        w.r.t. input is the forward conv3d. This means:
        - Output gradient impulse at coord G
        - Input gradient = conv3d(output_grad, kernel) at appropriate position

        For an impulse output gradient at G, the input gradient at C is:
            d_input[C] = sum_k kernel[k] where G = C + (k - K//2)
            i.e., d_input[C] = kernel[G - C + K//2] (if in bounds)

        This produces the FLIPPED kernel pattern (same as forward conv3d).
        """
        device = resolve_device(device)

        coord = torch.tensor(self.SINGLE_COORD, device=device, dtype=torch.int32)
        impulse_field = torch.zeros((1,) + self.SINGLE_VOLUME_SHAPE, device=device, dtype=dtype)
        impulse_field[0, coord[0], coord[1], coord[2]] = 1
        impulse_field.requires_grad_(True)

        kernel = fourier_anti_symmetric_kernel(self.KERNEL_SIZE, dtype=dtype, device=device)
        self.assertFalse(has_any_symmetry(kernel))
        kernel_with_channels = kernel.reshape(1, 1, *self.KERNEL_SIZE)
        padding = tuple(k // 2 for k in self.KERNEL_SIZE)

        with disable_tf32():
            # Forward pass
            output = torch.nn.functional.conv_transpose3d(
                input=impulse_field, weight=kernel_with_channels, padding=padding
            )

            # Create output gradient with impulse at the same coord
            output_grad = torch.zeros_like(output)
            output_grad[0, coord[0], coord[1], coord[2]] = 1

            # Backward pass
            output.backward(output_grad)

        input_grad = impulse_field.grad
        assert input_grad is not None

        # Extract the region around the impulse coordinate
        kernel_half = tuple(k // 2 for k in self.KERNEL_SIZE)
        start_coords = tuple(int(coord[i].item()) - kernel_half[i] for i in range(3))
        end_coords = tuple(int(coord[i].item()) + kernel_half[i] + 1 for i in range(3))

        input_grad_region = input_grad[
            0, start_coords[0] : end_coords[0], start_coords[1] : end_coords[1], start_coords[2] : end_coords[2]
        ]

        self.assertEqual(input_grad_region.shape, kernel.shape)

        # The backward of conv_transpose w.r.t. input is conv3d, which produces
        # the FLIPPED kernel (because conv3d is cross-correlation)
        flipped_kernel = torch.flip(kernel, dims=[0, 1, 2])
        tols = get_tolerances(dtype)
        torch.testing.assert_close(
            input_grad_region, flipped_kernel, rtol=tols["input_grad"][0], atol=tols["input_grad"][1]
        )

    @parameterized.expand(ALL_DEVICE_DTYPE_COMBOS)
    def test_single_impulse_backward_weight_grad(self, device: DeviceIdentifier, dtype: torch.dtype):
        """
        Test that backward pass w.r.t. weights produces expected gradients.

        For conv_transpose with input impulse at C and output gradient impulse at G:
            weight_grad[k] = input[C] * output_grad[G] if G = C + (k - K//2)

        With both impulses at the same coord and value 1, the non-zero weight grad
        is at kernel position K//2 (the center), where the offset is zero.
        """
        device = resolve_device(device)

        coord = torch.tensor(self.SINGLE_COORD, device=device, dtype=torch.int32)
        impulse_field = torch.zeros((1,) + self.SINGLE_VOLUME_SHAPE, device=device, dtype=dtype)
        impulse_field[0, coord[0], coord[1], coord[2]] = 1

        kernel = fourier_anti_symmetric_kernel(self.KERNEL_SIZE, dtype=dtype, device=device)
        kernel_with_channels = kernel.reshape(1, 1, *self.KERNEL_SIZE).clone()
        kernel_with_channels.requires_grad_(True)
        padding = tuple(k // 2 for k in self.KERNEL_SIZE)

        with disable_tf32():
            # Forward pass
            output = torch.nn.functional.conv_transpose3d(
                input=impulse_field, weight=kernel_with_channels, padding=padding
            )

            # Create output gradient with impulse at the same coord
            output_grad = torch.zeros_like(output)
            output_grad[0, coord[0], coord[1], coord[2]] = 1

            # Backward pass
            output.backward(output_grad)

        weight_grad = kernel_with_channels.grad
        assert weight_grad is not None

        # With input and output grad both at coord C, the weight grad is non-zero
        # only at kernel center where offset = 0
        kernel_center = tuple(k // 2 for k in self.KERNEL_SIZE)
        expected_grad = torch.zeros_like(weight_grad)
        expected_grad[0, 0, kernel_center[0], kernel_center[1], kernel_center[2]] = 1

        tols = get_tolerances(dtype)
        torch.testing.assert_close(weight_grad, expected_grad, rtol=tols["kernel_grad"][0], atol=tols["kernel_grad"][1])

    @parameterized.expand(ALL_DEVICE_DTYPE_COMBOS)
    def test_single_impulse_backward_weight_grad_offset(self, device: DeviceIdentifier, dtype: torch.dtype):
        """
        Test weight gradient when input and output gradient impulses are at different coords.

        For conv_transpose: output[G] receives contribution from input[C] via kernel[k]
        where G = C + (k - K//2), i.e., k = G - C + K//2

        So weight_grad[k] = input[C] * output_grad[G] when k = G - C + K//2
        """
        device = resolve_device(device)

        coord_in = torch.tensor(self.SINGLE_COORD, device=device, dtype=torch.int32)
        impulse_field = torch.zeros((1,) + self.SINGLE_VOLUME_SHAPE, device=device, dtype=dtype)
        impulse_field[0, coord_in[0], coord_in[1], coord_in[2]] = 1

        kernel = fourier_anti_symmetric_kernel(self.KERNEL_SIZE, dtype=dtype, device=device)
        kernel_with_channels = kernel.reshape(1, 1, *self.KERNEL_SIZE).clone()
        kernel_with_channels.requires_grad_(True)

        kernel_half = tuple(k // 2 for k in self.KERNEL_SIZE)
        padding = kernel_half

        # Test a few different offset positions
        test_offsets = [
            (0, 0, 0),  # Center
            (1, 0, 0),  # Offset in first axis
            (0, -1, 1),  # Mixed offset
        ]

        for offset in test_offsets:
            # Reset gradient
            if kernel_with_channels.grad is not None:
                kernel_with_channels.grad.zero_()

            # Output gradient impulse at offset from input
            coord_out = (
                int(coord_in[0].item()) + offset[0],
                int(coord_in[1].item()) + offset[1],
                int(coord_in[2].item()) + offset[2],
            )

            with disable_tf32():
                output = torch.nn.functional.conv_transpose3d(
                    input=impulse_field, weight=kernel_with_channels, padding=padding
                )

                output_grad = torch.zeros_like(output)
                output_grad[0, coord_out[0], coord_out[1], coord_out[2]] = 1

                output.backward(output_grad)

            weight_grad = kernel_with_channels.grad
            assert weight_grad is not None

            # For conv_transpose: k = G - C + K//2 = offset + K//2
            expected_kernel_pos = tuple(offset[i] + kernel_half[i] for i in range(3))

            # Check bounds
            in_bounds = all(0 <= expected_kernel_pos[i] < self.KERNEL_SIZE[i] for i in range(3))

            if in_bounds:
                # Should have exactly one non-zero gradient
                nonzero_mask = weight_grad[0, 0] != 0
                nonzero_coords = torch.nonzero(nonzero_mask)
                self.assertEqual(nonzero_coords.shape[0], 1, f"Expected 1 non-zero grad for offset {offset}")

                actual_pos = tuple(nonzero_coords[0].tolist())
                self.assertEqual(actual_pos, expected_kernel_pos, f"Wrong grad position for offset {offset}")
            else:
                # Should have no gradient (offset out of kernel range)
                self.assertTrue(torch.all(weight_grad == 0), f"Expected zero grad for out-of-bounds offset {offset}")

    @parameterized.expand(REDUCED_DEVICE_DTYPE_COMBOS)
    def test_multiple_impulses_backward(self, device: DeviceIdentifier, dtype: torch.dtype):
        """Test that non-overlapping impulse gradients produce independent patterns."""
        device = resolve_device(device)

        impulse_coords, impulse_field = generate_hermit_impulses_dense(
            num_candidates=self.NUM_CANDIDATES,
            volume_shape=self.VOLUME_SHAPE,
            kernel_size=self.KERNEL_SIZE,
            impulse_value=1,
            dtype=dtype,
            device=device,
        )

        num_impulses = len(impulse_coords)

        impulse_field_with_channel = impulse_field.reshape(1, *self.VOLUME_SHAPE)
        impulse_field_with_channel = impulse_field_with_channel.clone().requires_grad_(True)

        kernel = fourier_anti_symmetric_kernel(self.KERNEL_SIZE, dtype=dtype, device=device)
        self.assertFalse(has_any_symmetry(kernel))
        kernel_with_channels = kernel.reshape(1, 1, *self.KERNEL_SIZE)
        padding = tuple(k // 2 for k in self.KERNEL_SIZE)

        with disable_tf32():
            # Forward pass
            output = torch.nn.functional.conv_transpose3d(
                input=impulse_field_with_channel, weight=kernel_with_channels, padding=padding
            )

            # Create output gradient with impulses at the same coords as input
            output_grad = torch.zeros_like(output)
            for i in range(num_impulses):
                c = impulse_coords[i]
                output_grad[0, c[0], c[1], c[2]] = 1

            # Backward pass
            output.backward(output_grad)

        input_grad = impulse_field_with_channel.grad
        assert input_grad is not None

        # Verify each impulse produces the expected pattern
        # The backward of conv_transpose w.r.t. input produces FLIPPED kernel
        flipped_kernel = torch.flip(kernel, dims=[0, 1, 2])
        kernel_half = tuple(k // 2 for k in self.KERNEL_SIZE)

        for i in range(num_impulses):
            coord = impulse_coords[i]
            start_coords = tuple(max(0, int(coord[j].item()) - kernel_half[j]) for j in range(3))
            end_coords = tuple(
                min(input_grad.shape[j + 1], int(coord[j].item()) + kernel_half[j] + 1) for j in range(3)
            )

            grad_region = input_grad[
                0, start_coords[0] : end_coords[0], start_coords[1] : end_coords[1], start_coords[2] : end_coords[2]
            ]

            # Only check if the region is fully within bounds
            expected_shape = tuple(end_coords[j] - start_coords[j] for j in range(3))
            if expected_shape == self.KERNEL_SIZE:
                tols = get_tolerances(dtype)
                torch.testing.assert_close(
                    grad_region, flipped_kernel, rtol=tols["input_grad"][0], atol=tols["input_grad"][1]
                )

    # =========================================================================
    # Strided Transposed Convolution Tests
    # =========================================================================
    #
    # KEY CONCEPT: Strided conv_transpose3d is "upsampling"
    #
    # For strided conv_transpose with stride=S:
    #   - Input at coordinate c produces outputs at c*S + (k - K//2) for each k
    #   - This "spreads out" each input to a larger output region
    #   - Output size = (input_size - 1) * S + kernel_size - 2*padding
    #
    # Compare to strided conv3d (downsampling):
    #   - Output at o receives inputs from o*S + (k - K//2)
    #   - This "gathers" from multiple inputs to each output
    #   - Output size = (input_size + 2*padding - kernel_size) / S + 1

    @parameterized.expand(REDUCED_DEVICE_DTYPE_COMBOS)
    def test_strided_impulse_output_location(self, device: DeviceIdentifier, dtype: torch.dtype):
        """
        Test that strided conv_transpose produces output at expected coordinates.

        With stride=S and half-padding, an impulse at input coordinate c produces
        output at coordinates: c*S + (k - K//2) for each kernel position k.

        For a 3x3x3 kernel with K//2=1, the output is centered at c*S and extends
        from c*S - 1 to c*S + 1 in each dimension.
        """
        device = resolve_device(device)

        kernel_size = (3, 3, 3)
        kernel_half = tuple(k // 2 for k in kernel_size)

        # Test different strides
        test_cases = [
            # (stride, input_coord, expected_output_center)
            ((2, 2, 2), (3, 3, 3), (6, 6, 6)),  # center = input * stride
            ((3, 3, 3), (2, 2, 2), (6, 6, 6)),
            ((2, 2, 2), (4, 5, 3), (8, 10, 6)),
        ]

        for stride, input_coord, expected_center in test_cases:
            # Volume size needs to accommodate output
            output_size = tuple(20 * s for s in stride)
            input_size = tuple(output_size[i] // stride[i] for i in range(3))

            impulse_field = torch.zeros((1, 1) + input_size, device=device, dtype=dtype)
            impulse_field[0, 0, input_coord[0], input_coord[1], input_coord[2]] = 1

            # All-ones kernel to see which outputs are activated
            kernel = torch.ones((1, 1) + kernel_size, device=device, dtype=dtype)

            with disable_tf32():
                output = torch.nn.functional.conv_transpose3d(
                    input=impulse_field, weight=kernel, stride=stride, padding=kernel_half
                )

            # Find non-zero output coordinates
            nonzero_coords = torch.nonzero(output[0, 0])

            # Expected outputs: center +/- kernel_half
            expected_outputs = []
            for k0 in range(kernel_size[0]):
                for k1 in range(kernel_size[1]):
                    for k2 in range(kernel_size[2]):
                        out_coord = (
                            expected_center[0] + (k0 - kernel_half[0]),
                            expected_center[1] + (k1 - kernel_half[1]),
                            expected_center[2] + (k2 - kernel_half[2]),
                        )
                        expected_outputs.append(out_coord)

            actual_outputs = [tuple(c.tolist()) for c in nonzero_coords]

            # Verify all expected outputs are present
            for expected in expected_outputs:
                self.assertIn(
                    expected,
                    actual_outputs,
                    f"stride={stride}, input={input_coord}: expected output at {expected} not found. "
                    f"Got {len(actual_outputs)} outputs.",
                )

            # Verify count matches
            self.assertEqual(len(actual_outputs), len(expected_outputs))

    @parameterized.expand(REDUCED_DEVICE_DTYPE_COMBOS)
    def test_strided_activation_and_weights(self, device: DeviceIdentifier, dtype: torch.dtype):
        """
        Test the relationship between kernel position, stride, and output coordinate.

        For strided conv_transpose with:
        - Impulse at input coordinate c
        - Kernel with single weight at position (k0, k1, k2)
        - Stride S and half-padding

        The output impulse appears at: c*S + (k - K//2)

        This is the INVERSE of strided conv3d where:
        output at o = (c - k + K//2) / S
        """
        device = resolve_device(device)

        kernel_size = (3, 3, 3)
        kernel_half = tuple(k // 2 for k in kernel_size)
        stride = (2, 2, 2)

        input_coord = (5, 5, 5)
        input_size = (12, 12, 12)

        impulse_field = torch.zeros((1, 1) + input_size, device=device, dtype=dtype)
        impulse_field[0, 0, input_coord[0], input_coord[1], input_coord[2]] = 1

        # Test each kernel position
        for k0 in range(kernel_size[0]):
            for k1 in range(kernel_size[1]):
                for k2 in range(kernel_size[2]):
                    kernel = torch.zeros((1, 1) + kernel_size, device=device, dtype=dtype)
                    kernel[0, 0, k0, k1, k2] = 1

                    with disable_tf32():
                        output = torch.nn.functional.conv_transpose3d(
                            input=impulse_field, weight=kernel, stride=stride, padding=kernel_half
                        )

                    nonzero_coords = torch.nonzero(output[0, 0])

                    # Expected output: c*S + (k - K//2)
                    expected_o = tuple(
                        input_coord[i] * stride[i] + ([k0, k1, k2][i] - kernel_half[i]) for i in range(3)
                    )

                    # Should have exactly one output
                    self.assertEqual(
                        len(nonzero_coords),
                        1,
                        f"kernel_pos=({k0},{k1},{k2}): expected 1 output, got {len(nonzero_coords)}",
                    )

                    actual_o = tuple(nonzero_coords[0].tolist())
                    self.assertEqual(
                        actual_o,
                        expected_o,
                        f"kernel_pos=({k0},{k1},{k2}): expected output at {expected_o}, got {actual_o}",
                    )

    @parameterized.expand(REDUCED_DEVICE_DTYPE_COMBOS)
    def test_strided_backward_input_grad(self, device: DeviceIdentifier, dtype: torch.dtype):
        """
        Test backward pass w.r.t. input for strided conv_transpose.

        The backward of strided conv_transpose w.r.t. input is strided conv3d.
        For an output gradient impulse at coord G, the input gradient at C receives
        contributions when G = C*S + (k - K//2), i.e., when (G - k + K//2) % S == 0.

        The gradient pattern depends on which G coordinates align with the stride grid.
        """
        device = resolve_device(device)

        kernel_size = (3, 3, 3)
        kernel_half = tuple(k // 2 for k in kernel_size)
        stride = (2, 2, 2)

        input_size = (12, 12, 12)
        input_coord = (5, 5, 5)

        impulse_field = torch.zeros((1, 1) + input_size, device=device, dtype=dtype)
        impulse_field[0, 0, input_coord[0], input_coord[1], input_coord[2]] = 1
        impulse_field.requires_grad_(True)

        kernel = fourier_anti_symmetric_kernel(kernel_size, dtype=dtype, device=device)
        kernel_5d = kernel.reshape(1, 1, *kernel_size)

        with disable_tf32():
            output = torch.nn.functional.conv_transpose3d(
                input=impulse_field, weight=kernel_5d, stride=stride, padding=kernel_half
            )

            # Output gradient impulse at the center of the output region
            output_coord = tuple(input_coord[i] * stride[i] for i in range(3))
            output_grad = torch.zeros_like(output)
            output_grad[0, 0, output_coord[0], output_coord[1], output_coord[2]] = 1

            output.backward(output_grad)

        input_grad = impulse_field.grad
        assert input_grad is not None

        # The backward of strided conv_transpose is strided conv3d
        # For output_grad at G = input*S, the input gradient is:
        # d_input[C] = kernel[(G - C*S + K//2)] if (G - C*S + K//2) is valid kernel index
        #
        # With G at input_coord * stride, d_input[input_coord] gets kernel[K//2] (center)
        # This should be a single value at the original input coordinate

        # Check that only the input coordinate has non-zero gradient
        nonzero_mask = input_grad[0, 0] != 0
        nonzero_coords = torch.nonzero(nonzero_mask)

        self.assertEqual(len(nonzero_coords), 1, "Expected exactly one non-zero gradient location")

        actual_coord = tuple(nonzero_coords[0].tolist())
        self.assertEqual(actual_coord, input_coord, "Gradient should be at the input coordinate")

        # The gradient value should be kernel[K//2] (the center value)
        expected_grad_value = kernel[kernel_half[0], kernel_half[1], kernel_half[2]]
        actual_grad_value = input_grad[0, 0, input_coord[0], input_coord[1], input_coord[2]]

        tols = get_tolerances(dtype)
        torch.testing.assert_close(
            actual_grad_value, expected_grad_value, rtol=tols["input_grad"][0], atol=tols["input_grad"][1]
        )

    @parameterized.expand(REDUCED_DEVICE_DTYPE_COMBOS)
    def test_strided_backward_weight_grad(self, device: DeviceIdentifier, dtype: torch.dtype):
        """
        Test backward pass w.r.t. weights for strided conv_transpose.

        For strided conv_transpose:
        - Output at G = C*S + (k - K//2) receives kernel[k] * input[C]
        - Weight gradient: dw[k] = sum over C,G of input[C] * output_grad[G]
                          where G = C*S + (k - K//2)

        With input impulse at C and output grad impulse at G:
        - dw[k] = 1 if G = C*S + (k - K//2), i.e., k = G - C*S + K//2
        """
        device = resolve_device(device)

        kernel_size = (3, 3, 3)
        kernel_half = tuple(k // 2 for k in kernel_size)
        stride = (2, 2, 2)

        input_size = (12, 12, 12)
        input_coord = (5, 5, 5)

        impulse_field = torch.zeros((1, 1) + input_size, device=device, dtype=dtype)
        impulse_field[0, 0, input_coord[0], input_coord[1], input_coord[2]] = 1

        kernel = fourier_anti_symmetric_kernel(kernel_size, dtype=dtype, device=device)
        kernel_5d = kernel.reshape(1, 1, *kernel_size).clone().requires_grad_(True)

        with disable_tf32():
            output = torch.nn.functional.conv_transpose3d(
                input=impulse_field, weight=kernel_5d, stride=stride, padding=kernel_half
            )

            # Output gradient impulse at the center of output region
            output_coord = tuple(input_coord[i] * stride[i] for i in range(3))
            output_grad = torch.zeros_like(output)
            output_grad[0, 0, output_coord[0], output_coord[1], output_coord[2]] = 1

            output.backward(output_grad)

        weight_grad = kernel_5d.grad
        assert weight_grad is not None

        # Expected kernel position: k = G - C*S + K//2
        # With G = C*S, we get k = K//2 (center)
        expected_k = kernel_half

        nonzero_mask = weight_grad[0, 0] != 0
        nonzero_coords = torch.nonzero(nonzero_mask)
        self.assertEqual(len(nonzero_coords), 1, "Expected exactly one non-zero weight gradient")

        actual_k = tuple(nonzero_coords[0].tolist())
        self.assertEqual(actual_k, expected_k, f"Expected weight grad at {expected_k}, got {actual_k}")

    @parameterized.expand(REDUCED_DEVICE_DTYPE_COMBOS)
    def test_strided_conv_transpose_output_size(self, device: DeviceIdentifier, dtype: torch.dtype):
        """
        Verify the output size formula for strided conv_transpose.

        For conv_transpose3d with:
        - input_size, kernel_size, stride, padding

        Output size = (input_size - 1) * stride + kernel_size - 2 * padding

        This is the inverse of the conv3d output size formula.
        """
        device = resolve_device(device)

        test_cases = [
            # (input_size, kernel_size, stride, padding, expected_output_size)
            ((10, 10, 10), (3, 3, 3), (2, 2, 2), (1, 1, 1), (19, 19, 19)),
            ((10, 10, 10), (3, 3, 3), (1, 1, 1), (1, 1, 1), (10, 10, 10)),  # stride=1 with same padding
            ((5, 7, 9), (3, 5, 7), (2, 2, 2), (1, 2, 3), (9, 13, 17)),
            ((8, 8, 8), (5, 5, 5), (3, 3, 3), (2, 2, 2), (22, 22, 22)),
        ]

        for input_size, kernel_size, stride, padding, expected_output in test_cases:
            x = torch.randn((1, 1) + input_size, device=device, dtype=dtype)
            kernel = torch.randn((1, 1) + kernel_size, device=device, dtype=dtype)

            output = torch.nn.functional.conv_transpose3d(input=x, weight=kernel, stride=stride, padding=padding)

            actual_output_size = output.shape[2:]
            self.assertEqual(
                actual_output_size,
                expected_output,
                f"Input {input_size}, kernel {kernel_size}, stride {stride}, padding {padding}: "
                f"expected {expected_output}, got {actual_output_size}",
            )

    @parameterized.expand(REDUCED_DEVICE_DTYPE_COMBOS)
    def test_strided_conv_transpose_is_adjoint_of_strided_conv(self, device: DeviceIdentifier, dtype: torch.dtype):
        """
        Verify that strided conv_transpose is the adjoint of strided conv.

        For strided conv3d with stride S that maps input_size -> output_size:
            output_size = floor((input_size + 2*padding - kernel_size) / S) + 1

        The adjoint (conv_transpose) maps output_size -> input_size:
            <conv(x), y> = <x, conv_transpose(y)>

        This property defines the transposed convolution.

        IMPORTANT: For strided operations, conv_transpose needs `output_padding` to
        guarantee the output size matches the original input size. The output_padding
        compensates for the ambiguity in inverting the floor division in the size formula.
        """
        device = resolve_device(device)

        kernel_size = (3, 3, 3)
        kernel_half = tuple(k // 2 for k in kernel_size)
        stride = (2, 2, 2)
        padding = kernel_half

        # Input for forward conv
        x_size = (10, 10, 10)
        x = torch.randn((1, 1) + x_size, device=device, dtype=dtype)

        kernel = fourier_anti_symmetric_kernel(kernel_size, dtype=dtype, device=device)
        kernel_5d = kernel.reshape(1, 1, *kernel_size)

        with disable_tf32():
            # Forward strided conv
            conv_x = torch.nn.functional.conv3d(input=x, weight=kernel_5d, stride=stride, padding=padding)

            # Create y of the output size
            y = torch.randn_like(conv_x)

            # Compute <conv(x), y>
            inner_conv = (conv_x * y).sum()

            # For conv_transpose to be the exact adjoint, we need output_padding to
            # ensure the output size matches x_size. The formula is:
            # output_size = (y_size - 1) * stride + kernel_size - 2*padding + output_padding
            # We want output_size = x_size, so:
            # output_padding = x_size - (y_size - 1) * stride - kernel_size + 2*padding
            y_size = conv_x.shape[2:]
            output_padding = tuple(
                x_size[i] - (y_size[i] - 1) * stride[i] - kernel_size[i] + 2 * padding[i] for i in range(3)
            )

            conv_t_y = torch.nn.functional.conv_transpose3d(
                input=y, weight=kernel_5d, stride=stride, padding=padding, output_padding=output_padding
            )

            # Verify sizes match
            self.assertEqual(conv_t_y.shape, x.shape, "conv_transpose output size should match x size")

            # Compute <x, conv_transpose(y)>
            inner_conv_t = (x * conv_t_y).sum()

        tols = get_tolerances(dtype)
        torch.testing.assert_close(inner_conv, inner_conv_t, rtol=tols["forward"][0], atol=tols["forward"][1])


if __name__ == "__main__":
    unittest.main()
