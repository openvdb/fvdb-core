# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""
Test the PyTorch ground truth for convolutions in 3D.

The tests in this file will validate that the PyTorch dense versions of convolution
do what we expect them to do, and establish a set of baseline demonstrations of
convolution properties.
"""

import io
import sys
import time
import unittest

import pytest
import torch
from fvdb.types import DeviceIdentifier, resolve_device
from fvdb.utils.tests import (
    ScopedTimer,
    fourier_anti_symmetric_kernel,
    generate_chebyshev_spaced_ijk,
    generate_chebyshev_spaced_ijk_batch,
    generate_hermit_impulses_dense,
    generate_hermit_impulses_dense_batch,
    has_any_symmetry,
)
from parameterized import parameterized

all_device_dtype_combos = [
    # ["cuda", torch.float16],
    ["cpu", torch.float32],
    ["cuda", torch.float32],
    ["cpu", torch.float64],
    ["cuda", torch.float64],
]


def _validate_impulse_convolution(
    impulse_coord: torch.Tensor,
    kernel: torch.Tensor,
    convolved: torch.Tensor,
    kernel_size: tuple,
    test_case: unittest.TestCase,
) -> None:
    """
    Validate that a convolution of an impulse at a specific coordinate produces the expected kernel.

    This function extracts the region around the impulse coordinate from the convolution result,
    flips it (since PyTorch conv3d performs cross-correlation), and compares it to the kernel.

    Args:
        impulse_coord: The coordinate of the impulse as a torch.Tensor
        kernel: The kernel used for convolution
        convolved: The result of the convolution
        kernel_size: The size of the kernel as a tuple
        test_case: The unittest.TestCase instance for assertions
    """
    # Extract the region around the impulse coordinate where the kernel should appear
    # The kernel is centered at the impulse coordinate
    kernel_half = tuple(k // 2 for k in kernel_size)

    # Define the slice boundaries for extracting the kernel region
    start_coords = tuple(max(0, impulse_coord[i].item() - kernel_half[i]) for i in range(3))
    end_coords = tuple(min(convolved.shape[i + 1], impulse_coord[i].item() + kernel_half[i] + 1) for i in range(3))

    # The non-zero region of convolved should be exactly inside the bounds we just computed.
    # Check that the bounds of the non-zero region exactly match the expected bounds
    non_zero_mask = convolved[0] != 0
    non_zero_coords = torch.nonzero(non_zero_mask)

    if non_zero_coords.shape[0] > 0:
        # Find the actual bounds of the non-zero region
        actual_start_coords = tuple(non_zero_coords[:, dim].min().item() for dim in range(3))
        actual_end_coords = tuple(non_zero_coords[:, dim].max().item() + 1 for dim in range(3))

        # The actual bounds should exactly match the expected bounds
        test_case.assertEqual(actual_start_coords, start_coords)
        test_case.assertEqual(actual_end_coords, end_coords)

    # Extract the convolved region around the impulse
    convolved_region = convolved[
        0, start_coords[0] : end_coords[0], start_coords[1] : end_coords[1], start_coords[2] : end_coords[2]
    ]

    test_case.assertEqual(convolved_region.shape, kernel.shape)

    # Since PyTorch conv3d performs cross-correlation, we need to flip to get the result to
    # match the kernel.
    convolved_region = torch.flip(convolved_region, dims=[0, 1, 2])

    # The flipped convolved region should match the kernel
    torch.testing.assert_close(convolved_region, kernel, rtol=1e-5, atol=1e-6)


class TestConvGroundTruth(unittest.TestCase):

    VOLUME_SHAPE = (71, 34, 58)
    KERNEL_SIZE = (3, 5, 7)
    SINGLE_VOLUME_SHAPE = (5, 7, 9)
    SINGLE_COORD = (2, 3, 4)
    NUM_CANDIDATES = 1000

    def setUp(self):
        torch.random.manual_seed(2024)

    @parameterized.expand(all_device_dtype_combos)
    def test_single_impulse(self, device: DeviceIdentifier, dtype: torch.dtype):
        device = resolve_device(device)

        # For single impulse, we just need to make sure it's far enough away from
        # the boundary of the volume.

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

        kernel_sum = torch.sum(kernel).item()

        kernel_with_channels = kernel.reshape(1, 1, *self.KERNEL_SIZE)

        # Do a single convolution
        _backend_setting = torch.backends.cudnn.allow_tf32
        # Disable TF32 for consistent precision across CPU and CUDA
        torch.backends.cudnn.allow_tf32 = False
        convolved = torch.nn.functional.conv3d(input=impulse_field, weight=kernel_with_channels, padding="same")
        self.assertEqual(impulse_field.shape, convolved.shape)
        torch.backends.cudnn.allow_tf32 = _backend_setting

        # We know where the impulse coordinate is, so we should be able to test that the
        # convolution matches the kernel. Even though PyTorch calls it conv3d, it's actually a
        # cross-correlation, per the documentation. Therefore, convolving an impulse with a kernel
        # should produce exactly the kernel. We can test this by extracting the region around the impulse
        # coordinate and comparing it to the kernel.

        # Use the helper function to validate the impulse convolution
        _validate_impulse_convolution(
            impulse_coord=coord,
            kernel=kernel,
            convolved=convolved,
            kernel_size=self.KERNEL_SIZE,
            test_case=self,
        )

    @parameterized.expand(all_device_dtype_combos)
    def test_multiple_impulses(self, device: DeviceIdentifier, dtype: torch.dtype):
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
        print(f"Number of generated impulses: {num_impulses}")

        total_value = torch.sum(impulse_field).item()
        print(f"Total sum of impulse_field: {total_value}")

        # Test that the impulse field's shape matches the volume shape
        self.assertEqual(impulse_field.shape, self.VOLUME_SHAPE)

        # Test that the total value of the impulse field matches the total number of impulses
        self.assertEqual(round(total_value), num_impulses)
