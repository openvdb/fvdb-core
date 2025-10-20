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
    generate_chebyshev_spaced_ijk,
    generate_chebyshev_spaced_ijk_batch,
    generate_hermit_impulses_dense,
    generate_hermit_impulses_dense_batch,
)
from parameterized import parameterized

all_device_dtype_combos = [
    ["cuda", torch.float16],
    ["cpu", torch.float32],
    ["cuda", torch.float32],
    ["cpu", torch.float64],
    ["cuda", torch.float64],
]


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
        impulse_field = torch.zeros(self.SINGLE_VOLUME_SHAPE, device=device, dtype=dtype)

        impulse_field[coord[0], coord[1], coord[2]] = 1

        self.assertEqual(impulse_field.sum().item(), 1)

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
