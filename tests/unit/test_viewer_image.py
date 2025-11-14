#!/usr/bin/env python3
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

import sys
import unittest
import warnings

import numpy as np
import pytest
import torch

import fvdb

PORT = 8081


class TestViewerImage(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            fvdb.viz.init(ip_address="127.0.0.1", port=PORT, verbose=False)
        except Exception as e:
            pytest.skip(f"Could not initialize viewer server: {e}")

    def test_add_image_gradient(self):
        """Test adding a gradient RGBA image to the viewer."""
        scene = fvdb.viz.Scene("test_image_gradient")

        width = 1440
        height = 720

        # Create an RGBA image with a gradient pattern (matching TEST_IMAGE2D)
        # Format: packed RGBA8 values in a 1D uint8 tensor
        rgba_data = np.zeros((height, width, 4), dtype=np.uint8)

        for j in range(height):
            for i in range(width):
                rgba_data[j, i, 0] = (255 * i) // (width - 1)   # Red gradient (horizontal)
                rgba_data[j, i, 1] = (255 * j) // (height - 1)  # Green gradient (vertical)
                rgba_data[j, i, 2] = 0                          # Blue = 0
                rgba_data[j, i, 3] = 255                        # Alpha = 255

        # Flatten to 1D and convert to torch tensor
        rgba_tensor = torch.from_numpy(rgba_data.reshape(-1))

        # Add to viewer
        scene.add_image("gradient_image", rgba_tensor, width, height)

        print(f"Successfully added gradient image ({width}x{height}) to viewer")

    def test_add_image_checkerboard(self):
        """Test adding a checkerboard RGBA image to the viewer."""
        scene = fvdb.viz.Scene("test_image_checkerboard")

        width = 512
        height = 512
        checker_size = 64

        # Create a checkerboard pattern
        rgba_data = np.zeros((height, width, 4), dtype=np.uint8)

        for j in range(height):
            for i in range(width):
                checker_x = (i // checker_size) % 2
                checker_y = (j // checker_size) % 2
                is_white = (checker_x + checker_y) % 2 == 0

                if is_white:
                    rgba_data[j, i, :3] = 255  # White
                else:
                    rgba_data[j, i, :3] = 0    # Black
                rgba_data[j, i, 3] = 255       # Alpha = 255

        # Flatten to 1D and convert to torch tensor
        rgba_tensor = torch.from_numpy(rgba_data.reshape(-1))

        # Add to viewer
        scene.add_image("checker_image", rgba_tensor, width, height)

        print(f"Successfully added checkerboard image ({width}x{height}) to viewer")

    def test_add_image_small(self):
        """Test adding a small RGBA image to the viewer."""
        scene = fvdb.viz.Scene("test_image_small")

        width = 64
        height = 64

        # Create a simple solid color image
        rgba_data = np.zeros((height, width, 4), dtype=np.uint8)
        rgba_data[:, :, 0] = 255  # Red
        rgba_data[:, :, 1] = 128  # Half green
        rgba_data[:, :, 2] = 0    # No blue
        rgba_data[:, :, 3] = 255  # Fully opaque

        # Flatten to 1D and convert to torch tensor
        rgba_tensor = torch.from_numpy(rgba_data.reshape(-1))

        # Add to viewer
        scene.add_image("small_image", rgba_tensor, width, height)

        print(f"Successfully added small image ({width}x{height}) to viewer")

    def test_add_image_validation(self):
        """Test that add_image validates input correctly."""
        scene = fvdb.viz.Scene("test_image_validation")

        width = 100
        height = 100

        # Test wrong size - should raise error
        wrong_size_tensor = torch.zeros(width * height * 3, dtype=torch.uint8)
        with pytest.raises((RuntimeError, ValueError)):
            scene.add_image("wrong_size", wrong_size_tensor, width, height)

        # Test wrong dtype - should raise error
        wrong_dtype_tensor = torch.zeros(width * height * 4, dtype=torch.float32)
        with pytest.raises((RuntimeError, TypeError)):
            scene.add_image("wrong_dtype", wrong_dtype_tensor, width, height)

        # Test wrong dimensions - should raise error
        wrong_dim_tensor = torch.zeros((height, width, 4), dtype=torch.uint8)
        with pytest.raises((RuntimeError, ValueError)):
            scene.add_image("wrong_dim", wrong_dim_tensor, width, height)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
