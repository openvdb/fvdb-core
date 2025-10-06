# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import unittest

import numpy as np
import torch
import torch.nn.functional as F
from fvdb.utils.tests import (
    generate_hermit_impulses_dense,
    generate_hermit_impulses_dense_batch,
)
from parameterized import parameterized

from fvdb import ConvolutionPlan, Grid, GridBatch, JaggedTensor

all_device_combos = [
    ["cpu"],
    ["cuda"],
]


class TestConvolutionTopology(unittest.TestCase):
    """
    Testing basic assumptions about the convolution topology.
    """

    @parameterized.expand(all_device_combos)
    def test_simple_convolution_topology(self, device):
        """
        Test the simple convolution topology for a small symmetric kernel.
        """

        # Use a simple kernel size for detailed analysis
        kernel_size = (3, 3, 3)
        kernel_volume = 27

        center_ijk = torch.tensor([[1, 1, 1]], dtype=torch.int32, device=device)
        grid = Grid.from_ijk(center_ijk, device=device)
        self.assertEqual(1, grid.num_voxels)
        print(f"input ijk: {grid.ijk.tolist()}")

        conv_output_grid = grid.conv_grid(kernel_size=kernel_size, stride=1)
        print(f"conv ijk: {conv_output_grid.ijk.tolist()}")
        self.assertEqual(kernel_volume, conv_output_grid.num_voxels)

    def test_detailed_voxel_positions(self):
        """
        Detailed examination of exact voxel positions for a small asymmetric kernel.
        This test explicitly checks every output voxel location to understand the
        exact mapping.
        """
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        device = "cpu"

        # Use a simple kernel size for detailed analysis
        kernel_size = (3, 3, 3)  # Asymmetric and even/odd mix
        center = np.array([10, 20, 30])

        kernel_volume = kernel_size[0] * kernel_size[1] * kernel_size[2]

        center_ijk = torch.tensor([center], dtype=torch.int32, device=device)
        grid = Grid.from_ijk(center_ijk, device=device)
        self.assertEqual(1, grid.num_voxels)
        print(f"input ijk: {grid.ijk.tolist()}")

        conv_output_grid = grid.conv_grid(kernel_size=kernel_size, stride=1)
        print(f"conv ijk: {conv_output_grid.ijk.tolist()}")
        self.assertEqual(kernel_volume, conv_output_grid.num_voxels)

        output_ijk = conv_output_grid.ijk.cpu().numpy()

        print(f"\nDetailed voxel positions for kernel {kernel_size}:")
        print(f"Input center: {center}")
        print(f"Output voxels ({len(output_ijk)} total):")

        # Sort for easier visualization
        output_sorted = output_ijk[np.lexsort((output_ijk[:, 2], output_ijk[:, 1], output_ijk[:, 0]))]

        for i, voxel in enumerate(output_sorted):
            offset = voxel - center
            print(f"  {i:3d}: {voxel} (offset: {offset})")

        # Check that we have the expected number
        kx, ky, kz = kernel_size
        self.assertEqual(len(output_ijk), kx * ky * kz)

        # Verify all expected voxels are present
        # For PyTorch conv3d with padding, even kernels typically start at different offsets
        # Let's discover what the actual pattern is
        expected_voxels = set()
        min_coords = output_ijk.min(axis=0)

        for dx in range(kx):
            for dy in range(ky):
                for dz in range(kz):
                    voxel = tuple(min_coords + np.array([dx, dy, dz]))
                    expected_voxels.add(voxel)

        actual_voxels = set(tuple(v) for v in output_ijk)

        self.assertEqual(
            expected_voxels,
            actual_voxels,
            f"Voxel sets don't match. Missing: {expected_voxels - actual_voxels}, Extra: {actual_voxels - expected_voxels}",
        )

    @parameterized.expand(
        [
            # Test name, kernel_size (x, y, z), stride, expected description
            ("symmetric_3x3x3", (3, 3, 3), (1, 1, 1)),
            ("asymmetric_3x5x7", (3, 5, 7), (1, 1, 1)),
            ("asymmetric_1x1x3", (1, 1, 3), (1, 1, 1)),
            ("asymmetric_1x3x5", (1, 3, 5), (1, 1, 1)),
            ("asymmetric_3x1x1", (3, 1, 1), (1, 1, 1)),
            ("asymmetric_5x3x1", (5, 3, 1), (1, 1, 1)),
            ("asymmetric_7x5x3", (7, 5, 3), (1, 1, 1)),
            ("even_2x4x6", (2, 4, 6), (1, 1, 1)),
            ("even_4x2x2", (4, 2, 2), (1, 1, 1)),
        ]
    )
    def test_single_voxel_conv_topology_grid(self, name, kernel_size, stride):
        """
        Test convolution topology by creating a Grid with a single voxel and examining
        the voxel locations created by conv_grid with asymmetric kernel sizes.

        This helps verify that kernel dimensions map correctly to spatial dimensions:
        - kernel_size[0] should affect x-axis
        - kernel_size[1] should affect y-axis
        - kernel_size[2] should affect z-axis
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Create a grid with a single voxel at a unique location
        center_ijk = torch.tensor([[11, 17, -31]], dtype=torch.int32, device=device)
        grid = Grid.from_ijk(center_ijk, voxel_size=1.0, origin=0.0, device=device)

        # Get the output grid topology from conv_grid
        conv_output_grid = grid.conv_grid(kernel_size=kernel_size, stride=stride)

        # Extract the voxel coordinates
        output_ijk = conv_output_grid.ijk.cpu().numpy()

        # For a single input voxel with stride (1,1,1), the output should contain
        # all voxels that would be touched by the kernel centered at the input voxel
        kx, ky, kz = kernel_size
        sx, sy, sz = stride

        # Expected output voxel count for stride 1
        expected_count = kx * ky * kz

        # For stride > 1, the output might be different - we're exploring this
        if stride == (1, 1, 1):
            self.assertEqual(
                len(output_ijk),
                expected_count,
                f"Expected {expected_count} output voxels for kernel {kernel_size}, got {len(output_ijk)}",
            )

        # Analyze the spatial extent in each dimension
        if len(output_ijk) > 0:
            min_coords = output_ijk.min(axis=0)
            max_coords = output_ijk.max(axis=0)
            extent = max_coords - min_coords + 1  # +1 because it's inclusive

            print(f"\n{name}:")
            print(f"  Kernel size (x,y,z): {kernel_size}")
            print(f"  Stride (x,y,z): {stride}")
            print(f"  Input voxel: {center_ijk.cpu().numpy()[0]}")
            print(f"  Output voxel count: {len(output_ijk)}")
            print(f"  Output extent (x,y,z): {extent}")
            print(f"  Output min (x,y,z): {min_coords}")
            print(f"  Output max (x,y,z): {max_coords}")

            # For stride (1,1,1), verify the extents match the kernel size
            if stride == (1, 1, 1):
                self.assertEqual(extent[0], kx, f"X extent {extent[0]} should match kernel_size[0]={kx}")
                self.assertEqual(extent[1], ky, f"Y extent {extent[1]} should match kernel_size[1]={ky}")
                self.assertEqual(extent[2], kz, f"Z extent {extent[2]} should match kernel_size[2]={kz}")

                # Also verify the centering - for odd kernels, should be symmetric
                # For even kernels, PyTorch conv3d has specific behavior
                center = center_ijk.cpu().numpy()[0]
                if kx % 2 == 1:  # odd kernel in x
                    expected_x_center = center[0]
                    actual_x_center = (min_coords[0] + max_coords[0]) // 2
                    self.assertEqual(
                        actual_x_center,
                        expected_x_center,
                        f"X center mismatch: expected {expected_x_center}, got {actual_x_center}",
                    )

                if ky % 2 == 1:  # odd kernel in y
                    expected_y_center = center[1]
                    actual_y_center = (min_coords[1] + max_coords[1]) // 2
                    self.assertEqual(
                        actual_y_center,
                        expected_y_center,
                        f"Y center mismatch: expected {expected_y_center}, got {actual_y_center}",
                    )

                if kz % 2 == 1:  # odd kernel in z
                    expected_z_center = center[2]
                    actual_z_center = (min_coords[2] + max_coords[2]) // 2
                    self.assertEqual(
                        actual_z_center,
                        expected_z_center,
                        f"Z center mismatch: expected {expected_z_center}, got {actual_z_center}",
                    )

    @parameterized.expand(
        [
            # Test name, kernel_size (Smajor, Sinner, Sminor), stride
            ("symmetric_3x3x3", (3, 3, 3), (1, 1, 1)),
            ("asymmetric_3x5x7", (3, 5, 7), (1, 1, 1)),
            ("asymmetric_1x1x3", (1, 1, 3), (1, 1, 1)),
            ("asymmetric_1x3x5", (1, 3, 5), (1, 1, 1)),
            ("asymmetric_3x1x1", (3, 1, 1), (1, 1, 1)),
            ("asymmetric_5x3x1", (5, 3, 1), (1, 1, 1)),
            ("asymmetric_7x5x3", (7, 5, 3), (1, 1, 1)),
            ("even_2x4x6", (2, 4, 6), (1, 1, 1)),
            ("even_4x2x2", (4, 2, 2), (1, 1, 1)),
        ]
    )
    def test_single_impulse_dense_torch_conv3d(self, name, kernel_size, stride):
        """
        Test convolution using dense torch grid and torch.nn.functional.conv3d.
        Creates a single impulse at (10, 13, 17) in a 25x35x45 grid, applies
        convolution, and finds all non-zero cells and their bounds.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Grid dimensions: 25x35x45
        grid_shape = (25, 35, 45)
        impulse_loc = (10, 13, 17)

        # Create dense grid with single impulse
        # PyTorch conv3d expects (batch, channels, depth, height, width)
        # For 3D spatial data, typical convention is (D, H, W) = (X, Y, Z)
        dense_grid = torch.zeros((1, 1, *grid_shape), dtype=torch.float32, device=device)
        dense_grid[0, 0, impulse_loc[0], impulse_loc[1], impulse_loc[2]] = 1.0

        # Create kernel (out_channels, in_channels, kD, kH, kW)
        kx, ky, kz = kernel_size
        kernel = torch.ones((1, 1, kx, ky, kz), dtype=torch.float32, device=device)

        # Apply convolution
        # PyTorch conv3d uses 'valid' padding by default (no padding)
        # To see where the kernel touches, we need to use appropriate padding
        # For 'same' output size with odd kernels, padding = (k-1)//2
        # For even kernels, PyTorch doesn't have direct 'same' mode
        sx, sy, sz = stride
        padding = ((kx - 1) // 2, (ky - 1) // 2, (kz - 1) // 2)

        output = F.conv3d(dense_grid, kernel, stride=stride, padding=padding)

        # Find all non-zero cells
        nonzero_indices = torch.nonzero(output[0, 0], as_tuple=False).cpu().numpy()

        if len(nonzero_indices) > 0:
            # Get bounds
            min_coords = nonzero_indices.min(axis=0)
            max_coords = nonzero_indices.max(axis=0)
            extent = max_coords - min_coords + 1

            print(f"\n{name} (Dense Torch Conv3d):")
            print(f"  Kernel size (x,y,z): {kernel_size}")
            print(f"  Stride (x,y,z): {stride}")
            print(f"  Impulse location: {impulse_loc}")
            print(f"  Grid shape (x,y,z): {grid_shape}")
            print(f"  Padding (x,y,z): {padding}")
            print(f"  Non-zero output voxel count: {len(nonzero_indices)}")
            print(f"  Output extent (x,y,z): {extent}")
            print(f"  Output min (x,y,z): {min_coords}")
            print(f"  Output max (x,y,z): {max_coords}")
            print(f"  Output values range: [{output[0, 0].min().item():.2f}, {output[0, 0].max().item():.2f}]")

            # With stride (1,1,1) and proper padding, we should get non-zero outputs
            # The exact behavior depends on padding strategy
            if stride == (1, 1, 1):
                # For odd kernels with 'same' padding, the impulse should affect
                # a region around the impulse location
                # For a kernel of size k, the affected region should be roughly k voxels
                # The actual count depends on boundary conditions

                # Check that we got some output
                self.assertGreater(len(nonzero_indices), 0, "Should have non-zero outputs")

                # For symmetric padding and stride 1, max value should be at or near impulse location
                max_val_idx = torch.argmax(output[0, 0])
                max_val_coords = np.unravel_index(max_val_idx.cpu().item(), output[0, 0].shape)
                print(f"  Max value location: {max_val_coords}")

            # Verify bounds are within grid
            self.assertTrue(np.all(min_coords >= 0), "Min coords should be non-negative")
            self.assertTrue(
                np.all(max_coords < np.array(grid_shape)),
                f"Max coords {max_coords} should be within grid shape {grid_shape}",
            )
        else:
            print(f"\n{name} (Dense Torch Conv3d): No non-zero outputs")


if __name__ == "__main__":
    unittest.main()
