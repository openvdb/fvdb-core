#!/usr/bin/env python3
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

import os

import torch
import tyro
from fvdb.utils.tests import (
    generate_center_frame_point_at_depth,
    generate_random_4x4_xform,
)

from fvdb import GaussianSplat3d


def save_tensors_torchscript(tensor_list, filepath, tensor_names=None):
    """
    Save a list of tensors using TorchScript format for C++ compatibility.

    Args:
        tensor_list: List of tensors to save
        filepath: Path to save the tensors to
        tensor_names: List of names for the tensors (if None, will use generic names)
    """

    # Create a container module
    class TensorContainer(torch.nn.Module):
        def __init__(self, tensors, names=None):
            super().__init__()
            self.tensor_names = []
            for i, tensor in enumerate(tensors):
                if names and i < len(names):
                    name = names[i]
                else:
                    name = f"tensor_{i}"
                self.tensor_names.append(name)
                self.register_buffer(name, tensor)

        def forward(self):
            # This forward method is not strictly necessary for accessing the tensors,
            # which are stored as named buffers. Returning None ensures that
            # loading scripts will fall back to iterating over named_buffers(),
            # which is a more reliable way to get named tensors.
            return None

        def get_tensor_names(self):
            # Helper method to expose tensor names
            return self.tensor_names

    # Use descriptive names if not provided
    if tensor_names is None:
        # Default names for input tensors
        if len(tensor_list) == 7:  # Input tensors
            tensor_names = [
                "means2d",
                "conics",
                "colors",
                "opacities",
                "tile_offsets",
                "tile_gaussian_ids",
                "image_dims",
            ]
        elif len(tensor_list) == 3:  # Output tensors
            tensor_names = ["rendered_colors", "rendered_alphas", "last_ids"]

    # Create and script the container
    container = TensorContainer(tensor_list, tensor_names)

    # Print tensor information for debugging
    print(f"Creating TorchScript container with tensors:")
    for i, tensor in enumerate(tensor_list):
        name = tensor_names[i] if tensor_names and i < len(tensor_names) else f"tensor_{i}"
        print(f"  - {name}: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}")

    try:
        # Script the module to handle the None return from forward().
        # This is more robust than tracing for modules with control flow.
        scripted_module = torch.jit.script(container)

        # Save the module
        torch.jit.save(scripted_module, filepath)
        print(f"Successfully saved TorchScript module to {filepath}")

        # Test load to verify it works
        try:
            test_load = torch.jit.load(filepath)
            print(f"Verified the saved module can be loaded")
        except Exception as e:
            print(f"WARNING: Saved module failed test loading: {e}")

    except Exception as e:
        print(f"ERROR during TorchScript saving: {e}")
        # Fall back to legacy format if TorchScript fails
        print(f"Falling back to legacy format...")
        legacy_path = filepath + ".legacy"
        torch.save(tensor_list, legacy_path, _use_new_zipfile_serialization=False)
        print(f"Saved tensors in legacy format to {legacy_path}")
        return legacy_path

    if tensor_names:
        print(f"Saved {len(tensor_list)} tensors with names: {tensor_names}")
    else:
        print(f"Saved {len(tensor_list)} tensors with generic names")

    return filepath


def main(output_path: str, h=512, w=1024):
    cam_to_world_xform = torch.from_numpy(generate_random_4x4_xform()).to("cuda")
    world_to_cam_xform = torch.linalg.inv(cam_to_world_xform).float()

    # Fix intrinsics to match the actual image size
    # For image size 1024x512, principal point should be around (512, 256)
    focal_length = 18.0  # Reasonable focal length for this image size
    intrinsics = torch.tensor(
        [[focal_length, 0.0, w / 2.0], [0.0, focal_length, h / 2.0], [0.0, 0.0, 1.0]], device="cuda"
    )

    num_layers = 5

    # means3d = torch.cat(
    #     [
    #         create_uniform_grid_points_at_depth(h, w, (i + 1) * 8, intrinsics, cam_to_world_xform, spacing=2).reshape(
    #             -1, 3
    #         )
    #         for i in range(num_layers)
    #     ],
    #     dim=0,
    # )

    means3d = torch.cat(
        [
            generate_center_frame_point_at_depth(h, w, (i + 1) * 8, intrinsics, cam_to_world_xform).reshape(-1, 3)
            for i in range(num_layers)
        ],
        dim=0,
    )

    # Fix opacity values to avoid logit(1.0) = inf and logit(0.0) = -inf
    # Use safe values: 0.2, 0.5, 0.8 for the three depth layers

    opacities = torch.cat(
        [torch.full((means3d.shape[0] // num_layers,), 0.4, device=means3d.device) for _ in range(num_layers)],
        dim=0,
    )
    logit_opacities = torch.logit(opacities)
    print(means3d.shape)
    print(logit_opacities.shape)

    # Generate identity quaternions (no rotation)
    # Identity quaternion is [x=0, y=0, z=0, w=1] representing no rotation
    quats = torch.zeros(means3d.shape[0], 4, device=means3d.device)
    quats[:, 3] = 1.0  # Set w component to 1, others remain 0

    # Fix log scales: ensure positive values before taking log to avoid NaN
    scales = torch.full(
        (means3d.shape[0], 3), 1e-30, device=means3d.device
    )  # torch.abs(torch.randn(means3d.shape[0], 3, device=means3d.device)) + 0.01  # Ensure positive
    log_scales = torch.log(scales)

    sh0 = torch.randn(means3d.shape[0], 1, 3, device=means3d.device)
    shN = torch.randn(means3d.shape[0], 1, 3, device=means3d.device)

    gs3d = GaussianSplat3d.from_tensors(means3d, quats, log_scales, logit_opacities, sh0, shN)

    state = gs3d.project_gaussians_for_images(
        world_to_cam_xform.unsqueeze(0).contiguous(), intrinsics.unsqueeze(0).contiguous(), w, h, 0.1, 10000.0
    )

    image_dims = torch.tensor([w, h], dtype=torch.int32, device="cuda")

    # Save the input tensors using TorchScript format
    inputs = [
        state.means2d,
        state.inv_covar_2d,
        state.opacities,
        state.tile_offsets,
        state.tile_gaussian_ids,
        image_dims,
    ]
    input_names = ["means2d", "conics", "opacities", "tile_offsets", "tile_gaussian_ids", "image_dims"]
    inputs_path = os.path.join(output_path, "gaussian_top_contributors_1point_input.pt")
    print(f"Saving inputs to {inputs_path}")
    save_tensors_torchscript(inputs, inputs_path, input_names)


if __name__ == "__main__":
    tyro.cli(main)
