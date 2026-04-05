# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import numpy as np
import torch


def generate_random_4x4_xform():
    """
    Generate a 4x4 transformation matrix with random translation between -100,100
    and random rotation.

    Returns:
        np.ndarray: 4x4 transformation matrix
    """
    # Generate random translation between -100 and 100
    translation = np.random.uniform(-100, 100, 3)

    # Generate a proper random rotation matrix using Rodrigues' rotation formula
    # First generate a random axis (normalized)
    axis = np.random.randn(3)
    axis = axis / np.linalg.norm(axis)

    # Random angle between 0 and 2*pi
    angle = np.random.uniform(0, 2 * np.pi)

    # Rodrigues' rotation formula to create rotation matrix
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)

    # Cross-product matrix of axis
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])

    # Rotation matrix using Rodrigues' formula: R = I + sin(θ)K + (1-cos(θ))K²
    R_3x3 = np.eye(3) + sin_angle * K + (1 - cos_angle) * np.dot(K, K)

    # Construct 4x4 transformation matrix
    transform = np.eye(4)
    transform[:3, :3] = R_3x3
    transform[:3, 3] = translation

    return transform


def create_uniform_grid_points_at_depth(h, w, depth, intrinsics, cam_to_world_xform, device="cuda", spacing=1):
    """
    Create a uniform grid of 3D world points at a specified depth from a camera.

    Args:
        h: Image height
        w: Image width
        depth: Depth from camera center
        intrinsics: Camera intrinsic matrix (3x3)
        cam_to_world_xform: Camera to world transformation matrix (4x4)
        device: Device to create tensors on
        spacing: Pixel spacing for grid sampling (1=every pixel, 2=every 2nd pixel, etc.)

    Returns:
        torch.Tensor: Grid of 3D world points (h//spacing, w//spacing, 3)
    """
    # Create pixel coordinates with specified spacing
    # Sample every 'spacing' pixels starting from 0
    x_coords = torch.arange(0, w, spacing, device=device, dtype=torch.float32)
    y_coords = torch.arange(0, h, spacing, device=device, dtype=torch.float32)

    # Create meshgrid - x varies along width, y varies along height
    y_grid, x_grid = torch.meshgrid(y_coords + 0.5, x_coords + 0.5, indexing="ij")

    # Stack to create (h, w, 2) grid where each pixel has [x, y] coordinates
    grid = torch.stack([x_grid, y_grid], dim=-1)  # Shape: (h, w, 2)

    # Reshape grid to (h*w, 2) for easier processing
    pixels = grid.reshape(-1, 2)  # Shape: (h*w, 2) where each row is [x, y]

    # Convert to homogeneous coordinates by adding ones
    pixels_homo = torch.cat([pixels, torch.ones(pixels.shape[0], 1, device=device)], dim=1)  # Shape: (h*w, 3) [x, y, 1]

    # Get inverse intrinsics to convert from pixel space to camera space
    intrinsics_inv = torch.inverse(intrinsics)

    # Convert pixels to normalized camera coordinates (camera rays)
    camera_rays = (intrinsics_inv @ pixels_homo.T).T  # Shape: (h*w, 3)

    # Scale rays by desired depth to get 3D camera space points
    camera_points_3d = camera_rays * depth  # Shape: (h*w, 3)

    # Convert to homogeneous coordinates for transformation
    camera_points_homo = torch.cat(
        [camera_points_3d, torch.ones(camera_points_3d.shape[0], 1, device=device)], dim=1
    )  # Shape: (h*w, 4)

    # Transform from camera space to world space using cam_to_world_xform
    cam_to_world_tensor = cam_to_world_xform.float()
    world_points_homo = (cam_to_world_tensor @ camera_points_homo.T).T  # Shape: (h*w, 4)

    # Extract 3D world coordinates (ignore homogeneous coordinate)
    world_points = world_points_homo[:, :3]  # Shape: (h*w, 3)

    # Reshape back to grid format using actual grid dimensions
    grid_h = len(y_coords)
    grid_w = len(x_coords)
    world_points_grid = world_points.reshape(grid_h, grid_w, 3)  # Shape: (grid_h, grid_w, 3)
    return world_points_grid


def generate_center_frame_point_at_depth(h, w, depth, intrinsics, cam_to_world_xform, device="cuda"):
    """
    Generate a single 3D world point at the center of the camera image at a specified depth.

    Args:
        h: Image height
        w: Image width
        depth: Depth from camera center
        intrinsics: Camera intrinsic matrix (3x3)
        cam_to_world_xform: Camera to world transformation matrix (4x4)
        device: Device to create tensors on

    Returns:
        torch.Tensor: Single 3D world point (1, 3)
    """
    # Calculate center pixel coordinates
    center_x = (w - 1) / 2.0
    center_y = (h - 1) / 2.0

    # Create center pixel coordinate as tensor
    center_pixel = torch.tensor([[center_x, center_y]], device=device)  # Shape: (1, 2)

    # Convert to homogeneous coordinates by adding ones
    center_pixel_homo = torch.cat([center_pixel, torch.ones(1, 1, device=device)], dim=1)  # Shape: (1, 3)

    # Get inverse intrinsics to convert from pixel space to camera space
    intrinsics_inv = torch.inverse(intrinsics)

    # Convert pixel to normalized camera coordinates (camera ray)
    camera_ray = (intrinsics_inv @ center_pixel_homo.T).T  # Shape: (1, 3)

    # Scale ray by desired depth to get 3D camera space point
    camera_point_3d = camera_ray * depth  # Shape: (1, 3)

    # Convert to homogeneous coordinates for transformation
    camera_point_homo = torch.cat([camera_point_3d, torch.ones(1, 1, device=device)], dim=1)  # Shape: (1, 4)

    # Transform from camera space to world space using cam_to_world_xform
    cam_to_world_tensor = cam_to_world_xform.float()
    world_point_homo = (cam_to_world_tensor @ camera_point_homo.T).T  # Shape: (1, 4)

    # Extract 3D world coordinates (ignore homogeneous coordinate)
    world_point = world_point_homo[:, :3]  # Shape: (1, 3)

    return world_point
