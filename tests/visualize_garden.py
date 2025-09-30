# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import logging
import pathlib
import time

import torch
import tyro
from fvdb.viz import Viewer

from fvdb import GaussianSplat3d


def main(
    ply_path: pathlib.Path,
    viewer_port: int = 8080,
    verbose: bool = False,
    device: str | torch.device = "cuda",
):
    """
    Visualize a scene in a saved checkpoint file.

    Args:
        ply_path (pathlib.Path): Path to a PLY file containing the Gaussian splat model.
        viewer_port (int): The port to expose the viewer server on
        verbose (bool): If True, then the viewer will log verbosely.
        device (str | torch.device): Device to use for computation (default is "cuda").
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s : %(message)s")

    viewer = Viewer(port=viewer_port, verbose=verbose)

    model, metadata = GaussianSplat3d.from_ply(ply_path, device)
    viewer.add_gaussian_splat_3d(name="model", gaussian_splat_3d=model)
    assert isinstance(metadata["camera_to_world_matrices"], torch.Tensor)
    assert isinstance(metadata["projection_matrices"], torch.Tensor)
    first_cam_pos = metadata["camera_to_world_matrices"][0, :3, 3]
    viewer.set_camera_lookat(
        camera_origin=first_cam_pos, lookat_point=model.means.mean(dim=0).cpu().numpy(), up_direction=[0, 0, 1]
    )

    camera_to_world_matrices = metadata["camera_to_world_matrices"].to(device)
    projection_matrices = metadata["projection_matrices"].to(device)
    cam_view = viewer.add_camera_view("test", camera_to_world_matrices, projection_matrices)

    cam_view.axis_length = 0.2
    cam_view.axis_thickness = 2.0
    cam_view.frustum_scale = 0.5
    cam_view.frustum_line_width = 4.0

    logger = logging.getLogger("visualize")
    logger.info("Viewer running... Ctrl+C to exit.")
    time.sleep(1000000)


if __name__ == "__main__":
    tyro.cli(main)
