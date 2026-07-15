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

PORT = 8080


class TestViewerServer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            fvdb.viz.init(ip_address="127.0.0.1", port=PORT, verbose=False)
        except Exception as e:
            pytest.skip(f"Could not initialize viewer server: {e}")

    def test_init(self):
        assert fvdb.viz._viewer_server._viewer_server_cpp is not None

        # Call init again - should show warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fvdb.viz.init(ip_address="127.0.0.1", port=PORT)

            assert len(w) == 1
            assert "already initialized" in str(w[0].message)

    def test_show(self):
        fvdb.viz.show()


class TestViewerScene(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            if fvdb.viz._viewer_server._viewer_server_cpp is None:
                fvdb.viz.init(ip_address="127.0.0.1", port=PORT, verbose=False)
        except Exception as e:
            pytest.skip(f"Could not initialize viewer server: {e}")

    def test_scene_creation(self):
        scene = fvdb.viz.Scene("test_scene_creation")
        assert scene._name == "test_scene_creation"

    def test_get_scene(self):
        scene = fvdb.viz.get_scene("test_get_scene")
        assert scene._name == "test_get_scene"

    def test_add_point_cloud(self):
        scene = fvdb.viz.Scene("test_point_cloud")

        points = np.random.randn(100, 3)
        colors = np.random.rand(100, 3)  # Colors in [0, 1]
        point_size = 2.0

        view = scene.add_point_cloud("test_pc", points, colors, point_size)
        assert view is not None

    def test_add_cameras(self):
        scene = fvdb.viz.Scene("test_cameras")

        num_cameras = 3
        camera_to_world = torch.eye(4).unsqueeze(0).repeat(num_cameras, 1, 1)
        projection = torch.eye(3).unsqueeze(0).repeat(num_cameras, 1, 1)

        view = scene.add_cameras("test_cams", camera_to_world, projection)
        assert view is not None

    def test_scene_reset(self):
        scene = fvdb.viz.Scene("test_reset")

        scene.reset()

    def test_multiple_scenes(self):
        scene1 = fvdb.viz.get_scene("Scene 1")
        scene2 = fvdb.viz.get_scene("Scene 2")

        points = torch.randn(20, 3)
        colors = torch.rand(20, 3)

        scene1.add_point_cloud("pc1", points, colors, 2.0)
        scene2.add_point_cloud("pc2", points * 2, colors, 2.0)

        assert scene1._name == "Scene 1"
        assert scene2._name == "Scene 2"

    def test_add_image(self):
        scene = fvdb.viz.Scene("test_image")

        width = 64
        height = 64

        rgba_data = np.zeros((height, width, 4), dtype=np.uint8)
        rgba_data[:, :, 0] = 255
        rgba_data[:, :, 1] = 128
        rgba_data[:, :, 2] = 0
        rgba_data[:, :, 3] = 255

        # Flatten to 1D numpy array
        rgba_flat = rgba_data.reshape(-1)

        view = scene.add_image("small_image", rgba_flat, width, height)
        assert view is not None
        assert isinstance(view, fvdb.viz.ImageView)
        assert view.name == "small_image"
        assert view.scene_name == "test_image"
        assert view.width == width
        assert view.height == height

        # Test update method
        rgba_data2 = np.zeros((height, width, 4), dtype=np.uint8)
        rgba_data2[:, :, 2] = 255
        rgba_data2[:, :, 3] = 255
        rgba_flat2 = rgba_data2.reshape(-1)
        view.update(rgba_flat2)

    def test_add_level_set(self):
        scene = fvdb.viz.Scene("test_level_set")

        ijk = torch.randint(0, 16, (50, 3), dtype=torch.int32)
        grid = fvdb.Grid.from_ijk(ijk)
        sdf = fvdb.JaggedTensor([torch.randn(grid.num_voxels, dtype=torch.float32)])

        view = scene.add_level_set("terrain", grid, sdf)
        assert view is not None
        assert isinstance(view, fvdb.viz.LevelSetView)
        assert view.name == "terrain"
        assert view.scene_name == "test_level_set"
        # A single grid maps to a single view named exactly `name` (no `[i]` suffix).
        assert view._view_names == ["terrain"]

        # Update with fresh SDF values on the same grid.
        sdf2 = fvdb.JaggedTensor([torch.randn(grid.num_voxels, dtype=torch.float32)])
        view.update(grid, sdf2)
        assert view._view_names == ["terrain"]

    def test_add_fog_volume(self):
        scene = fvdb.viz.Scene("test_fog_volume")

        ijk = torch.randint(0, 16, (50, 3), dtype=torch.int32)
        grid = fvdb.Grid.from_ijk(ijk)
        density = fvdb.JaggedTensor([torch.rand(grid.num_voxels, dtype=torch.float32)])

        view = scene.add_fog_volume("cloud", grid, density)
        assert view is not None
        assert isinstance(view, fvdb.viz.FogVolumeView)
        assert view.name == "cloud"
        assert view.scene_name == "test_fog_volume"
        assert view._view_names == ["cloud"]

    def test_add_level_set_gridbatch(self):
        # A multi-grid GridBatch is expanded into one editor view per grid, named `name[i]`,
        # because the nanovdb-editor renders only one grid per view.
        scene = fvdb.viz.Scene("test_level_set_batch")

        ijk0 = torch.randint(0, 16, (40, 3), dtype=torch.int32)
        ijk1 = torch.randint(0, 16, (30, 3), dtype=torch.int32)
        grid_batch = fvdb.GridBatch.from_ijk(fvdb.JaggedTensor([ijk0, ijk1]))
        assert grid_batch.grid_count == 2

        sdf = grid_batch.jagged_like(torch.randn(grid_batch.total_voxels, dtype=torch.float32))

        view = scene.add_level_set("surf", grid_batch, sdf)
        assert isinstance(view, fvdb.viz.LevelSetView)
        assert view._view_names == ["surf[0]", "surf[1]"]

    def test_add_level_set_invalid(self):
        scene = fvdb.viz.Scene("test_level_set_invalid")

        ijk = torch.randint(0, 16, (50, 3), dtype=torch.int32)
        grid = fvdb.Grid.from_ijk(ijk)

        # Wrong dtype (float64 instead of float32).
        with pytest.raises(TypeError):
            scene.add_level_set(
                "bad_dtype", grid, fvdb.JaggedTensor([torch.randn(grid.num_voxels, dtype=torch.float64)])
            )

        # Wrong length (does not match voxel count).
        with pytest.raises(ValueError):
            scene.add_level_set(
                "bad_len", grid, fvdb.JaggedTensor([torch.randn(grid.num_voxels + 1, dtype=torch.float32)])
            )

        # Not a Grid or GridBatch.
        with pytest.raises(TypeError):
            scene.add_level_set(
                "bad_grid",
                ijk,  # type: ignore[arg-type]
                fvdb.JaggedTensor([torch.randn(grid.num_voxels, dtype=torch.float32)]),
            )

    def test_camera_fov(self):
        scene = fvdb.viz.Scene("test_camera_fov")

        fov = 1.2
        scene.camera_fov = fov
        assert scene.camera_fov == pytest.approx(fov)

        scene.camera_fov = 0.5
        assert scene.camera_fov == pytest.approx(0.5)

        with pytest.raises(ValueError):
            scene.camera_fov = 0.0

        with pytest.raises(ValueError):
            scene.camera_fov = -1.0

        with pytest.raises(ValueError):
            scene.camera_fov = np.pi

        with pytest.raises(ValueError):
            scene.camera_fov = float("inf")

        with pytest.raises(ValueError):
            scene.camera_fov = float("nan")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
