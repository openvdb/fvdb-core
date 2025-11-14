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

        points = torch.randn(100, 3)
        colors = torch.rand(100, 3)  # Colors in [0, 1]
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

        # Flatten to 1D and convert to torch tensor
        rgba_tensor = torch.from_numpy(rgba_data.reshape(-1))

        scene.add_image("small_image", rgba_tensor, width, height)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
