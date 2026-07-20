#!/usr/bin/env python3
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

import sys
import unittest
import warnings
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

import fvdb

PORT = 8080


def _make_gaussian_splat_view_tensors(sh_ordering="rgb_rgb_rgb"):
    num_gaussians = 3
    num_channels = 3
    num_higher_order_coefficients = 8
    tensors = {
        "means": torch.randn(num_gaussians, 3),
        "quats": torch.randn(num_gaussians, 4),
        "log_scales": torch.randn(num_gaussians, 3),
        "logit_opacities": torch.randn(num_gaussians),
    }
    if sh_ordering == "rgb_rgb_rgb":
        tensors["sh0"] = torch.randn(num_gaussians, 1, num_channels)
        tensors["shN"] = torch.randn(num_gaussians, num_higher_order_coefficients, num_channels)
    else:
        tensors["sh0"] = torch.randn(num_gaussians, num_channels, 1)
        tensors["shN"] = torch.randn(num_gaussians, num_channels, num_higher_order_coefficients)
    return tensors


class _SceneWithoutServer(fvdb.viz.Scene):
    def __init__(self, name="test_gaussian_splat_contract"):
        self._name = name

    def __del__(self):
        pass


class TestGaussianSplatViewData:
    def test_is_frozen_and_keeps_zero_copy_tensor_references(self):
        tensors = _make_gaussian_splat_view_tensors()

        data = fvdb.viz.GaussianSplatViewData(**tensors)

        for name, tensor in tensors.items():
            assert getattr(data, name) is tensor
        with pytest.raises(FrozenInstanceError):
            data.means = torch.empty_like(data.means)

    @pytest.mark.parametrize("sh_ordering", ["rgb_rgb_rgb", "rrr_ggg_bbb"])
    def test_accepts_both_spherical_harmonics_layouts(self, sh_ordering):
        tensors = _make_gaussian_splat_view_tensors(sh_ordering)

        data = fvdb.viz.GaussianSplatViewData(**tensors, sh_ordering=sh_ordering)

        assert data.sh_ordering == sh_ordering

    @pytest.mark.parametrize(
        ("field", "replacement", "error_type"),
        [
            ("means", torch.randn(3, 2), ValueError),
            ("quats", torch.randn(2, 4), ValueError),
            ("log_scales", torch.ones(3, 3, dtype=torch.int64), TypeError),
            ("logit_opacities", torch.randn(3, 1), ValueError),
            ("sh0", torch.randn(3, 3), ValueError),
            ("shN", torch.randn(3, 8, 4), ValueError),
        ],
    )
    def test_rejects_invalid_tensor_contracts(self, field, replacement, error_type):
        tensors = _make_gaussian_splat_view_tensors()
        tensors[field] = replacement

        with pytest.raises(error_type):
            fvdb.viz.GaussianSplatViewData(**tensors)

    def test_rejects_mixed_dtypes_and_unknown_spherical_harmonics_layout(self):
        tensors = _make_gaussian_splat_view_tensors()
        tensors["quats"] = tensors["quats"].double()
        with pytest.raises(TypeError, match="same dtype|must have dtype"):
            fvdb.viz.GaussianSplatViewData(**tensors)

        tensors = _make_gaussian_splat_view_tensors()
        with pytest.raises(ValueError, match="sh_ordering"):
            fvdb.viz.GaussianSplatViewData(**tensors, sh_ordering="unknown")


class TestGaussianSplatSceneContract:
    @pytest.mark.parametrize("mode", list(fvdb.viz.ShOrderingMode))
    def test_view_stores_spherical_harmonics_layout(self, mode):
        view = object.__new__(fvdb.viz.GaussianSplat3dView)

        with patch.object(view, "_get_view", return_value=SimpleNamespace()):
            view.sh_ordering_mode = mode

        assert view.sh_ordering_mode is mode

    def test_raw_tensor_api_forwards_renderer_inputs(self):
        scene = _SceneWithoutServer()
        tensors = _make_gaussian_splat_view_tensors("rrr_ggg_bbb")

        with patch("fvdb.viz._scene.GaussianSplat3dView", autospec=True) as view_type:
            result = scene.add_gaussian_splat_tensors(
                "raw tensors",
                **tensors,
                sh_ordering="rrr_ggg_bbb",
                tile_size=8,
                min_radius_2d=0.25,
                eps_2d=0.1,
                antialias=True,
                sh_degree_to_use=2,
            )

        assert result is view_type.return_value
        view_type.assert_called_once()
        forwarded = dict(view_type.call_args.kwargs)
        forwarded.pop("_private")
        assert forwarded == {
            "scene_name": scene._name,
            "name": "raw tensors",
            **tensors,
            "tile_size": 8,
            "min_radius_2d": 0.25,
            "eps_2d": 0.1,
            "antialias": True,
            "sh_degree_to_use": 2,
            "sh_ordering_mode": fvdb.viz.ShOrderingMode.RRR_GGG_BBB,
        }

    def test_view_data_api_delegates_to_raw_tensor_api(self):
        scene = _SceneWithoutServer()
        tensors = _make_gaussian_splat_view_tensors("rrr_ggg_bbb")
        data = fvdb.viz.GaussianSplatViewData(**tensors, sh_ordering="rrr_ggg_bbb")

        with patch.object(
            _SceneWithoutServer,
            "add_gaussian_splat_tensors",
            autospec=True,
            return_value=object(),
        ) as add_tensors:
            result = scene.add_gaussian_splat_3d(
                "view data",
                data,
                tile_size=32,
                min_radius_2d=0.5,
                eps_2d=0.2,
                antialias=True,
                sh_degree_to_use=1,
            )

        assert result is add_tensors.return_value
        add_tensors.assert_called_once_with(
            scene,
            "view data",
            **tensors,
            sh_ordering="rrr_ggg_bbb",
            tile_size=32,
            min_radius_2d=0.5,
            eps_2d=0.2,
            antialias=True,
            sh_degree_to_use=1,
        )

    def test_legacy_model_shape_warns_and_remains_compatible(self):
        scene = _SceneWithoutServer()
        tensors = _make_gaussian_splat_view_tensors()
        legacy_model = SimpleNamespace(**tensors)

        with patch.object(
            _SceneWithoutServer,
            "add_gaussian_splat_tensors",
            autospec=True,
            return_value=object(),
        ) as add_tensors:
            with pytest.warns(DeprecationWarning, match="GaussianSplatViewData"):
                scene.add_gaussian_splat_3d("legacy", legacy_model)

        add_tensors.assert_called_once_with(
            scene,
            "legacy",
            **tensors,
            sh_ordering="rgb_rgb_rgb",
            tile_size=16,
            min_radius_2d=0.0,
            eps_2d=0.3,
            antialias=False,
            sh_degree_to_use=-1,
        )


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


class TestSceneParamWidgets(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            if fvdb.viz._viewer_server._viewer_server_cpp is None:
                fvdb.viz.init(ip_address="127.0.0.1", port=PORT, verbose=False)
        except Exception as e:
            pytest.skip(f"Could not initialize viewer server: {e}")

    def test_add_slider_round_trip(self):
        scene = fvdb.viz.Scene("test_widget_slider")
        slider = scene.add_slider("mask_blend", min=0.0, max=1.0, initial=0.5, step=0.01)

        assert isinstance(slider, fvdb.viz.SliderView)
        assert slider.name == "mask_blend"
        assert slider.scene_name == "test_widget_slider"
        assert slider.min == pytest.approx(0.0)
        assert slider.max == pytest.approx(1.0)
        assert slider.step == pytest.approx(0.01)
        assert slider.value == pytest.approx(0.5)

        slider.value = 0.75
        assert slider.value == pytest.approx(0.75)

        with pytest.raises(ValueError):
            slider.value = 2.0
        with pytest.raises(ValueError):
            scene.add_slider("bad_bounds", min=1.0, max=0.0, initial=0.5)
        with pytest.raises(ValueError):
            scene.add_slider("bad_step", min=0.0, max=1.0, initial=0.5, step=0.0)

    def test_add_number_round_trip(self):
        scene = fvdb.viz.Scene("test_widget_number")
        number = scene.add_number("mask_scale", initial=1.0, min=0.1, max=10.0, step=0.1)

        assert isinstance(number, fvdb.viz.NumberView)
        assert number.name == "mask_scale"
        assert number.min == pytest.approx(0.1)
        assert number.max == pytest.approx(10.0)
        assert number.step == pytest.approx(0.1)
        assert number.value == pytest.approx(1.0)

        number.value = 5.0
        assert number.value == pytest.approx(5.0)
        with pytest.raises(ValueError):
            number.value = 100.0

        unbounded = scene.add_number("bare_number", initial=42.0)
        assert unbounded.min is None
        assert unbounded.max is None
        unbounded.value = -1e6  # No bounds, anything goes
        assert unbounded.value == pytest.approx(-1e6)

    def test_add_text_round_trip(self):
        scene = fvdb.viz.Scene("test_widget_text")
        text = scene.add_text("language_query", initial="a red chair", max_length=128)

        assert isinstance(text, fvdb.viz.TextView)
        assert text.name == "language_query"
        assert text.max_length == 128
        assert text.value == "a red chair"
        assert text.commit_on_enter is False

        text.value = "the green vase"
        assert text.value == "the green vase"
        text.value = ""
        assert text.value == ""

        # max_length includes NUL terminator, so capacity-1 is the limit.
        text.value = "x" * (text.max_length - 1)
        assert text.value == "x" * (text.max_length - 1)
        with pytest.raises(ValueError):
            text.value = "x" * text.max_length

        with pytest.raises(ValueError):
            scene.add_text("too_short", max_length=1)
        with pytest.raises(ValueError):
            scene.add_text("too_long_initial", initial="x" * 64, max_length=8)

    def test_text_commit_on_enter_metadata(self):
        scene = fvdb.viz.Scene("test_widget_text_commit_metadata")
        text = scene.add_text("language_query", initial="hello", max_length=64, commit_on_enter=True)
        server = fvdb.viz._viewer_server._get_viewer_server_cpp()

        assert text.commit_on_enter is True
        assert text.value == "hello"
        assert list(server.scene_widget_names("test_widget_text_commit_metadata")) == ["language_query"]
        assert int(server.get_text("language_query").submit_counter) == 0

        text.value = "world"
        assert int(server.get_text("language_query").submit_counter) == 0

    def test_text_on_submit_requires_commit_on_enter(self):
        scene = fvdb.viz.Scene("test_widget_text_on_submit_guard")
        live = scene.add_text("live", initial="", max_length=32)
        with pytest.raises(RuntimeError):
            live.on_submit(lambda v: None)

    def test_text_on_submit_fires_on_counter_increment(self):
        scene = fvdb.viz.Scene("test_widget_text_on_submit_fires")
        text = scene.add_text("query", initial="alpha", max_length=64, commit_on_enter=True)
        seen: list[str] = []

        @text.on_submit
        def _on_submit(value: str) -> None:
            seen.append(value)

        scene.poll_widgets()
        assert seen == []

        text.value = "beta"
        scene.poll_widgets()
        assert seen == []

        # Simulate an editor-side Enter press.
        text._last_submit_counter = -1
        scene.poll_widgets()
        assert seen == ["beta"]

        scene.poll_widgets()
        assert seen == ["beta"]

    def test_add_checkbox_round_trip(self):
        scene = fvdb.viz.Scene("test_widget_checkbox")
        checkbox = scene.add_checkbox("lock_pca", initial=False)

        assert isinstance(checkbox, fvdb.viz.CheckboxView)
        assert checkbox.name == "lock_pca"
        assert checkbox.value is False

        checkbox.value = True
        assert checkbox.value is True
        checkbox.value = False
        assert checkbox.value is False

    def test_widgets(self):
        scene = fvdb.viz.Scene("test_widget_palette")

        mask_blend = scene.add_slider("mask_blend", min=0.0, max=1.0, initial=0.5, step=0.01)
        scene.add_slider("mask_scale", min=0.1, max=10.0, initial=1.0, step=0.1)
        language_query = scene.add_text("language_query", initial="a red chair", max_length=128)
        lock_pca = scene.add_checkbox("lock_pca", initial=False)

        widget_names = list(fvdb.viz._viewer_server._get_viewer_server_cpp().scene_widget_names("test_widget_palette"))
        assert widget_names == ["mask_blend", "mask_scale", "language_query", "lock_pca"]

        mask_blend.value = 0.7
        mask_scale_again = scene.add_slider("mask_scale", min=0.1, max=20.0, initial=1.0, step=0.1)
        assert mask_blend.value == pytest.approx(0.7)
        assert mask_scale_again.max == pytest.approx(20.0)

        language_query.value = "the green vase"
        lock_pca.value = True
        assert language_query.value == "the green vase"
        assert lock_pca.value is True

    def test_on_update_callback_via_poll(self):
        scene = fvdb.viz.Scene("test_widget_callback")
        slider = scene.add_slider("mask_blend", min=0.0, max=1.0, initial=0.5)
        text = scene.add_text("query", initial="hello", max_length=64)

        seen_slider: list[float] = []
        seen_text: list[str] = []

        @slider.on_update
        def _on_slider(value: float) -> None:
            seen_slider.append(value)

        @text.on_update
        def _on_text(value: str) -> None:
            seen_text.append(value)

        changed = scene.poll_widgets()
        assert changed == []
        assert seen_slider == []
        assert seen_text == []

        slider.value = 0.8
        text.value = "world"
        changed = sorted(scene.poll_widgets())
        assert changed == ["mask_blend", "query"]
        assert seen_slider == [pytest.approx(0.8)]
        assert seen_text == ["world"]

        changed = scene.poll_widgets()
        assert changed == []
        assert len(seen_slider) == 1
        assert len(seen_text) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
