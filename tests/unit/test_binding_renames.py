# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Smoke tests that every renamed/new C++ binding is accessible and every
deleted binding is absent."""

import unittest

from fvdb import _fvdb_cpp as _C


class TestBindingRenames(unittest.TestCase):
    """Verify the Gaussian splat C++ bindings match the refactored names."""

    EXPECTED_BINDINGS = [
        # Utility
        "relocate_gaussians",
        "add_noise_to_gaussian_means",
        "save_gaussians_ply",
        "load_gaussians_ply",
        # Analysis
        "count_contributing_gaussians",
        "count_contributing_gaussians_sparse",
        "identify_contributing_gaussians",
        "identify_contributing_gaussians_sparse",
        # Analytic projection fwd/bwd
        "project_gaussians_analytic_fwd",
        "project_gaussians_analytic_bwd",
        "project_gaussians_analytic_jagged_fwd",
        "project_gaussians_analytic_jagged_bwd",
        # UT projection fwd
        "project_gaussians_ut_fwd",
        # SH evaluation fwd/bwd
        "eval_gaussian_sh_fwd",
        "eval_gaussian_sh_bwd",
        # Dense rasterization fwd/bwd
        "rasterize_screen_space_gaussians_fwd",
        "rasterize_screen_space_gaussians_bwd",
        # Sparse rasterization fwd/bwd
        "rasterize_screen_space_gaussians_sparse_fwd",
        "rasterize_screen_space_gaussians_sparse_bwd",
        # World-space rasterization fwd/bwd
        "rasterize_world_space_gaussians_fwd",
        "rasterize_world_space_gaussians_bwd",
        # Tile intersection
        "intersect_gaussian_tiles",
        "intersect_gaussian_tiles_sparse",
        "build_sparse_gaussian_tile_layout",
    ]

    DELETED_BINDINGS = [
        # Old gsplat_ prefixed names
        "check_gaussian_state",
        "gsplat_check_state",
        "gsplat_projection_fwd",
        "gsplat_projection_bwd",
        "gsplat_projection_jagged_fwd",
        "gsplat_projection_jagged_bwd",
        "gsplat_rasterize_fwd",
        "gsplat_rasterize_bwd",
        "gsplat_rasterize_sparse_fwd",
        "gsplat_rasterize_sparse_bwd",
        "gsplat_rasterize_from_world_fwd",
        "gsplat_rasterize_from_world_bwd",
        "gsplat_tile_intersection",
        "gsplat_sh_eval_fwd",
        "gsplat_sh_eval_bwd",
        "gsplat_render_crop_from_projected",
        "render_crop_from_projected_gaussians",
        "gsplat_render_num_contributing",
        "gsplat_render_contributing_ids",
        "gsplat_sparse_render_num_contributing",
        "gsplat_sparse_render_contributing_ids",
        "gsplat_load_ply",
        "gsplat_save_ply",
        "gsplat_relocate_gaussians",
        "gsplat_add_noise_to_means",
        # Deleted pipeline / utility bindings
        "gsplat_eval_sh",
        "gsplat_project_gaussians_analytic",
        "gsplat_project_gaussians_ut",
        "gsplat_project_gaussians_for_camera_with_accum",
        "gsplat_sparse_project_gaussians_for_camera",
        "gsplat_sparse_project_gaussians_ut",
        "gsplat_sparse_render",
        "gsplat_rasterize_from_world",
        "gsplat_render_depth_from_world",
        "evaluate_spherical_harmonics",
        # Old query_ prefixed names
        "query_num_contributing_gaussians",
        "query_num_contributing_gaussians_sparse",
        "query_contributing_gaussian_ids",
        "query_contributing_gaussian_ids_sparse",
    ]

    def test_expected_bindings_exist(self):
        for name in self.EXPECTED_BINDINGS:
            with self.subTest(name=name):
                self.assertTrue(hasattr(_C, name), f"Binding '{name}' should exist on _fvdb_cpp but is missing")
                self.assertTrue(callable(getattr(_C, name)), f"Binding '{name}' should be callable")

    def test_deleted_bindings_absent(self):
        for name in self.DELETED_BINDINGS:
            with self.subTest(name=name):
                self.assertFalse(hasattr(_C, name), f"Binding '{name}' should have been deleted but still exists")


if __name__ == "__main__":
    unittest.main()
