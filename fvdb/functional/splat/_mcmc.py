# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Functional API for MCMC-based Gaussian operations.

Note: relocate_gaussians and add_noise_to_means are currently thin dispatches
on the C++ GaussianSplat3d class. They will become direct pybind free functions
when the class is removed in Milestone 9. For now, they remain accessible only
through the GaussianSplat3d Python class.
"""
