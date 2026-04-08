# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Functional API for MCMC-based Gaussian operations.

relocate_gaussians and add_noise_to_means dispatch directly to C++ free
functions via pybind (gaussianRelocation, gaussianMCMCAddNoise).
"""
