// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_VOLUMERENDER_H
#define FVDB_DETAIL_OPS_VOLUMERENDER_H

#include <torch/types.h>

#include <tuple>

namespace fvdb {
namespace detail {
namespace ops {

/// @brief Forward volume render along jagged per-ray sample sequences.
///
/// Composites per-sample radiance with Beer-Lambert transmittance computed
/// from the per-sample densities and ray-segment lengths, terminating each
/// ray once its accumulated transmittance ``T`` satisfies
/// ``T <= tsmtThreshold``.
///
/// @note When ``needsBackward`` is false, the kernel skips the per-sample
/// ``ws`` store and the per-ray ``depth`` / ``totalSamples`` stores that are
/// only consumed by the backward pass, and returns size-0 placeholder tensors
/// for those three outputs. This lets pure inference callers avoid the large
/// per-sample ``ws`` allocation and its corresponding global-memory traffic.
///
/// @param sigmas         Per-sample density, shape [N_samples].
/// @param rgbs           Per-sample radiance, shape [N_samples, C], where
///                       ``1 <= C <= MAX_VOLUME_RENDER_CHANNELS`` (currently
///                       16; enforced by the host-side checks).
/// @param deltaTs        Per-sample ray-segment lengths, shape [N_samples].
/// @param ts             Per-sample ray t-values (used for the depth output),
///                       shape [N_samples].
/// @param jOffsets       CSR-style offsets delimiting each ray's sample span,
///                       shape [N_rays + 1].
/// @param tsmtThreshold  Transmittance early-termination threshold.
/// @param needsBackward  If true, populate all five outputs; otherwise return
///                       size-0 placeholders for ``depth``, ``ws`` and
///                       ``totalSamples`` (see note above).
/// @return Tuple ``(rgb, depth, opacity, ws, totalSamples)``.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
volumeRender(const torch::Tensor &sigmas,
             const torch::Tensor &rgbs,
             const torch::Tensor &deltaTs,
             const torch::Tensor &ts,
             const torch::Tensor &jOffsets,
             double tsmtThreshold,
             bool needsBackward = true);

/// @brief Backward pass corresponding to ``volumeRender``.
///
/// Computes gradients of a downstream loss with respect to the per-sample
/// densities ``sigmas`` and radiances ``rgbs``, given the upstream gradients
/// on the five forward outputs and the tensors saved from the forward pass.
/// The forward must have been invoked with ``needsBackward = true`` so that
/// ``ws``, ``depth`` and ``opacity`` are populated.
///
/// @param dLdOpacity     Upstream gradient on the forward ``opacity`` output,
///                       shape [N_rays].
/// @param dLdDepth       Upstream gradient on the forward ``depth`` output,
///                       shape [N_rays].
/// @param dLdRgb         Upstream gradient on the forward ``rgb`` output,
///                       shape [N_rays, C].
/// @param dLdWs          Upstream gradient on the forward ``ws`` output,
///                       shape [N_samples].
/// @param sigmas         Per-sample density from the forward, shape [N_samples].
/// @param rgbs           Per-sample radiance from the forward,
///                       shape [N_samples, C].
/// @param ws             Per-sample compositing weights saved by the forward,
///                       shape [N_samples].
/// @param deltas         Per-sample ray-segment lengths from the forward,
///                       shape [N_samples].
/// @param ts             Per-sample ray t-values from the forward,
///                       shape [N_samples].
/// @param jOffsets       CSR-style offsets delimiting each ray's sample span,
///                       shape [N_rays + 1].
/// @param opacity        Forward ``opacity`` output saved for backward,
///                       shape [N_rays].
/// @param depth          Forward ``depth`` output saved for backward,
///                       shape [N_rays].
/// @param rgb            Forward ``rgb`` output saved for backward,
///                       shape [N_rays, C].
/// @param tsmtThreshold  Transmittance early-termination threshold (must match
///                       the value used in the corresponding forward call).
/// @return Tuple ``(dLdSigmas, dLdRgbs)``.
std::tuple<torch::Tensor, torch::Tensor> volumeRenderBackward(const torch::Tensor &dLdOpacity,
                                                              const torch::Tensor &dLdDepth,
                                                              const torch::Tensor &dLdRgb,
                                                              const torch::Tensor &dLdWs,
                                                              const torch::Tensor &sigmas,
                                                              const torch::Tensor &rgbs,
                                                              const torch::Tensor &ws,
                                                              const torch::Tensor &deltas,
                                                              const torch::Tensor &ts,
                                                              const torch::Tensor &jOffsets,
                                                              const torch::Tensor &opacity,
                                                              const torch::Tensor &depth,
                                                              const torch::Tensor &rgb,
                                                              float tsmtThreshold);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_VOLUMERENDER_H
