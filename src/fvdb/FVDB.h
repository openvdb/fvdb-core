// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_FVDB_H
#define FVDB_FVDB_H

#include <fvdb/GridBatchData.h>
#include <fvdb/JaggedTensor.h>

#include <vector>

namespace fvdb {

torch::Device parseDeviceString(const std::string &string);

/// @brief Forward volume render along jagged per-ray sample sequences.
///
/// Composites per-sample radiance with Beer-Lambert transmittance computed
/// from the per-sample densities and ray-segment lengths, terminating each
/// ray once its accumulated transmittance ``T`` satisfies
/// ``T <= transmittanceThresh``.
///
/// @note When ``needsBackward`` is false, the kernel skips the per-sample
/// ``ws`` store and the per-ray ``depth`` / ``totalSamples`` stores that are
/// only consumed by the backward pass, and returns size-0 placeholder tensors
/// for those three outputs. This lets pure inference callers avoid the large
/// per-sample ``ws`` allocation and its corresponding global-memory traffic.
///
/// @param sigmas              Per-sample density, shape [N_samples].
/// @param rgbs                Per-sample radiance, shape [N_samples, C], where
///                            ``1 <= C <= MAX_VOLUME_RENDER_CHANNELS``
///                            (currently 16; enforced by the host-side
///                            checks).
/// @param deltaTs             Per-sample ray-segment lengths,
///                            shape [N_samples].
/// @param ts                  Per-sample ray t-values (used for the depth
///                            output), shape [N_samples].
/// @param packInfo            CSR-style offsets delimiting each ray's sample
///                            span, shape [N_rays + 1].
/// @param transmittanceThresh Transmittance early-termination threshold.
/// @param needsBackward       If true, populate all five outputs; otherwise
///                            return size-0 placeholders for ``depth``,
///                            ``ws`` and ``totalSamples`` (see note above).
/// @return Tuple ``(rgb, depth, opacity, ws, totalSamples)``.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
volumeRenderForward(const torch::Tensor &sigmas,
                    const torch::Tensor &rgbs,
                    const torch::Tensor &deltaTs,
                    const torch::Tensor &ts,
                    const torch::Tensor &packInfo,
                    double transmittanceThresh,
                    bool needsBackward);

/// @brief Backward pass corresponding to ``volumeRenderForward``.
///
/// Computes gradients of a downstream loss with respect to the per-sample
/// densities ``sigmas`` and radiances ``rgbs``, given the upstream gradients
/// on the five forward outputs and the tensors saved from the forward pass.
/// The forward must have been invoked with ``needsBackward = true`` so that
/// ``ws``, ``depth`` and ``opacity`` are populated.
///
/// @param dLdOpacity          Upstream gradient on the forward ``opacity``
///                            output, shape [N_rays].
/// @param dLdDepth            Upstream gradient on the forward ``depth``
///                            output, shape [N_rays].
/// @param dLdRgb              Upstream gradient on the forward ``rgb`` output,
///                            shape [N_rays, C].
/// @param dLdWs               Upstream gradient on the forward ``ws`` output,
///                            shape [N_samples].
/// @param sigmas              Per-sample density from the forward,
///                            shape [N_samples].
/// @param rgbs                Per-sample radiance from the forward,
///                            shape [N_samples, C].
/// @param ws                  Per-sample compositing weights saved by the
///                            forward, shape [N_samples].
/// @param deltas              Per-sample ray-segment lengths from the forward,
///                            shape [N_samples].
/// @param ts                  Per-sample ray t-values from the forward,
///                            shape [N_samples].
/// @param packInfo            CSR-style offsets delimiting each ray's sample
///                            span, shape [N_rays + 1].
/// @param opacity             Forward ``opacity`` output saved for backward,
///                            shape [N_rays].
/// @param depth               Forward ``depth`` output saved for backward,
///                            shape [N_rays].
/// @param rgb                 Forward ``rgb`` output saved for backward,
///                            shape [N_rays, C].
/// @param transmittanceThresh Transmittance early-termination threshold (must
///                            match the value used in the corresponding
///                            forward call).
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
                                                              const torch::Tensor &packInfo,
                                                              const torch::Tensor &opacity,
                                                              const torch::Tensor &depth,
                                                              const torch::Tensor &rgb,
                                                              double transmittanceThresh);

/// @brief Concatenate a list of grid batches into a single grid batch
/// @param vec A list of grid batches to concatenate
/// @return A GridBatchData representing the concatenated grid batch
c10::intrusive_ptr<GridBatchData> jcat(const std::vector<c10::intrusive_ptr<GridBatchData>> &vec);

/// @brief Concatenate a list of JaggedTensor into a single JaggedTensor
/// @param vec A list of JaggedTensor to concatenate
/// @param dim The dimension to concatenate along or nullptr to concatenate the outermost tensor
/// lists
/// @return A JaggedTensor representing the concatenated JaggedTensor
JaggedTensor jcat(const std::vector<JaggedTensor> &vec, std::optional<int64_t> dim = std::nullopt);

/// @brief Create a JaggedTensor filled with random numbers from a uniform distribution
///        on the interval [0, 1) with the specified lshape an rshape
/// @param lsizes The lshape of the JaggedTensor (number of elements per tensor)
/// @param rsizes The rshape of the JaggedTensor (feature dimension of each tensor)
/// @param options The options to use for the created tensor
/// @return A JaggedTensor filled with random numbers from the uniform distribution on [0, 1).
JaggedTensor jrand(const std::vector<int64_t> &lsizes,
                   const std::vector<int64_t> rsizes = {},
                   at::TensorOptions options         = {});
JaggedTensor jrand(const std::vector<std::vector<int64_t>> &lsizes,
                   const std::vector<int64_t> rsizes = {},
                   at::TensorOptions options         = {});

/// @brief Create a JaggedTensor filled with random numbers from a normal distribution
///        with mean 0 and variance 1 (also called the standard normal distribution).
/// @param lsizes The lshape of the JaggedTensor (number of elements per tensor)
/// @param rsizes The rshape of the JaggedTensor (feature dimension of each tensor)
/// @param options The options to use for the created tensor
/// @return A JaggedTensor filled with random numbers from the standard normal distribution.
JaggedTensor jrandn(const std::vector<int64_t> &lsizes,
                    const std::vector<int64_t> rsizes = {},
                    at::TensorOptions options         = {});
JaggedTensor jrandn(const std::vector<std::vector<int64_t>> &lsizes,
                    const std::vector<int64_t> rsizes = {},
                    at::TensorOptions options         = {});

/// @brief Create a JaggedTensor filled with zeros.
/// @param lsizes The lshape of the JaggedTensor (number of elements per tensor)
/// @param rsizes The rshape of the JaggedTensor (feature dimension of each tensor)
/// @param options The options to use for the created tensor
/// @return A JaggedTensor filled with zeros.
JaggedTensor jzeros(const std::vector<int64_t> &lsizes,
                    const std::vector<int64_t> rsizes = {},
                    at::TensorOptions options         = {});
JaggedTensor jzeros(const std::vector<std::vector<int64_t>> &lsizes,
                    const std::vector<int64_t> rsizes = {},
                    at::TensorOptions options         = {});

/// @brief Create a JaggedTensor filled with ones.
/// @param lsizes The lshape of the JaggedTensor (number of elements per tensor)
/// @param rsizes The rshape of the JaggedTensor (feature dimension of each tensor)
/// @param options The options to use for the created tensor
/// @return A JaggedTensor filled with ones.
JaggedTensor jones(const std::vector<int64_t> &lsizes,
                   const std::vector<int64_t> rsizes = {},
                   at::TensorOptions options         = {});
JaggedTensor jones(const std::vector<std::vector<int64_t>> &lsizes,
                   const std::vector<int64_t> rsizes = {},
                   at::TensorOptions options         = {});

/// @brief Create an empty JaggedTensor with uninitialized values.
/// @param lsizes The lshape of the JaggedTensor (number of elements per tensor)
/// @param rsizes The rshape of the JaggedTensor (feature dimension of each tensor)
/// @param options The options to use for the created tensor
/// @return A JaggedTensor filled with uninitialized values.
JaggedTensor jempty(const std::vector<int64_t> &lsizes,
                    const std::vector<int64_t> rsizes = {},
                    at::TensorOptions options         = {});
JaggedTensor jempty(const std::vector<std::vector<int64_t>> &lsizes,
                    const std::vector<int64_t> rsizes = {},
                    at::TensorOptions options         = {});

/// @brief Return a grid batch with voxels which contain a point in an input set of point clouds
/// @param points A JaggedTensor with shape [B, -1, 3] containing one point set per grid to create
/// @param voxel_sizes A tensor of shape [B, 3] or [3,] containing the voxel size of each grid in
/// the batch or one voxel size for all grids
/// @param origins A tensor of shape [B, 3] or [3,] containing the world space coordinate of the [0,
/// 0, 0] voxel
///                for each grid in the batch, or one origin for all grids
/// @return A GridBatchData containing the created grid batch
c10::intrusive_ptr<GridBatchData>
gridbatch_from_points(const JaggedTensor &points,
                      const std::vector<nanovdb::Vec3d> &voxel_sizes,
                      const std::vector<nanovdb::Vec3d> &origins);

/// @brief Return a grid batch with the eight nearest voxels to each point in an input set of point
/// clouds
/// @param points A JaggedTensor with shape [B, -1, 3] containing one point set per grid to create
/// @param voxel_sizes A tensor of shape [B, 3] or [3,] containing the voxel size of each grid in
/// the batch or one voxel size for all grids
/// @param origins A tensor of shape [B, 3] or [3,] containing the world space coordinate of the [0,
/// 0, 0] voxel for each grid in the batch, or one origin for all grids
/// @return A GridBatchData containing the created grid batch
c10::intrusive_ptr<GridBatchData>
gridbatch_from_nearest_voxels_to_points(const JaggedTensor &points,
                                        const std::vector<nanovdb::Vec3d> &voxel_sizes,
                                        const std::vector<nanovdb::Vec3d> &origins);

/// @brief Return a grid batch with the specified voxel coordinates
/// @param coords A JaggedTensor of shape [B, -1, 3] specifying the coordinates of each voxel to
/// insert
/// @param voxel_sizes A tensor of shape [B, 3] or [3,] containing the voxel size of each grid in
/// the batch or one voxel size for all grids
/// @param origins A tensor of shape [B, 3] or [3,] containing the world space coordinate of the [0,
/// 0, 0] voxel
///                for each grid in the batch, or one origin for all grids
/// @return A GridBatchData containing the created grid batch
c10::intrusive_ptr<GridBatchData> gridbatch_from_ijk(const JaggedTensor &ijk,
                                                     const std::vector<nanovdb::Vec3d> &voxel_sizes,
                                                     const std::vector<nanovdb::Vec3d> &origins);

/// @brief Return a grid batch densely from ijkMin to ijkMin + size
/// @param numGrids The number of grids to create in the batch
/// @param denseDims The size of each dense grid (shape [3,] = [W, H, D])
/// @param ijkMin The minimum ijk coordinate of each dense grid in the batch (shape [3,])
/// @param voxel_sizes A tensor of shape [B, 3] or [3,] containing the voxel size of each grid in
/// the batch or one voxel size for all grids
/// @param origins A tensor of shape [B, 3] or [3,] containing the world space coordinate of the [0,
/// 0, 0] voxel
///                     for each grid in the batch, or one origin for all grids
/// @param mask Optional mask of shape [W, H, D] to specify voxels which are included in the dense
/// grid.
///             Note that the same mask will be re-used for all the grids in the batch.
/// @param device Which device to build the grid batch on
/// @return A GridBatchData containing a batch of dense grids
c10::intrusive_ptr<GridBatchData>
gridbatch_from_dense(const int64_t numGrids,
                     const nanovdb::Coord &denseDims,
                     const nanovdb::Coord &ijkMin,
                     const std::vector<nanovdb::Vec3d> &voxel_sizes,
                     const std::vector<nanovdb::Vec3d> &origins,
                     std::optional<torch::Tensor> mask = std::nullopt,
                     const torch::Device &device       = torch::kCPU);

/// @brief Return a grid batch from a jagged batch of triangle meshes (i.e. each voxel intersects
/// the mesh)
/// @param vertices A JaggedTensor of shape [B, -1, 3] containing the vertices of each mesh in the
/// batch
/// @param faces A JaggedTensor of shape [B, -1, 3] containing the faces of each mesh in the batch
/// @param voxel_sizes A tensor of shape [B, 3] or [3,] containing the voxel size of each grid in
/// the batch or one voxel size for all grids
/// @param origins A tensor of shape [B, 3] or [3,] containing the world space coordinate of the [0,
/// 0, 0] voxel for each grid in the batch, or one origin for all grids
/// @return A GridBatchData containing the created grid batch
c10::intrusive_ptr<GridBatchData>
gridbatch_from_mesh(const JaggedTensor &vertices,
                    const JaggedTensor &faces,
                    const std::vector<nanovdb::Vec3d> &voxel_sizes,
                    const std::vector<nanovdb::Vec3d> &origins);

/// @brief Return a grid batch, tensors of data, and names from a nanovdb grid handle
/// @param handle nanovdb grid handle
/// @return A triple (gridbatch_data, data, names) where gridbatch_data is a GridBatchData
/// containing the converted grids,
///         data is a JaggedTensor containing the data of the grids, and names is a list of strings
///         containing the name of each grid
std::tuple<c10::intrusive_ptr<GridBatchData>, JaggedTensor, std::vector<std::string>>
from_nanovdb(nanovdb::GridHandle<nanovdb::HostBuffer> &handle);

/// @brief Return a nanovdb grid handle created from a grid batch, optional jagged tensor of data,
/// and optional
///        list of names
/// @param gridBatchData The grid batch data to convert
/// @param maybeData Optional JaggedTensor of data to save with the grid batch (one element per
/// voxel)
/// @param maybeNames  Optional list of names for each grid in the batch (or a single name to use
/// for every grid)
/// @return A nanovdb grid handle, whose type is inferred from the data, containing the converted
/// grids
nanovdb::GridHandle<nanovdb::HostBuffer>
to_nanovdb(const GridBatchData &gridBatchData,
           const std::optional<JaggedTensor> maybeData = std::nullopt,
           const std::vector<std::string> &names       = {});

/// @brief Save a grid batch and optional jagged tensor to a .nvdb file. Will overwrite existing
/// files.
/// @param path The path to save the file to.
/// @param gridBatchData The grid batch data to save
/// @param maybeData Optional JaggedTensor of data to save with the grid batch (one element per
/// voxel)
/// @param names Optional list of names for each grid in the batch
/// @param compressed Whether to compress the stored grid using Blosc (https://www.blosc.org/)
/// @param verbose Whether to print information about the saved grids
void save(const std::string &path,
          const GridBatchData &gridBatchData,
          const std::optional<JaggedTensor> maybeData = std::nullopt,
          const std::vector<std::string> &names       = {},
          bool compressed                             = false,
          bool verbose                                = false);

/// @brief Save a grid batch and optional jagged tensor to a .nvdb file. Will overwrite existing
/// files.
/// @param path The path to save the file to.
/// @param gridBatchData The grid batch data to save
/// @param maybeData Optional JaggedTensor of data to save with the grid batch (one element per
/// voxel)
/// @param names Optional single name to use for every grid in the batch
/// @param compressed Whether to compress the stored grid using Blosc (https://www.blosc.org/)
/// @param verbose Whether to print information about the saved grids
void save(const std::string &path,
          const GridBatchData &gridBatchData,
          const std::optional<JaggedTensor> maybeData = std::nullopt,
          const std::string &name                     = std::string(),
          bool compressed                             = false,
          bool verbose                                = false);

/// @brief Load a grid batch from a .nvdb file.
/// @param path The path to the .nvdb file to load
/// @param indices The list of indices to load from the file
/// @param device Which device to load the grid batch on
/// @param verbose If set to true, print information about the loaded grids
/// @return A triple (gridbatch_data, data, names)
std::tuple<c10::intrusive_ptr<GridBatchData>, JaggedTensor, std::vector<std::string>>
load(const std::string &path,
     const std::vector<uint64_t> &indices,
     const torch::Device &device,
     bool verbose = false);
/// @brief Load a grid batch from a .nvdb file.
/// @param path The path to the .nvdb file to load
/// @param names The list of names to load from the file
/// @param device Which device to load the grid batch on
/// @param verbose If set to true, print information about the loaded grids
/// @return A triple (gridbatch_data, data, names)
std::tuple<c10::intrusive_ptr<GridBatchData>, JaggedTensor, std::vector<std::string>>
load(const std::string &path,
     const std::vector<std::string> &names,
     const torch::Device &device,
     bool verbose = false);
/// @brief Load a grid batch from a .nvdb file.
/// @param path The path to the .nvdb file to load
/// @param device Which device to load the grid batch on
/// @param verbose If set to true, print information about the loaded grids
/// @return A triple (gridbatch_data, data, names)
std::tuple<c10::intrusive_ptr<GridBatchData>, JaggedTensor, std::vector<std::string>>
load(const std::string &path, const torch::Device &device, bool verbose = false);

/// @brief Convert a tensor of ijk coordinates to a tensor of morton codes
/// @param ijk An int32 tensor of shape [N, 3] containing the ijk coordinates to convert
/// @return An int64 tensor of shape [N] containing the morton codes
torch::Tensor morton(torch::Tensor const &ijk);

/// @brief Convert a tensor of ijk coordinates to a tensor of hilbert codes
/// @param ijk An int32 tensor of shape [N, 3] containing the ijk coordinates to convert
/// @return An int64 tensor of shape [N] containing the hilbert codes
torch::Tensor hilbert(torch::Tensor const &ijk);

} // namespace fvdb

#endif // FVDB_FVDB_H
