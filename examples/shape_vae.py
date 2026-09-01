# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Sparse 3D shape variational autoencoder with generative transposed convolutions.

This example is the fvdb analog of MinkowskiEngine's ``examples/vae.py``. An
encoder compresses each sparse voxelized shape into a single latent vector; the
decoder generates the shape's voxel topology back from the latent alone, growing
coordinates with *generative* transposed convolutions and trimming them with
per-level occupancy classifiers and :class:`fvdb.nn.Prune`. Because the decoder
sees only the latent (no skip connections), sampling ``z ~ N(0, I)`` generates
novel shapes - demonstrated at the end of the script.

MinkowskiEngine to fvdb mapping (see ``shape_completion.py`` for the shared
decoder pattern and lattice/anchoring notes):

- ``MinkowskiGenerativeConvolutionTranspose`` ->
  :meth:`fvdb.ConvolutionPlan.from_grid_batch_transposed` with ``target_grid=None``.
- ``MinkowskiPruning`` -> :class:`fvdb.nn.Prune`.
- ``MinkowskiGlobalPooling`` -> a jagged global average pool
  (:meth:`fvdb.JaggedTensor.jsum` divided by voxel counts). MinkowskiEngine
  instead strides all the way down to a single voxel per shape at
  ``tensor_stride = resolution``; global pooling over the jagged batch is the
  idiomatic fvdb equivalent.
- MinkowskiEngine seeds its decoder with one voxel per shape at the coarsest
  stride; here the decoder starts from a small dense "neck" grid (4^3 voxels at
  stride 16) built with :meth:`fvdb.GridBatch.from_dense`. The latent vector is
  broadcast to every neck voxel and combined with a learned per-voxel positional
  embedding (without it the broadcast features would be spatially identical and
  the decoder could not tell neck positions apart). The first decoder level then
  learns which neck children to keep.

The dataset is the 254-shoe "Shoe" category of Scanned Objects by Google Research
(CC-BY 4.0), bundled with fvdb-example-data — a single object category with real
intra-class variation (runners, boat shoes, ballet flats, cleats, boots), which is
the same role ModelNet40's chair class plays in MinkowskiEngine's example. Training
runs small random minibatches over the pre-voxelized dataset. Deviations from
MinkowskiEngine, chosen so the example trains in minutes: Adam instead of SGD, a
down-weighted KL term (``KLD_WEIGHT``), and a compressed iteration budget — crank
up ``NUM_ITERATIONS`` for better samples.
"""

import logging

import polyscope as ps
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from fvdb.utils.examples import load_gso_shoes

import fvdb
import fvdb.nn as fvnn
from fvdb import ConvolutionPlan, GridBatch, JaggedTensor

RESOLUTION = 64  # voxels along the longest axis of each shape
NUM_LEVELS = 4  # stride-2 pyramid depth; the decoder neck lattice is (RESOLUTION / 2**NUM_LEVELS)^3 = 4^3
CHANNELS = [16, 32, 64, 128, 256]  # CHANNELS[0] = full resolution, CHANNELS[NUM_LEVELS] = coarsest
LATENT_DIM = 128
KLD_WEIGHT = 1e-2
NUM_ITERATIONS = 1500
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
LOG_EVERY = 100
NUM_PRIOR_SAMPLES = 4
NUM_EVAL_SHAPES = 8  # reconstruction / visualization subset


def normalize_vertices(vertices: torch.Tensor) -> torch.Tensor:
    """Uniformly scale vertices into [0.02, 0.98]^3 so all voxel ijk are in [0, RESOLUTION)."""
    vertices = vertices - vertices.amin(dim=0)
    vertices = vertices / vertices.amax()
    return vertices * 0.96 + 0.02


def prepare_shapes(device: torch.device) -> GridBatch:
    """Voxelize the GSO shoe meshes (CC-BY 4.0) into a single GridBatch 'dataset'."""
    meshes = load_gso_shoes(device=device)
    vertices = JaggedTensor([normalize_vertices(v) for v, _ in meshes])
    faces = JaggedTensor([f.int() for _, f in meshes])
    return GridBatch.from_mesh(vertices, faces, voxel_sizes=1.0 / RESOLUTION, origins=0.0)


def build_gt_pyramid(gt_grid: GridBatch) -> list[GridBatch]:
    """Ground-truth topology at every pyramid level; level d has voxel size 2^d / RESOLUTION."""
    pyramid = [gt_grid]
    for _ in range(NUM_LEVELS):
        pyramid.append(pyramid[-1].conv_grid(2, 2))
    return pyramid


class ConvBnElu(nn.Module):
    """A sparse convolution followed by batch norm and ELU, building its plan per call."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int):
        super().__init__()
        self.conv = fvnn.SparseConv3d(in_channels, out_channels, kernel_size, stride)
        self.norm = fvnn.BatchNorm(out_channels)
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, data: JaggedTensor, grid: GridBatch) -> tuple[JaggedTensor, GridBatch]:
        if self.stride == 1:
            plan = ConvolutionPlan.from_grid_batch(self.kernel_size, 1, grid, grid)
        else:
            plan = ConvolutionPlan.from_grid_batch(self.kernel_size, self.stride, grid)
        data = self.conv(data, plan)
        out_grid = plan.target_grid_batch
        return out_grid.jagged_like(F.elu(self.norm(data, out_grid).jdata)), out_grid


class GenerativeUpBlock(nn.Module):
    """Generative transposed conv (uncropped support) + BN + ELU + k3 conv + BN + ELU."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = fvnn.SparseConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.up_norm = fvnn.BatchNorm(out_channels)
        self.conv = ConvBnElu(out_channels, out_channels, kernel_size=3, stride=1)

    def forward(self, data: JaggedTensor, grid: GridBatch) -> tuple[JaggedTensor, GridBatch]:
        # target_grid=None -> COMPLETE topology policy: the plan *generates* new coordinates,
        # the fvdb equivalent of MinkowskiGenerativeConvolutionTranspose.
        plan = ConvolutionPlan.from_grid_batch_transposed(2, 2, grid)
        data = self.up(data, plan)
        out_grid = plan.target_grid_batch
        data = out_grid.jagged_like(F.elu(self.up_norm(data, out_grid).jdata))
        return self.conv(data, out_grid)


class Encoder(nn.Module):
    """Strided sparse convolutions to the coarsest level, then a global average pool and mu/logvar heads."""

    def __init__(self):
        super().__init__()
        self.stem = ConvBnElu(1, CHANNELS[0], kernel_size=3, stride=1)
        self.down = nn.ModuleList(
            ConvBnElu(CHANNELS[i], CHANNELS[i + 1], kernel_size=2, stride=2) for i in range(NUM_LEVELS)
        )
        self.conv = nn.ModuleList(
            ConvBnElu(CHANNELS[i + 1], CHANNELS[i + 1], kernel_size=3, stride=1) for i in range(NUM_LEVELS)
        )
        self.fc_mu = nn.Linear(CHANNELS[NUM_LEVELS], LATENT_DIM)
        self.fc_logvar = nn.Linear(CHANNELS[NUM_LEVELS], LATENT_DIM)

    def forward(self, data: JaggedTensor, grid: GridBatch) -> tuple[torch.Tensor, torch.Tensor]:
        data, grid = self.stem(data, grid)
        for i in range(NUM_LEVELS):
            data, grid = self.down[i](data, grid)
            data, grid = self.conv[i](data, grid)
        # Global average pool over each grid's voxels (MinkowskiGlobalPooling analog).
        counts = grid.num_voxels.to(data.jdata.dtype).clamp_min(1).unsqueeze(1)
        pooled = data.jsum(0).jdata / counts
        return self.fc_mu(pooled), self.fc_logvar(pooled)


class Decoder(nn.Module):
    """Generates shape topology from a latent vector alone (no skip connections)."""

    def __init__(self):
        super().__init__()
        neck_extent = RESOLUTION // 2**NUM_LEVELS
        self.fc_seed = nn.Linear(LATENT_DIM, CHANNELS[NUM_LEVELS])
        # Learned per-voxel positional embedding for the dense neck. This breaks spatial
        # symmetry: with only the broadcast latent, every neck voxel would carry identical
        # features and the decoder could not distinguish positions. (MinkowskiEngine's
        # single-voxel seed gets this for free - every descendant has a unique kernel-tap
        # path from the root.)
        self.neck_position = nn.Parameter(0.02 * torch.randn(neck_extent**3, CHANNELS[NUM_LEVELS]))
        self.up = nn.ModuleList(GenerativeUpBlock(CHANNELS[i + 1], CHANNELS[i]) for i in range(NUM_LEVELS))
        self.head = nn.ModuleList(fvnn.SparseConv3d(CHANNELS[i], 1, kernel_size=1, stride=1) for i in range(NUM_LEVELS))
        self.prune = fvnn.Prune()

    def make_seed_grid(self, batch_size: int, device: torch.device) -> GridBatch:
        neck_extent = RESOLUTION // 2**NUM_LEVELS
        return GridBatch.from_dense(
            batch_size, [neck_extent] * 3, voxel_sizes=float(2**NUM_LEVELS) / RESOLUTION, origins=0.0, device=device
        )

    def forward(
        self, z: torch.Tensor, gt_pyramid: list[GridBatch] | None
    ) -> tuple[list[JaggedTensor], list[JaggedTensor], JaggedTensor, GridBatch]:
        """Decode latents. ``gt_pyramid`` supplies per-level targets (and teacher forcing while
        training); pass ``None`` for pure generation. Returns per-level (logits, targets) plus
        the final features and grid."""
        grid = self.make_seed_grid(z.shape[0], z.device)
        voxels_per_grid = grid.total_voxels // z.shape[0]
        seed = torch.repeat_interleave(F.elu(self.fc_seed(z)), voxels_per_grid, dim=0)
        data = grid.jagged_like(seed + self.neck_position.tile(z.shape[0], 1))

        logits_per_level: list[JaggedTensor] = []
        targets_per_level: list[JaggedTensor] = []
        for i in reversed(range(NUM_LEVELS)):
            if grid.total_voxels == 0:
                break  # every voxel was pruned (possible when generating from the prior)
            data, grid = self.up[i](data, grid)
            head_plan = ConvolutionPlan.from_grid_batch(1, 1, grid, grid)
            logits = self.head[i](data, head_plan)
            keep = logits.jdata.squeeze(-1) > 0
            if gt_pyramid is not None:
                target = gt_pyramid[i].coords_in_grid(grid.ijk)
                logits_per_level.append(logits)
                targets_per_level.append(target)
                if self.training:
                    keep |= target.jdata  # teacher forcing, as in MinkowskiEngine
            data, grid = self.prune(data, grid, grid.jagged_like(keep))
        return logits_per_level, targets_per_level, data, grid


class ShapeVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, data: JaggedTensor, grid: GridBatch, gt_pyramid: list[GridBatch] | None):
        mu, logvar = self.encoder(data, grid)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu) if self.training else mu
        logits, targets, out_data, out_grid = self.decoder(z, gt_pyramid)
        return logits, targets, out_data, out_grid, mu, logvar


def vae_loss(
    logits_per_level: list[JaggedTensor],
    targets_per_level: list[JaggedTensor],
    mu: torch.Tensor,
    logvar: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bce = torch.stack(
        [
            F.binary_cross_entropy_with_logits(logits.jdata.squeeze(-1), target.jdata.float())
            for logits, target in zip(logits_per_level, targets_per_level)
        ]
    ).mean()
    kld = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
    return bce + KLD_WEIGHT * kld, bce, kld


def grid_iou(predicted: GridBatch, gt: GridBatch) -> float:
    intersection = int(gt.coords_in_grid(predicted.ijk).jdata.sum().item())
    union = predicted.total_voxels + gt.total_voxels - intersection
    return intersection / max(union, 1)


def visualize(gt_grid: GridBatch, recon_grid: GridBatch, sample_grid: GridBatch) -> None:
    ps.init()
    for name, grid, offset in (
        ("gt", gt_grid, -1.2),
        ("reconstruction", recon_grid, 0.0),
        ("sample", sample_grid, 1.2),
    ):
        centers = grid.voxel_to_world(grid.ijk.float())
        for b in range(grid.grid_count):
            points = centers[b].jdata.cpu().numpy()
            points = points + [offset, 1.2 * b, 0.0]
            ps.register_point_cloud(f"{name}_{b}", points, radius=0.0025)
    ps.show()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    logging.addLevelName(logging.INFO, "\033[1;32m%s\033[1;0m" % logging.getLevelName(logging.INFO))
    torch.random.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gt_grid = prepare_shapes(device)
    # Build the ground-truth pyramid ONCE for the whole dataset; minibatches sub-index each
    # level (GridBatch indexing is cheap, rebuilding conv_grid pyramids per iteration is not).
    gt_pyramid = build_gt_pyramid(gt_grid)
    logging.info(f"dataset: {gt_grid.grid_count} shapes, {gt_grid.total_voxels} voxels total")

    model = ShapeVAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for iteration in tqdm.trange(NUM_ITERATIONS, desc="training"):
        idx = torch.randperm(gt_grid.grid_count)[:BATCH_SIZE]
        batch_gt = gt_grid[idx]
        batch_pyramid = [level[idx] for level in gt_pyramid]
        batch_features = batch_gt.jagged_like(torch.ones(batch_gt.total_voxels, 1, device=device))

        logits, targets, _, _, mu, logvar = model(batch_features, batch_gt, batch_pyramid)
        loss, bce, kld = vae_loss(logits, targets, mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if iteration % LOG_EVERY == 0 or iteration == NUM_ITERATIONS - 1:
            logging.info(
                f"iteration {iteration}: loss = {loss.item():.4f} (bce {bce.item():.4f}, kld {kld.item():.4f})"
            )

    model.eval()
    with torch.no_grad():
        # Reconstructions: encode a fixed subset of shapes and decode from z = mu.
        eval_gt = gt_grid[list(range(NUM_EVAL_SHAPES))]
        eval_features = eval_gt.jagged_like(torch.ones(eval_gt.total_voxels, 1, device=device))
        _, _, _, recon_grid, _, _ = model(eval_features, eval_gt, None)
        logging.info(f"reconstruction IoU on {NUM_EVAL_SHAPES} shapes: {grid_iou(recon_grid, eval_gt):.3f}")

        # Generation: decode novel shapes from the prior.
        z = torch.randn(NUM_PRIOR_SAMPLES, LATENT_DIM, device=device)
        _, _, _, sample_grid = model.decoder(z, None)
        logging.info(f"prior samples decoded to {sample_grid.num_voxels.tolist()} voxels")
    visualize(eval_gt, recon_grid, sample_grid)


if __name__ == "__main__":
    main()
