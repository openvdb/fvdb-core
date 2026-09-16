# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Sparse 3D shape completion with generative transposed convolutions.

A sparse encoder-decoder network is given a partial shape (a slab cropped along x)
and learns to generate the voxel topology of the complete shape. The decoder grows
topology with *generative* transposed convolutions and trims it with per-level
occupancy classifiers.
"""

import logging

import fvdb.nn as fvnn
import polyscope as ps
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from fvdb.utils.examples import load_dragon_mesh, load_happy_mesh

import fvdb
from fvdb import ConvolutionPlan, GridBatch, JaggedTensor

RESOLUTION = 64  # voxels along the longest axis of each shape
NUM_LEVELS = 5  # stride-2 pyramid depth; coarsest lattice is RESOLUTION / 2**NUM_LEVELS = 2^3
CHANNELS = [16, 32, 64, 128, 256, 256]  # CHANNELS[0] = full resolution, CHANNELS[NUM_LEVELS] = coarsest
CROP_FRACTION = 0.55  # keep voxels with x below this fraction of the shape's x-extent
NUM_ITERATIONS = 500
LEARNING_RATE = 1e-2
LOG_EVERY = 50


def normalize_vertices(vertices: torch.Tensor) -> torch.Tensor:
    """Uniformly scale vertices into [0.02, 0.98]^3 so all voxel ijk are in [0, RESOLUTION)."""
    vertices = vertices - vertices.amin(dim=0)
    vertices = vertices / vertices.amax()
    return vertices * 0.96 + 0.02


def prepare_shapes(device: torch.device) -> tuple[GridBatch, GridBatch]:
    """Voxelize the bundled meshes and slab-crop them along x.

    Returns:
        gt_grid (GridBatch): Complete (ground-truth) shape topologies.
        partial_grid (GridBatch): The cropped, partial input topologies.
    """
    voxel_size = 1.0 / RESOLUTION
    meshes = [load_dragon_mesh(mode="vf", device=device), load_happy_mesh(mode="vf", device=device)]
    vertices = JaggedTensor([normalize_vertices(v) for v, _ in meshes])
    faces = JaggedTensor([f.int() for _, f in meshes])
    gt_grid = GridBatch.from_mesh(vertices, faces, voxel_sizes=voxel_size, origins=0.0)

    # Crop away the top (1 - CROP_FRACTION) of each shape along x, in voxel space.
    ijk = gt_grid.ijk
    keep = torch.zeros(gt_grid.total_voxels, dtype=torch.bool, device=device)
    for b in range(gt_grid.grid_count):
        in_b = ijk.jidx == b
        x = ijk.jdata[in_b, 0].float()
        keep[in_b] = x < x.min() + CROP_FRACTION * (x.max() - x.min())
    partial_grid = gt_grid.pruned_grid(gt_grid.jagged_like(keep))
    return gt_grid, partial_grid


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

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.up = fvnn.SparseConvTranspose3d(in_channels, out_channels, kernel_size, stride=2)
        self.up_norm = fvnn.BatchNorm(out_channels)
        self.conv = ConvBnElu(out_channels, out_channels, kernel_size=3, stride=1)
        self.kernel_size = kernel_size

    def forward(self, data: JaggedTensor, grid: GridBatch) -> tuple[JaggedTensor, GridBatch]:
        # target_grid=None -> COMPLETE topology policy: the plan *generates* new coordinates
        plan = ConvolutionPlan.from_grid_batch_transposed(self.kernel_size, 2, grid)
        data = self.up(data, plan)
        out_grid = plan.target_grid_batch
        data = out_grid.jagged_like(F.elu(self.up_norm(data, out_grid).jdata))
        return self.conv(data, out_grid)


class CompletionNet(nn.Module):
    """Sparse encoder-decoder that completes a partial shape."""

    def __init__(self):
        super().__init__()
        self.stem = ConvBnElu(1, CHANNELS[0], kernel_size=3, stride=1)
        self.enc_down = nn.ModuleList(
            ConvBnElu(CHANNELS[i], CHANNELS[i + 1], kernel_size=2, stride=2) for i in range(NUM_LEVELS)
        )
        self.enc_conv = nn.ModuleList(
            ConvBnElu(CHANNELS[i + 1], CHANNELS[i + 1], kernel_size=3, stride=1) for i in range(NUM_LEVELS)
        )
        # Decoder level i produces the level-i grid from level i+1. The coarsest transpose
        # uses kernel size 4; the rest use kernel size 2.
        self.dec_up = nn.ModuleList(
            GenerativeUpBlock(CHANNELS[i + 1], CHANNELS[i], kernel_size=4 if i == NUM_LEVELS - 1 else 2)
            for i in range(NUM_LEVELS)
        )
        self.dec_head = nn.ModuleList(
            fvnn.SparseConv3d(CHANNELS[i], 1, kernel_size=1, stride=1) for i in range(NUM_LEVELS)
        )
        self.prune = fvnn.Prune()

    def forward(
        self, data: JaggedTensor, grid: GridBatch, gt_pyramid: list[GridBatch]
    ) -> tuple[list[JaggedTensor], list[JaggedTensor], JaggedTensor, GridBatch]:
        """Run completion. Returns per-level (logits, targets) plus the final features and grid."""
        # Encoder: store per-level features for the additive U-Net skip connections.
        data, grid = self.stem(data, grid)
        skips: list[tuple[JaggedTensor, GridBatch]] = [(data, grid)]
        for i in range(NUM_LEVELS):
            data, grid = self.enc_down[i](data, grid)
            data, grid = self.enc_conv[i](data, grid)
            skips.append((data, grid))

        # Decoder: generate, classify, and prune one level at a time.
        logits_per_level: list[JaggedTensor] = []
        targets_per_level: list[JaggedTensor] = []
        for i in reversed(range(NUM_LEVELS)):
            if grid.total_voxels == 0:
                break  # everything was pruned (possible when not teacher-forced)
            data, grid = self.dec_up[i](data, grid)
            # Additive skip: gather encoder features onto the generated decoder topology
            # (voxels absent from the encoder grid contribute zero)
            skip_data, skip_grid = skips[i]
            data = grid.jagged_like(data.jdata + grid.inject_from(skip_grid, skip_data).jdata)

            head_plan = ConvolutionPlan.from_grid_batch(1, 1, grid, grid)
            logits = self.dec_head[i](data, head_plan)
            target = gt_pyramid[i].coords_in_grid(grid.ijk)
            logits_per_level.append(logits)
            targets_per_level.append(target)

            keep = logits.jdata.squeeze(-1) > 0
            if self.training:
                keep |= target.jdata  # teacher forcing
            data, grid = self.prune(data, grid, grid.jagged_like(keep))
        return logits_per_level, targets_per_level, data, grid


def occupancy_loss(logits_per_level: list[JaggedTensor], targets_per_level: list[JaggedTensor]) -> torch.Tensor:
    losses = [
        F.binary_cross_entropy_with_logits(logits.jdata.squeeze(-1), target.jdata.float())
        for logits, target in zip(logits_per_level, targets_per_level)
    ]
    return torch.stack(losses).mean()


def grid_iou(predicted: GridBatch, gt: GridBatch) -> float:
    intersection = int(gt.coords_in_grid(predicted.ijk).jdata.sum().item())
    union = predicted.total_voxels + gt.total_voxels - intersection
    return intersection / max(union, 1)


def visualize(partial_grid: GridBatch, predicted_grid: GridBatch, gt_grid: GridBatch) -> None:
    ps.init()
    for name, grid, offset in (("input", partial_grid, -1.2), ("predicted", predicted_grid, 0.0), ("gt", gt_grid, 1.2)):
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

    gt_grid, partial_grid = prepare_shapes(device)
    gt_pyramid = build_gt_pyramid(gt_grid)
    input_features = partial_grid.jagged_like(torch.ones(partial_grid.total_voxels, 1, device=device))
    logging.info(f"GT voxels: {gt_grid.total_voxels}, partial-input voxels: {partial_grid.total_voxels}")

    model = CompletionNet().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)

    model.train()
    for iteration in tqdm.trange(NUM_ITERATIONS, desc="training"):
        logits, targets, _, _ = model(input_features, partial_grid, gt_pyramid)
        loss = occupancy_loss(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if iteration % LOG_EVERY == 0 or iteration == NUM_ITERATIONS - 1:
            logging.info(f"iteration {iteration}: loss = {loss.item():.4f}")

    # Evaluation: no teacher forcing - the decoder keeps only what it predicts.
    model.eval()
    with torch.no_grad():
        _, _, _, predicted_grid = model(input_features, partial_grid, gt_pyramid)
    logging.info(f"completed-shape IoU vs ground truth: {grid_iou(predicted_grid, gt_grid):.3f}")

    visualize(partial_grid, predicted_grid, gt_grid)


if __name__ == "__main__":
    main()
