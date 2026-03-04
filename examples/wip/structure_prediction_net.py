# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

import os

import fvdb.nn as fvnn
import numpy as np
import point_cloud_utils as pcu
import polyscope as ps
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_cudamanaged

import fvdb

torch.backends.cudnn.deterministic = True
np.random.seed(42)
torch.manual_seed(42)


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, order: str, num_groups: int):
        super().__init__()
        for i, char in enumerate(order):
            if char == "r":
                self.add_module("ReLU", fvnn.ReLU(inplace=True))
            elif char == "c":
                conv = fvnn.SparseConv3d(in_channels, out_channels, 3, 1, bias="g" not in order)
                conv.backend = "halo"
                self.add_module("Conv", conv)
            elif char == "g":
                num_channels = in_channels if i < order.index("c") else out_channels
                if num_channels < num_groups:
                    num_groups = 1
                self.add_module("GroupNorm", fvnn.GroupNorm(num_groups=num_groups, num_channels=num_channels))
            else:
                raise NotImplementedError


class SparseDoubleConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, order: str, num_groups: int, encoder: bool):
        super().__init__()
        if encoder:
            conv1_in_channels = in_channels
            conv1_out_channels = out_channels // 2
            if conv1_out_channels < in_channels:
                conv1_out_channels = in_channels
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        else:
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels

        self.add_module("SingleConv1", ConvBlock(conv1_in_channels, conv1_out_channels, order, num_groups))
        self.add_module("SingleConv2", ConvBlock(conv2_in_channels, conv2_out_channels, order, num_groups))


class StructPredictionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_conv = fvnn.SparseConv3d(3, 32, 3, 1)
        self.pre_conv.backend = "halo"
        self.dconv1 = SparseDoubleConv(32, 32, "gcr", 8, True)
        self.dconv2 = SparseDoubleConv(32, 64, "gcr", 8, True)
        self.dconv3 = SparseDoubleConv(64, 128, "gcr", 8, True)

        self.max_pool = fvnn.MaxPool(2)
        self.up_sample = fvnn.UpsamplingNearest(2)
        self.up_sample0 = fvnn.UpsamplingNearest(1)

        self.pad = fvnn.FillFromGrid()

        self.dconvd2 = SparseDoubleConv(192, 64, "gcr", 8, False)
        self.dconvd1 = SparseDoubleConv(96, 32, "gcr", 8, False)

        self.struct_conv3 = fvnn.SparseConv3d(128, 2, 3, 1)
        self.struct_conv3.backend = "halo"
        self.struct_conv2 = fvnn.SparseConv3d(64, 2, 3, 1)
        self.struct_conv2.backend = "halo"
        self.struct_conv1 = fvnn.SparseConv3d(32, 2, 3, 1)
        self.struct_conv1.backend = "halo"

    @classmethod
    def struct_to_mask(cls, struct_pred: fvnn.VDBTensor):
        # 0 is exist, 1 is non-exist
        mask = struct_pred.jdata[:, 0] > struct_pred.jdata[:, 1]
        return struct_pred.grid.jagged_like(mask)

    def forward(self, x: fvnn.VDBTensor):
        x = self.pre_conv(x)
        enc1 = x = self.dconv1(x)
        x = self.max_pool(x)
        enc2 = x = self.dconv2(x)
        x = self.max_pool(x)
        x = self.dconv3(x)

        x = self.pad(x, neck_grid)
        struct2 = self.struct_conv3(x)

        x = self.up_sample(x, self.struct_to_mask(struct2))
        x = fvdb.jcat([x, self.pad(enc2, x)], dim=1)
        x = self.dconvd2(x)
        struct1 = self.struct_conv2(x)

        x = self.up_sample(x, self.struct_to_mask(struct1))
        x = fvdb.jcat([x, self.pad(enc1, x)], dim=1)
        x = self.dconvd1(x)
        struct0 = self.struct_conv1(x)

        x = self.up_sample0(x, self.struct_to_mask(struct0))

        return x, struct0, struct1, struct2


def normalize_pts(xyz: np.ndarray):
    xyz_min = np.min(xyz, axis=0)
    xyz_max = np.max(xyz, axis=0)
    xyz_center = (xyz_min + xyz_max) / 2
    xyz_scale = np.max(xyz_max - xyz_min) * 1.2
    return (xyz - xyz_center) / xyz_scale + 0.5


def visualize_grid(grid: fvdb.GridBatch):
    ps.init()
    xyz = grid.voxel_to_world(grid.ijk.float())
    for batch_idx in range(grid.grid_count):
        pts = xyz[batch_idx].jdata.cpu().numpy()
        ps.register_point_cloud(f"grid_{batch_idx}", pts, radius=0.0025)
    ps.show()


def compute_loss(pd_struct: fvnn.VDBTensor, gt_grid: fvdb.GridBatch):
    assert torch.allclose(pd_struct.grid.origins, gt_grid.origins)
    assert torch.allclose(pd_struct.grid.voxel_sizes, gt_grid.voxel_sizes)
    idx_mask = gt_grid.ijk_to_index(pd_struct.grid.ijk).jdata == -1
    idx_mask = idx_mask.long()
    loss = F.cross_entropy(pd_struct.jdata, idx_mask)
    return 0.0 if idx_mask.size(0) == 0 else loss


if __name__ == "__main__":
    grid_resolution = 256
    device = "cudamanaged:0"

    voxel_size = 1 / grid_resolution
    dragon_pts = pcu.load_mesh_v("/home/mcong/voxel-foundation/fvdb-example-data/meshes/dragon.ply")
    dragon_pts = normalize_pts(dragon_pts)
    dragon_pts = torch.from_numpy(dragon_pts).to(device).to(torch.float32)

    bunny_pts = pcu.load_mesh_v("/home/mcong/voxel-foundation/fvdb-example-data/meshes/bunny.ply")
    bunny_pts = normalize_pts(bunny_pts)
    bunny_pts = torch.from_numpy(bunny_pts).to(device).to(torch.float32)

    gt_grid = fvdb.GridBatch.from_points(
        fvdb.JaggedTensor([dragon_pts, bunny_pts]), voxel_sizes=voxel_size, origins=[voxel_size / 2.0] * 3
    )
    # visualize_grid(gt_grid)

    gt_grid0 = gt_grid
    gt_grid1 = gt_grid.coarsened_grid(2)
    gt_grid2 = gt_grid.coarsened_grid(4)

    neck_grid = fvdb.GridBatch.from_dense(
        gt_grid.grid_count,
        [grid_resolution // 2 // 2] * 3,
        device=device,
        voxel_sizes=voxel_size * 4,
        origins=[voxel_size * 2.0] * 3,
    )
    assert torch.allclose(gt_grid.coarsened_grid(4).origins, neck_grid.origins)
    # visualize_grid(neck_grid)

    input_x = fvnn.VDBTensor(gt_grid, gt_grid.ijk.float())
    net = StructPredictionNet().to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    for i in range(1000):
        output_x, pd_struct0, pd_struct1, pd_struct2 = net(input_x)
        if i % 100 == 0:
            visualize_grid(output_x.grid)

        loss = (
            compute_loss(pd_struct0, gt_grid0) + compute_loss(pd_struct1, gt_grid1) + compute_loss(pd_struct2, gt_grid2)
        )
        print(f"step = {i}, loss: {loss.item()}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
