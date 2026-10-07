# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for the Point Transformer V3 layers and model in fvdb.nn."""

import unittest

import torch
import torch.nn.functional as F
from fvdb.utils.tests.convolution_utils import conv_ground_truth_stride_1, disable_tf32
from parameterized import parameterized

import fvdb
import fvdb.nn as fvnn
from fvdb import ConvolutionPlan, GridBatch, JaggedTensor

from .test_patch_attention import CAN_RUN_VARLEN, _ref_patch_attention

DEVICES = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
CUDA = torch.cuda.is_available()


def _make_grid(sizes, device, extent=24, seed=0) -> GridBatch:
    """Random grids whose coordinates start at 0, like Pointcept's grid_coord."""
    gen = torch.Generator().manual_seed(seed)
    ijks = []
    for n in sizes:
        ijk = torch.randint(0, extent, (n, 3), generator=gen, dtype=torch.int32)
        if n > 0:
            ijk[0] = 0
        ijks.append(ijk.to(device))
    return GridBatch.from_ijk(JaggedTensor(ijks))


def _features(grid: GridBatch, channels: int, seed=1) -> JaggedTensor:
    gen = torch.Generator().manual_seed(seed)
    return grid.jagged_like(torch.randn(grid.total_voxels, channels, generator=gen).to(grid.device))


def _parent_index(fine: GridBatch, coarse: GridBatch, stride: int = 2) -> torch.Tensor:
    ijk = fine.ijk
    return coarse.ijk_to_index(ijk.jagged_like(ijk.jdata // stride), cumulative=True).jdata


def _manual_attention(module: fvnn.SerializedAttention, data: JaggedTensor) -> torch.Tensor:
    """Global attention per grid computed directly in float32."""
    heads, dim = module.num_heads, module.channels // module.num_heads
    qkv = F.linear(data.jdata, module.qkv.weight, module.qkv.bias).view(-1, 3, heads, dim)
    outs = []
    offsets = data.joffsets.tolist()
    for s, e in zip(offsets[:-1], offsets[1:]):
        q, k, v = (qkv[s:e, i].transpose(0, 1) for i in range(3))
        attn = torch.softmax((q @ k.transpose(-1, -2)) * module.scale, dim=-1)
        outs.append((attn @ v).transpose(0, 1).reshape(e - s, module.channels))
    return F.linear(torch.cat(outs), module.proj.weight, module.proj.bias)


@unittest.skipUnless(CUDA, "requires CUDA")
class TestSerializedAttention(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_global_attention_matches_manual(self):
        grid = _make_grid([60, 0, 45], "cuda")
        data = _features(grid, 16)
        ser = fvdb.serialize(grid, ("z", "hilbert"))
        for order_index in (0, 1):
            module = fvnn.SerializedAttention(16, 4, patch_size=0, order_index=order_index).cuda()
            with disable_tf32():
                out = module(data, ser)
                expected = _manual_attention(module, data)
            self.assertTrue(torch.equal(out.joffsets, data.joffsets))
            torch.testing.assert_close(out.jdata, expected, atol=1e-5, rtol=1e-5)

    @unittest.skipUnless(CAN_RUN_VARLEN, "requires torch.nn.attention.varlen and an SM80+ GPU")
    def test_patch_attention_matches_reference(self):
        grid = _make_grid([300, 7, 150], "cuda")
        data = _features(grid, 32)
        ser = fvdb.serialize(grid, ("z-trans", "hilbert-trans"))
        module = fvnn.SerializedAttention(32, 2, patch_size=64, order_index=1).cuda()
        out = module(data, ser)

        perm, inv = ser.perm(1), ser.inv_perm(1)
        qkv = module.qkv(data.jdata)[perm].view(-1, 3, 2, 16).bfloat16()
        lshape = grid.num_voxels.tolist()
        attn = _ref_patch_attention(qkv[:, 0], qkv[:, 1], qkv[:, 2], lshape, 64, module.scale)
        expected = module.proj(attn.reshape(-1, 32)[inv])
        torch.testing.assert_close(out.jdata, expected, atol=3e-2, rtol=3e-2)

    @unittest.skipUnless(CAN_RUN_VARLEN, "requires torch.nn.attention.varlen and an SM80+ GPU")
    def test_patch_covering_grid_matches_global(self):
        grid = _make_grid([80, 120], "cuda")
        data = _features(grid, 16)
        ser = fvdb.serialize(grid, "hilbert")
        module = fvnn.SerializedAttention(16, 2, patch_size=1024).cuda()
        with disable_tf32():
            expected = _manual_attention(module, data)
        torch.testing.assert_close(module(data, ser).jdata, expected, atol=3e-2, rtol=3e-2)

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            fvnn.SerializedAttention(10, 3)
        with self.assertRaises(ValueError):
            fvnn.SerializedAttention(16, 2, patch_size=8, window_size=8)


class TestConditionalPositionEncoding(unittest.TestCase):
    @parameterized.expand(DEVICES)
    def test_conv_matches_dense_conv3d(self, device):
        grid = _make_grid([200], device, extent=12)
        data = _features(grid, 8)
        module = fvnn.ConditionalPositionEncoding(8).to(device)
        assert module.conv is not None
        plan = ConvolutionPlan.from_grid_batch(3, 1, grid, grid)
        with disable_tf32():
            conv = module.conv(data, plan)
            _, dense = conv_ground_truth_stride_1(grid, data, module.conv.weight.detach(), ijk_min=(0, 0, 0))
        ijk = grid.ijk.jdata.long()
        expected = dense[0, :, ijk[:, 0], ijk[:, 1], ijk[:, 2]].T + module.conv.bias
        torch.testing.assert_close(conv.jdata, expected, atol=1e-5, rtol=1e-5)

        out = module(data, plan)
        torch.testing.assert_close(out.jdata, module.norm(module.linear(conv.jdata)), atol=1e-6, rtol=1e-6)

    @parameterized.expand(DEVICES)
    def test_without_conv(self, device):
        grid = _make_grid([40], device)
        data = _features(grid, 8)
        module = fvnn.ConditionalPositionEncoding(8, use_conv=False).to(device)
        out = module(data, None)
        torch.testing.assert_close(out.jdata, module.norm(module.linear(data.jdata)))

    def test_conv_requires_plan(self):
        grid = _make_grid([10], "cpu")
        with self.assertRaises(ValueError):
            fvnn.ConditionalPositionEncoding(4)(_features(grid, 4), None)


class TestPoolingAndEmbedding(unittest.TestCase):
    @parameterized.expand(DEVICES)
    def test_pooling_is_max_of_projected_children(self, device):
        grid = _make_grid([150, 0, 90], device)
        data = _features(grid, 6)
        module = fvnn.SerializedPooling(6, 10, norm="layer").to(device)
        out, coarse = module(data, grid)
        self.assertTrue(torch.equal(coarse.ijk.jdata, grid.coarsened_grid(2).ijk.jdata))

        parent = _parent_index(grid, coarse)
        proj = module.proj(data.jdata)
        pooled = torch.full((coarse.total_voxels, 10), -torch.inf, device=grid.device)
        pooled = pooled.scatter_reduce(0, parent[:, None].expand(-1, 10), proj, reduce="amax")
        expected = F.gelu(module.norm.norm(pooled))
        torch.testing.assert_close(out.jdata, expected, atol=1e-5, rtol=1e-5)

    @parameterized.expand(DEVICES)
    def test_unpooling_adds_parent_features_to_skip(self, device):
        fine = _make_grid([150, 90], device)
        coarse = fine.coarsened_grid(2)
        coarse_data, skip_data = _features(coarse, 12), _features(fine, 6, seed=2)
        module = fvnn.SerializedUnpooling(12, 6, 8, norm="layer").to(device)
        out, out_grid = module(coarse_data, coarse, skip_data, fine)
        self.assertIs(out_grid, fine)

        up = F.gelu(module.norm.norm(module.proj(coarse_data.jdata)))
        skip = F.gelu(module.norm_skip.norm(module.proj_skip(skip_data.jdata)))
        expected = skip + up[_parent_index(fine, coarse)]
        torch.testing.assert_close(out.jdata, expected, atol=1e-5, rtol=1e-5)

    @parameterized.expand(DEVICES)
    def test_batch_norm_option(self, device):
        grid = _make_grid([100, 80], device)
        module = fvnn.SerializedPooling(4, 8, norm="batch").to(device)
        out, coarse = module(_features(grid, 4), grid)
        self.assertEqual(out.jdata.shape, (coarse.total_voxels, 8))
        self.assertIsInstance(module.norm.norm, fvnn.BatchNorm)
        with self.assertRaises(ValueError):
            fvnn.SerializedPooling(4, 8, norm="group")  # type: ignore[arg-type]

    @parameterized.expand(DEVICES)
    def test_embedding_modes(self, device):
        grid = _make_grid([120, 30], device)
        data = _features(grid, 6)
        linear = fvnn.PointTransformerV3Embedding(6, 16, mode="linear", norm="layer").to(device)
        expected = F.gelu(linear.norm.norm(linear.stem(data.jdata)))
        torch.testing.assert_close(linear(data, grid, None).jdata, expected)

        conv = fvnn.PointTransformerV3Embedding(6, 16).to(device)
        self.assertIsNone(conv.stem.bias)
        plan = ConvolutionPlan.from_grid_batch(5, 1, grid, grid)
        self.assertEqual(conv(data, grid, plan).jdata.shape, (grid.total_voxels, 16))
        with self.assertRaises(ValueError):
            conv(data, grid, None)


@unittest.skipUnless(CUDA, "requires CUDA")
class TestPointTransformerV3Block(unittest.TestCase):
    @parameterized.expand([[True], [False]])
    def test_residual_structure(self, pre_norm):
        grid = _make_grid([90, 60], "cuda")
        data = _features(grid, 16)
        ser = fvdb.serialize(grid, ("z", "z-trans"))
        plan = ConvolutionPlan.from_grid_batch(3, 1, grid, grid)
        block = fvnn.PointTransformerV3Block(16, 2, patch_size=0, pre_norm=pre_norm, order_index=1).cuda().eval()
        out = block(data, ser, plan)

        x = data + block.cpe(data, plan)
        if pre_norm:
            x = x + block.attn(block.norm1(x), ser)
            x = x + block.mlp(block.norm2(x))
        else:
            x = block.norm1(x + block.attn(x, ser))
            x = block.norm2(x + block.mlp(x))
        torch.testing.assert_close(out.jdata, x.jdata)

    def test_backward(self):
        grid = _make_grid([90, 60], "cuda")
        data = _features(grid, 16)
        ser = fvdb.serialize(grid, "hilbert")
        plan = ConvolutionPlan.from_grid_batch(3, 1, grid, grid)
        block = fvnn.PointTransformerV3Block(16, 4, patch_size=0, drop_path=0.2).cuda()
        block(data, ser, plan).jdata.square().sum().backward()
        for name, param in block.named_parameters():
            self.assertIsNotNone(param.grad, name)
            assert param.grad is not None
            self.assertTrue(bool(torch.isfinite(param.grad).all()), name)


SMALL_CONFIG = dict(
    in_channels=4,
    order=("z", "hilbert-trans"),
    stride=(2, 2),
    enc_depths=(1, 1, 1),
    enc_channels=(8, 16, 32),
    enc_num_head=(1, 2, 4),
    enc_patch_size=16,
    dec_depths=(1, 1),
    dec_channels=(8, 16),
    dec_num_head=(1, 2),
    dec_patch_size=16,
    drop_path=0.1,
)


@unittest.skipUnless(CUDA, "requires CUDA")
class TestPointTransformerV3(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.grid = _make_grid([400, 0, 250, 3], "cuda", extent=20)
        self.data = _features(self.grid, 4)

    def _check_backward(self, model, out):
        out.jdata.square().mean().backward()
        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad, name)
            assert param.grad is not None
            self.assertTrue(bool(torch.isfinite(param.grad).all()), name)

    def _check_output(self, out, out_grid, channels):
        self.assertIs(out_grid, self.grid)
        self.assertTrue(torch.equal(out.joffsets, self.data.joffsets))
        self.assertEqual(out.jdata.shape, (self.grid.total_voxels, channels))
        self.assertTrue(bool(torch.isfinite(out.jdata).all()))

    @unittest.skipUnless(CAN_RUN_VARLEN, "requires torch.nn.attention.varlen and an SM80+ GPU")
    def test_patch_attention_forward_backward(self):
        model = fvnn.PointTransformerV3(**SMALL_CONFIG).cuda()
        out, out_grid = model(self.data, self.grid)
        self._check_output(out, out_grid, 8)
        self._check_backward(model, out)

    @unittest.skipUnless(CAN_RUN_VARLEN, "requires torch.nn.attention.varlen and an SM80+ GPU")
    def test_window_attention(self):
        model = fvnn.PointTransformerV3(**SMALL_CONFIG, window_size=8).cuda()
        self.assertEqual(model.enc[0].blocks[0].attn.window_size, 8)
        self.assertEqual(model.enc[0].blocks[0].attn.patch_size, 0)
        out, out_grid = model(self.data, self.grid)
        self._check_output(out, out_grid, 8)
        self._check_backward(model, out)

    def test_global_attention_float32_backward(self):
        config = dict(SMALL_CONFIG, enc_patch_size=0, dec_patch_size=0)
        model = fvnn.PointTransformerV3(**config).cuda()
        out, out_grid = model(self.data, self.grid)
        self._check_output(out, out_grid, 8)
        self._check_backward(model, out)

    def test_linear_stem_layer_norm_without_cpe_conv(self):
        # The configuration fvdb-examples compared against its modified Pointcept model.
        config = dict(SMALL_CONFIG, enc_patch_size=0, dec_patch_size=0, order=("z",))
        model = fvnn.PointTransformerV3(**config, embedding_mode="linear", norm="layer", cpe_conv=False).cuda()
        self.assertIsNone(model.enc[0].blocks[0].cpe.conv)
        out, out_grid = model(self.data, self.grid)
        self._check_output(out, out_grid, 8)

    def test_cls_mode_returns_coarsest_grid(self):
        config = dict(SMALL_CONFIG, enc_patch_size=0)
        model = fvnn.PointTransformerV3(**config, cls_mode=True).cuda()
        self.assertEqual(len(model.dec), 0)
        out, out_grid = model(self.data, self.grid)
        expected = self.grid.coarsened_grid(2).coarsened_grid(2)
        self.assertTrue(torch.equal(out_grid.ijk.jdata, expected.ijk.jdata))
        self.assertEqual(out.jdata.shape, (expected.total_voxels, 32))

    def test_eval_is_deterministic_without_shuffling(self):
        config = dict(SMALL_CONFIG, enc_patch_size=0, dec_patch_size=0)
        model = fvnn.PointTransformerV3(**config, shuffle_orders=False, shuffle_pooled_orders=False).cuda().eval()
        with torch.no_grad():
            first, _ = model(self.data, self.grid)
            second, _ = model(self.data, self.grid)
        self.assertTrue(torch.equal(first.jdata, second.jdata))

    @unittest.skipUnless(CAN_RUN_VARLEN, "requires torch.nn.attention.varlen and an SM80+ GPU")
    def test_autocast_bfloat16(self):
        model = fvnn.PointTransformerV3(**SMALL_CONFIG).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out, _ = model(self.data, self.grid)
        self.assertTrue(bool(torch.isfinite(out.jdata.float()).all()))
        self._check_backward(model, out.jagged_like(out.jdata.float()))

    def test_default_config_builds(self):
        model = fvnn.PointTransformerV3()
        self.assertEqual(len(model.enc), 5)
        self.assertEqual(len(model.dec), 4)
        self.assertEqual([len(stage.blocks) for stage in model.enc], [2, 2, 2, 6, 2])
        self.assertEqual(model.order, ("z", "z-trans", "hilbert", "hilbert-trans"))

    def test_invalid_configs(self):
        # fvdb-examples' PTV3 defaults: four encoder depths but five channel entries.
        with self.assertRaisesRegex(ValueError, "enc_channels"):
            fvnn.PointTransformerV3(enc_depths=(2, 2, 2, 2), stride=(2, 2, 2))
        with self.assertRaisesRegex(ValueError, "stride"):
            fvnn.PointTransformerV3(stride=(2, 2, 3, 2))
        with self.assertRaisesRegex(ValueError, "attn_drop"):
            fvnn.PointTransformerV3(attn_drop=0.1)
        with self.assertRaisesRegex(ValueError, "dec_channels"):
            fvnn.PointTransformerV3(dec_channels=(64, 64, 128))


if __name__ == "__main__":
    unittest.main()
