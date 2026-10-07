# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for patch and window attention in fvdb.scaled_dot_product_attention."""

import importlib.util
import unittest

import torch
import torch.nn.functional as F
from parameterized import parameterized

import fvdb
from fvdb import JaggedTensor
from fvdb.attention import _patch_layout, _window_bounds

HAS_VARLEN = importlib.util.find_spec("torch.nn.attention.varlen") is not None
HAS_SM80 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
CAN_RUN_VARLEN = HAS_VARLEN and HAS_SM80
DEVICES = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]

TOLERANCES = {torch.bfloat16: 2e-2, torch.float16: 5e-3, torch.float32: 2e-2}


def _offsets(lshape, device) -> torch.Tensor:
    return F.pad(torch.cumsum(torch.tensor(lshape, dtype=torch.int64), 0), (1, 0)).to(device)


def _pointcept_padding(lshape, patch_size, device):
    """Port of Pointcept's SerializedAttention.get_padding_and_inverse, used as an oracle."""
    bincount = torch.tensor(lshape, dtype=torch.long)
    offset = torch.cumsum(bincount, 0)
    bincount_pad = torch.div(bincount + patch_size - 1, patch_size, rounding_mode="trunc") * patch_size
    mask_pad = bincount > patch_size
    bincount_pad = ~mask_pad * bincount + mask_pad * bincount_pad
    _offset = F.pad(offset, (1, 0))
    _offset_pad = F.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))
    pad = torch.arange(int(_offset_pad[-1]))
    unpad = torch.arange(int(_offset[-1]))
    cu_seqlens = []
    for i in range(len(offset)):
        unpad[_offset[i] : _offset[i + 1]] += _offset_pad[i] - _offset[i]
        if bincount[i] != bincount_pad[i]:
            r = int(bincount[i] % patch_size)
            end = int(_offset_pad[i + 1])
            pad[end - patch_size + r : end] = pad[end - 2 * patch_size + r : end - patch_size]
        pad[_offset_pad[i] : _offset_pad[i + 1]] -= _offset_pad[i] - _offset[i]
        cu_seqlens.append(torch.arange(int(_offset_pad[i]), int(_offset_pad[i + 1]), step=patch_size))
    cu = F.pad(torch.cat(cu_seqlens), (0, 1), value=int(_offset_pad[-1]))
    return pad.to(device), unpad.to(device), cu.to(torch.int32).to(device)


def _attend(q, k, v, scale, mask=None):
    """Math-backend attention on (L, H, D) tensors in float32."""
    out = F.scaled_dot_product_attention(
        q.transpose(0, 1).float(), k.transpose(0, 1).float(), v.transpose(0, 1).float(), attn_mask=mask, scale=scale
    )
    return out.transpose(0, 1)


def _ref_patch_attention(q, k, v, lshape, patch_size, scale):
    """Reference patch attention: loop over sequences and patches, with Pointcept padding."""
    outs, start = [], 0
    for n in lshape:
        qs, ks, vs = q[start : start + n], k[start : start + n], v[start : start + n]
        start += n
        if n == 0:
            continue
        padded = n if n <= patch_size else -(-n // patch_size) * patch_size
        if padded > n:
            src = torch.arange(padded, device=q.device)
            src = torch.where(src < n, src, src - patch_size)
            qs, ks, vs = qs[src], ks[src], vs[src]
        chunks = [
            _attend(qs[a : a + patch_size], ks[a : a + patch_size], vs[a : a + patch_size], scale)
            for a in range(0, padded, patch_size)
        ]
        outs.append(torch.cat(chunks)[:n])
    return torch.cat(outs)


def _ref_window_attention(q, k, v, lshape, left, right, scale):
    """Reference window attention with an explicit band mask per sequence."""
    outs, start = [], 0
    for n in lshape:
        if n == 0:
            continue
        i = torch.arange(n, device=q.device)
        mask = torch.ones(n, n, dtype=torch.bool, device=q.device)
        if left >= 0:
            mask &= i[None, :] >= i[:, None] - left
        if right >= 0:
            mask &= i[None, :] <= i[:, None] + right
        outs.append(_attend(q[start : start + n], k[start : start + n], v[start : start + n], scale, mask))
        start += n
    return torch.cat(outs)


def _random_tensors(lshape, heads, dim, dtype, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return [torch.randn(sum(lshape), heads, dim, device="cuda", generator=gen).to(dtype) for _ in range(3)]


def _random_qkv(lshape, heads, dim, dtype, seed=0):
    offsets = _offsets(lshape, "cuda")
    return [JaggedTensor.from_data_and_offsets(t, offsets) for t in _random_tensors(lshape, heads, dim, dtype, seed)]


class TestPatchLayout(unittest.TestCase):
    @parameterized.expand(DEVICES)
    def test_no_padding_when_sequences_fit(self, device):
        lshape = [5, 8, 0, 3]
        layout = _patch_layout(_offsets(lshape, device), lshape, 8)
        self.assertTrue(torch.equal(layout.gather_idx.cpu(), torch.arange(16)))
        self.assertTrue(torch.equal(layout.unpad_idx.cpu(), torch.arange(16)))
        self.assertEqual(layout.cu_seqlens.tolist(), [0, 5, 13, 16])
        self.assertEqual(layout.max_seqlen, 8)

    @parameterized.expand(DEVICES)
    def test_exact_multiple(self, device):
        layout = _patch_layout(_offsets([16], device), [16], 8)
        self.assertEqual(layout.cu_seqlens.tolist(), [0, 8, 16])
        self.assertTrue(torch.equal(layout.gather_idx.cpu(), torch.arange(16)))

    @parameterized.expand(DEVICES)
    def test_padding_repeats_previous_patch(self, device):
        layout = _patch_layout(_offsets([11], device), [11], 4)
        self.assertEqual(layout.gather_idx.tolist(), list(range(11)) + [7])
        self.assertEqual(layout.cu_seqlens.tolist(), [0, 4, 8, 12])
        self.assertEqual(layout.unpad_idx.tolist(), list(range(11)))

        layout = _patch_layout(_offsets([6, 11], device), [6, 11], 4)
        self.assertEqual(layout.gather_idx[8:].tolist(), [6 + i for i in list(range(11)) + [7]])
        self.assertEqual(layout.unpad_idx[6:].tolist(), list(range(8, 19)))
        self.assertEqual(layout.cu_seqlens.tolist(), [0, 4, 8, 12, 16, 20])

    @parameterized.expand(DEVICES)
    def test_matches_pointcept_padding(self, device):
        gen = torch.Generator().manual_seed(0)
        for _ in range(50):
            patch_size = int(torch.randint(1, 12, (1,), generator=gen))
            lshape = torch.randint(0, 40, (int(torch.randint(1, 6, (1,), generator=gen)),), generator=gen).tolist()
            if sum(lshape) == 0:
                continue
            layout = _patch_layout(_offsets(lshape, device), lshape, patch_size)
            pad, unpad, cu = _pointcept_padding(lshape, patch_size, device)
            self.assertTrue(torch.equal(layout.gather_idx, pad), (lshape, patch_size))
            self.assertTrue(torch.equal(layout.unpad_idx, unpad), (lshape, patch_size))
            self.assertTrue(torch.equal(layout.cu_seqlens, cu), (lshape, patch_size))

    def test_int32_overflow_raises(self):
        with self.assertRaisesRegex(ValueError, "int32"):
            _patch_layout(torch.tensor([0, 2**31 + 1]), [2**31 + 1], 1024)

    def test_window_bounds(self):
        self.assertEqual(_window_bounds(8), (4, 4))
        self.assertEqual(_window_bounds(9), (4, 4))
        self.assertEqual(_window_bounds((3, 0)), (3, 0))
        with self.assertRaises(ValueError):
            _window_bounds(-2)
        with self.assertRaises(ValueError):
            _window_bounds((-3, 1))


class TestPatchAttentionArguments(unittest.TestCase):
    def _qkv_cpu(self):
        jt = JaggedTensor([torch.randn(5, 2, 8), torch.randn(3, 2, 8)])
        return jt, jt, jt

    def test_patch_and_window_together_raise(self):
        q, k, v = self._qkv_cpu()
        with self.assertRaisesRegex(ValueError, "at most one"):
            fvdb.scaled_dot_product_attention(q, k, v, 1.0, patch_size=4, window_size=4)

    def test_negative_patch_size_raises(self):
        q, k, v = self._qkv_cpu()
        with self.assertRaises(ValueError):
            fvdb.scaled_dot_product_attention(q, k, v, 1.0, patch_size=-1)

    def test_global_attention_accepts_non_contiguous_inputs(self):
        qkv = torch.randn(8, 3, 2, 4)
        offsets = torch.tensor([0, 5, 8])
        q, k, v = (JaggedTensor.from_data_and_offsets(t, offsets) for t in qkv.unbind(1))
        self.assertFalse(q.jdata.is_contiguous())
        out = fvdb.scaled_dot_product_attention(q, k, v, 0.5)
        contiguous = [JaggedTensor.from_data_and_offsets(t.contiguous(), offsets) for t in qkv.unbind(1)]
        expected = fvdb.scaled_dot_product_attention(*contiguous, 0.5)
        torch.testing.assert_close(out.jdata, expected.jdata)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_global_attention_float32_backward(self):
        # The memory-efficient backend has no nested-tensor backward, so float32 training falls back to math.
        data = torch.randn(30, 2, 16, device="cuda", requires_grad=True)
        jt = JaggedTensor.from_data_and_offsets(data, torch.tensor([0, 12, 30], device="cuda"))
        fvdb.scaled_dot_product_attention(jt, jt, jt, 0.25).jdata.square().sum().backward()
        assert data.grad is not None

        ref_data = data.detach().clone().requires_grad_(True)
        outs = [_attend(t, t, t, 0.25) for t in (ref_data[:12], ref_data[12:])]
        torch.cat(outs).square().sum().backward()
        torch.testing.assert_close(data.grad, ref_data.grad, atol=1e-4, rtol=1e-4)

    def test_cpu_or_old_torch_raises(self):
        q, k, v = self._qkv_cpu()
        message = "CUDA" if HAS_VARLEN else "2.11"
        with self.assertRaisesRegex(RuntimeError, message):
            fvdb.scaled_dot_product_attention(q, k, v, 1.0, patch_size=4)


@unittest.skipUnless(CAN_RUN_VARLEN, "requires torch.nn.attention.varlen and an SM80+ GPU")
class TestPatchAttentionKernel(unittest.TestCase):
    HEADS = 4
    DIM = 64
    LSHAPE = [37, 64, 0, 5, 128]

    @parameterized.expand([[torch.bfloat16], [torch.float16], [torch.float32]])
    def test_patch_matches_reference(self, dtype):
        q, k, v = _random_qkv(self.LSHAPE, self.HEADS, self.DIM, dtype)
        scale = self.DIM**-0.5
        out = fvdb.scaled_dot_product_attention(q, k, v, scale, patch_size=32)
        self.assertEqual(out.dtype, dtype)
        self.assertTrue(torch.equal(out.joffsets, q.joffsets))

        # Compare against inputs rounded to the kernel dtype.
        kernel_dtype = torch.bfloat16 if dtype == torch.float32 else dtype
        qr, kr, vr = (t.jdata.to(kernel_dtype) for t in (q, k, v))
        ref = _ref_patch_attention(qr, kr, vr, self.LSHAPE, 32, scale)
        tol = TOLERANCES[dtype]
        torch.testing.assert_close(out.jdata.float(), ref, atol=tol, rtol=tol)

    def test_fp16_compute_dtype(self):
        q, k, v = _random_qkv(self.LSHAPE, self.HEADS, self.DIM, torch.float32)
        scale = self.DIM**-0.5
        out = fvdb.scaled_dot_product_attention(q, k, v, scale, patch_size=32, compute_dtype=torch.float16)
        self.assertEqual(out.dtype, torch.float32)
        qr, kr, vr = (t.jdata.half() for t in (q, k, v))
        ref = _ref_patch_attention(qr, kr, vr, self.LSHAPE, 32, scale)
        torch.testing.assert_close(out.jdata, ref, atol=5e-3, rtol=5e-3)

    def test_patch_larger_than_sequences_equals_global(self):
        q, k, v = _random_qkv(self.LSHAPE, self.HEADS, self.DIM, torch.bfloat16)
        scale = self.DIM**-0.5
        local = fvdb.scaled_dot_product_attention(q, k, v, scale, patch_size=4096)
        qf, kf, vf = (t.jagged_like(t.jdata.float()) for t in (q, k, v))
        global_ = fvdb.scaled_dot_product_attention(qf, kf, vf, scale)
        torch.testing.assert_close(local.jdata.float(), global_.jdata, atol=2e-2, rtol=2e-2)

    @parameterized.expand([[8], [9]])
    def test_window_matches_reference(self, window):
        q, k, v = _random_qkv(self.LSHAPE, self.HEADS, self.DIM, torch.bfloat16)
        scale = self.DIM**-0.5
        out = fvdb.scaled_dot_product_attention(q, k, v, scale, window_size=window)
        ref = _ref_window_attention(q.jdata, k.jdata, v.jdata, self.LSHAPE, window // 2, window // 2, scale)
        torch.testing.assert_close(out.jdata.float(), ref, atol=2e-2, rtol=2e-2)

    def test_window_tuple_passthrough(self):
        q, k, v = _random_qkv(self.LSHAPE, self.HEADS, self.DIM, torch.bfloat16)
        scale = self.DIM**-0.5
        out = fvdb.scaled_dot_product_attention(q, k, v, scale, window_size=(3, 0))
        ref = _ref_window_attention(q.jdata, k.jdata, v.jdata, self.LSHAPE, 3, 0, scale)
        torch.testing.assert_close(out.jdata.float(), ref, atol=2e-2, rtol=2e-2)

    def test_single_sequence(self):
        q, k, v = _random_qkv([100], self.HEADS, self.DIM, torch.bfloat16)
        scale = self.DIM**-0.5
        out = fvdb.scaled_dot_product_attention(q, k, v, scale, patch_size=16)
        ref = _ref_patch_attention(q.jdata, k.jdata, v.jdata, [100], 16, scale)
        torch.testing.assert_close(out.jdata.float(), ref, atol=2e-2, rtol=2e-2)

    def test_all_sequences_empty(self):
        q, k, v = _random_qkv([0, 0], self.HEADS, self.DIM, torch.bfloat16)
        out = fvdb.scaled_dot_product_attention(q, k, v, 1.0, patch_size=16)
        self.assertEqual(out.jdata.shape, (0, self.HEADS, self.DIM))

    def test_backward_matches_reference(self):
        lshape = [37, 70, 5]
        offsets = _offsets(lshape, "cuda")
        leaves = [t.requires_grad_(True) for t in _random_tensors(lshape, self.HEADS, self.DIM, torch.bfloat16)]
        q, k, v = (JaggedTensor.from_data_and_offsets(t, offsets) for t in leaves)
        scale = self.DIM**-0.5
        out = fvdb.scaled_dot_product_attention(q, k, v, scale, patch_size=16)
        grad_out = torch.randn_like(out.jdata.float())
        (out.jdata.float() * grad_out).sum().backward()

        ref_leaves = [t.detach().float().requires_grad_(True) for t in leaves]
        ref = _ref_patch_attention(*ref_leaves, lshape, 16, scale)
        (ref * grad_out).sum().backward()
        for ours, theirs in zip(leaves, ref_leaves):
            assert ours.grad is not None and theirs.grad is not None
            torch.testing.assert_close(ours.grad.float(), theirs.grad, atol=5e-2, rtol=5e-2)

    def test_fp64_raises(self):
        q, k, v = _random_qkv([10], self.HEADS, self.DIM, torch.float64)
        with self.assertRaises(TypeError):
            fvdb.scaled_dot_product_attention(q, k, v, 1.0, patch_size=4)

    def test_serialized_end_to_end(self):
        gen = torch.Generator().manual_seed(2)
        ijks = [torch.randint(0, 24, (n, 3), generator=gen, dtype=torch.int32).cuda() for n in (300, 90)]
        grid = fvdb.GridBatch.from_ijk(JaggedTensor(ijks))
        ser = fvdb.serialize(grid, ("z", "hilbert"))
        lshape = grid.num_voxels.tolist()
        feats = [grid.jagged_like(torch.randn(grid.total_voxels, 2, 32, device="cuda").bfloat16()) for _ in range(3)]
        scale = 32**-0.5
        for i in range(len(ser)):
            q, k, v = (fvdb.permute_jagged(f, ser.perm(i)) for f in feats)
            out = fvdb.permute_jagged(fvdb.scaled_dot_product_attention(q, k, v, scale, patch_size=64), ser.inv_perm(i))
            ref = _ref_patch_attention(q.jdata, k.jdata, v.jdata, lshape, 64, scale)[ser.inv_perm(i)]
            torch.testing.assert_close(out.jdata.float(), ref, atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    unittest.main()
