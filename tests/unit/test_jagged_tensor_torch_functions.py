# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
# Tests for type-safe torch-compatible functions on fvdb.JaggedTensor
# (style and patterns follow existing tests)
#
# These tests use fvdb.* functions which provide proper type inference
# for both JaggedTensor and regular Tensor arguments.

import unittest
from typing import cast

import numpy as np
import torch
from parameterized import parameterized

import fvdb

all_device_dtype_combos = [
    ["cuda", torch.float16],
    ["cpu", torch.float32],
    ["cuda", torch.float32],
    ["cpu", torch.float64],
    ["cuda", torch.float64],
]


class TestJaggedTensorTorchFunctions(unittest.TestCase):
    def setUp(self):
        torch.random.manual_seed(123)
        np.random.seed(123)

    # -------- helpers --------
    def _mk_list(self, device, dtype, count=5, edims=(3, 4)) -> list[torch.Tensor]:
        ls = [torch.randn(100 + np.random.randint(10), *edims, device=device, dtype=dtype) for _ in range(count)]
        return ls

    def _mk_jt(self, device, dtype, count=5, edims=(3, 4)) -> tuple[fvdb.JaggedTensor, list[torch.Tensor]]:
        data = self._mk_list(device, dtype, count=count, edims=edims)
        jt = fvdb.JaggedTensor(data)
        return jt, data

    def _assert_preserved_layout(self, a: fvdb.JaggedTensor, b: fvdb.JaggedTensor) -> None:
        # Primary dim and jagged layout should match
        self.assertTrue(torch.equal(a.joffsets, b.joffsets))
        self.assertTrue(torch.equal(a.jidx, b.jidx))
        self.assertEqual(a.lshape, b.lshape)

    # -------------- unary pointwise --------------

    @parameterized.expand(all_device_dtype_combos)
    def test_relu(self, device, dtype):
        jt, _ = self._mk_jt(device, dtype)
        out = fvdb.relu(jt)
        self._assert_preserved_layout(out, jt)
        self.assertTrue(torch.allclose(out.jdata, torch.relu(jt.jdata)))

    @parameterized.expand(all_device_dtype_combos)
    def test_relu_inplace(self, device, dtype):
        jt, _ = self._mk_jt(device, dtype)
        before = jt.jdata.clone()
        ret = fvdb.relu_(jt)
        self.assertIs(ret, jt)
        self.assertTrue(torch.allclose(jt.jdata, torch.relu(before)))

    @parameterized.expand(all_device_dtype_combos)
    def test_sigmoid(self, device, dtype):
        jt, _ = self._mk_jt(device, dtype)
        out = fvdb.sigmoid(jt)
        self._assert_preserved_layout(out, jt)
        self.assertTrue(torch.allclose(out.jdata, torch.sigmoid(jt.jdata)))

    @parameterized.expand(all_device_dtype_combos)
    def test_tanh(self, device, dtype):
        jt, _ = self._mk_jt(device, dtype)
        out = fvdb.tanh(jt)
        self._assert_preserved_layout(out, jt)
        self.assertTrue(torch.allclose(out.jdata, torch.tanh(jt.jdata)))

    @parameterized.expand(all_device_dtype_combos)
    def test_exp_log(self, device, dtype):
        jt, _ = self._mk_jt(device, dtype)
        out_exp = fvdb.exp(jt)
        self._assert_preserved_layout(out_exp, jt)
        self.assertTrue(torch.allclose(out_exp.jdata, torch.exp(jt.jdata)))
        tensors = cast(list[torch.Tensor], jt.unbind())  # resolving unbind ambiguity
        pos = fvdb.JaggedTensor([t.abs() + 1e-4 for t in tensors])
        out_log = fvdb.log(pos)
        self._assert_preserved_layout(out_log, pos)
        self.assertTrue(torch.allclose(out_log.jdata, torch.log(pos.jdata)))

    @parameterized.expand(all_device_dtype_combos)
    def test_sqrt(self, device, dtype):
        jt, _ = self._mk_jt(device, dtype)
        tensors = cast(list[torch.Tensor], jt.unbind())  # resolving unbind ambiguity
        jt_pos = fvdb.JaggedTensor([t.abs() for t in tensors])
        out = fvdb.sqrt(jt_pos)
        self._assert_preserved_layout(out, jt_pos)
        self.assertTrue(torch.allclose(out.jdata, torch.sqrt(jt_pos.jdata)))

    @parameterized.expand(all_device_dtype_combos)
    def test_floor_and_ceil_and_round(self, device, dtype):
        jt, _ = self._mk_jt(device, dtype)
        out_f = fvdb.floor(jt)
        out_c = fvdb.ceil(jt)
        out_r = fvdb.round(jt)
        self._assert_preserved_layout(out_f, jt)
        self._assert_preserved_layout(out_c, jt)
        self._assert_preserved_layout(out_r, jt)
        self.assertTrue(torch.allclose(out_f.jdata, torch.floor(jt.jdata)))
        self.assertTrue(torch.allclose(out_c.jdata, torch.ceil(jt.jdata)))
        self.assertTrue(torch.allclose(out_r.jdata, torch.round(jt.jdata)))

    @parameterized.expand(all_device_dtype_combos)
    def test_nan_to_num_and_clamp(self, device, dtype):
        jt, _ = self._mk_jt(device, dtype)
        tensors = cast(list[torch.Tensor], jt.unbind())  # resolving unbind ambiguity
        jt2 = fvdb.JaggedTensor([t.clone() for t in tensors])
        # introduce some NaNs / infs safely
        if jt2.jdata.numel() > 0:
            jt2.jdata.view(-1)[::17] = float("nan")
            jt2.jdata.view(-1)[::29] = float("inf")
        out = fvdb.nan_to_num(jt2, nan=0.0, posinf=1.0, neginf=-1.0)
        self._assert_preserved_layout(out, jt2)
        self.assertTrue(torch.allclose(out.jdata, torch.nan_to_num(jt2.jdata, nan=0.0, posinf=1.0, neginf=-1.0)))
        out2 = fvdb.clamp(jt, min=-0.5, max=0.5)
        self._assert_preserved_layout(out2, jt)
        self.assertTrue(torch.allclose(out2.jdata, torch.clamp(jt.jdata, min=-0.5, max=0.5)))

    # -------------- binary/ternary elementwise --------------

    @parameterized.expand(all_device_dtype_combos)
    def test_add_sub_mul_div_with_scalar(self, device, dtype):
        jt, _ = self._mk_jt(device, dtype)
        self.assertTrue(torch.allclose(fvdb.add(jt, 2.0).jdata, torch.add(jt.jdata, 2.0)))
        self.assertTrue(torch.allclose(fvdb.sub(jt, 3).jdata, torch.sub(jt.jdata, 3)))
        self.assertTrue(torch.allclose(fvdb.mul(jt, 4).jdata, torch.mul(jt.jdata, 4)))
        self.assertTrue(torch.allclose(fvdb.true_divide(jt, 5).jdata, torch.true_divide(jt.jdata, 5)))

    @parameterized.expand(all_device_dtype_combos)
    def test_add_sub_mul_div_with_jagged(self, device, dtype):
        jt, _ = self._mk_jt(device, dtype)
        tensors = cast(list[torch.Tensor], jt.unbind())  # resolving unbind ambiguity
        jt_b = fvdb.JaggedTensor([torch.rand_like(t) + 1e-5 for t in tensors])
        self._assert_preserved_layout(jt_b, jt)
        self.assertTrue(torch.allclose(fvdb.add(jt, jt_b).jdata, torch.add(jt.jdata, jt_b.jdata)))
        self.assertTrue(torch.allclose(fvdb.sub(jt, jt_b).jdata, torch.sub(jt.jdata, jt_b.jdata)))
        self.assertTrue(torch.allclose(fvdb.mul(jt, jt_b).jdata, torch.mul(jt.jdata, jt_b.jdata)))
        self.assertTrue(torch.allclose(fvdb.true_divide(jt, jt_b).jdata, torch.true_divide(jt.jdata, jt_b.jdata)))

    @parameterized.expand(all_device_dtype_combos)
    def test_floor_div_remainder_pow(self, device, dtype):
        jt, _ = self._mk_jt(device, dtype)
        tensors = cast(list[torch.Tensor], jt.unbind())  # resolving unbind ambiguity
        jt_b = fvdb.JaggedTensor([torch.rand_like(t) + 0.5 for t in tensors])
        out1 = fvdb.floor_divide(jt, 2)
        out2 = fvdb.remainder(jt, 3)
        out3 = fvdb.pow(jt_b, 2.0)
        self._assert_preserved_layout(out1, jt)
        self._assert_preserved_layout(out2, jt)
        self._assert_preserved_layout(out3, jt_b)
        self.assertTrue(torch.allclose(out1.jdata, torch.floor_divide(jt.jdata, 2)))
        self.assertTrue(torch.allclose(out2.jdata, torch.remainder(jt.jdata, 3)))
        self.assertTrue(torch.allclose(out3.jdata, torch.pow(jt_b.jdata, 2.0)))

    @parameterized.expand(all_device_dtype_combos)
    def test_minimum_maximum(self, device, dtype):
        jt, _ = self._mk_jt(device, dtype)
        tensors = cast(list[torch.Tensor], jt.unbind())  # resolving unbind ambiguity
        jt_b = fvdb.JaggedTensor([torch.rand_like(t) for t in tensors])
        out_min = fvdb.minimum(jt, jt_b)
        out_max = fvdb.maximum(jt, jt_b)
        self._assert_preserved_layout(out_min, jt)
        self._assert_preserved_layout(out_max, jt)
        self.assertTrue(torch.allclose(out_min.jdata, torch.minimum(jt.jdata, jt_b.jdata)))
        self.assertTrue(torch.allclose(out_max.jdata, torch.maximum(jt.jdata, jt_b.jdata)))

    @parameterized.expand(all_device_dtype_combos)
    def test_comparisons(self, device, dtype):
        jt, _ = self._mk_jt(device, dtype)
        tensors = cast(list[torch.Tensor], jt.unbind())  # resolving unbind ambiguity
        jt_b = fvdb.JaggedTensor([torch.rand_like(t) for t in tensors])
        for fn, fvdb_fn in [
            (torch.eq, fvdb.eq),
            (torch.ne, fvdb.ne),
            (torch.lt, fvdb.lt),
            (torch.le, fvdb.le),
            (torch.gt, fvdb.gt),
            (torch.ge, fvdb.ge),
        ]:
            out = fvdb_fn(jt, jt_b)
            self._assert_preserved_layout(out, jt)
            self.assertTrue(torch.all(out.jdata == fn(jt.jdata, jt_b.jdata)))

    @parameterized.expand(all_device_dtype_combos)
    def test_where(self, device, dtype):
        jt, _ = self._mk_jt(device, dtype)
        tensors = cast(list[torch.Tensor], jt.unbind())  # resolving unbind ambiguity
        jt_b = fvdb.JaggedTensor([torch.rand_like(t) for t in tensors])
        # cond must broadcast across last dims; use full boolean tensor:
        cond = fvdb.JaggedTensor([torch.rand_like(t, dtype=torch.float32) > 0.5 for t in tensors])
        out = fvdb.where(cond, jt, jt_b)
        self._assert_preserved_layout(out, jt)
        self.assertTrue(torch.equal(out.jdata, torch.where(cond.jdata, jt.jdata, jt_b.jdata)))

    @parameterized.expand(all_device_dtype_combos)
    def test_out_argument(self, device, dtype):
        jt, _ = self._mk_jt(device, dtype)
        expected = torch.add(jt.jdata, 1.0)
        out_buf = jt.jagged_like(torch.empty_like(expected))
        ret = fvdb.add(jt, 1.0, out=out_buf)
        self.assertIs(ret, out_buf)
        self._assert_preserved_layout(out_buf, jt)
        self.assertTrue(torch.allclose(out_buf.jdata, expected))

    # -------------- reductions over non-leading dims --------------

    @parameterized.expand(all_device_dtype_combos)
    def test_sum_mean_keep_primary_dim(self, device, dtype):
        jt, _ = self._mk_jt(device, dtype)
        out_sum = fvdb.sum(jt, dim=-1)
        out_mean = fvdb.mean(jt, dim=-1)
        self._assert_preserved_layout(out_sum, jt)
        self._assert_preserved_layout(out_mean, jt)
        self.assertTrue(torch.allclose(out_sum.jdata, torch.sum(jt.jdata, dim=-1)))
        self.assertTrue(torch.allclose(out_mean.jdata, torch.mean(jt.jdata, dim=-1)))

    @parameterized.expand(all_device_dtype_combos)
    def test_amin_amax_argmin_argmax(self, device, dtype):
        jt, _ = self._mk_jt(device, dtype)
        amax = fvdb.amax(jt, dim=-1)
        amin = fvdb.amin(jt, dim=-1)
        argmax = fvdb.argmax(jt, dim=-1)
        argmin = fvdb.argmin(jt, dim=-1)
        for out in (amax, amin, argmax, argmin):
            self._assert_preserved_layout(out, jt)
        self.assertTrue(torch.allclose(amax.jdata, torch.amax(jt.jdata, dim=-1)))
        self.assertTrue(torch.allclose(amin.jdata, torch.amin(jt.jdata, dim=-1)))
        self.assertTrue(torch.equal(argmax.jdata, torch.argmax(jt.jdata, dim=-1)))
        self.assertTrue(torch.equal(argmin.jdata, torch.argmin(jt.jdata, dim=-1)))

    @parameterized.expand(all_device_dtype_combos)
    def test_all_any(self, device, dtype):
        jt, _ = self._mk_jt(device, dtype)
        # make boolean data
        tensors = cast(list[torch.Tensor], jt.unbind())  # resolving unbind ambiguity
        cond = fvdb.JaggedTensor([torch.rand_like(t, dtype=torch.float32) > 0.5 for t in tensors])
        out_all = fvdb.all(cond, dim=-1)
        out_any = fvdb.any(cond, dim=-1)
        self._assert_preserved_layout(out_all, cond)
        self._assert_preserved_layout(out_any, cond)
        self.assertTrue(torch.equal(out_all.jdata, torch.all(cond.jdata, dim=-1)))
        self.assertTrue(torch.equal(out_any.jdata, torch.any(cond.jdata, dim=-1)))

    @parameterized.expand(all_device_dtype_combos)
    def test_var_std_norm(self, device, dtype):
        jt, _ = self._mk_jt(device, dtype)
        out_var = fvdb.var(jt, dim=-1, unbiased=False)
        out_std = fvdb.std(jt, dim=-1, unbiased=False)
        out_norm = fvdb.norm(jt, dim=-1)
        self._assert_preserved_layout(out_var, jt)
        self._assert_preserved_layout(out_std, jt)
        self._assert_preserved_layout(out_norm, jt)
        self.assertTrue(torch.allclose(out_var.jdata, torch.var(jt.jdata, dim=-1, unbiased=False)))
        self.assertTrue(torch.allclose(out_std.jdata, torch.std(jt.jdata, dim=-1, unbiased=False)))
        self.assertTrue(torch.allclose(out_norm.jdata, torch.norm(jt.jdata, dim=-1)))

    # -------------- disallowed cases --------------

    @parameterized.expand(all_device_dtype_combos)
    def test_sum_over_all_dims_disallowed(self, device, dtype):
        jt, _ = self._mk_jt(device, dtype)
        with self.assertRaises(RuntimeError):
            _ = fvdb.sum(jt)  # would collapse primary dim -> should raise
