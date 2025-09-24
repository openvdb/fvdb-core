#!/usr/bin/env python3
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

import io
import sys
import time
import unittest

import pytest
import torch
from fvdb.utils.tests import ScopedTimer


class TestScopedTimer(unittest.TestCase):
    def test_split_without_context_raises(self):
        t = ScopedTimer()
        with pytest.raises(RuntimeError):
            _ = t.split()

    def test_timer_elapsed_time_basic(self):
        with ScopedTimer() as timer:
            time.sleep(0.01)
        assert timer.elapsed_time is not None
        assert timer.elapsed_time > 0.0

    def test_timer_split_positive(self):
        with ScopedTimer() as timer:
            time.sleep(0.002)
            s1 = timer.split()
            time.sleep(0.002)
            s2 = timer.split()
        assert s1 > 0.0 and s2 > 0.0

    def test_timer_prints_message_on_exit_cpu(self):
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            with ScopedTimer(message="CPU scope"):
                time.sleep(0.001)
        finally:
            sys.stdout = old_stdout

        out = buf.getvalue()
        assert "CPU scope:" in out
        # Ensure we printed a floating seconds value
        assert "seconds" in out

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_timer_cuda_timing(self):
        device = torch.device("cuda")
        a = torch.randn(512, 512, device=device)
        b = torch.randn(512, 512, device=device)
        with ScopedTimer(cuda=True) as timer:
            _ = a @ b
        assert timer.elapsed_time is not None and timer.elapsed_time > 0.0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_timer_prints_message_on_exit_cuda(self):
        device = torch.device("cuda")
        a = torch.randn(256, 256, device=device)
        b = torch.randn(256, 256, device=device)

        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            with ScopedTimer(message="GPU scope", cuda=True):
                _ = a @ b
        finally:
            sys.stdout = old_stdout

        out = buf.getvalue()
        assert "GPU scope:" in out
        assert "seconds" in out
