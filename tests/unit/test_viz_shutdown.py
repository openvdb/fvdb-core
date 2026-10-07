# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

import base64
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import time

import pytest


def _create_point_cloud(port):
    import torch

    import fvdb
    from fvdb.viz._viewer_server import _get_viewer_server_cpp

    fvdb.viz.init(ip_address="127.0.0.1", port=port)
    viewer = _get_viewer_server_cpp()
    viewer.add_scene("shutdown")
    viewer.add_gaussian_splat_3d_view(
        scene_name="shutdown",
        name="point",
        means=torch.zeros((1, 3)),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        log_scales=torch.full((1, 3), -20.0),
        logit_opacities=torch.full((1,), 10.0),
        sh0=torch.zeros((1, 3)),
        shN=torch.empty((1, 0, 3)),
    )


def _wait_for_frames(port):
    deadline = time.monotonic() + 60
    while True:
        try:
            stream = socket.create_connection(("127.0.0.1", port), timeout=5)
            break
        except OSError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.1)

    with stream, stream.makefile("rb") as response:
        key = base64.b64encode(os.urandom(16)).decode()
        stream.sendall(
            (
                f"GET /ws HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nUpgrade: websocket\r\n"
                f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n"
            ).encode()
        )
        assert response.readline().startswith(b"HTTP/1.1 101"), "WebSocket upgrade failed"
        while True:
            line = response.readline()
            assert line, "Viewer closed the WebSocket during handshake"
            if line == b"\r\n":
                break
            assert time.monotonic() < deadline, "WebSocket handshake timed out"

        # The next render iteration processes the queued GPU upload before it sends a new frame.
        frame_ids = set()
        while len(frame_ids) < 2:
            remaining = deadline - time.monotonic()
            assert remaining > 0, "Viewer did not send two frames before shutdown"
            stream.settimeout(remaining)
            header = response.read(2)
            assert len(header) == 2, "Viewer closed the WebSocket before shutdown"
            opcode, size = header
            size &= 0x7F
            if size == 126:
                size = struct.unpack(">H", response.read(2))[0]
            elif size == 127:
                size = struct.unpack(">Q", response.read(8))[0]
            payload = response.read(size)
            assert len(payload) == size, "Viewer sent an incomplete WebSocket frame"
            if opcode & 0x0F == 1:
                metadata = json.loads(payload)
                if "frameid" in metadata:
                    frame_ids.add(metadata["frameid"])


def _run_shutdown(shutdown):
    from fvdb.viz import _viewer_server

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    _create_point_cloud(port)
    _wait_for_frames(port)
    print("Viewer ready for shutdown", flush=True)
    if shutdown == "explicit":
        _viewer_server._viewer_server_cpp = None


@pytest.mark.parametrize("shutdown", ["explicit", "interpreter"])
def test_viewer_shutdown_with_gpu_buffers(shutdown):
    pytest.importorskip("nanovdb_editor")
    result = subprocess.run(
        [sys.executable, "-u", "-X", "faulthandler", str(Path(__file__).resolve()), shutdown],
        capture_output=True,
        text=True,
        timeout=150,
    )
    assert "Viewer ready for shutdown" in result.stdout, result.stdout + result.stderr
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    _run_shutdown(sys.argv[1])
