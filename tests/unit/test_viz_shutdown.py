# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

import base64
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import threading
import time
from urllib.request import urlopen
import zlib

import numpy as np
import pytest


def _screenshot_pixels(png):
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    width, height, bits, color, compression, filtering, interlace = struct.unpack_from(">IIBBBBB", png, 16)
    assert (bits, color, compression, filtering, interlace) == (8, 6, 0, 0, 0)
    offset = 8
    data = []
    while offset < len(png):
        size, kind = struct.unpack_from(">I4s", png, offset)
        if kind == b"IDAT":
            data.append(png[offset + 8 : offset + 8 + size])
        offset += size + 12
    rows = np.frombuffer(zlib.decompress(b"".join(data)), dtype=np.uint8).reshape(height, 1 + width * 4)
    # The editor screenshot endpoint emits unfiltered RGBA8 scanlines.
    assert np.all(rows[:, 0] == 0)
    return rows[:, 1:].reshape(height, width, 4)[..., :3].astype(np.int16)


def _render_point_cloud():
    import torch

    from fvdb.viz._viewer_server import ViewerCpp

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    viewer = ViewerCpp(ip_address="127.0.0.1", port=port, device_id=0, verbose=False)
    scene = "shutdown"
    viewer.add_scene(scene)
    axis = np.linspace(-0.6, 0.6, 23)
    x, y = np.meshgrid(axis, axis)
    keep = x * x + y * y < 0.36
    disc = np.stack([x[keep], y[keep], np.zeros(keep.sum())], axis=1)
    points = torch.tensor(np.concatenate([disc + [-1.5, 0, 0], disc, disc + [1.5, 0, 0]]), dtype=torch.float32)
    colors = torch.eye(3).repeat_interleave(len(disc), dim=0)
    quats = torch.zeros((len(points), 4))
    quats[:, 0] = 1.0
    cloud = viewer.add_gaussian_splat_3d_view(
        scene_name=scene,
        name="RGB points",
        means=points,
        quats=quats,
        log_scales=torch.full_like(points, -20.0),
        logit_opacities=torch.full((len(points),), 10.0),
        sh0=(colors - 0.5) / 0.28209479177387814,
        shN=torch.empty((len(points), 0, 3)),
    )
    cloud.eps_2d = 4.0
    cloud.tile_size = 16
    cloud.sh_degree_to_use = 0
    viewer.set_camera_orbit_center(scene, 0, 0, 0)
    viewer.set_camera_view_direction(scene, 0, 0, 1)
    viewer.set_camera_orbit_radius(scene, 7.0)
    viewer.set_camera_up_direction(scene, 0, 1, 0)

    deadline = time.monotonic() + 60
    while True:
        try:
            stream = socket.create_connection(("127.0.0.1", port), timeout=5)
            break
        except OSError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.1)
    stop = threading.Event()
    thread = None
    try:
        key = base64.b64encode(os.urandom(16)).decode()
        stream.sendall(
            (
                f"GET /ws HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nUpgrade: websocket\r\n"
                f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n"
            ).encode()
        )
        header = b""
        while b"\r\n\r\n" not in header:
            part = stream.recv(4096)
            assert part, "Viewer closed the WebSocket during handshake"
            header += part
        assert header.startswith(b"HTTP/1.1 101"), header
        stream.settimeout(0.5)

        def drain():
            while not stop.is_set():
                try:
                    if not stream.recv(65536):
                        return
                except socket.timeout:
                    continue
                except OSError:
                    return

        thread = threading.Thread(target=drain)
        thread.start()
        # A rendered point cloud proves that the asynchronous GPU upload has completed.
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            with urlopen(f"http://127.0.0.1:{port}/screenshot.png", timeout=10) as response:
                png = response.read()
            if png:
                rgb = _screenshot_pixels(png)
                counts = [
                    np.count_nonzero((rgb[..., c] > 150) & (rgb[..., c] > np.delete(rgb, c, axis=2).max(axis=2) + 65))
                    for c in range(3)
                ]
                if min(counts) > 150:
                    print("Point cloud rendered", flush=True)
                    return viewer
            time.sleep(0.1)
        raise AssertionError("Point cloud did not render before shutdown")
    finally:
        stop.set()
        stream.close()
        if thread is not None:
            thread.join(5)
            assert not thread.is_alive(), "WebSocket reader did not stop"


@pytest.mark.parametrize("shutdown", ["explicit", "interpreter"])
def test_viewer_shutdown_after_rendering(shutdown):
    pytest.importorskip("nanovdb_editor")
    result = subprocess.run(
        [sys.executable, "-u", "-X", "faulthandler", str(Path(__file__).resolve()), shutdown],
        capture_output=True,
        text=True,
        timeout=150,
    )
    assert "Point cloud rendered" in result.stdout, result.stdout + result.stderr
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    viewer = _render_point_cloud()
    if sys.argv[1] == "explicit":
        del viewer
