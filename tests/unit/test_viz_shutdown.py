# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

import base64
import gc
import io
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import time
from unittest.mock import Mock
import weakref

import pytest
import torch


def _create_point_cloud(port):
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


def _read_exact(response, size):
    data = bytearray()
    while len(data) < size:
        part = response.read(size - len(data))
        assert part, f"Viewer closed the WebSocket with {size - len(data)} bytes missing"
        data.extend(part)
    return bytes(data)


def _read_text_message(stream, response, deadline):
    message = bytearray()
    message_opcode = None
    while True:
        remaining = deadline - time.monotonic()
        assert remaining > 0, "Viewer did not send frame metadata before shutdown"
        stream.settimeout(remaining)
        first, second = _read_exact(response, 2)
        final, opcode = bool(first & 0x80), first & 0x0F
        assert not second & 0x80, "Viewer sent a masked WebSocket server frame"
        size = second & 0x7F
        if size == 126:
            size = struct.unpack(">H", _read_exact(response, 2))[0]
        elif size == 127:
            size = struct.unpack(">Q", _read_exact(response, 8))[0]
        payload = _read_exact(response, size)
        if opcode >= 8:
            assert final and size <= 125, "Viewer sent an invalid WebSocket control frame"
            assert opcode != 8, "Viewer closed the WebSocket before shutdown"
            assert opcode in (9, 10), f"Unexpected WebSocket control opcode: {opcode}"
            if opcode == 9:
                mask = os.urandom(4)
                pong = bytes(value ^ mask[i % 4] for i, value in enumerate(payload))
                stream.sendall(bytes([0x8A, 0x80 | size]) + mask + pong)
            continue
        if opcode in (1, 2):
            assert message_opcode is None, "Viewer started a message before completing the previous one"
            message_opcode = opcode
        else:
            assert opcode == 0 and message_opcode is not None, "Unexpected WebSocket continuation frame"
        if message_opcode == 1:
            message.extend(payload)
        if final:
            if message_opcode == 1:
                return bytes(message)
            message_opcode = None


def _wait_for_frames(requested_port):
    from fvdb.viz._viewer_server import _get_viewer_server_cpp

    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        port = _get_viewer_server_cpp().port()
        assert port > 0, f"Viewer failed to bind a valid port: {port}"
        if port != requested_port:
            try:
                stream = socket.create_connection(("127.0.0.1", port), timeout=5)
                break
            except OSError:
                pass
        time.sleep(0.1)
    else:
        raise AssertionError("Viewer did not bind and report its fallback port")

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
            metadata = json.loads(_read_text_message(stream, response, deadline))
            if "frameid" in metadata:
                frame_ids.add(metadata["frameid"])


def _run_shutdown(shutdown):
    from fvdb.viz import _viewer_server

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        probe.listen()
        requested_port = probe.getsockname()[1]
        _create_point_cloud(requested_port)
        _wait_for_frames(requested_port)
    print("Viewer ready for shutdown", flush=True)
    if shutdown == "explicit":
        viewer = _viewer_server._viewer_server_cpp
        reference = weakref.ref(viewer)
        _viewer_server._viewer_server_cpp = None
        del viewer
        gc.collect()
        assert reference() is None, "Viewer still has live references after explicit shutdown"


class _ShortReader(io.BytesIO):
    def read(self, size):
        return super().read(min(size, 1))


def test_read_fragmented_websocket_metadata():
    wire = b'\x82\x03abc\x01\x06{"fram\x8a\x00\x80\x07eid":7}'
    assert _read_text_message(Mock(), _ShortReader(wire), time.monotonic() + 5) == b'{"frameid":7}'


def test_reply_to_websocket_ping():
    stream = Mock()
    wire = b'\x89\x02hi\x81\x0d{"frameid":7}'
    assert _read_text_message(stream, _ShortReader(wire), time.monotonic() + 5) == b'{"frameid":7}'
    pong = stream.sendall.call_args.args[0]
    assert pong[:2] == b"\x8a\x82"
    assert bytes(value ^ pong[2 + i % 4] for i, value in enumerate(pong[6:])) == b"hi"


@pytest.mark.parametrize(
    ("wire", "error"),
    [(b"\x81\x80\x00\x00\x00\x00", "masked"), (b'\x81\x0d{"frame', "bytes missing")],
)
def test_reject_invalid_websocket_metadata(wire, error):
    with pytest.raises(AssertionError, match=error):
        _read_text_message(Mock(), _ShortReader(wire), time.monotonic() + 5)


@pytest.mark.parametrize("port", [-1, 0])
def test_reject_invalid_viewer_port(monkeypatch, port):
    from fvdb.viz import _viewer_server

    viewer = Mock()
    viewer.port.return_value = port
    monkeypatch.setattr(_viewer_server, "_get_viewer_server_cpp", lambda: viewer)
    with pytest.raises(AssertionError, match="Viewer failed to bind a valid port"):
        _wait_for_frames(8080)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires an NVIDIA GPU for viewer streaming")
@pytest.mark.parametrize("shutdown", ["explicit", "interpreter"])
def test_viewer_shutdown_with_gpu_buffers(shutdown):
    pytest.importorskip("nanovdb_editor")
    try:
        result = subprocess.run(
            [sys.executable, "-u", "-X", "faulthandler", str(Path(__file__).resolve()), shutdown],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired as error:
        output = (error.stdout or b"").decode(errors="replace")
        errors = (error.stderr or b"").decode(errors="replace")
        pytest.fail(f"Viewer shutdown timed out after {error.timeout}s\n{output}{errors}", pytrace=False)
    assert "Viewer ready for shutdown" in result.stdout, result.stdout + result.stderr
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    _run_shutdown(sys.argv[1])
