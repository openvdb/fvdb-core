# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import hashlib
import importlib
import json
import logging
import timeit
from pathlib import Path
from types import ModuleType
from typing import List, Tuple, Union

import numpy as np
import torch
from fvdb.types import NumericMaxRank2
from fvdb.utils._data_repo import fetch_data_repo

from fvdb import GridBatch, JaggedTensor

_EXAMPLE_DATA_REPO = "voxel-foundation/fvdb-example-data"
_EXAMPLE_DATA_REVISION = "42ea11a3210677f7c010f93a2febf9760faa1641"


def _import_optional(module_name: str) -> ModuleType:
    """Import one of the optional example dependencies, or explain how to install it.

    These are imported lazily so that importing fvdb, or this module, does not require
    dependencies that only the example helpers need.
    """
    try:
        return importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(
            f"'{module_name}' is required for this fvdb example helper but is not installed. "
            f"Install the example dependencies with: pip install 'fvdb-core[examples]'"
        ) from e


def get_fvdb_example_data_path() -> Path:
    """Get the path to the downloaded fvdb-example-data snapshot."""
    return fetch_data_repo(_EXAMPLE_DATA_REPO, _EXAMPLE_DATA_REVISION, "fvdb_example_data")


def _get_md5_checksum(file_path: Path):
    md5_hash = hashlib.md5(open(file_path, "rb").read())
    return md5_hash.hexdigest()


def make_grid_batch_from_points(
    points: JaggedTensor, padding: int, voxel_sizes: NumericMaxRank2, origins: NumericMaxRank2
) -> GridBatch:
    logging.info("Building GridBatch from points...")
    start = timeit.default_timer()
    grid_batch = GridBatch.from_points(points, voxel_sizes=voxel_sizes, origins=origins)
    grid_batch = grid_batch.dilated_grid(padding)
    torch.cuda.synchronize()
    logging.info(f"Done in {timeit.default_timer() - start}s")
    logging.info(f"GridBatch has {grid_batch.total_voxels} voxels")

    return grid_batch


def make_ray_grid(
    nrays: int,
    origin: Union[torch.Tensor, Tuple, List],
    minb=(-0.3, -0.3),
    maxb=(0.3, 0.3),
    device: Union[str, torch.device] = "cpu",
    dtype=torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    ray_o = torch.tensor([origin] * nrays**2)

    ray_d = torch.from_numpy(
        np.stack(
            [a.ravel() for a in np.mgrid[minb[0] : maxb[0] : nrays * 1j, minb[1] : maxb[1] : nrays * 1j]]
            + [np.ones(nrays**2)],
            axis=-1,
        ).astype(np.float32)
    )
    ray_d /= torch.norm(ray_d, dim=-1, keepdim=True)

    ray_o, ray_d = ray_o.to(device).to(dtype), ray_d.to(device).to(dtype)

    return ray_o, ray_d


def load_pointcloud(
    data_path,
    skip_every=1,
    shuffle=False,
    device=torch.device("cuda"),
    dtype=torch.float32,
) -> torch.Tensor:
    pcu = _import_optional("point_cloud_utils")

    logging.info(f"Loading pointlcoud {data_path}...")
    start = timeit.default_timer()
    pts = pcu.load_mesh_v(data_path)
    if shuffle:
        pts = pts[np.random.permutation(pts.shape[0])]
    pts = pts[::skip_every]
    logging.info(f"Done in {timeit.default_timer() - start}s")
    return torch.from_numpy(pts).to(device).to(dtype)


def load_mesh(
    data_path, expected_md5, skip_every=1, mode="vn", device=torch.device("cuda"), dtype=torch.float32
) -> List[torch.Tensor]:
    pcu = _import_optional("point_cloud_utils")

    if _get_md5_checksum(data_path) != expected_md5:
        raise ValueError(f"Checksum for {data_path} is incorrect, expected {expected_md5}")
    logging.info(f"Loading mesh {data_path}...")
    start = timeit.default_timer()
    if mode == "v":
        attrs = [pcu.load_mesh_v(data_path)]
    elif mode == "vf":
        attrs = pcu.load_mesh_vf(data_path)
    elif mode == "vn":
        attrs = pcu.load_mesh_vn(data_path)
    else:
        raise ValueError(f"Unsupported mode {mode}")
    for a in attrs:
        if a is None:
            raise ValueError(f"Failed to load mesh {data_path}, missing attributes")
    if mode == "vf":
        attrs = [
            torch.from_numpy(attrs[0][::skip_every]).to(device).to(dtype),
            torch.from_numpy(attrs[1][::skip_every]).to(device),
        ]
    else:
        attrs = [torch.from_numpy(a[::skip_every]).to(device).to(dtype) for a in attrs]
    logging.info(f"Done in {timeit.default_timer() - start}s")

    return attrs


def load_dragon_mesh(skip_every=1, mode="vn", device=torch.device("cuda"), dtype=torch.float32) -> List[torch.Tensor]:
    data_path = get_fvdb_example_data_path() / "meshes" / "dragon.ply"
    return load_mesh(
        data_path,
        expected_md5="0222e7d2147eebcb2eacdaf6263a9512",
        mode=mode,
        skip_every=skip_every,
        device=device,
        dtype=dtype,
    )


def load_happy_mesh(skip_every=1, mode="vn", device=torch.device("cuda"), dtype=torch.float32) -> List[torch.Tensor]:
    data_path = get_fvdb_example_data_path() / "meshes" / "happy.ply"
    return load_mesh(
        data_path,
        expected_md5="5cfe3c9c0b58bad9a77b47ae04454160",
        mode=mode,
        skip_every=skip_every,
        device=device,
        dtype=dtype,
    )


def load_bunny_mesh(skip_every=1, mode="vn", device=torch.device("cuda"), dtype=torch.float32) -> List[torch.Tensor]:
    data_path = get_fvdb_example_data_path() / "meshes" / "bunny.ply"
    return load_mesh(
        data_path,
        expected_md5="fe2f062a8e22b7dab895a1945c32cd58",
        mode=mode,
        skip_every=skip_every,
        device=device,
        dtype=dtype,
    )


def load_car_1_mesh(skip_every=1, mode="vf", device=torch.device("cuda"), dtype=torch.float32) -> List[torch.Tensor]:
    data_path = get_fvdb_example_data_path() / "meshes" / "car-mesh-1.ply"
    return load_mesh(
        data_path,
        expected_md5="969f91abdf00bad792ca2af347c58499",
        mode=mode,
        skip_every=skip_every,
        device=device,
        dtype=dtype,
    )


def load_car_2_mesh(skip_every=1, mode="vf", device=torch.device("cuda"), dtype=torch.float32) -> List[torch.Tensor]:
    data_path = get_fvdb_example_data_path() / "meshes" / "car-mesh-2.ply"
    return load_mesh(
        data_path,
        expected_md5="d4aa0dd4f4609ea1b19aca7d8618d22a",
        mode=mode,
        skip_every=skip_every,
        device=device,
        dtype=dtype,
    )


def load_car_3_mesh(skip_every=1, mode="vf", device=torch.device("cuda"), dtype=torch.float32) -> List[torch.Tensor]:
    data_path = get_fvdb_example_data_path() / "meshes" / "car-mesh-3.ply"
    return load_mesh(
        data_path,
        expected_md5="a058d534da71748167799db0351f21f4",
        mode=mode,
        skip_every=skip_every,
        device=device,
        dtype=dtype,
    )


def load_car_4_mesh(skip_every=1, mode="vf", device=torch.device("cuda"), dtype=torch.float32) -> List[torch.Tensor]:
    data_path = get_fvdb_example_data_path() / "meshes" / "car-mesh-4.ply"
    return load_mesh(
        data_path,
        expected_md5="6238478fcf1f963e38a95b52a1521b5d",
        mode=mode,
        skip_every=skip_every,
        device=device,
        dtype=dtype,
    )


def load_gso_shoes(
    limit: Union[int, None] = None, device=torch.device("cuda"), dtype=torch.float32
) -> List[List[torch.Tensor]]:
    """Load the Google Scanned Objects "Shoe" meshes (254 scans, CC-BY 4.0).

    The subset lives in ``meshes/gso_shoes`` of the example-data snapshot; per-model
    attribution and source URLs are recorded in its ``ATTRIBUTION.json`` manifest
    (individual files are integrity-protected by the pinned data-repo revision, so
    no per-file checksums are kept here).

    Args:
        limit: Load only the first ``limit`` meshes in manifest order, or ``None`` for all 254.
        device: Device for the returned tensors.
        dtype: Floating dtype for the vertex tensors.

    Returns:
        meshes: A list of ``[vertices, faces]`` tensor pairs, one per shoe.
    """
    pcu = _import_optional("point_cloud_utils")

    shoes_dir = get_fvdb_example_data_path() / "meshes" / "gso_shoes"
    with open(shoes_dir / "ATTRIBUTION.json") as fp:
        entries = json.load(fp)["models"]
    if limit is not None:
        entries = entries[:limit]
    logging.info(f"Loading {len(entries)} GSO shoe meshes...")
    start = timeit.default_timer()
    meshes = []
    for entry in entries:
        v, f = pcu.load_mesh_vf(str(shoes_dir / entry["file"]))
        meshes.append([torch.from_numpy(v).to(device=device, dtype=dtype), torch.from_numpy(f).to(device)])
    logging.info(f"Done in {timeit.default_timer() - start}s")
    return meshes


def plot_ray_segments(ray_o, ray_d, times, plot_every=1):
    ps = _import_optional("polyscope")

    for i in range(0, ray_o.shape[0], plot_every):
        t0s = times[i].jdata[:, 0].unsqueeze(-1)
        t1s = times[i].jdata[:, 1].unsqueeze(-1)
        roi = ray_o[i].unsqueeze(0)
        rdi = ray_d[i].unsqueeze(0)
        rp = torch.cat([roi + t0s * rdi, roi + t1s * rdi])
        re = torch.stack(
            [torch.arange(t0s.shape[0]), torch.arange(t0s.shape[0]) + t0s.shape[0]],
            dim=-1,
        )

        ray_segs = ps.register_curve_network(f"ray segments {i}", rp, re, radius=0.001)
        rv = torch.zeros(re.shape[0])
        rv[::2] = 1.0
        ray_segs.add_scalar_quantity(f"segment colors {i}", rv, defined_on="edges", enabled=True, cmap="jet")


__all__ = [
    "get_fvdb_example_data_path",
    "make_ray_grid",
    "load_pointcloud",
    "load_mesh",
    "load_dragon_mesh",
    "load_happy_mesh",
    "load_bunny_mesh",
    "load_car_1_mesh",
    "load_car_2_mesh",
    "load_gso_shoes",
    "plot_ray_segments",
]
