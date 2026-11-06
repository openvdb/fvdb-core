# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
from typing import TYPE_CHECKING, Any

import torch

from ._nanovdb_grid_view import _sub_view_name, remove_stale_views
from ._viewer_server import _get_viewer_server_cpp

if TYPE_CHECKING:
    from .. import JaggedTensor


def _split_meshes(
    vertices: "torch.Tensor | JaggedTensor",
    faces: "torch.Tensor | JaggedTensor",
    colors: "torch.Tensor | JaggedTensor | None",
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]]:
    """Split single or batched mesh inputs into one ``(vertices, faces, colors)`` tuple per mesh."""
    from .. import JaggedTensor

    if isinstance(vertices, JaggedTensor) or isinstance(faces, JaggedTensor):
        if not isinstance(vertices, JaggedTensor) or not isinstance(faces, JaggedTensor):
            raise TypeError("vertices and faces must both be torch.Tensor or both be fvdb.JaggedTensor")
        if colors is not None and not isinstance(colors, JaggedTensor):
            raise TypeError("colors must be a fvdb.JaggedTensor when vertices and faces are")
        if vertices.ldim != 1 or faces.ldim != 1 or (colors is not None and colors.ldim != 1):
            raise ValueError("batched mesh inputs must be JaggedTensors with ldim == 1")
        count = vertices.num_tensors
        if faces.num_tensors != count or (colors is not None and colors.num_tensors != count):
            raise ValueError(
                f"vertices, faces and colors must hold the same number of meshes, got {count} vertex lists "
                f"and {faces.num_tensors} face lists"
            )
        return [
            (vertices[i].jdata, faces[i].jdata, colors[i].jdata if colors is not None else None) for i in range(count)
        ]

    if not isinstance(vertices, torch.Tensor) or not isinstance(faces, torch.Tensor):
        raise TypeError(
            f"vertices and faces must be torch.Tensor or fvdb.JaggedTensor, got "
            f"{type(vertices).__name__} and {type(faces).__name__}"
        )
    if colors is not None and not isinstance(colors, torch.Tensor):
        raise TypeError(f"colors must be a torch.Tensor, got {type(colors).__name__}")
    return [(vertices, faces, colors)]


def _validate_mesh(vertices: torch.Tensor, faces: torch.Tensor, colors: torch.Tensor | None) -> torch.Tensor | None:
    """Check one mesh and return its colors as float32 in ``[0, 1]``."""
    if not vertices.is_floating_point():
        raise TypeError(f"vertices must have a floating point dtype, got {vertices.dtype}")
    if vertices.dim() != 2 or vertices.shape[1] != 3 or vertices.shape[0] == 0:
        raise ValueError(f"vertices must be a non-empty (V, 3) tensor, got shape {tuple(vertices.shape)}")
    if faces.is_floating_point() or faces.is_complex() or faces.dtype == torch.bool:
        raise TypeError(f"faces must have an integer dtype, got {faces.dtype}")
    if faces.dim() != 2 or faces.shape[1] != 3 or faces.shape[0] == 0:
        raise ValueError(f"faces must be a non-empty (F, 3) tensor, got shape {tuple(faces.shape)}")

    num_vertices = vertices.shape[0]
    if int(faces.min()) < 0 or int(faces.max()) >= num_vertices:
        raise ValueError(f"faces must index vertices in [0, {num_vertices})")

    if colors is None:
        return None
    if colors.shape != (num_vertices, 3):
        raise ValueError(f"colors must have shape ({num_vertices}, 3), got {tuple(colors.shape)}")
    if colors.dtype == torch.uint8:
        return colors.to(torch.float32) / 255.0
    if not colors.is_floating_point():
        raise TypeError(f"colors must be floating point in [0, 1] or uint8 in [0, 255], got {colors.dtype}")
    return colors.to(torch.float32)


def add_mesh_views(
    scene_name: str,
    name: str,
    vertices: "torch.Tensor | JaggedTensor",
    faces: "torch.Tensor | JaggedTensor",
    colors: "torch.Tensor | JaggedTensor | None",
    resolution: int,
    inflation_radius: float,
) -> list[str]:
    """Add one editor view per mesh and return the view names.

    A single mesh uses ``name``. A batch of meshes uses ``name[i]`` for mesh ``i``.
    """
    if not 1 <= resolution <= 4096:
        raise ValueError(f"resolution must be in [1, 4096], got {resolution}")
    if not inflation_radius >= 0.0:
        raise ValueError(f"inflation_radius must be non-negative, got {inflation_radius}")

    meshes = _split_meshes(vertices, faces, colors)
    checked = [(v, f, _validate_mesh(v, f, c)) for v, f, c in meshes]

    server = _get_viewer_server_cpp()
    view_names: list[str] = []
    for i, (v, f, c) in enumerate(checked):
        sub_name = _sub_view_name(name, i, len(checked))
        server.add_mesh_view(scene_name, sub_name, v, f, c, resolution, inflation_radius)
        view_names.append(sub_name)
    return view_names


class MeshView:
    """
    A view for rendering a triangle mesh in the viewer.

    The nanovdb-editor builds a VoxelBVH over the triangles in its render loop and ray-traces
    them, so the mesh appears shortly after it is added. Per-vertex colors are interpolated
    across each triangle, and shading uses face normals.

    A batch of meshes given as :class:`~fvdb.JaggedTensor` inputs is expanded into one view per
    mesh, named ``name[i]``.
    """

    __PRIVATE__ = object()

    def __init__(
        self,
        scene_name: str,
        name: str,
        view_names: list[str],
        resolution: int,
        inflation_radius: float,
        _private: Any = None,
    ):
        """
        .. warning::

            This constructor is private.  Use :meth:`fvdb.viz.Scene.add_surface_mesh` instead.
        """
        if _private is not self.__PRIVATE__:
            raise ValueError("MeshView constructor is private. Use Scene.add_surface_mesh().")
        self._scene_name = scene_name
        self._name = name
        # The editor view names backing this mesh view (one per mesh in a batch).
        self._view_names = view_names
        self._resolution = resolution
        self._inflation_radius = inflation_radius

    @property
    def name(self) -> str:
        """The logical view name shared by all meshes in this view."""
        return self._name

    @property
    def scene_name(self) -> str:
        """The name of the scene containing this mesh."""
        return self._scene_name

    @torch.no_grad()
    def update(
        self,
        vertices: "torch.Tensor | JaggedTensor",
        faces: "torch.Tensor | JaggedTensor",
        colors: "torch.Tensor | JaggedTensor | None" = None,
    ) -> None:
        """
        Replace the mesh data in the viewer.

        Views left over from a larger previous batch are removed.

        Args:
            vertices: ``(V, 3)`` vertex positions, or a :class:`~fvdb.JaggedTensor` of them.
            faces: ``(F, 3)`` integer vertex indices per triangle, or a :class:`~fvdb.JaggedTensor` of them.
            colors: Optional ``(V, 3)`` per-vertex RGB, floating point in ``[0, 1]`` or ``uint8``.
        """
        view_names = add_mesh_views(
            self._scene_name, self._name, vertices, faces, colors, self._resolution, self._inflation_radius
        )
        remove_stale_views(self._scene_name, self._view_names, view_names)
        self._view_names = view_names
