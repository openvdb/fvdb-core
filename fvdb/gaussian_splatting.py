# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import pathlib
from typing import Any, Mapping, Sequence, overload

import torch
from fvdb.enums import ProjectionType

from . import JaggedTensor as JaggedTensorCpp
from ._fvdb_cpp import GaussianSplat3d as GaussianSplat3dCpp
from ._fvdb_cpp import JaggedTensor
from ._fvdb_cpp import ProjectedGaussianSplats as ProjectedGaussianSplatsCpp
from .grid import Grid
from .grid_batch import GridBatch
from .jagged_tensor import JaggedTensor
from .types import DeviceIdentifier, cast_check, resolve_device


class ProjectedGaussianSplats:
    """
    A class representing a set of Gaussian splats projected onto a batch of 2D image planes.

    A :class:`ProjectedGaussianSplats` instance contains the 2D projections of 3D Gaussian splats, which can be used to render
    images onto the image planes. Instances of this class are created by calling the :meth:`GaussianSplat3d.project_gaussians_for_images`,
    :meth:`GaussianSplat3d.project_gaussians_for_images_and_depths`, etc. methods.

    .. note::

        The reason to have a separate class for projected Gaussian splats is to be able to run projection once, and then render
        the splats multiple times (e.g. rendering crops) without re-projecting them each time. This can save significant computation time.
    """

    __PRIVATE__ = object()

    def __init__(self, impl: ProjectedGaussianSplatsCpp, _private: Any = None) -> None:
        """
        Private constructor. Use :meth:`GaussianSplat3d.project_gaussians_for_images` or similar methods to create instances.

        Args:
            impl (ProjectedGaussianSplatsCpp): The underlying C++ implementation.
            _private (Any): A private object to prevent direct construction. Must be :attr:`ProjectedGaussianSplats.__PRIVATE__`.
        """
        if _private is not self.__PRIVATE__:
            raise ValueError(
                "ProjectedGaussianSplats constructor is private. Use GaussianSplat3d.project_gaussians_for_images or similar methods instead."
            )
        self._impl = impl

    @property
    def antialias(self) -> bool:
        """
        Return whether antialiasing was enabled during the projection of the Gaussian splats.

        Returns:
            antialias (bool): ``True`` if antialiasing was enabled during projection, ``False`` otherwise.
        """
        return self._impl.antialias

    @property
    def inv_covar_2d(self) -> torch.Tensor:
        """
        The inverse of the 2D covariance matrices of the Gaussians projected into each image plane. These define the
        spatial extent of ellipses for each splatted Gaussian. Note that
        since covariance matrices are symmetric, we pack them into a tensor of shape ``(num_projected_gaussians, 3)``
        where each covariance matrix is represented as ``(Cxx, Cxy, Cyy)``.

        Returns:
            inv_covar_2d (torch.Tensor): A tensor of shape ``(C, N, D)`` representing the packed inverse 2D covariance matrices,
                where ``C`` is the number of image planes, ``N`` is the number of projected Gaussians, and ``D`` is number of feature channels for each
                Gaussian (see :attr:`GaussianSplat3d.num_channels`).
        """
        return self._impl.conics

    @property
    def depths(self) -> torch.Tensor:
        """
        Return the depth of each projected Gaussian in each image plane. The depth is defined as the
        distance from the camera to the mean of the Gaussian along the camera's viewing direction.

        Returns:
            depths (torch.Tensor): A tensor of shape ``(C, N)`` representing the depth of each projected Gaussian, where
                ``C`` is the number of image planes, and ``N`` is the number of projected Gaussians.
        """
        return self._impl.depths

    @property
    def eps_2d(self) -> float:
        """
        Return the epsilon value used during the projection of the Gaussian splats to avoid
        numerical issues. This value is used to clamp very small radii during projection.

        Returns:
            eps_2d (float): The epsilon value used during projection.
        """
        return self._impl.eps_2d

    @property
    def far_plane(self) -> float:
        """
        Return the far plane distance used during the projection of the Gaussian splats.

        Returns:
            far_plane (float): The far plane distance.
        """
        return self._impl.far_plane

    @property
    def image_height(self) -> int:
        """
        Return the height of the image planes used during the projection of the Gaussian splats.

        Returns:
            image_height (int): The height of the image planes.
        """
        return self._impl.image_height

    @property
    def image_width(self) -> int:
        """
        Return the width of the image planes used during the projection of the Gaussian splats.

        Returns:
            image_width (int): The width of the image planes.
        """
        return self._impl.image_width

    @property
    def means2d(self) -> torch.Tensor:
        """
        Return the 2D projected means (in pixel units) of the Gaussians in each image plane.

        Returns:
            means2d (torch.Tensor): A tensor of shape ``(C, N, 2)`` representing the 2D projected means,
                where ``C`` is the number of image planes, ``N`` is the number of projected Gaussians,
                and the last dimension contains the (x, y) coordinates of the means in pixel space.
        """
        return self._impl.means2d

    @property
    def min_radius_2d(self) -> float:
        """
        Return the minimum radius (in pixels) used to clip Gaussians during projection. Gaussians
        whose radius projected to less than this value are ignored to avoid numerical issues.

        Returns:
            min_radius_2d (float): The minimum radius used during projection.
        """
        return self._impl.min_radius_2d

    @property
    def near_plane(self) -> float:
        """
        Return the near plane distance used during the projection of the Gaussian splats.

        Returns:
            near_plane (float): The near plane distance.
        """
        return self._impl.near_plane

    @property
    def opacities(self) -> torch.Tensor:
        """
        Return the opacities of each projected Gaussian in each image plane.

        Returns:
            opacities (torch.Tensor): A tensor of shape ``(C, N)`` representing the opacity of each projected Gaussian, where
                ``C`` is the number of image planes, and ``N`` is the number of projected Gaussians.
        """
        return self._impl.opacities

    @property
    def projection_type(self) -> ProjectionType:
        """
        Return the projection type used during the projection of the Gaussian splats.

        Returns:
            projection_type (ProjectionType): The projection type (*e.g.* :attr:`ProjectionType.PERSPECTIVE` or :attr:`ProjectionType.ORTHOGRAPHIC`).
        """
        return GaussianSplat3d._proj_type_from_cpp(self._impl.projection_type)

    @property
    def radii(self) -> torch.Tensor:
        """
        Return the 2D radii (in pixels) of each projected Gaussian in each image plane. The radius of a Gaussian is the maximum extent
        of the Gaussian along any direction in the image plane.

        Returns:
            radii (torch.Tensor): A tensor of shape ``(C, N)`` representing the 2D radius of each projected Gaussian, where
                ``C`` is the number of image planes, and ``N`` is the number of projected Gaussians.
        """
        return self._impl.radii

    @property
    def render_quantities(self) -> torch.Tensor:
        """
        Return the render quantities of each projected Gaussian in each image plane. The render quantities
        are used for shading and lighting calculations during rendering.

        Returns:
            render_quantities (torch.Tensor): A tensor of shape ``(C, N, D)`` representing the render quantities of each projected Gaussian,
                where ``C`` is the number of image planes, ``N`` is the number of projected Gaussians, and ``D`` is the number of feature
                channels for each Gaussian (see :attr:`GaussianSplat3d.num_channels`).
        """
        return self._impl.render_quantities

    @property
    def sh_degree_to_use(self) -> int:
        """
        Return the spherical harmonic degree used during the projection of the Gaussian splats.

        .. note::

            This indicates up to which degree the spherical harmonics coefficients were projected
            for each Gaussian. For example, if this value is ``0``, only the diffuse (degree 0) coefficients
            were projected. If this value is ``2``, coefficients up to degree 2 were projected.

        Returns:
            sh_degree_to_use (int): The spherical harmonic degree used during projection.
        """
        return self._impl.sh_degree_to_use

    @property
    def tile_gaussian_ids(self) -> torch.Tensor:
        """
        Return a tensor containing the ID of each tile/gaussian intersection.

        Returns:
            tile_gaussian_ids (torch.Tensor): A tensor of shape ``(M,)`` containing the IDs of the Gaussians.
        """
        return self._impl.tile_gaussian_ids

    @property
    def tile_offsets(self) -> torch.Tensor:
        """
        Return the starting offset of the set of intersections for each tile into :attr:`tile_gaussian_ids`.

        Returns:
            tile_offsets (torch.Tensor): A tensor of shape ``(C, TH, TW,)`` where ``C`` is the number of image planes,
                ``TH`` is the number of tiles in the height dimension, and ``TW`` is the number of tiles in the width dimension.
        """
        return self._impl.tile_offsets


class GaussianSplat3d:
    """
    An efficient data structure representing a Gaussian splat radiance field in 3D space.

    A :class:`GaussianSplat3d` instance contains a set of 3D Gaussian splats, each defined by its mean position,
    orientation (quaternion), scale, opacity, and spherical harmonics coefficients for color representation.

    Together, these define a radiance field which can be volume rendered to produce images and depths from
    arbitrary viewpoints. This class provides a variety of methods for rendering and manipulating Gaussian splats radiance fields.
    These include:

    - Rendering images with arbitrary channels using spherical harmonics for view-dependent color
      representation (:meth:`render_images`, :meth:`render_images_and_depths`).
    - Rendering depth maps (:meth:`render_depths`, :meth:`render_images_and_depths`).
    - Rendering features at arbitrary sparse pixel locations (:meth:`sparse_render_features`).
    - Rendering depths at arbitrary sparse pixel locations (:meth:`sparse_render_depths`).
    - Computing which gaussians contribute to each pixel in an image plane
      (:meth:`render_num_contributing_gaussians`, :meth:`render_contributing_gaussian_ids`).
    - Computing the set of Gaussians which contribute to a set of sparse pixel locations
      (:meth:`sparse_render_num_contributing_gaussians`, :meth:`sparse_render_contributing_gaussian_ids`).
    - Saving and loading Gaussian splat data to/from PLY files (:meth:`save_to_ply`, :meth:`from_ply`).
    - Slicing, indexing, and masking Gaussians to create new :class:`GaussianSplat3d` instances.
    - Concatenating multiple :class:`GaussianSplat3d` instances into a single instance (:meth:`cat`).

    Background
    -----------

    Mathematically, the radiance field represented by a :class:`GaussianSplat3d` is defined as a sum of anisotropic 3D Gaussians,
    with view-dependent features represented using spherical harmonics. The radiance field :math:`R(x, v)` accepts as
    input a 3D position :math:`x \\in \\mathbb{R}^3` and a viewing direction :math:`v \\in \\mathbb{S}^2`, and is defined as:

    .. math::

        R(x, v) = \\sum_{i=1}^{N} o_i \\cdot \\alpha_i(x) \\cdot SH(v; C_i)

        \\alpha_i(x) = \\exp\\left(-\\frac{1}{2}(x - \\mu_i)^T \\Sigma_i^{-1} (x - \\mu_i)\\right)

        \\Sigma_i = R(q_i)^T \\cdot \\text{diag}(S_i) \\cdot R(q_i)

    where:

    - :math:`N` is the number of Gaussians (see :attr:`num_gaussians`).
    - :math:`\\mu_i \\in \\mathbb{R}^3` is the mean of the i-th Gaussian (see :attr:`means`).
    - :math:`\\Sigma_i \\in \\mathbb{R}^{3 \\times 3}` is the covariance matrix of the i-th Gaussian,
      defined by its scale diagonal scale :math:`S_i \\in \\mathbb{R}^3` (see :attr:`scales`) and orientation
      quaternion :math:`q_i \\in \\mathbb{R}^4` (see :attr:`quats`).
    - :math:`o_i \\in [0, 1]` is the opacity of the i-th Gaussian (see :attr:`opacities`).
    - :math:`SH(v; C_i)` is the spherical harmonics function evaluated at direction :math:`v` with coefficients :math:`C_i`.
    - :math:`R(q_i)` is the rotation matrix corresponding to the quaternion :math:`q_i`.

    To render images from a :class:`GaussianSplat3d`, you volume render the radiance field using

    .. math::

        I(u, v) = \\int_{t \\in r(u, v)} T(t) R(r(t), d) dt

    where :math:`r(u, v)` is the camera ray through pixel :math:`(u, v)`, :math:`d` is the viewing direction of the ray,
    and :math:`T(t) = \\exp\\left(-\\int_{0}^{t} R(r(s), s) ds\\right)` is the accumulated transmittance along the ray up to distance :math:`t`.

    and to render depths you compute

    .. math::

        D(u, v) = \\int_{t \\in r(u, v)} t \\cdot T(t) \\sum_{i=1}^{N} o_i \\cdot \\alpha_i(r(t), d) dt

    """

    PLY_VERSION_STRING = "fvdb_ply 1.0.0"
    """
    Version string written to PLY files saved using the :meth:`save_to_ply` method.
    This string will be written in the comment section of the PLY file to identify
    the version of the fvdb library used to save the file. The comment will have the form
    ``comment fvdb_gs_ply <PLY_VERSION_STRING>``.
    """

    __PRIVATE__ = object()

    def __init__(
        self,
        impl: GaussianSplat3dCpp,
        _private: Any = None,
    ) -> None:
        """
        Initializes the :class:`GaussianSplat3d` with an existing C++ implementation.
        This constructor is used to wrap an existing instance of :class:`GaussianSplat3dCpp`.
        It is only called internally within this class and should not be used directly.

        .. note::

            You should never call this constructor directly. Instead, use the
            :meth:`from_tensors` or :meth:`from_ply` class methods to create new instances of
            :class:`GaussianSplat3d`.

        Args:
            impl (GaussianSplat3dCpp): An instance of the C++ implementation.
        """
        if _private is not self.__PRIVATE__:
            raise ValueError("GaussianSplat3d constructor is private. Use from_tensors or from_ply instead.")
        self._impl = impl

    @classmethod
    def from_tensors(
        cls,
        means: torch.Tensor,
        quats: torch.Tensor,
        log_scales: torch.Tensor,
        logit_opacities: torch.Tensor,
        sh0: torch.Tensor,
        shN: torch.Tensor,
        accumulate_mean_2d_gradients: bool = False,
        accumulate_max_2d_radii: bool = False,
        detach: bool = False,
    ) -> "GaussianSplat3d":
        """
        Create a new :class:`GaussianSplat3d` from the provided tensors. This constructs a new
        Gaussian splat radiance field with the specified means, orientations, scales, opacities, and spherical harmonics coefficients.

        .. note::

            The :class:`GaussianSplat3d` stores the log of scales scales (:attr:`log_scales`) rather than the scales
            directly. This ensures numerical stability, especially when optimizing the scales, since each gaussian
            is defined as :math:`\\exp(R(q)^T S R(q))` where :math:`R(q)` is rotation matrix defined by the unit quaternion of the Gaussian,
            and :math:`S = diag(exp(log_scales))`.


        .. note::

            The :class:`GaussianSplat3d` stores the logit of opacities (:attr:`logit_opacities`) rather than the opacities
            directly. The actual opacities are obtained by applying the sigmoid function to the logit opacities.
            This ensures opacities are always in the range ``[0, 1]`` and improves numerical stability during optimization.

        Args:
            means (torch.Tensor): Tensor of shape ``(N, 3)`` representing the means of the gaussians, where ``N`` is the number of gaussians.
            quats (torch.Tensor): Tensor of shape ``(N, 4)`` representing the quaternions (orientations) of the gaussians, where ``N`` is the number of gaussians.
            log_scales (torch.Tensor): Tensor of shape ``(N, 3)`` representing the log scales of the gaussians, where ``N`` is the number of gaussians.
            logit_opacities (torch.Tensor): Tensor of shape ``(N,)`` representing the logit opacities of the gaussians, where ``N`` is the number of gaussians.
            sh0 (torch.Tensor): Tensor of shape ``(N, 1, D)`` representing the diffuse SH coefficients
                where ``D`` is the number of channels (see :attr:`num_channels`).
            shN (torch.Tensor): Tensor of shape ``(N, K-1, D)`` representing the directionally
                varying SH coefficients where ``D`` is the number of channels (see :attr:`num_channels`),
                and ``K`` is the number of spherical harmonic bases (see :attr:`num_sh_bases`).
            accumulate_mean_2d_gradients (bool, optional): If ``True``, tracks the average norm of the
                gradient of projected means for each Gaussian during the backward pass of projection.
                This is useful for some optimization techniques, such as the one in the `original paper <https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/>`_.
                Defaults to ``False``.
            accumulate_max_2d_radii (bool, optional): If ``True``, tracks the maximum 2D radii for each Gaussian
                during the backward pass of projection. This is useful for some optimization techniques, such as the one in the `original paper <https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/>`_.
                Defaults to ``False``.
            detach (bool, optional): If ``True``, creates copies of the input tensors and detaches them
                from the computation graph. Defaults to ``False``.
        """

        return GaussianSplat3d(
            GaussianSplat3dCpp(
                means=means,
                quats=quats,
                log_scales=log_scales,
                logit_opacities=logit_opacities,
                sh0=sh0,
                shN=shN,
                accumulate_mean_2d_gradients=accumulate_mean_2d_gradients,
                accumulate_max_2d_radii=accumulate_max_2d_radii,
                detach=detach,
            ),
            _private=cls.__PRIVATE__,
        )

    @classmethod
    def from_ply(
        cls, filename: pathlib.Path | str, device: DeviceIdentifier = "cuda"
    ) -> "tuple[GaussianSplat3d, dict[str, str | int | float | torch.Tensor]]":
        """
        Create a `GaussianSplat3d` instance from a PLY file.

        Args:
            filename (str): The name of the file to load the PLY data from.
            device (torch.device): The device to load the data onto. Default is "cuda".

        Returns:
            splats (GaussianSplat3d): An instance of GaussianSplat3d initialized with the data from the PLY file.
            metadata (dict[str, str | int | float | torch.Tensor]): A dictionary of metadata where the keys are strings and the
                values are either strings, ints, floats, or tensors. Can be empty if no metadata is saved in the PLY file.
        """
        device = resolve_device(device)
        if isinstance(filename, pathlib.Path):
            filename = str(filename)

        gs_impl, metadata = GaussianSplat3dCpp.from_ply(filename=filename, device=device)

        return cls(impl=gs_impl, _private=cls.__PRIVATE__), metadata

    @overload
    def __getitem__(self, index: slice) -> "GaussianSplat3d": ...

    @overload
    def __getitem__(self, index: torch.Tensor) -> "GaussianSplat3d": ...

    def __getitem__(self, index: slice | torch.Tensor) -> "GaussianSplat3d":
        """
        Select Gaussians using either an integer index tensor, a boolean mask tensor, or a slice.

        .. note::

            If :attr:`accumulate_mean_2d_gradients` or :attr:`accumulate_max_2d_radii` is enabled on this
            :class:`GaussianSplat3d` instance, the returned :class:`GaussianSplat3d` will also contain
            the corresponding accumulated values.

        Example usage:

        .. code-block:: python

            # Using a slice
            gs_subset = gsplat3d[10:20] # Selects Gaussians from index 10 to 19

            # Using an integer index tensor
            indices = torch.tensor([0, 2, 4, 6])
            gs_subset = gsplat3d[indices] # Selects Gaussians at indices 0, 2, 4, and 6

            # Using a boolean mask tensor

            mask = torch.tensor([True, False, True, False, ...]) # Length must be num_gaussians
            gs_subset = gsplat3d[mask] # Selects Gaussians where mask is True

        Args:
            index (slice | torch.Tensor): A slice object or a 1D tensor containing either integer indices or a boolean mask.

        Returns:
            gaussian_splat_3d (GaussianSplat3d): A new instance of :class:`GaussianSplat3d` containing only the selected Gaussians.

        """
        if isinstance(index, slice):
            return GaussianSplat3d(
                impl=self._impl.slice_select(
                    index.start if index.start is not None else 0,
                    index.stop if index.stop is not None else self.num_gaussians,
                    index.step if index.step is not None else 1,
                ),
                _private=self.__PRIVATE__,
            )
        elif isinstance(index, torch.Tensor):
            if index.dim() != 1:
                raise ValueError("Expected 'index' to be a 1D tensor.")

            if index.dtype == torch.bool:
                if len(index) != self.num_gaussians:
                    raise ValueError(
                        f"Expected 'index_or_mask' to have the same length as the number of Gaussians ({self.num_gaussians}), "
                        f"but got {len(index)}."
                    )
                return GaussianSplat3d(impl=self._impl.mask_select(index), _private=self.__PRIVATE__)
            elif index.dtype == torch.int64 or index.dtype == torch.int32:
                return GaussianSplat3d(impl=self._impl.index_select(index), _private=self.__PRIVATE__)
            else:
                raise ValueError("Expected 'index' to be a boolean or integer (int32 or int64) tensor.")
        else:
            raise TypeError("Expected 'index' to be a slice or a torch.Tensor.")

    @overload
    def __setitem__(self, index: slice, value: "GaussianSplat3d") -> None: ...

    @overload
    def __setitem__(self, index: torch.Tensor, value: "GaussianSplat3d") -> None: ...

    def __setitem__(self, index: torch.Tensor | slice, value: "GaussianSplat3d") -> None:
        """
        Set the values of Gaussians in this :class:`GaussianSplat3d` instance using either an integer index tensor,
        a boolean mask tensor, or a slice.

        .. note::

            If using integer indices with duplicate indices, the Gaussian set from ``value`` at the duplicate indices will
            overwrite in a random order.

        .. note::

            If :attr:`accumulate_mean_2d_gradients` or :attr:`accumulate_max_2d_radii` is enabled on this
            :class:`GaussianSplat3d` instance, the corresponding accumulated values will also be updated
            for the selected Gaussians, based on the values from the ``value`` instance. If ``value`` does not have
            these accumulations enabled, the accumulated values for the selected Gaussians will be reset to zero.

        Example:

        .. code-block:: python

            # Using a slice
            gs_subset: GaussianSplat3d = ...  # Some GaussianSplat3d instance with 10 Gaussians
            gsplat3d[10:20] = gs_subset  # Sets Gaussians from index 10 to 19

            # Using an integer index tensor
            indices = torch.tensor([0, 2, 4, 6])
            gs_subset: GaussianSplat3d = ...  # Some GaussianSplat3d instance with 4 Gaussians
            gsplat3d[indices] = gs_subset  # Sets Gaussians at indices 0, 2, 4, and 6

            # Using a boolean mask tensor
            mask = torch.tensor([True, False, True, False, ...])  # Length must be num_gaussians
            gs_subset: GaussianSplat3d = ...  # Some GaussianSplat3d instance with num unmasked Gaussians
            gsplat3d[mask] = gs_subset  # Sets Gaussians where mask is True

        Args:
            index (torch.Tensor | slice): A slice object or a 1D tensor containing either integer indices or a boolean mask.
            value (GaussianSplat3d): The :class:`GaussianSplat3d` instance containing the new values to set.
                Must have the same number of Gaussians as the selected indices or mask.
        """
        if isinstance(index, slice):
            self._impl.slice_set(
                index.start if index.start is not None else 0,
                index.stop if index.stop is not None else self.num_gaussians,
                index.step if index.step is not None else 1,
                value._impl,
            )
            return
        elif isinstance(index, torch.Tensor):

            if index.dim() != 1:
                raise ValueError("Expected 'index' to be a 1D tensor.")

            if index.dtype == torch.bool:
                if len(index) != self.num_gaussians:
                    raise ValueError(
                        f"Expected 'index' to have the same length as the number of Gaussians ({self.num_gaussians}), "
                        f"but got {len(index)}."
                    )
                self._impl.mask_set(index, value._impl)
            elif index.dtype == torch.int64 or index.dtype == torch.int32:
                self._impl.index_set(index, value._impl)
            else:
                raise ValueError("Expected 'index' to be a boolean or integer (int32 or int64) tensor.")
        else:
            raise TypeError("Expected 'index' to be a slice or a torch.Tensor")

    def detach(self) -> "GaussianSplat3d":
        """
        Return a new :class:`GaussianSplat3d` instance whose tensors are detached from the computation graph.
        This is useful when you want to stop tracking gradients for this instance.

        Returns:
            gaussian_splat (GaussianSplat3d): A new :class:`GaussianSplat3d` instance whose
                tensors are detached.
        """
        return GaussianSplat3d(impl=self._impl.detach(), _private=self.__PRIVATE__)

    def detach_(self) -> None:
        """
        Detaches this :class:`GaussianSplat3d` instance from the computation graph in place.
        This modifies the current instance to stop tracking gradients.

        .. note::

            This method modifies the current instance and does not return a new instance.

        """
        self._impl.detach_in_place()

    @staticmethod
    def cat(
        splats: "Sequence[GaussianSplat3d]",
        accumulate_mean_2d_gradients: bool = False,
        accumulate_max_2d_radii: bool = False,
        detach: bool = False,
    ) -> "GaussianSplat3d":
        """
        Concatenates a sequence of :class:`GaussianSplat3d` instances into a single :class:`GaussianSplat3d` instance.

        The returned :class:`GaussianSplat3d` will contain all the Gaussians from the input instances,
        in the order they were provided.

        .. note::

            All input :class:`GaussianSplat3d` instances must have the same number of channels
            and spherical harmonic degree.

        .. note::

            If ``accumulate_mean_2d_gradients`` is ``True``, the concatenated instance will track the average norm
            of projected mean gradients for each Gaussian during the backward pass of projection. This value
            is copied over from each input instance if they were tracking it, and initialized to zero otherwise.

        .. note::

            If ``accumulate_max_2d_radii`` is ``True``, the concatenated instance will track the maximum 2D radii
            for each Gaussian during the backward pass of projection. This value is copied over from each input
            instance if they were tracking it, and initialized to zero otherwise.


        Args:
            splats (Sequence[GaussianSplat3d]): A sequence of :class:`GaussianSplat3d` instances to concatenate.
            accumulate_mean_2d_gradients (bool): If True, copies over the accumulated mean 2D gradients
                for each :class:`GaussianSplat3d` into the new one, or initializes it to zero if the input
                instance was not tracking it.
                Defaults to ``False``.
            accumulate_max_2d_radii (bool): If ``True``, copies the accumulated maximum 2D radii
                for each :class:`GaussianSplat3d` into the concatenated one, or initializes it to zero if the input
                instance was not tracking it.
                Defaults to ``False``.
            detach (bool): If ``True``, detaches the concatenated :class:`GaussianSplat3d` from the computation graph.
                Defaults to ``False``.

        Returns:
            GaussianSplat3d: A new instance of GaussianSplat3d containing the concatenated Gaussians.
        """
        splat_list = [splat._impl for splat in splats]
        return GaussianSplat3d(
            impl=GaussianSplat3dCpp.cat(splat_list, accumulate_mean_2d_gradients, accumulate_max_2d_radii, detach),
            _private=GaussianSplat3d.__PRIVATE__,
        )

    @classmethod
    def from_state_dict(cls, state_dict: dict[str, torch.Tensor]) -> "GaussianSplat3d":
        """
        Creates a :class:`GaussianSplat3d` instance from a state dictionary generated by :meth:`state_dict`.
        This method is typically used to load a saved state of the :class:`GaussianSplat3d` instance.

        A state dictionary must contains the following keys which are all the required parameters to initialize a :class:`GaussianSplat3d`.
        Here ``N`` denotes the number of Gaussians (see :attr:`num_gaussians`)

        - ``'means'``: Tensor of shape ``(N, 3)`` representing the means of the Gaussians.
        - ``'quats'``: Tensor of shape ``(N, 4)`` representing the quaternions of the Gaussians.
        - ``'log_scales'``: Tensor of shape ``(N, 3)`` representing the log scales of the Gaussians.
        - ``'logit_opacities'``: Tensor of shape ``(N,)`` representing the logit opacities of the Gaussians.
        - ``'sh0'``: Tensor of shape ``(N, 1, D)`` representing the diffuse SH coefficients
          where ``D`` is the number of channels (see :attr:`num_channels`).
        - ``'shN'``: Tensor of shape ``(N, K-1, D)`` representing the directionally varying SH
          coefficients where ``D`` is the number of channels (see :attr:`num_channels`), and ``K``
          is the number of spherical harmonic bases (see :attr:`num_sh_bases`).
        - ``'accumulate_max_2d_radii'``: bool Tensor with a single element indicating
          whether to track the maximum 2D radii for gradients.
        - ``'accumulate_mean_2d_gradients'``: bool Tensor with a single element indicating whether
          to track the average norm of the gradient of projected means for each Gaussian.

        It can also optionally contain the following keys:

        - ``'accumulated_gradient_step_counts'``: Tensor of shape ``(N,)`` representing the
          accumulated gradient step counts for each Gaussian.
        - ``'accumulated_max_2d_radii'``: Tensor of shape ``(N,)`` representing the maximum
          2D projected radius for each Gaussian across every iteration of optimization.
        - ``'accumulated_mean_2d_gradient_norms'``: Tensor of shape ``(N,)`` representing the
          average norm of the gradient of projected means for each Gaussian across every iteration of optimization.

        Args:
            state_dict (dict[str, torch.Tensor]): A dictionary containing the state of the :class:`GaussianSplat3d` instance, usually generated via the :meth:`state_dict` method.

        Returns:
            gaussian_splat (GaussianSplat3d): An instance of :class:`GaussianSplat3d` initialized with the provided state dictionary.
        """
        return cls(impl=GaussianSplat3dCpp.from_state_dict(state_dict), _private=cls.__PRIVATE__)

    @property
    def device(self) -> torch.device:
        """
        Returns the device on which the Tensors managed by this :class:`GaussianSplat3d` instance is stored.

        Returns:
            device (torch.device): The device of this :class:`GaussianSplat3d` instance.
        """
        return self._impl.device

    @property
    def dtype(self) -> torch.dtype:
        """
        Returns the data type of of the tensors managed by this :class:`GaussianSplat3d` instance
        (e.g., ``torch.float32``, ``torch.float64``).

        Returns:
            torch.dtype: The data type of the tensors managed by this :class:`GaussianSplat3d` instance.
        """
        return self._impl.dtype

    @property
    def sh_degree(self) -> int:
        """
        Returns the degree of the spherical harmonics used in the Gaussian splatting representation.
        This value is 0 for diffuse SH coefficients and >= 1 for directionally varying SH coefficients.

        .. note::

            This is **not** the same as the number of spherical harmonics bases (see :attr:`num_sh_bases`).
            The relationship between the degree and the number of bases is given by
            :math:`K = (sh\\_degree + 1)^2`, where :math:`K` is the number of spherical harmonics bases.

        Returns:
            sh_degree (int): The degree of the spherical harmonics.
        """
        return self._impl.sh_degree

    @property
    def num_channels(self) -> int:
        """
        Returns the number of channels in the Gaussian splatting representation.
        For example, if you are rendering RGB images, this method will return 3.

        Returns:
            num_channels (int): The number of channels.
        """
        return self._impl.num_channels

    @property
    def num_gaussians(self) -> int:
        """
        Returns the number of Gaussians in the Gaussian splatting representation.
        This is the total number of individual gaussian splats that are being used to represent the scene.

        Returns:
            num_gaussians (int): The number of Gaussians.
        """
        return self._impl.num_gaussians

    @property
    def num_sh_bases(self) -> int:
        """
        Returns the number of spherical harmonics (SH) bases used in the Gaussian splatting representation.

        .. note::

            The number of SH bases is related to the SH degree (see :attr:`sh_degree`) by the formula
            :math:`K = (sh\\_degree + 1)^2`, where :math:`K` is the number of spherical harmonics bases.

        Returns:
            num_sh_bases (int): The number of spherical harmonics bases.
        """
        return self._impl.num_sh_bases

    @property
    def log_scales(self) -> torch.Tensor:
        """
        Returns the log of the scales for each Gaussian. Gaussians are represented in 3D space,
        as ellipsoids defined by their means, orientations (quaternions), and scales. *i.e.*

        .. math::

            g_i(x) = \\exp(-0.5 (x - \\mu_i)^T \\Sigma_i^{-1} (x - \\mu_i))

        where :math:`\\mu_i` is the mean and :math:`\\Sigma_i = R(q_i)^T S_i R(q_i)` is the covariance of the i-th Gaussian
        with :math:`R(q_i)` being the rotation matrix defined by the unit
        quaternion :math:`q_i` of the Gaussian, and :math:`S_i = diag(\\exp(log\\_scales_i))`.

        .. note::

            The :class:`GaussianSplat3d` stores the log of scales scales (:attr:`log_scales`) rather than the scales
            directly. This ensures numerical stability, especially when optimizing the scales.
            To read the scales directly, see the :attr:`scales` property (which is read-only).

        Returns:
            log_scales (torch.Tensor): A tensor of shape ``(N, 3)`` where ``N`` is the number
                of Gaussians (see :attr:`num_gaussians`). Each row represents the log of the scale of a Gaussian in 3D space.
        """
        return self._impl.log_scales

    @log_scales.setter
    def log_scales(self, value: torch.Tensor) -> None:
        """
        Sets the log of the scales for each Gaussian. Gaussians are represented in 3D space,
        as ellipsoids defined by their means, orientations (quaternions), and scales. *i.e.*

        .. math::

            g_i(x) = \\exp(-0.5 (x - \\mu_i)^T \\Sigma_i^{-1} (x - \\mu_i))

        where :math:`\\mu_i` is the mean and :math:`\\Sigma_i = R(q_i)^T S_i R(q_i)` is the covariance of the i-th Gaussian
        with :math:`R(q_i)` being the rotation matrix defined by the unit
        quaternion :math:`q_i` of the Gaussian, and :math:`S_i = diag(\\exp(log\\_scales_i))`.

        .. note::

            The :class:`GaussianSplat3d` stores the log of scales scales (:attr:`log_scales`) rather than the scales
            directly. This ensures numerical stability, especially when optimizing the scales.
            To read the scales directly, see the :attr:`scales` property (which is read-only).

        Args:
            value (torch.Tensor): A tensor of shape ``(N, 3)`` where ``N`` is the number
                of Gaussians (see :attr:`num_gaussians`). Each row represents the log of the
                scale of a Gaussian in 3D space.

        """
        self._impl.log_scales = cast_check(value, torch.Tensor, "log_scales")

    @property
    def logit_opacities(self) -> torch.Tensor:
        """
        Return the logit (inverse of sigmoid) of the opacities of each Gaussian in the scene.

        .. note::

            The :class:`GaussianSplat3d` stores the logit of opacities (:attr:`logit_opacities`) rather than the opacities
            directly. The actual opacities are obtained by applying the sigmoid function to the logit opacities.
            To read the opacities directly, see the :attr:`opacities` property (which is read-only).

        Returns:
            logit_opacities (torch.Tensor): A tensor of shape ``(N,)`` where ``N`` is the number
                of Gaussians (see :attr:`num_gaussians`). Each row represents the logit of the opacity of a Gaussian in 3D space.
        """
        return self._impl.logit_opacities

    @logit_opacities.setter
    def logit_opacities(self, value: torch.Tensor) -> None:
        """
        Set the logit (inverse of sigmoid) of the opacities of each Gaussian in the scene.

        .. note::

            The :class:`GaussianSplat3d` stores the logit of opacities (:attr:`logit_opacities`) rather than the opacities
            directly. The actual opacities are obtained by applying the sigmoid function to the logit opacities.
            To read the opacities directly, see the :attr:`opacities` property (which is read-only).

        Args:
            value (torch.Tensor): A tensor of shape ``(N,)`` where ``N`` is the number
                of Gaussians (see :attr:`num_gaussians`). Each row represents the logit of the opacity of a Gaussian in 3D space.
        """
        self._impl.logit_opacities = cast_check(value, torch.Tensor, "logit_opacities")

    @property
    def means(self) -> torch.Tensor:
        """
        Return the means (3d positions) of the Gaussians in this :class:`GaussianSplat3d`.
        The means represent the center of each Gaussian in 3D space.
        *i.e* each Gaussian :math:`g_i` is defined as:

        .. math::

            g_i(x) = \\exp(-0.5 (x - \\mu_i)^T \\Sigma_i^{-1} (x - \\mu_i))

        where :math:`\\mu_i` is the mean and :math:`\\Sigma_i = R(q_i)^T S_i R(q_i)` is the covariance of the i-th Gaussian
        with :math:`R(q_i)` being the rotation matrix defined by the unit
        quaternion :math:`q_i` of the Gaussian, and :math:`S_i = diag(\\exp(log\\_scales_i))`.


        Returns:
            torch.Tensor: A tensor of shape (N, 3) where N is the number
                of Gaussians (see `num_gaussians`). Each row represents the mean of a Gaussian in 3D space.
        """
        return self._impl.means

    @means.setter
    def means(self, value: torch.Tensor) -> None:
        """
        Sets the means (3d positions) of the Gaussians in this :class:`GaussianSplat3d`.
        The means represent the center of each Gaussian in 3D space.
        *i.e* each Gaussian :math:`g_i` is defined as:

        .. math::

            g_i(x) = \\exp(-0.5 (x - \\mu_i)^T \\Sigma_i^{-1} (x - \\mu_i))

        where :math:`\\mu_i` is the mean and :math:`\\Sigma_i = R(q_i)^T S_i R(q_i)` is the covariance of the i-th Gaussian
        with :math:`R(q_i)` being the rotation matrix defined by the unit
        quaternion :math:`q_i` of the Gaussian, and :math:`S_i = diag(\\exp(log\\_scales_i))`.

        Args:
            value (torch.Tensor): A tensor of shape ``(N, 3)`` where ``N`` is the number
                of Gaussians (see :attr:`num_gaussians`). Each row represents the mean of a Gaussian in 3D space.
        """
        self._impl.means = cast_check(value, torch.Tensor, "means")

    @property
    def quats(self) -> torch.Tensor:
        """
        Returns the unit quaternions representing the orientation of the covariance of the Gaussians in this :class:`GaussianSplat3d`.
        *i.e* each Gaussian :math:`g_i` is defined as:

        .. math::

            g_i(x) = \\exp(-0.5 (x - \\mu_i)^T \\Sigma_i^{-1} (x - \\mu_i))

        where :math:`\\mu_i` is the mean and :math:`\\Sigma_i = R(q_i)^T S_i R(q_i)` is the covariance of the i-th Gaussian
        with :math:`R(q_i)` being the rotation matrix defined by the unit
        quaternion :math:`q_i` of the Gaussian, and :math:`S_i = diag(\\exp(log\\_scales_i))`.

        Returns:
            quats (torch.Tensor): A tensor of shape ``(N, 4)`` where ``N`` is the number
                of Gaussians (see :attr:`num_gaussians`). Each row represents the unit quaternion of a Gaussian in 3D space.
        """
        return self._impl.quats

    @quats.setter
    def quats(self, value: torch.Tensor) -> None:
        """
        Sets the unit quaternions representing the orientation of the covariance of the Gaussians in this :class:`GaussianSplat3d`.
        *i.e* each Gaussian :math:`g_i` is defined as:

        .. math::

            g_i(x) = \\exp(-0.5 (x - \\mu_i)^T \\Sigma_i^{-1} (x - \\mu_i))

        where :math:`\\mu_i` is the mean and :math:`\\Sigma_i = R(q_i)^T S_i R(q_i)` is the covariance of the i-th Gaussian
        with :math:`R(q_i)` being the rotation matrix defined by the unit
        quaternion :math:`q_i` of the Gaussian, and :math:`S_i = diag(\\exp(log\\_scales_i))`.

        Args:
            value (torch.Tensor): A tensor of shape ``(N, 4)`` where ``N`` is the number
                of Gaussians (see :attr:`num_gaussians`). Each row represents the unit quaternion of a Gaussian in 3D space.
        """
        self._impl.quats = cast_check(value, torch.Tensor, "quats")

    @property
    def requires_grad(self) -> bool:
        """
        Returns whether the tensors tracked by this :class:`GaussianSplat3d` instance are set to require gradients.
        This is typically set to True if you want to optimize the parameters of the Gaussians.

        Example:

        .. code-block:: python

            gsplat3d = GaussianSplat3d(...)  # Some GaussianSplat3d instance
            gsplat3d.requires_grad = True  # Enable gradient tracking for optimization

            assert gsplat3d.means.requires_grad  # Now the means will require gradients
            assert gsplat3d.covariances.requires_grad  # Now the covariances will require gradients
            assert gsplat3d.logit_opacities.requires_grad  # Now the logit opacities will require gradients
            assert gsplat3d.log_scales.requires_grad  # Now the log scales will require gradients
            assert gsplat3d.sh0.requires_grad  # Now the SH coefficients will require gradients
            assert gsplat3d.shN.requires_grad  # Now the SH coefficients will require gradients

        Returns:
            requires_grad (bool): ``True`` if gradients are required, ``False`` otherwise.
        """
        return self._impl.requires_grad

    @requires_grad.setter
    def requires_grad(self, value: bool) -> None:
        """
        Sets whether the tensors tracked by this :class:`GaussianSplat3d` instance require gradients.
        This is typically set to True if you want to optimize the parameters of the Gaussians.

        Example:

        .. code-block:: python

            gsplat3d = GaussianSplat3d(...)  # Some GaussianSplat3d instance
            gsplat3d.requires_grad = True  # Enable gradient tracking for optimization

            assert gsplat3d.means.requires_grad  # Now the means will require gradients
            assert gsplat3d.covariances.requires_grad  # Now the covariances will require gradients
            assert gsplat3d.logit_opacities.requires_grad  # Now the logit opacities will require gradients
            assert gsplat3d.log_scales.requires_grad  # Now the log scales will require gradients
            assert gsplat3d.sh0.requires_grad  # Now the SH coefficients will require gradients
            assert gsplat3d.shN.requires_grad  # Now the SH coefficients will require gradients

        Returns:
            requires_grad (bool): ``True`` if gradients are required, ``False`` otherwise.
        """
        self._impl.requires_grad = cast_check(value, bool, "requires_grad")

    @property
    def sh0(self) -> torch.Tensor:
        """
        Returns the diffuse spherical harmonics coefficients of the Gaussians in this :class:`GaussianSplat3d`.
        These coefficients are used to represent the diffuse color/feature of each Gaussian.

        Returns:
            sh0 (torch.Tensor): A tensor of shape ``(N, 1, D)`` where ``N`` is the number
                of Gaussians (see :attr:`num_gaussians`), and ``D`` is the number of channels (see :attr:`num_channels`).
                Each row represents the diffuse SH coefficients for a Gaussian.
        """
        return self._impl.sh0

    @sh0.setter
    def sh0(self, value: torch.Tensor) -> None:
        """
        Sets the diffuse spherical harmonics coefficients of the Gaussians in this :class:`GaussianSplat3d`.
        These coefficients are used to represent the diffuse color/feature of each Gaussian.

        Args:
            value (torch.Tensor): A tensor of shape ``(N, 1, D)`` where ``N`` is the number
                of Gaussians (see :attr:`num_gaussians`), and ``D`` is the number of channels (see :attr:`num_channels`).
                Each row represents the diffuse SH coefficients for a Gaussian.
        """
        self._impl.sh0 = cast_check(value, torch.Tensor, "sh0")

    @property
    def shN(self) -> torch.Tensor:
        """
        Returns the directionally varying spherical harmonics coefficients of the Gaussians in the scene.
        These coefficients are used to represent a direction dependent color/feature of each Gaussian.

        Returns:
            torch.Tensor: A tensor of shape (N, K-1, D) where N is the number
                of Gaussians (see `num_gaussians`), D is the number of channels (see `num_channels`),
                and K is the number of spherical harmonic bases (see `num_sh_bases`).
                Each row represents the directionally varying SH coefficients for a Gaussian.
        """
        return self._impl.shN

    @shN.setter
    def shN(self, value: torch.Tensor) -> None:
        """
        Sets the directionally varying spherical harmonics coefficients of the Gaussians in this :class:`GaussianSplat3d`.
        These coefficients are used to represent a direction dependent color/feature of each Gaussian.

        Args:
            value (torch.Tensor): A tensor of shape ``(N, K-1, D)`` where ``N`` is the number
                of Gaussians (see :attr:`num_gaussians`), ``D`` is the number of channels (see :attr:`num_channels`),
                and ``K`` is the number of spherical harmonic bases (see :attr:`num_sh_bases`).
                Each row represents the directionally varying SH coefficients for a Gaussian.
        """
        self._impl.shN = cast_check(value, torch.Tensor, "shN")

    @property
    def opacities(self) -> torch.Tensor:
        """
        Returns the opacities of the Gaussians in the Gaussian splatting representation.
        The opacities encode the visibility of each Gaussian in the scene.

        .. note::

            This property is read only. :class:`GaussianSplat3d` stores the logit (inverse of sigmoid)
            of the opacities to ensure numerical stability, which you can modify. See :attr:`logit_opacities`.

        Returns:
            opacities (torch.Tensor): A tensor of shape ``(N,)`` where ``N`` is the number of Gaussians (see :attr:`num_gaussians`).
                Each element represents the opacity of a Gaussian.
        """
        return self._impl.opacities

    @property
    def scales(self) -> torch.Tensor:
        """
        Returns the scales of the Gaussians in the Gaussian splatting representation. The scales are
        the eigenvalues of the covariance matrix of each Gaussian.
        *i.e* each Gaussian :math:`g_i` is defined as:

        .. math::

            g_i(x) = \\exp(-0.5 (x - \\mu_i)^T \\Sigma_i^{-1} (x - \\mu_i))

        where :math:`\\mu_i` is the mean and :math:`\\Sigma_i = R(q_i)^T S_i R(q_i)` is the covariance of the i-th Gaussian
        with :math:`R(q_i)` being the rotation matrix defined by the unit
        quaternion :math:`q_i` of the Gaussian, and :math:`S_i = diag(\\exp(log\\_scales_i))`.

        .. note::

            This property is read only. :class:`GaussianSplat3d` stores the log of scales to ensure numerical stability,
            which you can modify. See :attr:`log_scales`.

        Returns:
            scales (torch.Tensor): A tensor of shape ``(N, 3)`` where ``N`` is the number
                of Gaussians. Each row represents the scale of a Gaussian in 3D space.
        """
        return self._impl.scales

    @property
    def accumulated_gradient_step_counts(self) -> torch.Tensor:
        """
        Returns the accumulated gradient step counts for each Gaussian.

        If this :class:`GaussianSplat3d` instance
        is set to track accumulated gradients (*i.e*  :attr:`accumulate_mean_2d_gradients` is ``True``),
        then this tensor contains the number of Gradient steps that have been applied to each Gaussian during optimization.

        If :attr:`accumulate_mean_2d_gradients` is ``False``, this property will be an empty tensor.

        .. note::

            To reset the counts, call call the :meth:`reset_accumulated_gradient_state` method.

        Returns:
            step_counts (torch.Tensor): A tensor of shape ``(N,)`` where ``N`` is the number of Gaussians (see :attr:`num_gaussians`).
                Each element represents the accumulated gradient step count for a Gaussian.
        """
        return self._impl.accumulated_gradient_step_counts

    @property
    def accumulated_max_2d_radii(self) -> torch.Tensor:
        """
        Returns the maximum 2D projected radius (in pixels) for each Gaussian across all calls to `render_*` functions.
        This is used by certain optimization techniques to ensure that the Gaussians do not become too large or too small during the optimization process.

        If :this :class:`GaussianSplat3d` instance is set to track maximum 2D radii
        (*i.e* :attr:`accumulate_max_2d_radii` is ``True``), then this tensor contains the maximum 2D radius for each Gaussian.

        If :attr:`accumulate_max_2d_radii` is ``False``, this property will be an empty tensor.

        .. note::

            To reset the maximum radii to zero, you can call the :meth:`reset_accumulated_gradient_state` method.

        Returns:
            max_radii (torch.Tensor): A tensor of shape ``(N,)`` where ``N`` is the number of Gaussians (see :attr:`num_gaussians`).
                Each element represents the maximum 2D radius for a Gaussian across all optimization iterations.

        """
        return self._impl.accumulated_max_2d_radii

    @property
    def accumulate_max_2d_radii(self) -> bool:
        """
        Returns whether to track the maximum 2D projected radius of each Gaussian across calls to `render_*` functions.
        This is used by certain optimization techniques to ensure that the Gaussians do not become too large or too small during the optimization process.


        .. seealso::

            See :attr:`accumulated_max_2d_radii` for the actual maximum radii values.

        Returns:
            accumulate_max_radii (bool): ``True`` if the maximum 2D radii are being tracked across rendering calls, ``False`` otherwise.
        """
        return self._impl.accumulate_max_2d_radii

    @accumulate_max_2d_radii.setter
    def accumulate_max_2d_radii(self, value) -> None:
        """
        Sets whether to track the maximum 2D projected radius of each Gaussian across calls to `render_*` functions.
        This is used by certain optimization techniques to ensure that the Gaussians do not become too large or too small during the optimization process.

        .. seealso::

            See :attr:`accumulated_max_2d_radii` for the actual maximum radii values.

        Args:
            value (bool): ``True`` if the maximum 2D radii are being tracked across rendering calls, ``False`` otherwise.
        """
        self._impl.accumulate_max_2d_radii = cast_check(value, bool, "accumulate_max_2d_radii")

    @property
    def accumulate_mean_2d_gradients(self) -> bool:
        """
        Returns whether to track the average norm of the gradient of projected means for each Gaussian during the backward pass of projection.
        This property is used by certain optimization techniques to split/prune/duplicate Gaussians.
        The accumulated 2d gradient norms are defined as follows:

        .. math::

            \\sum_{t=1}^{T} \\| \\partial_{L_t} \\mu_i^{2D} \\|_2

        where :math:`\\mu_i^{2D}` is the projection of the mean of Gaussian :math:`g_i` onto the image plane,
        and :math:`L_t` is the loss at iteration :math:`t`.

        .. seealso::

            See :attr:`accumulated_mean_2d_gradient_norms` for the actual average norms of the gradients.

        Returns:
            accumulate_mean_2d_grads (bool): ``True`` if the average norm of the gradient of projected means is being tracked, ``False`` otherwise.
        """
        return self._impl.accumulate_mean_2d_gradients

    @accumulate_mean_2d_gradients.setter
    def accumulate_mean_2d_gradients(self, value: bool) -> None:
        """
        Sets whether to track the average norm of the gradient of projected means for each Gaussian during the backward pass of projection.
        This property is used by certain optimization techniques to split/prune/duplicate Gaussians.
        The accumulated 2d gradient norms are defined as follows:

        .. math::

            \\sum_{t=1}^{T} \\| \\partial_{L_t} \\mu_i^{2D} \\|_2

        where :math:`\\mu_i^{2D}` is the projection of the mean of Gaussian :math:`g_i` onto the image plane,
        and :math:`L_t` is the loss at iteration :math:`t`.

        .. seealso::

            See :attr:`accumulated_mean_2d_gradient_norms` for the actual average norms of the gradients.

        Args:
            value (bool): ``True`` if the average norm of the gradient of projected means is being tracked, ``False`` otherwise.
        """
        self._impl.accumulate_mean_2d_gradients = cast_check(value, bool, "accumulate_mean_2d_gradients")

    @property
    def accumulated_mean_2d_gradient_norms(self) -> torch.Tensor:
        """
        Returns the average norm of the gradient of projected (2D) means for each Gaussian across every backward pass.
        This is used by certain optimization techniques to split/prune/duplicate Gaussians.
        The accumulated 2d gradient norms are defined as follows:

        .. math::

            \\sum_{t=1}^{T} \\| \\partial_{L_t} \\mu_i^{2D} \\|_2

        where :math:`\\mu_i^{2D}` is the projection of the mean of Gaussian :math:`g_i` onto the image plane,
        and :math:`L_t` is the loss at iteration :math:`t`.

        .. note::

            To reset the accumulated norms, call the :meth:`reset_accumulated_gradient_state` method.

        Returns:
            accumulated_grad_2d_norms (torch.Tensor): A tensor of shape ``(N,)`` where ``N`` is the number of Gaussians (see :attr:`num_gaussians`).
                Each element represents the average norm of the gradient of projected means for a Gaussian across all optimization iterations.
                The norm is computed in 2D space, i.e., the projected means.
        """
        return self._impl.accumulated_mean_2d_gradient_norms

    def project_gaussians_for_depths(
        self,
        world_to_camera_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        image_width: int,
        image_height: int,
        near: float,
        far: float,
        projection_type=ProjectionType.PERSPECTIVE,
        min_radius_2d: float = 0.0,
        eps_2d: float = 0.3,
        antialias: bool = False,
    ) -> ProjectedGaussianSplats:
        """
        Projects this :class:`GaussianSplat3d` onto one or more image planes for rendering depth images in those planes.
        You can render depth images from the projected Gaussians by calling :meth:`render_projected_gaussians`.

        .. note::

            The reason to have a separate projection and rendering step is to enable rendering crops of an image without
            having to project the Gaussians again.


        .. note::

            All images being rendered must have the same width and height.


        .. seealso::

            :class:`fvdb.ProjectedGaussianSplats` for the projected Gaussians representation.

        .. code-block:: python

            # Assume gaussian_splat_3d is an instance of GaussianSplat3d
            # Project the Gaussians for rendering depth images onto C image planes
            projected_gaussians = gaussian_splat_3d.project_gaussians_for_depths(
                world_to_camera_matrices, # tensor of shape [C, 4, 4]
                projection_matrices, # tensor of shape [C, 3, 3]
                image_width, # width of the C images
                image_height, # height of the C images
                near, # near clipping plane
                far) # far clipping plane

            # Now render a crop of size 100x100 starting at (10, 10) from the projected Gaussians
            # in each image plane.
            # Returns a tensor of shape [C, 100, 100, 1] containing the depth images,
            # and a tensor of shape [C, 100, 100, 1] containing the final alpha (opacity) values
            # of each pixel.
            cropped_depth_images_1, cropped_alphas = gaussian_splat_3d.render_from_projected_gaussians(
                projected_gaussians,
                crop_width=100,
                crop_height=100,
                crop_origin_w=10,
                crop_origin_h=10)

            # To get the depth images, divide the last channel by the alpha values
            true_depths_1 = cropped_images_1[..., -1:] / cropped_alphas

        Args:
            world_to_camera_matrices (torch.Tensor): Tensor of shape ``(C, 4, 4)`` representing the world-to-camera transformation matrices for ``C`` cameras.
                Each matrix transforms points from world coordinates to camera coordinates.
            projection_matrices (torch.Tensor): Tensor of shape ``(C, 3, 3)`` representing the projection matrices for ``C`` cameras.
                Each matrix projects points in camera space into homogeneous pixel coordinates.
            image_width (int): The width of the images to be rendered. Note that all images must have the same width.
            image_height (int): The height of the images to be rendered. Note that all images must have the same height.
            near (float): The near clipping plane distance for the projection.
            far (float): The far clipping plane distance for the projection.
            projection_type (ProjectionType): The type of projection to use. Default is :attr:`ProjectionType.PERSPECTIVE`.
            min_radius_2d (float): The minimum radius (in pixels) below which Gaussians are ignored during rendering.
            eps_2d (float): A value used to pad Gaussians when projecting them onto the image plane, to avoid very projected Gaussians which create artifacts and
                numerical issues.
            antialias (bool): If ``True``, applies opacity correction to the projected Gaussians when using ``eps_2d > 0.0``.

        Returns:
            projected_gaussians (ProjectedGaussianSplats): An instance of ProjectedGaussianSplats containing the projected Gaussians.
                This object contains the projected 2D representations of the Gaussians, which can be used for rendering depth images or further processing.

        """
        return ProjectedGaussianSplats(
            self._impl.project_gaussians_for_depths(
                world_to_camera_matrices=world_to_camera_matrices,
                projection_matrices=projection_matrices,
                image_width=image_width,
                image_height=image_height,
                near=near,
                far=far,
                projection_type=self._proj_type_to_cpp(projection_type),
                min_radius_2d=min_radius_2d,
                eps_2d=eps_2d,
                antialias=antialias,
            ),
            _private=ProjectedGaussianSplats.__PRIVATE__,
        )

    def project_gaussians_for_images(
        self,
        world_to_camera_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        image_width: int,
        image_height: int,
        near: float,
        far: float,
        projection_type=ProjectionType.PERSPECTIVE,
        sh_degree_to_use: int = -1,
        min_radius_2d: float = 0.0,
        eps_2d: float = 0.3,
        antialias: bool = False,
    ) -> ProjectedGaussianSplats:
        """
        Projects this :class:`GaussianSplat3d` onto one or more image planes for rendering multi-channel (see :attr:`num_channels`) images in those planes.
        You can render images from the projected Gaussians by calling :meth:`render_projected_gaussians`.

        .. note::

            The reason to have a separate projection and rendering step is to enable rendering crops of an image without
            having to project the Gaussians again.


        .. note::

            All images being rendered must have the same width and height.


        .. seealso::

            :class:`fvdb.ProjectedGaussianSplats` for the projected Gaussians representation.

        .. code-block:: python

            # Assume gaussian_splat_3d is an instance of GaussianSplat3d
            # Project the Gaussians for rendering images onto C image planes
            projected_gaussians = gaussian_splat_3d.project_gaussians_for_images(
                world_to_camera_matrices, # tensor of shape [C, 4, 4]
                projection_matrices, # tensor of shape [C, 3, 3]
                image_width, # width of the C images
                image_height, # height of the C images
                near, # near clipping plane
                far) # far clipping plane

            # Now render a crop of size 100x100 starting at (10, 10) from the projected Gaussians
            # in each image plane.
            # Returns a tensor of shape [C, 100, 100, D] containing the images (where D is num_channels),
            # and a tensor of shape [C, 100, 100, 1] containing the final alpha (opacity) values
            # of each pixel.
            cropped_images_1, cropped_alphas = gaussian_splat_3d.render_from_projected_gaussians(
                projected_gaussians,
                crop_width=100,
                crop_height=100,
                crop_origin_w=10,
                crop_origin_h=10)

        Args:
            world_to_camera_matrices (torch.Tensor): Tensor of shape ``(C, 4, 4)`` representing the world-to-camera transformation matrices for ``C`` cameras.
                Each matrix transforms points from world coordinates to camera coordinates.
            projection_matrices (torch.Tensor): Tensor of shape ``(C, 3, 3)`` representing the projection matrices for ``C`` cameras.
                Each matrix projects points in camera space into homogeneous pixel coordinates.
            image_width (int): The width of the images to be rendered. Note that all images must have the same width.
            image_height (int): The height of the images to be rendered. Note that all images must have the same height.
            near (float): The near clipping plane distance for the projection.
            far (float): The far clipping plane distance for the projection.
            projection_type (ProjectionType): The type of projection to use. Default is :attr:`ProjectionType.PERSPECTIVE`.
            sh_degree_to_use (int): The degree of spherical harmonics to use for rendering. -1 means use all available SH bases.
                0 means use only the first SH base (constant color). Note that you can't use more SH bases than available in the GaussianSplat3d instance.
                Default is -1.
            min_radius_2d (float): The minimum radius (in pixels) below which Gaussians are ignored during rendering.
            eps_2d (float): A value used to pad Gaussians when projecting them onto the image plane, to avoid very projected Gaussians which create artifacts and
                numerical issues.
            antialias (bool): If ``True``, applies opacity correction to the projected Gaussians when using ``eps_2d > 0.0``.

        Returns:
            projected_gaussians (ProjectedGaussianSplats): An instance of ProjectedGaussianSplats containing the projected Gaussians.
                This object contains the projected 2D representations of the Gaussians, which can be used for rendering images or further processing.

        """
        return ProjectedGaussianSplats(
            self._impl.project_gaussians_for_images(
                world_to_camera_matrices=world_to_camera_matrices,
                projection_matrices=projection_matrices,
                image_width=image_width,
                image_height=image_height,
                near=near,
                far=far,
                projection_type=self._proj_type_to_cpp(projection_type),
                sh_degree_to_use=sh_degree_to_use,
                min_radius_2d=min_radius_2d,
                eps_2d=eps_2d,
                antialias=antialias,
            ),
            _private=ProjectedGaussianSplats.__PRIVATE__,
        )

    def project_gaussians_for_images_and_depths(
        self,
        world_to_camera_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        image_width: int,
        image_height: int,
        near: float,
        far: float,
        projection_type=ProjectionType.PERSPECTIVE,
        sh_degree_to_use: int = -1,
        min_radius_2d: float = 0.0,
        eps_2d: float = 0.3,
        antialias: bool = False,
    ) -> ProjectedGaussianSplats:
        """
        Projects this :class:`GaussianSplat3d` onto one or more image planes for rendering multi-channel (see :attr:`num_channels`) images with depths
        in the last channel.
        You can render images+depths from the projected Gaussians by calling :meth:`render_projected_gaussians`.

        .. note::

            The reason to have a separate projection and rendering step is to enable rendering crops of an image without
            having to project the Gaussians again.


        .. note::

            All images being rendered must have the same width and height.


        .. seealso::

            :class:`fvdb.ProjectedGaussianSplats` for the projected Gaussians representation.

        .. code-block:: python

            # Assume gaussian_splat_3d is an instance of GaussianSplat3d
            # Project the Gaussians for rendering images onto C image planes
            projected_gaussians = gaussian_splat_3d.project_gaussians_for_images_and_depths(
                world_to_camera_matrices, # tensor of shape [C, 4, 4]
                projection_matrices, # tensor of shape [C, 3, 3]
                image_width, # width of the C images
                image_height, # height of the C images
                near, # near clipping plane
                far) # far clipping plane

            # Now render a crop of size 100x100 starting at (10, 10) from the projected Gaussians
            # in each image plane.
            # Returns a tensor of shape [C, 100, 100, D] containing the images (where D is num_channels + 1 for depth),
            # and a tensor of shape [C, 100, 100, 1] containing the final alpha (opacity) values
            # of each pixel.
            cropped_images_1, cropped_alphas = gaussian_splat_3d.render_from_projected_gaussians(
                projected_gaussians,
                crop_width=100,
                crop_height=100,
                crop_origin_w=10,
                crop_origin_h=10)

            cropped_images = cropped_images_1[..., :-1]  # Extract image channels

            # Divide by alpha to get the final true depth values
            cropped_depths = cropped_images_1[..., -1:] / cropped_alphas  # Extract depth channel

        Args:
            world_to_camera_matrices (torch.Tensor): Tensor of shape ``(C, 4, 4)`` representing the world-to-camera transformation matrices for ``C`` cameras.
                Each matrix transforms points from world coordinates to camera coordinates.
            projection_matrices (torch.Tensor): Tensor of shape ``(C, 3, 3)`` representing the projection matrices for ``C`` cameras.
                Each matrix projects points in camera space into homogeneous pixel coordinates.
            image_width (int): The width of the images to be rendered. Note that all images must have the same width.
            image_height (int): The height of the images to be rendered. Note that all images must have the same height.
            near (float): The near clipping plane distance for the projection.
            far (float): The far clipping plane distance for the projection.
            projection_type (ProjectionType): The type of projection to use. Default is :attr:`fvdb.ProjectionType.PERSPECTIVE`.
            sh_degree_to_use (int): The degree of spherical harmonics to use for rendering. -1 means use all available SH bases.
                0 means use only the first SH base (constant color). Note that you can't use more SH bases than available in the GaussianSplat3d instance.
                Default is -1.
            min_radius_2d (float): The minimum radius (in pixels) below which Gaussians are ignored during rendering.
            eps_2d (float): A value used to pad Gaussians when projecting them onto the image plane, to avoid very projected Gaussians which create artifacts and
                numerical issues.
            antialias (bool): If ``True``, applies opacity correction to the projected Gaussians when using ``eps_2d > 0.0``.

        Returns:
            projected_gaussians (ProjectedGaussianSplats): An instance of ProjectedGaussianSplats containing the projected Gaussians.
                This object contains the projected 2D representations of the Gaussians, which can be used for rendering images or further processing.

        """
        return ProjectedGaussianSplats(
            self._impl.project_gaussians_for_images_and_depths(
                world_to_camera_matrices=world_to_camera_matrices,
                projection_matrices=projection_matrices,
                image_width=image_width,
                image_height=image_height,
                near=near,
                far=far,
                projection_type=self._proj_type_to_cpp(projection_type),
                sh_degree_to_use=sh_degree_to_use,
                min_radius_2d=min_radius_2d,
                eps_2d=eps_2d,
                antialias=antialias,
            ),
            _private=ProjectedGaussianSplats.__PRIVATE__,
        )

    def render_from_projected_gaussians(
        self,
        projected_gaussians: ProjectedGaussianSplats,
        crop_width: int = -1,
        crop_height: int = -1,
        crop_origin_w: int = -1,
        crop_origin_h: int = -1,
        tile_size: int = 16,
        backgrounds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Render a set of images from Gaussian splats that have already been projected onto image planes
        (See for example :meth:`project_gaussians_for_images`).
        This method is useful when you want to render images from pre-computed projected Gaussians,
        for example, when rendering crops of images without having to re-project the Gaussians.

        .. note::

            If you want to render the full image, pass negative values for ``crop_width``, ``crop_height``,
            ``crop_origin_w``, and ``crop_origin_h`` (default behavior). To render full images,
            all these values must be negative or this method will raise an error.

        .. note::

            If your crop goes beyond the image boundaries, the resulting image will be clipped to
            be within the image boundaries.


        Example:

        .. code-block:: python

            # Assume gaussian_splat_3d is an instance of GaussianSplat3d
            # Project the Gaussians for rendering images onto C image planes
            projected_gaussians = gaussian_splat_3d.project_gaussians_for_images_and_depths(
                world_to_camera_matrices, # tensor of shape [C, 4, 4]
                projection_matrices, # tensor of shape [C, 3, 3]
                image_width, # width of the C images
                image_height, # height of the C images
                near, # near clipping plane
                far) # far clipping plane

            # Now render a crop of size 100x100 starting at (10, 10) from the projected Gaussians
            # in each image plane.
            # Returns a tensor of shape [C, 100, 100, D] containing the images (where D is num_channels + 1 for depth),
            # and a tensor of shape [C, 100, 100, 1] containing the final alpha (opacity) values
            # of each pixel.
            cropped_images_1, cropped_alphas = gaussian_splat_3d.render_from_projected_gaussians(
                projected_gaussians,
                crop_width=100,
                crop_height=100,
                crop_origin_w=10,
                crop_origin_h=10)

            cropped_images = cropped_images_1[..., :-1]  # Extract image channels

            # Divide by alpha to get the final true depth values
            cropped_depths = cropped_images_1[..., -1:] / cropped_alphas  # Extract depth channel


        Args:
            projected_gaussians (ProjectedGaussianSplats): An instance of :class:`fvdb.ProjectedGaussianSplats`
                containing the projected Gaussians after spherical harmonic evaluation. This object should have been created by calling
                :meth:`project_gaussians_for_images`, :meth:`project_gaussians_for_depths`,
                :meth:`project_gaussians_for_images_and_depths`, etc.
            crop_width (int): The width of the crop to render. If -1, the full image width is used.
                Default is -1.
            crop_height (int): The height of the crop to render. If -1, the full image height is used.
                Default is -1.
            crop_origin_w (int): The x-coordinate of the top-left corner of the crop. If -1, the crop starts at (0, 0).
                Default is -1.
            crop_origin_h (int): The y-coordinate of the top-left corner of the crop. If -1, the crop starts at (0, 0).
                Default is -1.
            tile_size (int): The size of the tiles to use for rendering. Default is 16.
                This parameter controls the size of the tiles used for rendering the images.
                You shouldn't set this parameter unless you really know what you are doing.

        Returns:
            rendered_images (torch.Tensor): A tensor of shape ``(C, H, W, D)`` where ``C`` is the number of image planes,
                ``H`` is the height of the rendered images, ``W`` is the width of the rendered images, and ``D`` is the
                number of channels (e.g., RGB, RGBD, etc.).
            alpha_images (torch.Tensor): A tensor of shape ``(C, H, W, 1)`` where ``C`` is the number of cameras,
                ``H`` is the height of the images, and ``W`` is the width of the images.
                Each element represents the alpha value (opacity) at a pixel such that 0 <= alpha < 1,
                and 0 means the pixel is fully transparent, and 1 means the pixel is fully opaque.
        """
        return self._impl.render_from_projected_gaussians(
            projected_gaussians=projected_gaussians._impl,
            crop_width=crop_width,
            crop_height=crop_height,
            crop_origin_w=crop_origin_w,
            crop_origin_h=crop_origin_h,
            tile_size=tile_size,
            backgrounds=backgrounds,
        )

    def render_depths(
        self,
        world_to_camera_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        image_width: int,
        image_height: int,
        near: float,
        far: float,
        projection_type=ProjectionType.PERSPECTIVE,
        tile_size: int = 16,
        min_radius_2d: float = 0.3,
        eps_2d: float = 0.3,
        antialias: bool = False,
        backgrounds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Render ``C`` depth maps from this :class:`GaussianSplat3d` from ``C`` camera views.

        .. note::

            All depth maps being rendered must have the same width and height.


        Example:

        .. code-block:: python

            # Assume gaussian_splat_3d is an instance of GaussianSplat3d
            # Render depth maps from C camera views
            # depth_images is a tensor of shape [C, H, W, 1]
            # alpha_images is a tensor of shape [C, H, W, 1]
            depth_images, alpha_images = gaussian_splat_3d.render_depths(
                world_to_camera_matrices, # tensor of shape [C, 4, 4]
                projection_matrices, # tensor of shape [C, 3, 3]
                image_width, # width of the depth maps
                image_height, # height of the depth maps
                near, # near clipping plane
                far) # far clipping plane

            true_depths = depth_images / alpha_images  # Get true depth values by dividing by alpha

        Args:
            world_to_camera_matrices (torch.Tensor): Tensor of shape ``(C, 4, 4)`` representing the
                world-to-camera transformation matrices for ``C`` cameras. Each matrix transforms points
                from world coordinates to camera coordinates.
            projection_matrices (torch.Tensor): Tensor of shape ``(C, 3, 3)`` representing the projection matrices for ``C`` cameras.
                Each matrix projects points in camera space into homogeneous pixel coordinates.
            image_width (int): The width of the depth maps to be rendered. Note these are the same for all depth maps being rendered.
            image_height (int): The height of the depth maps to be rendered. Note these are the same for all depth maps being rendered.
            near (float): The near clipping plane distance for the projection.
            far (float): The far clipping plane distance for the projection.
            projection_type (ProjectionType): The type of projection to use. Default is :attr:`fvdb.ProjectionType.PERSPECTIVE`.
            tile_size (int): The size of the tiles to use for rendering. Default is 16. You shouldn't set this parameter unless you really know what you are doing.
            min_radius_2d (float): The minimum radius (in pixels) below which Gaussians are ignored during rendering.
            eps_2d (float): A value used to pad Gaussians when projecting them onto the image plane, to avoid very projected Gaussians which create artifacts and
                numerical issues.
            antialias (bool): If ``True``, applies opacity correction to the projected Gaussians when using ``eps_2d > 0.0``.

        Returns:
            depth_images (torch.Tensor): A tensor of shape ``(C, H, W, 1)`` where ``C`` is the number of camera views,
                ``H`` is the height of the depth maps, and ``W`` is the width of the depth maps.
                Each element represents the depth value at that pixel in the depth map.
            alpha_images (torch.Tensor): A tensor of shape ``(C, H, W, 1)`` where ``C`` is the number of camera views,
                ``H`` is the height of the images, and ``W`` is the width of the images.
                Each element represents the alpha value (opacity) at a pixel such that ``0 <= alpha < 1``,
                and 0 means the pixel is fully transparent, and 1 means the pixel is fully opaque.
        """
        return self._impl.render_depths(
            world_to_camera_matrices=world_to_camera_matrices,
            projection_matrices=projection_matrices,
            image_width=image_width,
            image_height=image_height,
            near=near,
            far=far,
            projection_type=self._proj_type_to_cpp(projection_type),
            tile_size=tile_size,
            min_radius_2d=min_radius_2d,
            eps_2d=eps_2d,
            antialias=antialias,
            backgrounds=backgrounds,
        )

    def render_images(
        self,
        world_to_camera_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        image_width: int,
        image_height: int,
        near: float,
        far: float,
        projection_type=ProjectionType.PERSPECTIVE,
        sh_degree_to_use: int = -1,
        tile_size: int = 16,
        min_radius_2d: float = 0.0,
        eps_2d: float = 0.3,
        antialias: bool = False,
        backgrounds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Render ``C`` multi-channel images (see :attr:`num_channels`) from this :class:`GaussianSplat3d` from ``C`` camera views.

        .. note::

            All images being rendered must have the same width and height.


        Example:

        .. code-block:: python

            # Assume gaussian_splat_3d is an instance of GaussianSplat3d
            # Render images from C camera views.
            # images is a tensor of shape [C, H, W, D] where D is the number of channels
            # alpha_images is a tensor of shape [C, H, W, 1]
            images, alpha_images = gaussian_splat_3d.render_images(
                world_to_camera_matrices, # tensor of shape [C, 4, 4]
                projection_matrices, # tensor of shape [C, 3, 3]
                image_width, # width of the images
                image_height, # height of the images
                near, # near clipping plane
                far) # far clipping plane

        Args:
            world_to_camera_matrices (torch.Tensor): Tensor of shape ``(C, 4, 4)`` representing the
                world-to-camera transformation matrices for C cameras. Each matrix transforms points
                from world coordinates to camera coordinates.
            projection_matrices (torch.Tensor): Tensor of shape ``(C, 3, 3)`` representing the projection matrices for ``C`` cameras.
                Each matrix projects points in camera space into homogeneous pixel coordinates.
            image_width (int): The width of the images to be rendered. Note these are the same for all images being rendered.
            image_height (int): The height of the images to be rendered. Note these are the same for all images being rendered.
            near (float): The near clipping plane distance for the projection.
            far (float): The far clipping plane distance for the projection.
            projection_type (ProjectionType): The type of projection to use. Default is :attr:`fvdb.ProjectionType.PERSPECTIVE`.
            sh_degree_to_use (int): The degree of spherical harmonics to use for rendering. -1 means use all available SH bases.
                0 means use only the first SH base (constant color). Note that you can't use more SH bases than available in the GaussianSplat3d instance.
                Default is -1.
            tile_size (int): The size of the tiles to use for rendering. Default is 16. You shouldn't set this parameter unless you really know what you are doing.
            min_radius_2d (float): The minimum radius (in pixels) below which Gaussians are ignored during rendering.
            eps_2d (float): A value used to pad Gaussians when projecting them onto the image plane, to avoid very projected Gaussians which create artifacts and
                numerical issues.
            antialias (bool): If ``True``, applies opacity correction to the projected Gaussians when using ``eps_2d > 0.0``.

        Returns:
            images (torch.Tensor): A tensor of shape ``(C, H, W, D)`` where ``C`` is the number of camera views,
                ``H`` is the height of the images, ``W`` is the width of the images, and ``D`` is the number of channels.
            alpha_images (torch.Tensor): A tensor of shape ``(C, H, W, 1)`` where ``C`` is the number of camera views,
                ``H`` is the height of the images, and ``W`` is the width of the images.
                Each element represents the alpha value (opacity) at a pixel such that ``0 <= alpha < 1``,
                and 0 means the pixel is fully transparent, and 1 means the pixel is fully opaque.
        """
        return self._impl.render_images(
            world_to_camera_matrices=world_to_camera_matrices,
            projection_matrices=projection_matrices,
            image_width=image_width,
            image_height=image_height,
            near=near,
            far=far,
            projection_type=self._proj_type_to_cpp(projection_type),
            sh_degree_to_use=sh_degree_to_use,
            tile_size=tile_size,
            min_radius_2d=min_radius_2d,
            eps_2d=eps_2d,
            antialias=antialias,
            backgrounds=backgrounds,
        )

    def render_images_and_depths(
        self,
        world_to_camera_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        image_width: int,
        image_height: int,
        near: float,
        far: float,
        projection_type=ProjectionType.PERSPECTIVE,
        sh_degree_to_use: int = -1,
        tile_size: int = 16,
        min_radius_2d: float = 0.0,
        eps_2d: float = 0.3,
        antialias: bool = False,
        backgrounds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Render ``C`` multi-channel images (see :attr:`num_channels`) with depth as the last channel from this :class:`GaussianSplat3d` from ``C`` camera views.

        .. note::

            All images being rendered must have the same width and height.


        Example:

        .. code-block:: python

            # Assume gaussian_splat_3d is an instance of GaussianSplat3d
            # Render images with depth maps from C camera views.
            # images is a tensor of shape [C, H, W, D + 1] where D is the number of channels
            # alpha_images is a tensor of shape [C, H, W, 1]
            images, alpha_images = gaussian_splat_3d.render_images(
                world_to_camera_matrices, # tensor of shape [C, 4, 4]
                projection_matrices, # tensor of shape [C, 3, 3]
                image_width, # width of the images
                image_height, # height of the images
                near, # near clipping plane
                far) # far clipping plane

            images = images[..., :-1]  # Extract image channels

            depths = images[..., -1:] / alpha_images  # Extract depth channel by dividing by alpha

        Args:
            world_to_camera_matrices (torch.Tensor): Tensor of shape ``(C, 4, 4)`` representing the
                world-to-camera transformation matrices for C cameras. Each matrix transforms points
                from world coordinates to camera coordinates.
            projection_matrices (torch.Tensor): Tensor of shape ``(C, 3, 3)`` representing the projection matrices for ``C`` cameras.
                Each matrix projects points in camera space into homogeneous pixel coordinates.
            image_width (int): The width of the images to be rendered. Note these are the same for all images being rendered.
            image_height (int): The height of the images to be rendered. Note these are the same for all images being rendered.
            near (float): The near clipping plane distance for the projection.
            far (float): The far clipping plane distance for the projection.
            projection_type (ProjectionType): The type of projection to use. Default is :attr:`fvdb.ProjectionType.PERSPECTIVE`.
            sh_degree_to_use (int): The degree of spherical harmonics to use for rendering. -1 means use all available SH bases.
                0 means use only the first SH base (constant color). Note that you can't use more SH bases than available in the GaussianSplat3d instance.
                Default is -1.
            tile_size (int): The size of the tiles to use for rendering. Default is 16. You shouldn't set this parameter unless you really know what you are doing.
            min_radius_2d (float): The minimum radius (in pixels) below which Gaussians are ignored during rendering.
            eps_2d (float): A value used to pad Gaussians when projecting them onto the image plane, to avoid very projected Gaussians which create artifacts and
                numerical issues.
            antialias (bool): If ``True``, applies opacity correction to the projected Gaussians when using ``eps_2d > 0.0``.

        Returns:
            images (torch.Tensor): A tensor of shape ``(C, H, W, D + 1)`` where ``C`` is the number of camera views,
                ``H`` is the height of the images, ``W`` is the width of the images, and ``D`` is the number of channels.
            alpha_images (torch.Tensor): A tensor of shape ``(C, H, W, 1)`` where ``C`` is the number of camera views,
                ``H`` is the height of the images, and ``W`` is the width of the images.
                Each element represents the alpha value (opacity) at a pixel such that ``0 <= alpha < 1``,
                and 0 means the pixel is fully transparent, and 1 means the pixel is fully opaque.
        """
        return self._impl.render_images_and_depths(
            world_to_camera_matrices=world_to_camera_matrices,
            projection_matrices=projection_matrices,
            image_width=image_width,
            image_height=image_height,
            near=near,
            far=far,
            projection_type=self._proj_type_to_cpp(projection_type),
            sh_degree_to_use=sh_degree_to_use,
            tile_size=tile_size,
            min_radius_2d=min_radius_2d,
            eps_2d=eps_2d,
            antialias=antialias,
            backgrounds=backgrounds,
        )

    def render_num_contributing_gaussians(
        self,
        world_to_camera_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        image_width: int,
        image_height: int,
        near: float,
        far: float,
        projection_type=ProjectionType.PERSPECTIVE,
        tile_size: int = 16,
        min_radius_2d: float = 0.0,
        eps_2d: float = 0.3,
        antialias: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Renders ``C`` images where each pixel contains the number of contributing Gaussians for that pixel from ``C`` camera views.

        .. note::

            All images being rendered must have the same width and height.


        Example:

        .. code-block:: python

            # Assume gaussian_splat_3d is an instance of GaussianSplat3d
            # Render images from C camera views.
            # images is a tensor of shape [C, H, W, D] where D is the number of channels
            # alpha_images is a tensor of shape [C, H, W, 1]
            num_gaussians, alpha_images = gaussian_splat_3d.render_images(
                world_to_camera_matrices, # tensor of shape [C, 4, 4]
                projection_matrices, # tensor of shape [C, 3, 3]
                image_width, # width of the images
                image_height, # height of the images
                near, # near clipping plane
                far) # far clipping plane

            num_gaussians_cij = num_gaussians[c, i, j, 0]  # Number of contributing Gaussians at pixel (i, j) in camera c

        Args:
            world_to_camera_matrices (torch.Tensor): Tensor of shape ``(C, 4, 4)`` representing the
                world-to-camera transformation matrices for C cameras. Each matrix transforms points
                from world coordinates to camera coordinates.
            projection_matrices (torch.Tensor): Tensor of shape ``(C, 3, 3)`` representing the projection matrices for ``C`` cameras.
                Each matrix projects points in camera space into homogeneous pixel coordinates.
            image_width (int): The width of the images to be rendered. Note these are the same for all images being rendered.
            image_height (int): The height of the images to be rendered. Note these are the same for all images being rendered.
            near (float): The near clipping plane distance for the projection.
            far (float): The far clipping plane distance for the projection.
            projection_type (ProjectionType): The type of projection to use. Default is :attr:`fvdb.ProjectionType.PERSPECTIVE`.
            tile_size (int): The size of the tiles to use for rendering. Default is 16. You shouldn't set this parameter unless you really know what you are doing.
            min_radius_2d (float): The minimum radius (in pixels) below which Gaussians are ignored during rendering.
            eps_2d (float): A value used to pad Gaussians when projecting them onto the image plane, to avoid very projected Gaussians which create artifacts and
                numerical issues.
            antialias (bool): If ``True``, applies opacity correction to the projected Gaussians when using ``eps_2d > 0.0``.

        Returns:
            images (torch.Tensor): A tensor of shape ``(C, H, W, 1)`` where ``C`` is the number of camera views,
                ``H`` is the height of the images, ``W`` is the width of the images.
                Each element represents the number of contributing Gaussians at that pixel.
            alpha_images (torch.Tensor): A tensor of shape ``(C, H, W, 1)`` where ``C`` is the number of camera views,
                ``H`` is the height of the images, and ``W`` is the width of the images.
                Each element represents the alpha value (opacity) at a pixel such that ``0 <= alpha < 1``,
                and 0 means the pixel is fully transparent, and 1 means the pixel is fully opaque.
        """
        return self._impl.render_num_contributing_gaussians(
            world_to_camera_matrices=world_to_camera_matrices,
            projection_matrices=projection_matrices,
            image_width=image_width,
            image_height=image_height,
            near=near,
            far=far,
            projection_type=self._proj_type_to_cpp(projection_type),
            tile_size=tile_size,
            min_radius_2d=min_radius_2d,
            eps_2d=eps_2d,
            antialias=antialias,
        )

    @overload
    def sparse_render_num_contributing_gaussians(
        self,
        pixels_to_render: torch.Tensor,
        world_to_camera_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        image_width: int,
        image_height: int,
        near: float,
        far: float,
        projection_type=ProjectionType.PERSPECTIVE,
        tile_size: int = 16,
        min_radius_2d: float = 0.0,
        eps_2d: float = 0.3,
        antialias: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    @overload
    def sparse_render_num_contributing_gaussians(
        self,
        pixels_to_render: JaggedTensor,
        world_to_camera_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        image_width: int,
        image_height: int,
        near: float,
        far: float,
        projection_type=ProjectionType.PERSPECTIVE,
        tile_size: int = 16,
        min_radius_2d: float = 0.0,
        eps_2d: float = 0.3,
        antialias: bool = False,
    ) -> tuple[JaggedTensor, JaggedTensor]: ...

    def sparse_render_num_contributing_gaussians(
        self,
        pixels_to_render: JaggedTensor | torch.Tensor,
        world_to_camera_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        image_width: int,
        image_height: int,
        near: float,
        far: float,
        projection_type=ProjectionType.PERSPECTIVE,
        tile_size: int = 16,
        min_radius_2d: float = 0.0,
        eps_2d: float = 0.3,
        antialias: bool = False,
    ) -> tuple[JaggedTensor | torch.Tensor, JaggedTensor | torch.Tensor]:
        """
        Renders the number of Gaussians which contribute to each pixel specified in the input.

        .. seealso::

            :meth:`render_num_contributing_gaussians` for rendering dense images of contributing Gaussians.


        Args:
            pixels_to_render (torch.Tensor | JaggedTensor): A :class:`fvdb.JaggedTensor` of shape ``(C, R_c, 2)`` representing the
                pixels to render for each camera, where ``C`` is the number of camera views and ``R_c`` is the
                number of pixels to render per camera. Each value is an (x, y) pixel coordinate.
            world_to_camera_matrices (torch.Tensor): Tensor of shape ``(C, 4, 4)`` representing the
                world-to-camera transformation matrices for C cameras. Each matrix transforms points
                from world coordinates to camera coordinates.
            projection_matrices (torch.Tensor): Tensor of shape ``(C, 3, 3)`` representing the projection matrices for ``C`` cameras.
                Each matrix projects points in camera space into homogeneous pixel coordinates.
            image_width (int): The width of the images to be rendered. Note these are the same for all images being rendered.
            image_height (int): The height of the images to be rendered. Note these are the same for all images being rendered.
            near (float): The near clipping plane distance for the projection.
            far (float): The far clipping plane distance for the projection.
            projection_type (ProjectionType): The type of projection to use. Default is :attr:`fvdb.ProjectionType.PERSPECTIVE`.
            tile_size (int): The size of the tiles to use for rendering. Default is 16. You shouldn't set this parameter unless you really know what you are doing.
            min_radius_2d (float): The minimum radius (in pixels) below which Gaussians are ignored during rendering.
            eps_2d (float): A value used to pad Gaussians when projecting them onto the image plane, to avoid very projected Gaussians which create artifacts and
                numerical issues.
            antialias (bool): If ``True``, applies opacity correction to the projected Gaussians when using ``eps_2d > 0.0``.

        Returns:
            num_contributing_gaussians (torch.Tensor | JaggedTensor): A tensor of shape ``(C, R)`` (if this method was called with ``pixels_to_render`` as a :class:`torch.Tensor`)
                or a :class:`fvdb.JaggedTensor` of shape ``(C, R_c)`` (if this method was called with ``pixels_to_render`` as a :class:`fvdb.JaggedTensor`)
                where ``C`` is the number of cameras, and ``R``/``R_c`` is the number of pixels to render per camera.
                Each element represents the number of contributing Gaussians at that pixel.
            alphas (torch.Tensor | JaggedTensor): A tensor of shape ``(C, R)`` (if this method was called with ``pixels_to_render`` as a :class:`torch.Tensor`)
                or a :class:`fvdb.JaggedTensor` of shape ``(C, R_c)`` (if this method was called with ``pixels_to_render`` as a :class:`fvdb.JaggedTensor`)
                where ``C`` is the number of cameras, and ``R``/``R_c`` is the number of pixels to render per camera.
                Each element represents the alpha value (opacity) at that pixel such that ``0 <= alpha < 1``,
                and 0 means the pixel is fully transparent, and 1 means the pixel is fully opaque.
        """
        if isinstance(pixels_to_render, torch.Tensor):
            C, R, _ = pixels_to_render.shape
            tensors = [pixels_to_render[i] for i in range(C)]
            pixels_to_render_jagged = JaggedTensor(tensors)

            result_num_contributing_gaussians, result_alphas = self._impl.sparse_render_num_contributing_gaussians(
                pixels_to_render=pixels_to_render_jagged._impl,
                world_to_camera_matrices=world_to_camera_matrices,
                projection_matrices=projection_matrices,
                image_width=image_width,
                image_height=image_height,
                near=near,
                far=far,
                projection_type=self._proj_type_to_cpp(projection_type),
                tile_size=tile_size,
                min_radius_2d=min_radius_2d,
                eps_2d=eps_2d,
                antialias=antialias,
            )

            num_contributing_gaussians_list = result_num_contributing_gaussians.unbind()
            alphas_list = result_alphas.unbind()
            dense_num_contributing_gaussians = torch.stack(num_contributing_gaussians_list, dim=0)  # type: ignore # Shape: (C, R)
            dense_alphas = torch.stack(alphas_list, dim=0)  # type: ignore # Shape: (C, R)

            return dense_num_contributing_gaussians, dense_alphas
        else:
            # Already a JaggedTensor, call C++ implementation directly
            result_num_contributing_gaussians_impl, result_alphas_impl = (
                self._impl.sparse_render_num_contributing_gaussians(
                    pixels_to_render=pixels_to_render._impl,
                    world_to_camera_matrices=world_to_camera_matrices,
                    projection_matrices=projection_matrices,
                    image_width=image_width,
                    image_height=image_height,
                    near=near,
                    far=far,
                    projection_type=self._proj_type_to_cpp(projection_type),
                    tile_size=tile_size,
                    min_radius_2d=min_radius_2d,
                    eps_2d=eps_2d,
                    antialias=antialias,
                )
            )
            return JaggedTensor(impl=result_num_contributing_gaussians_impl), JaggedTensor(impl=result_alphas_impl)

    def render_contributing_gaussian_ids(
        self,
        world_to_camera_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        image_width: int,
        image_height: int,
        near: float,
        far: float,
        projection_type=ProjectionType.PERSPECTIVE,
        tile_size: int = 16,
        min_radius_2d: float = 0.0,
        eps_2d: float = 0.3,
        antialias: bool = False,
        top_k_contributors: int = 0,
    ) -> tuple[JaggedTensor, JaggedTensor]:
        """
        Render the IDs of the Gaussians that are the contributors to the rendered images' pixels and the value of their weighted contributions to the rendered pixels.

        Args:
            world_to_camera_matrices (torch.Tensor): Tensor of shape ``(C, 4, 4)`` representing the
                world-to-camera transformation matrices for C cameras. Each matrix transforms points
                from world coordinates to camera coordinates.
            projection_matrices (torch.Tensor): Tensor of shape ``(C, 3, 3)`` representing the projection matrices for ``C`` cameras.
                Each matrix projects points in camera space into homogeneous pixel coordinates.
            image_width (int): The width of the images to be rendered. Note these are the same for all images being rendered.
            image_height (int): The height of the images to be rendered. Note these are the same for all images being rendered.
            near (float): The near clipping plane distance for the projection.
            far (float): The far clipping plane distance for the projection.
            projection_type (ProjectionType): The type of projection to use. Default is :attr:`fvdb.ProjectionType.PERSPECTIVE`.
            tile_size (int): The size of the tiles to use for rendering. Default is 16. You shouldn't set this parameter unless you really know what you are doing.
            min_radius_2d (float): The minimum radius (in pixels) below which Gaussians are ignored during rendering.
            eps_2d (float): A value used to pad Gaussians when projecting them onto the image plane, to avoid very projected Gaussians which create artifacts and
                numerical issues.
            antialias (bool): If ``True``, applies opacity correction to the projected Gaussians when using ``eps_2d > 0.0``.
            top_k_contributors (int): If greater than 0, returns only the top ``k`` most opaque Gaussians contributing to each pixel.
                If 0 (default), returns all contributing Gaussians per pixel.

        Returns:
            ids (fvdb.JaggedTensor): A ``[[C1P1 + C1P2 + ... C1P(imageWidth * imageHeight), 1], ... [CNP1 + CNP2 + ... CNP(imageWidth * imageHeight), 1]]``
                jagged tensor containing the IDs of the contributing Gaussians of each rendered pixel for each camera.
            weights (fvdb.JaggedTensor): A ``[[C1P1 + C1P2 + ... C1P(imageWidth * imageHeight), 1], ... [CNP1 + CNP2 + ... CNP(imageWidth * imageHeight), 1]]``
                jagged tensor containing the weights of the contributing Gaussians of each rendered pixel for each camera. The weights are in row-major order and
                sum to 1 for each pixel if that pixel is opaque (alpha=1).
        """
        ids, weights = self._impl.render_contributing_gaussian_ids(
            world_to_camera_matrices=world_to_camera_matrices,
            projection_matrices=projection_matrices,
            image_width=image_width,
            image_height=image_height,
            near=near,
            far=far,
            projection_type=self._proj_type_to_cpp(projection_type),
            tile_size=tile_size,
            min_radius_2d=min_radius_2d,
            eps_2d=eps_2d,
            antialias=antialias,
            top_k_contributors=top_k_contributors,
        )
        return JaggedTensor(impl=ids), JaggedTensor(impl=weights)

    @overload
    def sparse_render_contributing_gaussian_ids(
        self,
        pixels_to_render: torch.Tensor,
        world_to_camera_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        image_width: int,
        image_height: int,
        near: float,
        far: float,
        projection_type=ProjectionType.PERSPECTIVE,
        tile_size: int = 16,
        min_radius_2d: float = 0.0,
        eps_2d: float = 0.3,
        antialias: bool = False,
        top_k_contributors: int = 0,
    ) -> tuple[JaggedTensor, JaggedTensor]: ...

    @overload
    def sparse_render_contributing_gaussian_ids(
        self,
        pixels_to_render: JaggedTensor,
        world_to_camera_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        image_width: int,
        image_height: int,
        near: float,
        far: float,
        projection_type=ProjectionType.PERSPECTIVE,
        tile_size: int = 16,
        min_radius_2d: float = 0.0,
        eps_2d: float = 0.3,
        antialias: bool = False,
        top_k_contributors: int = 0,
    ) -> tuple[JaggedTensor, JaggedTensor]: ...

    def sparse_render_contributing_gaussian_ids(
        self,
        pixels_to_render: JaggedTensor | torch.Tensor,
        world_to_camera_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        image_width: int,
        image_height: int,
        near: float,
        far: float,
        projection_type=ProjectionType.PERSPECTIVE,
        tile_size: int = 16,
        min_radius_2d: float = 0.0,
        eps_2d: float = 0.3,
        antialias: bool = False,
        top_k_contributors: int = 0,
    ) -> tuple[JaggedTensor, JaggedTensor]:
        """
        Render the IDs of the Gaussians that are the contributors to the rendered images'
        pixels and the value of their weighted contributions to the rendered pixels. This
        function will render only a sparse subset of the pixels in the overall image, as specified
        by the ``pixels_to_render`` parameter.

        Args:
            pixels_to_render (torch.Tensor | JaggedTensor): A :class:`torch.Tensor` of shape ``(C, R, 2)``
                or a :class:`fvdb.JaggedTensor` of shape ``(C, R_c, 2)`` representing the
                pixels to render for each camera, where ``C`` is the number of camera views and ``R``/``R_c`` is the
                number of pixels to render per camera. Each value is an (x, y) pixel coordinate.
            world_to_camera_matrices (torch.Tensor): Tensor of shape ``(C, 4, 4)`` representing the
                world-to-camera transformation matrices for ``C`` cameras. Each matrix transforms points
                from world coordinates to camera coordinates.
            projection_matrices (torch.Tensor): Tensor of shape ``(C, 3, 3)`` representing the projection matrices for ``C`` cameras.
                Each matrix projects points in camera space into homogeneous pixel coordinates.
            image_width (int): The width of the images to be rendered. Note these are the same for all images being rendered.
            image_height (int): The height of the images to be rendered. Note these are the same for all images being rendered.
            near (float): The near clipping plane distance for the projection.
            far (float): The far clipping plane distance for the projection.
            projection_type (ProjectionType): The type of projection to use. Default is :attr:`fvdb.ProjectionType.PERSPECTIVE`.
            tile_size (int): The size of the tiles to use for rendering. Default is 16. You shouldn't set this parameter unless you really know what you are doing.
            min_radius_2d (float): The minimum radius (in pixels) below which Gaussians are ignored during rendering.
            eps_2d (float): A value used to pad Gaussians when projecting them onto the image plane, to avoid very projected Gaussians which create artifacts and
                numerical issues.
            antialias (bool): If ``True``, applies opacity correction to the projected Gaussians when using ``eps_2d > 0.0``.
            top_k_contributors (int): If greater than 0, returns only the top ``k`` most opaque Gaussians contributing to each pixel,
                If 0 (default), returns all contributing Gaussians per pixel.

        Returns:
            ids (fvdb.JaggedTensor): A ``[[C1P1 + C1P2 + ... C1PN1, 1], ... [CNP1 + CNP2 + ... CNPNN, 1]]`` jagged tensor
                containing the IDs of the contributing Gaussians of each rendered pixel for each camera. The IDs are in row-major order.
            weights (fvdb.JaggedTensor): A ``[[C1P1 + C1P2 + ... C1PN1, 1], ... [CNP1 + CNP2 + ... CNPNN, 1]]`` jagged tensor
                containing the weights of the contributing Gaussians of each rendered pixel for each camera. The weights are in row-major order and sum to 1 for each pixel if that pixel is opaque (alpha=1).
        """
        if isinstance(pixels_to_render, torch.Tensor):
            C, R, _ = pixels_to_render.shape
            tensors = [pixels_to_render[i] for i in range(C)]
            pixels_to_render_jagged = JaggedTensor(tensors)

            result_ids, result_weights = self._impl.sparse_render_contributing_gaussian_ids(
                pixels_to_render=pixels_to_render_jagged._impl,
                world_to_camera_matrices=world_to_camera_matrices,
                projection_matrices=projection_matrices,
                image_width=image_width,
                image_height=image_height,
                near=near,
                far=far,
                projection_type=self._proj_type_to_cpp(projection_type),
                tile_size=tile_size,
                min_radius_2d=min_radius_2d,
                eps_2d=eps_2d,
                antialias=antialias,
                top_k_contributors=top_k_contributors,
            )

            return JaggedTensor(impl=result_ids), JaggedTensor(impl=result_weights)
        else:
            # Already a JaggedTensor, call C++ implementation directly
            result_ids_impl, result_weights_impl = self._impl.sparse_render_contributing_gaussian_ids(
                pixels_to_render=pixels_to_render._impl,
                world_to_camera_matrices=world_to_camera_matrices,
                projection_matrices=projection_matrices,
                image_width=image_width,
                image_height=image_height,
                near=near,
                far=far,
                projection_type=self._proj_type_to_cpp(projection_type),
                tile_size=tile_size,
                min_radius_2d=min_radius_2d,
                eps_2d=eps_2d,
                antialias=antialias,
                top_k_contributors=top_k_contributors,
            )
            return JaggedTensor(impl=result_ids_impl), JaggedTensor(impl=result_weights_impl)

    def reset_accumulated_gradient_state(self) -> None:
        """
        Reset the accumulated projected gradients of the mans if :attr:`accumulate_mean_2d_gradients` is ``True``,
        and the accumulated max 2D radii if :attr:`accumulate_max_2d_radii` is ``True``.

        The values of :attr:`accumulated_projected_mean_2d_gradients`, :attr:`accumulated_max_2d_radii`,
        and :attr:`accumulated_gradient_step_counts` will be zeroed out after this call.

        .. seealso::
            :meth:`accumulate_mean_2d_gradients` :meth:`accumulate_max_2d_radii` which control if
            we accumulate these values during rendering and backward passes.

        .. seealso::
            :attr:`accumulated_mean_2d_gradient_norms` :attr:`accumulated_max_2d_radii` :attr:`accumulated_gradient_step_counts`
            for the actual accumulated state being reset.

        """
        self._impl.reset_accumulated_gradient_state()

    def save_ply(
        self, filename: pathlib.Path | str, metadata: Mapping[str, str | int | float | torch.Tensor] | None = None
    ) -> None:
        """
        Save this :class:`GaussianSplat3d` to a PLY file. and include any metadata provided.

        Args:
            filename (pathlib.Path | str): The path to the PLY file to save.
            metadata (dict[str, str | int | float | torch.Tensor] | None): An optional dictionary of metadata
                where the keys are strings and the values are either strings, ints, floats, or tensors. Defaults to ``None``,
        """
        if isinstance(filename, pathlib.Path):
            filename = str(filename)
        self._impl.save_ply(filename, metadata)  # type: ignore -- mapping to dict is fine here

    @overload
    def to(self, dtype: torch.dtype | None = None) -> "GaussianSplat3d": ...

    @overload
    def to(
        self,
        device: DeviceIdentifier | None = None,
        dtype: torch.dtype | None = None,
    ) -> "GaussianSplat3d": ...

    @overload
    def to(
        self,
        other: torch.Tensor,
    ) -> "GaussianSplat3d": ...

    @overload
    def to(
        self,
        other: "GaussianSplat3d",
    ) -> "GaussianSplat3d": ...

    @overload
    def to(
        self,
        other: Grid,
    ) -> "GaussianSplat3d": ...

    @overload
    def to(
        self,
        other: GridBatch,
    ) -> "GaussianSplat3d": ...

    @overload
    def to(
        self,
        other: JaggedTensor,
    ) -> "GaussianSplat3d": ...

    def to(
        self,
        *args,
        **kwargs,
    ) -> "GaussianSplat3d":
        """
        Move the :class:`GaussianSplat3d` instance to a different device or change its data type or both.

        Args:
            other (DeviceIdentifier | torch.Tensor | GaussianSplat3d | Grid | GridBatch | JaggedTensor):
                The target :class:`torch.Device`, :class:`torch.Tensor`, :class:`~fvdb.Grid`,
                :class:`~fvdb.GridBatch`, :class:`~fvdb.JaggedTensor`, or :class:`~fvdb.GaussianSplat3d` instance
                to which the :class:`GaussianSplat3d` instance should be moved.
            device (DeviceIdentifier, optional): The target ``device`` to move the :class:`GaussianSplat3d` instance to.
            dtype (torch.dtype, optional): The target data type for the :class:`GaussianSplat3d` instance.

        Returns:
           gaussian_splat_3d (GaussianSplat3d): A new instance of :class:`GaussianSplat3d` with the specified device and/or data type.
        """

        # All values passed by keyword arguments
        if len(args) == 0:
            if len(kwargs) == 1:
                # .to(device=...) or .to(other=...)
                if "device" in kwargs:
                    device = kwargs["device"]
                    dtype = kwargs.get("dtype", self.dtype)
                elif "other" in kwargs:
                    other = kwargs["other"]
                    if isinstance(other, (torch.Tensor, JaggedTensor, GaussianSplat3d)):
                        device = other.device
                        dtype = other.dtype
                    elif isinstance(other, (Grid, GridBatch)):
                        device = other.device
                        dtype = self.dtype
                else:
                    raise TypeError(
                        f"Invalid keyword arguments for to(): {kwargs}. Expected 'device' or 'other' and optionally 'dtype'."
                    )
            elif len(kwargs) == 2:
                # .to(device=..., dtype=...) or .to(dtype=..., device=...)
                if "device" in kwargs and "dtype" in kwargs:
                    device = kwargs["device"]
                    dtype = kwargs["dtype"]
                else:
                    raise TypeError(
                        f"Invalid keyword arguments for to(): {kwargs}. Expected 'device' or 'other' and optionally 'dtype'."
                    )
            else:
                raise TypeError(
                    f"Invalid keyword arguments for to(): {kwargs}. Expected 'device' or 'other' and optionally 'dtype'."
                )

        elif len(args) == 1 and isinstance(args[0], (torch.Tensor, GaussianSplat3d, JaggedTensor)):
            # .to(other)
            device = args[0].device
            dtype = args[0].dtype
        elif len(args) == 1 and isinstance(args[0], (Grid, GridBatch)):
            # .to(other)
            device = args[0].device
            dtype = self.dtype
        elif len(args) == 1:
            # .to(device)
            device = args[0]
            dtype = kwargs.get("dtype", self.dtype)
        elif len(args) == 2:
            # .to(device, dtype)
            device = args[0]
            dtype = args[1]
        else:
            raise TypeError(
                f"Invalid arguments for to(): {args}. Expected a DeviceIdentifier, torch.Tensor, GaussianSplat3d, Grid, GridBatch, or JaggedTensor."
            )

        device = resolve_device(device, inherit_from=self)
        dtype = self.dtype if dtype is None else cast_check(dtype, torch.dtype, "dtype")

        return GaussianSplat3d(
            impl=self._impl.to(
                device=device,
                dtype=dtype,
            ),
            _private=GaussianSplat3d.__PRIVATE__,
        )

    def set_state(
        self,
        means: torch.Tensor,
        quats: torch.Tensor,
        log_scales: torch.Tensor,
        logit_opacities: torch.Tensor,
        sh0: torch.Tensor,
        shN: torch.Tensor,
    ) -> None:
        """
        Set the underlying tensors managed by this :class:`GaussianSplat3d` instance.

        Note: If :attr:`accumulate_mean_2d_gradients` and/or :attr:`accumulate_max_2d_radii` are ``True``,
        this method will reset the gradient state (see :meth:`reset_accumulated_gradient_state`).

        Args:
            means (torch.Tensor): Tensor of shape ``(N, 3)`` representing the means of the Gaussians.
                ``N`` is the number of Gaussians (see :attr:`num_gaussians`).
            quats (torch.Tensor): Tensor of shape ``(N, 4)`` representing the quaternions of the Gaussians.
                ``N`` is the number of Gaussians (see :attr:`num_gaussians`).
            log_scales (torch.Tensor): Tensor of shape ``(N, 3)`` representing the log scales of the Gaussians.
                ``N`` is the number of Gaussians (see :attr:`num_gaussians`).
            logit_opacities (torch.Tensor): Tensor of shape ``(N,)`` representing the logit opacities of the Gaussians.
                ``N`` is the number of Gaussians (see :attr:`num_gaussians`).
            sh0 (torch.Tensor): Tensor of shape ``(N, 1, D)`` representing the diffuse SH coefficients
                where ``N`` is the number of Gaussians (see :attr:`num_gaussians`), and ``D`` is the number of channels (see :attr:`num_channels`).
            shN (torch.Tensor): Tensor of shape ``(N, K-1, D)`` representing the directionally
                varying SH coefficients where ``N`` is the number of Gaussians (see :attr:`num_gaussians`),
                ``D`` is the number of channels (see :attr:`num_channels`),
                and ``K`` is the number of spherical harmonic bases (see :attr:`num_sh_bases`).
        """
        self._impl.set_state(
            means=means,
            quats=quats,
            log_scales=log_scales,
            logit_opacities=logit_opacities,
            sh0=sh0,
            shN=shN,
        )

    def state_dict(self) -> dict[str, torch.Tensor]:
        """
        Return a dictionary containing the state of the GaussianSplat3d instance.
        This is useful for serializing the state of the object for saving or transferring.

        A state dictionary always contains the following keys where ``N`` denotes the number of Gaussians (see :attr:`num_gaussians`):

        - ``'means'``: Tensor of shape ``(N, 3)`` representing the means of the Gaussians.
        - ``'quats'``: Tensor of shape ``(N, 4)`` representing the quaternions of the Gaussians.
        - ``'log_scales'``: Tensor of shape ``(N, 3)`` representing the log scales of the Gaussians.
        - ``'logit_opacities'``: Tensor of shape ``(N,)`` representing the logit opacities of the Gaussians.
        - ``'sh0'``: Tensor of shape ``(N, 1, D)`` representing the diffuse SH coefficients
          where ``D`` is the number of channels (see :attr:`num_channels`).
        - ``'shN'``: Tensor of shape ``(N, K-1, D)`` representing the directionally varying SH
          coefficients where ``D`` is the number of channels (see :attr:`num_channels`), and ``K``
          is the number of spherical harmonic bases (see :attr:`num_sh_bases`).
        - ``'accumulate_max_2d_radii'``: bool Tensor with a single element indicating
          whether to track the maximum 2D radii for gradients.
        - ``'accumulate_mean_2d_gradients'``: bool Tensor with a single element indicating whether
          to track the average norm of the gradient of projected means for each Gaussian.

        It can also optionally contain the following keys if :attr:`accumulate_mean_2d_gradients` and/or :attr:`accumulate_max_2d_radii` are set to ``True``:

        - ``'accumulated_gradient_step_counts'``: Tensor of shape ``(N,)`` representing the
          accumulated gradient step counts for each Gaussian.
        - ``'accumulated_max_2d_radii'``: Tensor of shape ``(N,)`` representing the maximum
          2D projected radius for each Gaussian across every iteration of optimization.
        - ``'accumulated_mean_2d_gradient_norms'``: Tensor of shape ``(N,)`` representing the
          average norm of the gradient of projected means for each Gaussian across every iteration of optimization.


        .. seealso:: :meth:`from_state_dict` for constructing a :class:`GaussianSplat3d` from a state dictionary.

        Returns:
            state_dict (dict[str, torch.Tensor]): A dictionary containing the state of
                the :class:`GaussianSplat3d` instance.
        """
        return self._impl.state_dict()

    @staticmethod
    def _proj_type_from_cpp(proj_type: GaussianSplat3dCpp.ProjectionType) -> ProjectionType:
        if proj_type == GaussianSplat3dCpp.ProjectionType.PERSPECTIVE:
            return ProjectionType.PERSPECTIVE
        elif proj_type == GaussianSplat3dCpp.ProjectionType.ORTHOGRAPHIC:
            return ProjectionType.ORTHOGRAPHIC
        else:
            raise ValueError(f"Invalid projection type: {proj_type}")

    @staticmethod
    def _proj_type_to_cpp(proj_type: ProjectionType) -> GaussianSplat3dCpp.ProjectionType:
        if proj_type == ProjectionType.PERSPECTIVE:
            return GaussianSplat3dCpp.ProjectionType.PERSPECTIVE
        elif proj_type == ProjectionType.ORTHOGRAPHIC:
            return GaussianSplat3dCpp.ProjectionType.ORTHOGRAPHIC
        else:
            raise ValueError(f"Invalid projection type: {proj_type}")
