# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import math
from typing import overload

import torch

from . import JaggedTensor
from ._Cpp import GaussianSplat3d as GaussianSplat3dCpp
from ._Cpp import ProjectedGaussianSplats


class GaussianSplat3d:

    @overload
    def __init__(
        self,
        means: torch.Tensor,
        quats: torch.Tensor,
        log_scales: torch.Tensor,
        logit_opacities: torch.Tensor,
        sh0: torch.Tensor,
        shN: torch.Tensor,
        requires_grad: bool = False,
    ) -> None:
        """
        Initializes the GaussianSplat3d with the provided parameters.

        Args:
            means (torch.Tensor): Tensor of shape (N, 3) representing the means of the gaussians.
            quats (torch.Tensor): Tensor of shape (N, 4) representing the quaternions (orientations) of the gaussians.
            log_scales (torch.Tensor): Tensor of shape (N, 3) representing the log scales of the gaussians.
            logit_opacities (torch.Tensor): Tensor of shape (N,) representing the logit opacities of the gaussians.
            sh0 (torch.Tensor): Tensor of shape (N, 1, D) representing the diffuse SH coefficients
                where D is the number of channels (see `num_channels`).
            shN (torch.Tensor): Tensor of shape (N, K-1, D) representing the directionally
                varying SH coefficients where D is the number of channels (see `num_channels`),
                and K is the number of spherical harmonic bases (see `num_sh_bases`).
            requires_grad (bool): If True, gradients will be computed for these parameters. Default is False.

        Raises:
            TypeError: If the provided tensors do not match the expected types.
            ValueError: If the number of gaussians (N) is not consistent across the provided tensors
                or if the tensors are incorrectly sized.
        """

    @overload
    def __init__(self, impl: GaussianSplat3dCpp) -> None:
        """
        Initializes the GaussianSplat3d with an existing C++ implementation.
        This constructor is used to wrap an existing instance of GaussianSplat3dCpp.
        It is only called internally within this class and should not be used directly.

        Args:
            impl (GaussianSplat3dCpp): An instance of the C++ implementation.

        Raises:
            TypeError: If the provided impl is not an instance of GaussianSplat3dCpp
                or if the arguments do not match the expected types.
            ValueError: If the impl is not provided or is None.
        """
        ...

    @overload
    def __init__(self, num_channels: int = 3, device=torch.device("cpu")) -> None:
        """
        Initializes an empty GaussianSplat3d instance with a given number of feature channels.
        This constructor is used to create an empty instance of GaussianSplat3d on the specified device.

        Args:
            num_channels (int): The number of feature channels for the Gaussians.
            device (torch.device): The device on which the GaussianSplat3d instance will be created.
                Default is torch.device("cpu"). This can be set to a GPU device if available.

        """
        ...

    @staticmethod
    def _make_empty_gaussian_splat_cpp(
        num_channels: int = 3, device: torch.device = torch.device("cpu")
    ) -> GaussianSplat3dCpp:
        return GaussianSplat3dCpp(
            means=torch.empty(0, 3, device=device),
            quats=torch.empty(0, 4, device=device),
            log_scales=torch.empty(0, 3, device=device),
            logit_opacities=torch.empty(0, device=device),
            sh0=torch.empty(0, 1, num_channels, device=device),
            shN=torch.empty(0, 0, num_channels, device=device),
            requires_grad=False,
        )

    def __init__(self, *args, **kwargs) -> None:
        if len(args) == 1 and len(kwargs) == 0:
            if isinstance(args[0], GaussianSplat3dCpp):
                self._impl = args[0]
                return
        elif len(args) == 0 and len(kwargs) == 1:
            if "impl" in kwargs:
                if isinstance(kwargs["impl"], GaussianSplat3dCpp):
                    self._impl = kwargs["impl"]
                    return

        try:
            self._impl = GaussianSplat3d._make_empty_gaussian_splat_cpp(*args, **kwargs)
        except TypeError:
            try:
                self._impl = GaussianSplat3dCpp(*args, **kwargs)
            except TypeError:
                # TODO: More informative error message
                raise TypeError("Invalid constructor arguments for GaussianSplat3d.")

    @property
    def device(self) -> torch.device:
        """
        Returns the device on which the GaussianSplat3d instance is stored.
        This is typically used to determine whether the instance is on CPU or GPU.

        Returns:
            torch.device: The device of the GaussianSplat3d instance.
        """
        return self._impl.means.device

    @property
    def dtype(self) -> torch.dtype:
        """
        Returns the data type of the GaussianSplat3d instance.
        This is typically used to determine whether the instance is using float32, float64, etc.

        Returns:
            torch.dtype: The data type of the GaussianSplat3d instance.
        """
        return self._impl.means.dtype

    @property
    def sh_degree(self) -> int:
        """
        Returns the degree of the spherical harmonics used in the Gaussian splatting representation.
        This is typically used to determine the complexity of the spherical harmonics representation.

        Note: this is **not** the same as `num_sh_bases` but is related. The degree of the
        spherical harmonics is defined as `sqrt(num_sh_bases) - 1`.

        Returns:
            int: The degree of the spherical harmonics.
        """
        return int(math.sqrt(self._impl.num_sh_bases)) - 1

    @staticmethod
    def from_state_dict(state_dict: dict[str, torch.Tensor]) -> "GaussianSplat3d":
        """
        Creates a GaussianSplat3d instance from a state dictionary.
        This method is typically used to load a saved state of the GaussianSplat3d instance.
        Args:
            state_dict (dict[str, torch.Tensor]): A dictionary containing the state of the GaussianSplat3d instance. Here N denotes the number of Gaussians (see `num_gaussians`).
                The keys should match the expected state variables of the GaussianSplat3d. At the minimum, it should contain:
                - 'means': Tensor of shape (N, 3) representing the means of the Gaussians.
                - 'quats': Tensor of shape (N, 4) representing the quaternions of the Gaussians.
                - 'log_scales': Tensor of shape (N, 3) representing the log scales of the Gaussians.
                - 'logit_opacities': Tensor of shape (N,) representing the logit opacities of the Gaussians.
                - 'sh0': Tensor of shape (N, 1, D) representing the diffuse SH coefficients where D is the number of channels (see `num_channels`).
                - 'shN': Tensor of shape (N, K-1, D) representing the directionally varying SH coefficients where D is the number of channels (see `num_channels`), and K is the number of spherical harmonic bases (see `num_sh_bases`).
                - 'requires_grad': bool Tensor with a single element indicating whether gradients should be computed for these parameters.
                - 'track_max_2d_radii_for_grad': bool Tensor with a single element indicating whether to track the maximum 2D radii for gradients.
                - 'track_max_2d_radii_for_grad': bool Tensor with a single element indicating whether to track the average norm of the gradient of projected means for each Gaussian.
                It can optionally contain:
                - 'accumulated_gradient_step_counts_for_grad': Tensor of shape (N,) representing the accumulated gradient step counts for each Gaussian.
                - 'accumulated_max_2d_radii_for_grad': Tensor of shape (N,) representing the maximum 2D projected radius for each Gaussian across every iteration of optimization.
                - 'accumulated_mean_2d_gradient_norms_for_grad': Tensor of shape (N,) representing the average norm of the gradient of projected means for each Gaussian across every iteration of optimization.

        Returns:
            GaussianSplat3d: An instance of GaussianSplat3d initialized with the provided state dictionary.
        """
        return GaussianSplat3d(GaussianSplat3dCpp.from_state_dict(state_dict))

    @property
    def num_channels(self) -> int:
        """
        Returns the number of channels in the Gaussian splatting representation.
        For example, if you are rendering RGB images, this methodß will return 3.

        Returns:
            int: The number of channels.
        """
        return self._impl.num_channels

    @property
    def num_gaussians(self) -> int:
        """
        Returns the number of Gaussians in the Gaussian splatting representation.
        This is the total number of individual gaussian splats that are being used to represent the scene.

        Returns:
            int: The number of Gaussians.
        """
        return self._impl.num_gaussians

    @property
    def num_sh_bases(self) -> int:
        """
        Returns the number of spherical harmonics (SH) bases used in the Gaussian splatting representation.

        Returns:
            int: The number of spherical harmonics bases.
        """
        return self._impl.num_sh_bases

    @property
    def log_scales(self) -> torch.Tensor:
        """
        Returns the log scales of the Gaussians in the Gaussian splatting representation.
        The log scales encode the diagonal of the covariance matrix for each gaussian,
        representing the spread of the gaussian in 3D space.

        Note: each Gaussians' covariance is defined as :math:`R(q)^T S R(q)` where :math:`R(q)` is
        rotation matrix defined by the unit quaternion of the Gaussian (see `quats`)
        and :math:`S` = `diag(exp(log_scales))`

        Note: We store the log of the scales to ensure numerical stability.

        Note: To read the scales, see the `scales` property (which is read-only).

        Returns:
            torch.Tensor: A tensor of shape (N, 3) where N is the number
                of Gaussians (see `num_gaussians`). Each row represents the log of the scale of a Gaussian in 3D space.
        """
        return self._impl.log_scales

    @log_scales.setter
    def log_scales(self, value: torch.Tensor) -> None:
        """
        Sets the log scales of the Gaussians in the Gaussian splatting representation.
        The log scales encode the diagonal of the covariance matrix for each gaussian,
        representing the spread of the gaussian in 3D space.

        Note: each Gaussians' covariance is defined as :math:`R(q)^T S R(q)` where :math:`R(q)` is
        rotation matrix defined by the unit quaternion of the Gaussian (see `quats`)
        and :math:`S` = `diag(exp(log_scales))`

        Note: We store the log of the scales to ensure numerical stability.

        Note: To read the scales, see the `scales` property (which is read-only).

        Args:
            value (torch.Tensor): A tensor of shape (N, 3) where N is the number
                of Gaussians (see `num_gaussians`). Each row represents the log of the scale of a gaussian in 3D space.
        Raises:
            TypeError: If the provided value is not a torch.Tensor.
            ValueError: If the shape of the provided value does not match the expected shape (N, 3).
        """
        if not isinstance(value, torch.Tensor):
            raise TypeError("Expected 'value' to be a torch.Tensor.")
        self._impl.log_scales = value

    @property
    def logit_opacities(self) -> torch.Tensor:
        """
        Return the logit (inverse of sigmoid) of the opacities of the Gaussians in the scene.

        Note: We store the logit of the opacities to ensure numerical stability.

        Note: To read the opacities, see the `opacities` property (which is read-only).

        Returns:
            torch.Tensor: A tensor of shape (N, 3) where N is the number
                of Gaussians (see `num_gaussians`). Each row represents the logit of the opacity of a Gaussian in 3D space.
        """
        return self._impl.logit_opacities

    @logit_opacities.setter
    def logit_opacities(self, value: torch.Tensor) -> None:
        """
        Sets the logit (inverse of sigmoid) of the opacities of the Gaussians in the scene.

        Note: We store the logit of the opacities to ensure numerical stability.

        Note: To read the opacities, see the `opacities` property (which is read-only).

        Args:
            value (torch.Tensor): A tensor of shape (N,) where N is the number of Gaussians (see `num_gaussians`).
                Each element represents the logit of the opacity of a gaussian.

        Raises:
            TypeError: If the provided value is not a torch.Tensor.
            ValueError: If the shape of the provided value does not match the expected shape (N,).
        """
        if not isinstance(value, torch.Tensor):
            raise TypeError("Expected 'value' to be a torch.Tensor.")
        self._impl.logit_opacities = value

    @property
    def means(self) -> torch.Tensor:
        """
        Return the mean (3d position) of each Gaussian in the scene.
        The means represent the center of each Gaussian in 3D space.

        Returns:
            torch.Tensor: A tensor of shape (N, 3) where N is the number
                of Gaussians (see `num_gaussians`). Each row represents the mean of a Gaussian in 3D space.
        """
        return self._impl.means

    @means.setter
    def means(self, value: torch.Tensor) -> None:
        """
        Sets the means of the Gaussians in the scene.
        The means represent the center of each gaussian in 3D space.

        Args:
            value (torch.Tensor): A tensor of shape (N, 3) where N is the number
                of Gaussians (see `num_gaussians`). Each row represents the mean of a Gaussian in 3D space.

        Raises:
            TypeError: If the provided value is not a torch.Tensor.
            ValueError: If the shape of the provided value does not match the expected shape (N, 3).
        """
        if not isinstance(value, torch.Tensor):
            raise TypeError("Expected 'value' to be a torch.Tensor.")
        self._impl.means = value

    @property
    def quats(self) -> torch.Tensor:
        """
        Returns the unit quaternion representing the orientation of each Gaussian in the scene.

        The quaternions define the rotation component of the covariance matrix for each Gaussian.

        Note: each Gaussians' covariance is defined as :math:`R(q)^T S R(q)` where :math:`R(q)` is
        rotation matrix defined by the unit quaternion :math:`q` for the Gaussian
        and :math:`S` = `diag(exp(log_scales))`

        Returns:
            torch.Tensor: A tensor of shape (N, 4) where N is the number
                of Gaussians (see `num_gaussians`). Each row represents the unit quaternion of a Gaussian in 3D space.
        """
        return self._impl.quats

    @quats.setter
    def quats(self, value: torch.Tensor) -> None:
        """
        Sets the unit quaternions representing the orientation of each Gaussian in the scene.

        The quaternions define the rotation component of the covariance matrix for each Gaussian.

        Note: each Gaussians' covariance is defined as :math:`R(q)^T S R(q)` where :math:`R(q)` is
        rotation matrix defined by the unit quaternion :math:`q` for the Gaussian
        and :math:`S` = `diag(exp(log_scales))`

        Args:
            value (torch.Tensor): A tensor of shape (N, 4) where N is the number
                of Gaussians (see `num_gaussians`). Each row represents the unit quaternion of a Gaussian in 3D space.

        Raises:
            TypeError: If the provided value is not a torch.Tensor.
            ValueError: If the shape of the provided value does not match the expected shape (N, 4).
        """
        if not isinstance(value, torch.Tensor):
            raise TypeError("Expected 'value' to be a torch.Tensor.")
        self._impl.quats = value

    @property
    def requires_grad(self) -> bool:
        """
        Returns whether the GaussianSplat3d instance requires gradients.
        This is typically set to True if you want to optimize the parameters of the Gaussians.

        Returns:
            bool: True if gradients are required, False otherwise.
        """
        return self._impl.requires_grad

    @requires_grad.setter
    def requires_grad(self, value: bool) -> None:
        """
        Sets whether the GaussianSplat3d instance requires gradients.
        This is typically set to True if you want to optimize the parameters of the Gaussians.

        Args:
            value (bool): True if gradients are required, False otherwise.

        Raises:
            TypeError: If the provided value is not a boolean.
        """
        if not isinstance(value, bool):
            raise TypeError("Expected 'value' to be a boolean.")
        self._impl.requires_grad = value

    @property
    def sh0(self) -> torch.Tensor:
        """
        Returns the diffuse spherical harmonics coefficients of the Gaussians in the scene.
        These coefficients are used to represent the diffuse color/feature of each Gaussian.

        Returns:
            torch.Tensor: A tensor of shape (N, 1, D) where N is the number
                of Gaussians (see `num_gaussians`), and D is the number of channels (see `num_channels`).
                Each row represents the diffuse SH coefficients for a Gaussian.
        """
        return self._impl.sh0

    @sh0.setter
    def sh0(self, value: torch.Tensor) -> None:
        """
        Sets the diffuse spherical harmonics coefficients of the Gaussians in the scene.
        These coefficients are used to represent the diffuse color/feature of each Gaussian.

        Args:
            value (torch.Tensor): A tensor of shape (N, 1, D) where N is the number of Gaussians (see `num_gaussians`),
                and D is the number of channels (see `num_channels`).
                Each row represents the diffuse SH coefficients for a Gaussian.

        Raises:
            TypeError: If the provided value is not a torch.Tensor.
            ValueError: If the shape of the provided value does not match the expected shape (N, 1, D).
        """
        if not isinstance(value, torch.Tensor):
            raise TypeError("Expected 'value' to be a torch.Tensor.")
        self._impl.sh0 = value

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
        Sets the directionally varying spherical harmonics coefficients of the Gaussians in the scene.
        These coefficients are used to represent a direction dependent color/feature of each Gaussian.

        Args:
            value (torch.Tensor): A tensor of shape (N, K-1, D) where N is the number
                of Gaussians (see `num_gaussians`), D is the number of channels (see `num_channels`),
                and K is the number of spherical harmonic bases (see `num_sh_bases`).
                Each row represents the directionally varying SH coefficients for a Gaussian.

        Raises:
            TypeError: If the provided value is not a torch.Tensor.
            ValueError: If the shape of the provided value does not match the expected shape (N, K-1, D).
        """
        if not isinstance(value, torch.Tensor):
            raise TypeError("Expected 'value' to be a torch.Tensor.")
        self._impl.shN = value

    @property
    def opacities(self) -> torch.Tensor:
        """
        Returns the opacities of the Gaussians in the Gaussian splatting representation.
        The opacities encode the visibility of each Gaussian in the scene.

        Note: This property is read only. We store the logit (inverse of sigmoid) of the opacities to ensure numerical stability.

        Note: To change the opacities, see the `logit_opacities` property.

        Returns:
            torch.Tensor: A tensor of shape (N,) where N is the number of Gaussians (see `num_gaussians`).
                Each element represents the opacity of a Gaussian.
        """
        return self._impl.opacities

    @property
    def scales(self) -> torch.Tensor:
        """
        Returns the scales of the Gaussians in the Gaussian splatting representation.
        The scales encode the diagonal of the covariance matrix for each Gaussian,
        representing the spread of the Gaussian in 3D space.

        Note: This property is read only. We store the log of the scales to ensure numerical stability.

        Note: To change the scales, see the `log_scales` property.


        Returns:
            torch.Tensor: A tensor of shape (N, 3) where N is the number
                of Gaussians. Each row represents the log scale of a Gaussian in 3D space.
        """
        return self._impl.scales

    @property
    def accumulated_gradient_step_counts_for_grad(self) -> torch.Tensor:
        """
        Returns the accumulated gradient step counts for each Gaussian.

        The gradient step counts are accumulated during the backward pass of the projection step
        of the pipeline, so this variable effectively counts the number of times each Gaussian has
        been projected.
        This tensor is used to track how many gradient steps have been applied to each Gaussian,
        which is useful for various optimization techniques.

        Note: To reset the counts, you can call the `reset_grad_state` method.

        Note: This property will be empty if the `requires_grad` flag was not set to True when initializing the GaussianSplat3d instance.

        Returns:
            torch.Tensor: A tensor of shape (N,) where N is the number of Gaussians (see `num_gaussians`).
                Each element represents the accumulated gradient step count for a Gaussian.
        """
        return self._impl.accumulated_gradient_step_counts_for_grad

    @property
    def accumulated_max_2d_radii_for_grad(self) -> torch.Tensor:
        """
        Returns the maximum 2D projected radius for each Gaussian across every iteration of optimization.
        This is used by certain optimization techniques to ensure that the Gaussians
        do not become too large or too small during the optimization process.
        This tensor is used to track the maximum 2D radius a Gaussian has been projected to during the optimization process.

        Note: To reset the maximum radii, you can call the `reset_grad_state` method.

        Note: This property will be empty if the `requires_grad` flag was not set to True when initializing the GaussianSplat3d instance.

        Note: This property will be empty if `track_max_2d_radii_for_grad` is set to False.

        Returns:
            torch.Tensor: A tensor of shape (N,) where N is the number of Gaussians (see `num_gaussians`).
                Each element represents the maximum 2D radius for a Gaussian across all optimization iterations.

        """
        return self._impl.accumulated_max_2d_radii_for_grad

    @property
    def track_max_2d_radii_for_grad(self) -> bool:
        """
        Returns whether to track the maximum 2D radii for gradients (False by default).

        This property is used by certain optimization techniques to ensure that the Gaussians
        do not become too large or too small during the optimization process.

        Note: See `accumulated_max_2d_radii_for_grad` for the actual maximum radii values.

        Returns:
            bool: True if the maximum 2D radii are being tracked, False otherwise.
        """
        return self._impl.track_max_2d_radii_for_grad

    @track_max_2d_radii_for_grad.setter
    def track_max_2d_radii_for_grad(self, value) -> bool:
        """
        Sets whether to track the maximum 2D radii for gradients (False by default).

        This property is used by certain optimization techniques to ensure that the Gaussians
        do not become too large or too small during the optimization process.

        Note: See `accumulated_max_2d_radii_for_grad` for the actual maximum radii values.

        Args:
            value (bool): True if the maximum 2D radii should be tracked, False otherwise.

        Raises:
            TypeError: If the provided value is not a boolean.
        """
        if not isinstance(value, bool):
            raise TypeError("Expected 'value' to be a boolean.")
        return self._impl.track_max_2d_radii_for_grad

    @property
    def accumulated_mean_2d_gradient_norms_for_grad(self) -> torch.Tensor:
        r"""
        Returns the average norm of the gradient of projected (2D) means for each Gaussian across every iteration of optimization.
        This is used by certain optimization techniques to split/prune/duplicate Gaussians.

        Mathematically, each Gaussian, :math:`G_i` is projected into :math:`C` 2D Gaussians :math:`G_i^1, \ldots, G_i^C` where :math:`C` is the number of cameras.
        If each projected Gaussian :math:`G_i^c` has a mean :math:`\mu_i^c`, then over :math:`T` iterations of optimization, this tensor stores:
        :math:`\frac{1}{T C} \sum_{t=1}^{T} \sum_{c=1}^{C} \| \partial_{L_t} \mu_i^c \|_2`
        where :math:`L_t` is the loss at iteration :math:`t`.

        Note: To reset the accumulated norms, you can call the `reset_grad_state` method.

        Note: This property will be empty if the `requires_grad` flag was not set to True
            when initializing the GaussianSplat3d instance.

        Returns:
            torch.Tensor: A tensor of shape (N,) where N is the number of Gaussians (see `num_gaussians`).
                Each element represents the average norm of the gradient of projected means for a Gaussian across all optimization iterations.
                The norm is computed in 2D space, i.e., the projected means.
        """
        return self._impl.accumulated_mean_2d_gradient_norms_for_grad

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        """
        Loads the state of the GaussianSplat3d instance from a state dictionary.
        This method is typically used to restore the state of the GaussianSplat3d instance
        after it has been saved or to initialize it with a specific state.

        Args:
            state_dict (dict[str, torch.Tensor]): A dictionary containing the state of the GaussianSplat3d instance. Here N denotes the number of Gaussians (see `num_gaussians`).
                The keys should match the expected state variables of the GaussianSplat3d. At the minimum, it should contain:
                - 'means': Tensor of shape (N, 3) representing the means of the Gaussians.
                - 'quats': Tensor of shape (N, 4) representing the quaternions of the Gaussians.
                - 'log_scales': Tensor of shape (N, 3) representing the log scales of the Gaussians.
                - 'logit_opacities': Tensor of shape (N,) representing the logit opacities of the Gaussians.
                - 'sh0': Tensor of shape (N, 1, D) representing the diffuse SH coefficients where D is the number of channels (see `num_channels`).
                - 'shN': Tensor of shape (N, K-1, D) representing the directionally varying SH coefficients where D is the number of channels (see `num_channels`), and K is the number of spherical harmonic bases (see `num_sh_bases`).
                - 'requires_grad': bool Tensor with a single element indicating whether gradients should be computed for these parameters.
                - 'track_max_2d_radii_for_grad': bool Tensor with a single element indicating whether to track the maximum 2D radii for gradients.
                - 'track_max_2d_radii_for_grad': bool Tensor with a single element indicating whether to track the average norm of the gradient of projected means for each Gaussian.
                It can optionally contain:
                - 'accumulated_gradient_step_counts_for_grad': Tensor of shape (N,) representing the accumulated gradient step counts for each Gaussian.
                - 'accumulated_max_2d_radii_for_grad': Tensor of shape (N,) representing the maximum 2D projected radius for each Gaussian across every iteration of optimization.
                - 'accumulated_mean_2d_gradient_norms_for_grad': Tensor of shape (N,) representing the average norm of the gradient of projected means for each Gaussian across every iteration of optimization.

        Raises:
            TypeError: If the provided state_dict does not match the expected types or keys.
            ValueError: If the state_dict is missing required keys or if the shapes of the tensors
                do not match the expected shapes for the GaussianSplat3d instance.
        """
        self._impl.load_state_dict(state_dict)

    def project_gaussians_for_depths(
        self,
        world_to_camera_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        image_width: int,
        image_height: int,
        near: float,
        far: float,
        projection_type="perspective",
        min_radius_2d: float = 0.0,
        eps_2d: float = 0.3,
        antialias: bool = False,
    ) -> ProjectedGaussianSplats:
        """
        Projects the Gaussians onto one ore more image planes for depth rendering (call `render_projected_gaussians` to actually render depth images).
        The reason to have a separate projection method is to enable rendering crops of an image without
        having to project the Gaussians again, which is useful for rendering crops of images.

        Args:
            world_to_camera_matrices (torch.Tensor): Tensor of shape (C, 4, 4) representing the world-to-camera transformation matrices for C cameras.
                Each matrix transforms points from world coordinates to camera coordinates.
            projection_matrices (torch.Tensor): Tensor of shape (C, 3, 3) representing the projection matrices for C cameras.
                Each matrix projects points in camera space into homogeneous pixel coordinates.
            image_width (int): The width of the images to be rendered.
            image_height (int): The height of the images to be rendered.
            near (float): The near clipping plane distance for the projection.
            far (float): The far clipping plane distance for the projection.
            projection_type (str): The type of projection to use. Default is "perspective".
                Other options could include "orthographic".
            min_radius_2d (float): The minimum radius in pixel space below which Gaussians are ignored during rendering.
            eps_2d (float): A small epsilon value to avoid numerical issues during projection.
            antialias (bool): If True, applies antialiasing to the projected Gaussians.
                This can help reduce artifacts in the rendered images, especially when the Gaussians
                are small or when the projection results in high-frequency details.

        Returns:
            ProjectedGaussianSplats: An instance of ProjectedGaussianSplats containing the projected Gaussians.
                This object contains the projected 2D representations of the Gaussians, which can be used for rendering depth images or further processing.

        Raises:
            TypeError: If the provided tensors do not match the expected types or shapes.
            ValueError: If the number of cameras (C) does not match the dimensions of the provided matrices,
                or if the image dimensions are not positive integers.
        """
        return self._impl.project_gaussians_for_depths(
            world_to_camera_matrices=world_to_camera_matrices,
            projection_matrices=projection_matrices,
            image_width=image_width,
            image_height=image_height,
            near=near,
            far=far,
            projection_type=projection_type,
            min_radius_2d=min_radius_2d,
            eps_2d=eps_2d,
            antialias=antialias,
        )

    def project_gaussians_for_images(
        self,
        world_to_camera_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        image_width: int,
        image_height: int,
        near: float,
        far: float,
        projection_type="perspective",
        sh_degree_to_use: int = -1,
        min_radius_2d: float = 0.0,
        eps_2d: float = 0.3,
        antialias: bool = False,
    ) -> ProjectedGaussianSplats:
        """
        Projects the Gaussians onto one ore more image planes for multi-channel image rendering (call `render_projected_gaussians` to actually render images).
        The reason to have a separate projection method is to enable rendering crops of an image without
        having to project the Gaussians again, which is useful for rendering crops of images.

        Args:
            world_to_camera_matrices (torch.Tensor): Tensor of shape (C, 4, 4) representing the world-to-camera transformation matrices for C cameras.
                Each matrix transforms points from world coordinates to camera coordinates.
            projection_matrices (torch.Tensor): Tensor of shape (C, 3, 3) representing the projection matrices for C cameras.
                Each matrix projects points in camera space into homogeneous pixel coordinates.
            image_width (int): The width of the images to be rendered.
            image_height (int): The height of the images to be rendered.
            near (float): The near clipping plane distance for the projection.
            far (float): The far clipping plane distance for the projection.
            projection_type (str): The type of projection to use. Default is "perspective".
                Other options could include "orthographic".
            sh_degree_to_use (int): The degree of spherical harmonics to use for rendering. -1 means use all available SH bases.
                0 means use only the first SH base (constant color). Note that you can't use more SH bases than available in the GaussianSplat3d instance.
                Default is -1.
            min_radius_2d (float): The minimum radius in pixel space below which Gaussians are ignored during rendering.
            eps_2d (float): A small epsilon value to avoid numerical issues during projection.
            antialias (bool): If True, applies antialiasing to the projected Gaussians.
                This can help reduce artifacts in the rendered images, especially when the Gaussians
                are small or when the projection results in high-frequency details.

        Returns:
            ProjectedGaussianSplats: An instance of ProjectedGaussianSplats containing the projected Gaussians.
                This object contains the projected 2D representations of the Gaussians, which can be used for rendering images or further processing.

        Raises:
            TypeError: If the provided tensors do not match the expected types or shapes.
            ValueError: If the number of cameras (C) does not match the dimensions of the provided matrices,
                or if the image dimensions are not positive integers.
        """
        return self._impl.project_gaussians_for_images(
            world_to_camera_matrices=world_to_camera_matrices,
            projection_matrices=projection_matrices,
            image_width=image_width,
            image_height=image_height,
            near=near,
            far=far,
            projection_type=projection_type,
            sh_degree_to_use=sh_degree_to_use,
            min_radius_2d=min_radius_2d,
            eps_2d=eps_2d,
            antialias=antialias,
        )

    def project_gaussians_for_images_and_depths(
        self,
        world_to_camera_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        image_width: int,
        image_height: int,
        near: float,
        far: float,
        projection_type="perspective",
        sh_degree_to_use: int = -1,
        min_radius_2d: float = 0.0,
        eps_2d: float = 0.3,
        antialias: bool = False,
    ) -> ProjectedGaussianSplats:
        """
        Projects the Gaussians onto one ore more image planes rendering multi-channel images and depth maps
        (call `render_projected_gaussians` to actually render images and depth maps). The reason to have a
        separate projection method is to enable rendering crops of an image without having to project the
        Gaussians again, which is useful for rendering crops of images.

        Args:
            world_to_camera_matrices (torch.Tensor): Tensor of shape (C, 4, 4) representing the world-to-camera
                transformation matrices for C cameras. Each matrix transforms points from world coordinates
                to camera coordinates.
            projection_matrices (torch.Tensor): Tensor of shape (C, 3, 3) representing the projection matrices
                for C cameras. Each matrix projects points in camera space into homogeneous pixel coordinates.
            image_width (int): The width of the images and depth maps to be rendered.
            image_height (int): The height of the images and depth maps to be rendered.
            near (float): The near clipping plane distance for the projection.
            far (float): The far clipping plane distance for the projection.
            projection_type (str): The type of projection to use. Default is "perspective".
                Other options could include "orthographic".
            sh_degree_to_use (int): The degree of spherical harmonics to use for rendering. -1 means use all available SH bases.
                0 means use only the first SH base (constant color). Note that you can't use more SH bases than available
                in the GaussianSplat3d instance. Default is -1.
            min_radius_2d (float): The minimum radius in pixel space below which Gaussians are ignored during rendering.
            eps_2d (float): A small epsilon value to avoid numerical issues during projection.
            antialias (bool): If True, applies antialiasing to the projected Gaussians.
                This can help reduce artifacts in the rendered images, especially when the Gaussians
                are small or when the projection results in high-frequency details.

        Returns:
            ProjectedGaussianSplats: An instance of ProjectedGaussianSplats containing the projected Gaussians.
                This object contains the projected 2D representations of the Gaussians, which can be used for
                rendering images and depth maps or further processing.

        Raises:
            TypeError: If the provided tensors do not match the expected types or shapes.
            ValueError: If the number of cameras (C) does not match the dimensions of the
                provided matrices, or if the image dimensions are not positive integers.
        """
        return self._impl.project_gaussians_for_images_and_depths(
            world_to_camera_matrices=world_to_camera_matrices,
            projection_matrices=projection_matrices,
            image_width=image_width,
            image_height=image_height,
            near=near,
            far=far,
            projection_type=projection_type,
            sh_degree_to_use=sh_degree_to_use,
            min_radius_2d=min_radius_2d,
            eps_2d=eps_2d,
            antialias=antialias,
        )

    def render_from_projected_gaussians(
        self,
        projected_gaussians: ProjectedGaussianSplats,
        crop_width: int = -1,
        crop_height: int = -1,
        crop_origin_w: int = -1,
        crop_origin_h: int = -1,
        tile_size: int = 16,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Render a set of images from Gaussian splats that have already been projected onto image planes.
        This method is useful when you want to render images from pre-computed projected Gaussians,
        for example, when rendering crops of images without having to re-project the Gaussians.

        Args:
            projected_gaussians (ProjectedGaussianSplats): An instance of ProjectedGaussianSplats
                containing the projected Gaussians. This object should have been created by calling
                `project_gaussians_for_images`, `project_gaussians_for_depths`,
                `project_gaussians_for_images_and_depths`, etc.
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

        Returns:
            rendered_images (torch.Tensor): A tensor of shape (C, H, W, D) where C is the number of cameras,
                H is the height of the rendered images, W is the width of the rendered images, and D is the
                number of channels (e.g., RGB, RGBD, etc.).
            alpha_images (torch.Tensor): A tensor of shape (C, H, W, 1) where C is the number of cameras,
                H is the height of the rendered images, and W is the width of the rendered images.
                Each element represents the alpha value (opacity) at that pixel in the rendered image.
        """
        return self._impl.render_from_projected_gaussians(
            projected_gaussians=projected_gaussians,
            crop_width=crop_width,
            crop_height=crop_height,
            crop_origin_w=crop_origin_w,
            crop_origin_h=crop_origin_h,
            tile_size=tile_size,
        )

    def render_depths(
        self,
        world_to_camera_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        image_width: int,
        image_height: int,
        near: float,
        far: float,
        projection_type="perspective",
        tile_size: int = 16,
        min_radius_2d: float = 0.3,
        eps_2d: float = 0.3,
        antialias: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Render :math:`C` depth maps from the Gaussian scene where :math:`C` is the number of cameras.

        Args:
            world_to_camera_matrices (torch.Tensor): Tensor of shape (C, 4, 4) representing the world-to-camera transformation matrices for C cameras.
                Each matrix transforms points from world coordinates to camera coordinates.
            projection_matrices (torch.Tensor): Tensor of shape (C, 3, 3) representing the projection matrices for C cameras.
                Each matrix projects points in camera space into homogeneous pixel coordinates.
            image_width (int): The width of the depth maps to be rendered.
            image_height (int): The height of the depth maps to be rendered.
            near (float): The near clipping plane distance for the projection.
            far (float): The far clipping plane distance for the projection.
            projection_type (str): The type of projection to use. Default is "perspective".
                Other options could include "orthographic".
            tile_size (int): The size of the tiles to use for rendering. Default is 16.
            min_radius_2d (float): The minimum radius in pixel space below which Gaussians are ignored during rendering.
                This helps to avoid rendering very small Gaussians that may not contribute significantly to the depth map.
            eps_2d (float): A small epsilon value to avoid numerical issues during projection.
            antialias (bool): If True, applies antialiasing to the rendered depth maps.
                This can help reduce artifacts in the depth maps, especially when the Gaussians
                are small or when the projection results in high-frequency details.

        Returns:
            depth_images (torch.Tensor): A tensor of shape (C, H, W, 1) where C is the number of cameras,
                H is the height of the depth maps, and W is the width of the depth maps.
                Each element represents the depth value at that pixel in the depth map.
            alpha_images (torch.Tensor): A tensor of shape (C, H, W, 1) where C is the number of cameras,
                H is the height of the depth maps, and W is the width of the depth maps.
                Each element represents the alpha value (opacity) at that pixel in the depth map.
        """
        return self._impl.render_depths(
            world_to_camera_matrices=world_to_camera_matrices,
            projection_matrices=projection_matrices,
            image_width=image_width,
            image_height=image_height,
            near=near,
            far=far,
            projection_type=projection_type,
            tile_size=tile_size,
            min_radius_2d=min_radius_2d,
            eps_2d=eps_2d,
            antialias=antialias,
        )

    def render_images(
        self,
        world_to_camera_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        image_width: int,
        image_height: int,
        near: float,
        far: float,
        projection_type="perspective",
        sh_degree_to_use: int = -1,
        tile_size: int = 16,
        min_radius_2d: float = 0.0,
        eps_2d: float = 0.3,
        antialias: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Render :math:`C` multi-channel images from the Gaussian scene where :math:`C` is the number of cameras.

        Args:
            world_to_camera_matrices (torch.Tensor): Tensor of shape (C, 4, 4) representing the world-to-camera transformation matrices for C cameras.
                Each matrix transforms points from world coordinates to camera coordinates.
            projection_matrices (torch.Tensor): Tensor of shape (C, 3, 3) representing the projection matrices for C cameras.
                Each matrix projects points in camera space into homogeneous pixel coordinates.
            image_width (int): The width of the images to be rendered.
            image_height (int): The height of the images to be rendered.
            near (float): The near clipping plane distance for the projection.
            far (float): The far clipping plane distance for the projection.
            projection_type (str): The type of projection to use. Default is "perspective".
                Other options could include "orthographic".
            tile_size (int): The size of the tiles to use for rendering. Default is 16.
            min_radius_2d (float): The minimum radius in pixel space below which Gaussians are ignored during rendering.
                This helps to avoid rendering very small Gaussians that may not contribute significantly to the depth map.
            eps_2d (float): A small epsilon value to avoid numerical issues during projection.
            antialias (bool): If True, applies antialiasing to the rendered images.
                This can help reduce artifacts in the images, especially when the Gaussians
                are small or when the projection results in high-frequency details.

        Returns:
            images (torch.Tensor): A tensor of shape (C, H, W, D) where C is the number of cameras,
                H is the height of the images, W is the width of the images, and D is the number of
                channels (e.g., 3 for RGB).
            alpha_images (torch.Tensor): A tensor of shape (C, H, W, 1) where C is the number of cameras,
                H is the height of the images, and W is the width of the images.
                Each element represents the alpha value (opacity) at that pixel in the depth map.
        """
        return self._impl.render_images(
            world_to_camera_matrices=world_to_camera_matrices,
            projection_matrices=projection_matrices,
            image_width=image_width,
            image_height=image_height,
            near=near,
            far=far,
            projection_type=projection_type,
            sh_degree_to_use=sh_degree_to_use,
            tile_size=tile_size,
            min_radius_2d=min_radius_2d,
            eps_2d=eps_2d,
            antialias=antialias,
        )

    def render_images_and_depths(
        self,
        world_to_camera_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        image_width: int,
        image_height: int,
        near: float,
        far: float,
        projection_type="perspective",
        sh_degree_to_use: int = -1,
        tile_size: int = 16,
        min_radius_2d: float = 0.0,
        eps_2d: float = 0.3,
        antialias: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Render :math:`C` multi-channel images with depth as the last channel from the Gaussian scene where :math:`C` is the number of cameras.

        Args:
            world_to_camera_matrices (torch.Tensor): Tensor of shape (C, 4, 4) representing the world-to-camera transformation matrices for C cameras.
                Each matrix transforms points from world coordinates to camera coordinates.
            projection_matrices (torch.Tensor): Tensor of shape (C, 3, 3) representing the projection matrices for C cameras.
                Each matrix projects points in camera space into homogeneous pixel coordinates.
            image_width (int): The width of the images to be rendered.
            image_height (int): The height of the images to be rendered.
            near (float): The near clipping plane distance for the projection.
            far (float): The far clipping plane distance for the projection.
            projection_type (str): The type of projection to use. Default is "perspective".
                Other options could include "orthographic".
            sh_degree_to_use (int): The degree of spherical harmonics to use for rendering. -1 means use all available SH bases.
                0 means use only the first SH base (constant color). Note that you can't use more SH bases than available in the GaussianSplat3d instance.
                Default is -1.
            tile_size (int): The size of the tiles to use for rendering. Default is 16.
            min_radius_2d (float): The minimum radius in pixel space below which Gaussians are ignored during rendering.
                This helps to avoid rendering very small Gaussians that may not contribute significantly to the depth map.
            eps_2d (float): A small epsilon value to avoid numerical issues during projection.
            antialias (bool): If True, applies antialiasing to the rendered images.
                This can help reduce artifacts in the images, especially when the Gaussians
                are small or when the projection results in high-frequency details.

        Returns:
            rendered_images (torch.Tensor): A tensor of shape (C, H, W, D+1) where C is the number of cameras,
                H is the height of the images, W is the width of the images, and D is the number of channels (e.g., RGB).
                The last channel contains the depth values.
            alpha_images (torch.Tensor): A tensor of shape (C, H, W, 1) where C is the number of cameras,
                H is the height of the images, and W is the width of the images.
                Each element represents the alpha value (opacity) at that pixel in the depth map.
        """
        return self._impl.render_images_and_depths(
            world_to_camera_matrices=world_to_camera_matrices,
            projection_matrices=projection_matrices,
            image_width=image_width,
            image_height=image_height,
            near=near,
            far=far,
            projection_type=projection_type,
            sh_degree_to_use=sh_degree_to_use,
            tile_size=tile_size,
            min_radius_2d=min_radius_2d,
            eps_2d=eps_2d,
            antialias=antialias,
        )

    def render_top_contributing_gaussian_ids(
        self,
        num_samples: int,
        world_to_camera_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        image_width: int,
        image_height: int,
        near: float,
        far: float,
        projection_type="perspective",
        tile_size: int = 16,
        min_radius_2d: float = 0.0,
        eps_2d: float = 0.3,
        antialias: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Renders the top contributing Gaussian ids for each pixel in the image. i.e the ids of the most opaque Gaussians along each ray.

        Args:
            num_samples (int): The number of top contributing Gaussians to return for each pixel.
            world_to_camera_matrices (torch.Tensor): Tensor of shape (C, 4, 4) representing the world-to-camera transformation matrices for C cameras.
                Each matrix transforms points from world coordinates to camera coordinates.
            projection_matrices (torch.Tensor): Tensor of shape (C, 3, 3) representing the projection matrices for C cameras.
                Each matrix projects points in camera space into homogeneous pixel coordinates.
            image_width (int): The width of the images to be rendered.
            image_height (int): The height of the images to be rendered.
            near (float): The near clipping plane distance for the projection.
            far (float): The far clipping plane distance for the projection.
            projection_type (str): The type of projection to use. Default is "perspective".
                Other options could include "orthographic".
            tile_size (int): The size of the tiles to use for rendering. Default is 16.
            min_radius_2d (float): The minimum radius in pixel space below which Gaussians are ignored during rendering.
                This helps to avoid rendering very small Gaussians that may not contribute significantly to the depth map.
            eps_2d (float): A small epsilon value to avoid numerical issues during projection.
            antialias (bool): If True, applies antialiasing to the rendered images.
                This can help reduce artifacts in the images, especially when the Gaussians
                are small or when the projection results in high-frequency details.

        Returns:
            top_contributing_gaussian_ids (torch.Tensor): A long tensor of shape (C, H, W, num_samples) where C is the number of cameras,
                H is the height of the images, W is the width of the images, and num_samples is the number of top contributing
                Gaussians to return for each pixel. Each element represents the id of a Gaussian that contributes to the pixel.
            weights (torch.Tensor): A tensor of shape (C, H, W, num_samples) where C is the number of cameras,
                H is the height of the images, W is the width of the images, and num_samples is the number of top contributing
                Gaussians to return for each pixel. Each element represents the transmittance-weighted opacity of the Gaussian
                that contributes to the pixel (i.e. its proportion of the visible contribution to the pixel).
        """
        return self._impl.render_top_contributing_gaussian_ids(
            num_samples=num_samples,
            world_to_camera_matrices=world_to_camera_matrices,
            projection_matrices=projection_matrices,
            image_width=image_width,
            image_height=image_height,
            near=near,
            far=far,
            projection_type=projection_type,
            tile_size=tile_size,
            min_radius_2d=min_radius_2d,
            eps_2d=eps_2d,
            antialias=antialias,
        )

    @overload
    def sparse_render_top_contributing_gaussian_ids(
        self,
        num_samples: int,
        pixels_to_render: torch.Tensor,
        world_to_camera_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        image_width: int,
        image_height: int,
        near: float,
        far: float,
        projection_type="perspective",
        tile_size: int = 16,
        min_radius_2d: float = 0.0,
        eps_2d: float = 0.3,
        antialias: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Renders the top contributing Gaussian ids for each pixel in the image. i.e the ids of the
        most opaque Gaussians along each ray.

        Args:
            num_samples (int): The number of top contributing Gaussians to return for each pixel.
            pixels_to_render (torch.Tensor): A dense tensor of shape (C, R, 2) representing the
                pixels to render for each camera, where C is the number of cameras and R is the
                number of pixels to render per camera (same for all cameras).
            world_to_camera_matrices (torch.Tensor): Tensor of shape (C, 4, 4) representing the
                world-to-camera transformation matrices for C cameras.
                Each matrix transforms points from world coordinates to camera coordinates.
            projection_matrices (torch.Tensor): Tensor of shape (C, 3, 3) representing the
                projection matrices for C cameras.
                Each matrix projects points in camera space into homogeneous pixel coordinates.
            image_width (int): The width of the images to be rendered.
            image_height (int): The height of the images to be rendered.
            near (float): The near clipping plane distance for the projection.
            far (float): The far clipping plane distance for the projection.
            projection_type (str): The type of projection to use. Default is "perspective".
                Other options could include "orthographic".
            tile_size (int): The size of the tiles to use for rendering. Default is 16.
            min_radius_2d (float): The minimum radius in pixel space below which Gaussians are
                ignored during rendering. This helps to avoid rendering very small Gaussians that
                may not contribute significantly to the depth map. Default is 0.0.
            eps_2d (float): A small epsilon value to avoid numerical issues during projection.
                Default is 0.3.
            antialias (bool): If True, applies antialiasing to the rendered images. Default is False.
                This can help reduce artifacts in the images, especially when the Gaussians
                are small or when the projection results in high-frequency details.

        Returns:
            top_contributing_gaussian_ids (torch.Tensor): A tensor of shape (C, R, num_samples) where C is the number of cameras,
                R is the number of pixels to render per camera, and num_samples is the number of top contributing
                Gaussians to return for each pixel. Each element represents the id of a Gaussian
                that contributes to the pixel.
            weights (torch.Tensor): A tensor of shape (C, R, num_samples) where C is the number of cameras,
                R is the number of pixels to render per camera, and num_samples is the number of
                top contributing Gaussians to return for each pixel. Each element represents the
                transmittance-weighted opacity of the Gaussian that contributes to the pixel (i.e.
                its proportion of the visible contribution to the pixel).
        """
        ...

    @overload
    def sparse_render_top_contributing_gaussian_ids(
        self,
        num_samples: int,
        pixels_to_render: JaggedTensor,
        world_to_camera_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        image_width: int,
        image_height: int,
        near: float,
        far: float,
        projection_type="perspective",
        tile_size: int = 16,
        min_radius_2d: float = 0.0,
        eps_2d: float = 0.3,
        antialias: bool = False,
    ) -> tuple[JaggedTensor, JaggedTensor]:
        """
        Renders the top contributing Gaussian ids for each pixel in the image. i.e the ids of the
        most opaque Gaussians along each ray.

        Args:
            num_samples (int): The number of top contributing Gaussians to return for each pixel.
            pixels_to_render (JaggedTensor): A JaggedTensor of ldim=1, num_tensors=[C] (the number
                of cameras in the batch) and each tensor is of shape [R_i, 2] representing the
                pixels to render for the i-th camera.
            world_to_camera_matrices (torch.Tensor): Tensor of shape (C, 4, 4) representing the
                world-to-camera transformation matrices for C cameras.
                Each matrix transforms points from world coordinates to camera coordinates.
            projection_matrices (torch.Tensor): Tensor of shape (C, 3, 3) representing the
                projection matrices for C cameras.
                Each matrix projects points in camera space into homogeneous pixel coordinates.
            image_width (int): The width of the images to be rendered.
            image_height (int): The height of the images to be rendered.
            near (float): The near clipping plane distance for the projection.
            far (float): The far clipping plane distance for the projection.
            projection_type (str): The type of projection to use. Default is "perspective".
                Other options could include "orthographic".
            tile_size (int): The size of the tiles to use for rendering. Default is 16.
            min_radius_2d (float): The minimum radius in pixel space below which Gaussians are
                ignored during rendering. This helps to avoid rendering very small Gaussians that
                This helps to avoid rendering very small Gaussians that may not contribute
                significantly to the depth map. Default is 0.0.
            eps_2d (float): A small epsilon value to avoid numerical issues during projection.
                Default is 0.3.
            antialias (bool): If True, applies antialiasing to the rendered images. Default is False.
                This can help reduce artifacts in the images, especially when the Gaussians
                are small or when the projection results in high-frequency details.

        Returns:
            top_contributing_gaussian_ids (fvdb.JaggedTensor): A JaggedTensor of ldim=1,
                num_tensors=[C] (the number of cameras in the batch) and each tensor is of shape
                [R_i, num_samples] where R_i is the number of pixels to render for the i-th camera,
                and num_samples is the number of top contributing
                Gaussians to return for each pixel. Each element represents the id of a Gaussian
                that contributes to the pixel.
            weights (fvdb.JaggedTensor): A JaggedTensor of ldim=1, num_tensors=[C] (the number of
                cameras in the batch) and each tensor is of shape [R_i, num_samples] where R_i is
                the number of pixels to render for the i-th camera, and num_samples is the number of
                top contributing Gaussians to return for each pixel. Each element represents the
                transmittance-weighted opacity of the Gaussian that contributes to the pixel (i.e.
                its proportion of the visible contribution to the pixel).
        """
        ...

    def sparse_render_top_contributing_gaussian_ids(
        self,
        num_samples: int,
        pixels_to_render: JaggedTensor | torch.Tensor,
        world_to_camera_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        image_width: int,
        image_height: int,
        near: float,
        far: float,
        projection_type="perspective",
        tile_size: int = 16,
        min_radius_2d: float = 0.0,
        eps_2d: float = 0.3,
        antialias: bool = False,
    ) -> tuple[JaggedTensor | torch.Tensor, JaggedTensor | torch.Tensor]:
        """
        Renders the top contributing Gaussian ids for each pixel in the image. i.e the ids of the
        most opaque Gaussians along each ray.

        Args:
            num_samples (int): The number of top contributing Gaussians to return for each pixel.
            pixels_to_render (JaggedTensor | torch.Tensor): A JaggedTensor or dense tensor of shape (C, R, 2) representing the
                pixels to render for each camera, where C is the number of cameras and R is the
                number of pixels to render per camera (same for all cameras).
            world_to_camera_matrices (torch.Tensor): Tensor of shape (C, 4, 4) representing the
                world-to-camera transformation matrices for C cameras.
                Each matrix transforms points from world coordinates to camera coordinates.
            projection_matrices (torch.Tensor): Tensor of shape (C, 3, 3) representing the
                projection matrices for C cameras.
                Each matrix projects points in camera space into homogeneous pixel coordinates.
            image_width (int): The width of the images to be rendered.
            image_height (int): The height of the images to be rendered.
            near (float): The near clipping plane distance for the projection.
            far (float): The far clipping plane distance for the projection.
            projection_type (str): The type of projection to use. Default is "perspective".
                Other options could include "orthographic".
            tile_size (int): The size of the tiles to use for rendering. Default is 16.
            min_radius_2d (float): The minimum radius in pixel space below which Gaussians are
                ignored during rendering. This helps to avoid rendering very small Gaussians that
                may not contribute significantly to the depth map. Default is 0.0.
            eps_2d (float): A small epsilon value to avoid numerical issues during projection.
                Default is 0.3.
            antialias (bool): If True, applies antialiasing to the rendered images. Default is False.
                This can help reduce artifacts in the images, especially when the Gaussians
                are small or when the projection results in high-frequency details.

        Returns:
            top_contributing_gaussian_ids (JaggedTensor | torch.Tensor): A JaggedTensor or dense tensor of shape (C, R, num_samples) where C is the number of cameras,
                R is the number of pixels to render per camera, and num_samples is the number of top contributing
                Gaussians to return for each pixel. Each element represents the id of a Gaussian
                that contributes to the pixel.
            weights (JaggedTensor | torch.Tensor): A JaggedTensor or dense tensor of shape (C, R, num_samples) where C is the number of cameras,
                R is the number of pixels to render per camera, and num_samples is the number of
                top contributing Gaussians to return for each pixel. Each element represents the
                transmittance-weighted opacity of the Gaussian that contributes to the pixel (i.e.
                its proportion of the visible contribution to the pixel).
        """
        if isinstance(pixels_to_render, torch.Tensor):
            C, R, _ = pixels_to_render.shape
            tensors = [pixels_to_render[i] for i in range(C)]
            pixels_to_render_jagged = JaggedTensor(tensors)

            result_ids, result_weights = self._impl.sparse_render_top_contributing_gaussian_ids(
                num_samples=num_samples,
                pixels_to_render=pixels_to_render_jagged,
                world_to_camera_matrices=world_to_camera_matrices,
                projection_matrices=projection_matrices,
                image_width=image_width,
                image_height=image_height,
                near=near,
                far=far,
                projection_type=projection_type,
                tile_size=tile_size,
                min_radius_2d=min_radius_2d,
                eps_2d=eps_2d,
                antialias=antialias,
            )

            ids_list = result_ids.unbind()
            weights_list = result_weights.unbind()
            dense_ids = torch.stack(ids_list, dim=0)  # Shape: (C, R, num_samples)
            dense_weights = torch.stack(weights_list, dim=0)  # Shape: (C, R, num_samples)

            return dense_ids, dense_weights
        else:
            # Already a JaggedTensor, call C++ implementation directly
            return self._impl.sparse_render_top_contributing_gaussian_ids(
                num_samples=num_samples,
                pixels_to_render=pixels_to_render,
                world_to_camera_matrices=world_to_camera_matrices,
                projection_matrices=projection_matrices,
                image_width=image_width,
                image_height=image_height,
                near=near,
                far=far,
                projection_type=projection_type,
                tile_size=tile_size,
                min_radius_2d=min_radius_2d,
                eps_2d=eps_2d,
                antialias=antialias,
            )

    def reset_grad_state(self) -> None:
        """
        Reset state tracked during backward passes if `requires_grad` was set to True when initializing the GaussianSplat3d instance.
        This method clears the accumulated gradient step counts (`accumulated_gradient_step_counts_for_grad`),
        maximum 2D radii (`accumulated_max_2d_radii_for_grad`), and average 2D gradient
        norms (`accumulated_mean_2d_gradient_norms_for_grad`) for each Gaussian.
        """
        self._impl.reset_grad_state()

    def save_ply(self, filename: str) -> None:
        """
        Save the current state of the GaussianSplat3d instance to a PLY file.
        Args:
            filename (str): The name of the file to save the PLY data to.
                The file will contain the means, quaternions, log scales, logit opacities, and
                spherical harmonics coefficients of the Gaussians.
        """
        self._impl.save_ply(filename)

    def load_ply(self, filename: str, device: torch.device = torch.device("cuda")) -> None:
        """
        Load the state of the GaussianSplat3d instance from a PLY file.
        Args:
            filename (str): The name of the file to load the PLY data from.
            device (torch.device): The device to load the data onto. Default is "cuda".
        Raises:
            FileNotFoundError: If the specified PLY file does not exist.
            ValueError: If the PLY file does not contain the expected data format or structure.
        """
        self._impl.load_ply(filename, device=device)

    def set_state(
        self,
        means: torch.Tensor,
        quats: torch.Tensor,
        log_scales: torch.Tensor,
        logit_opacities: torch.Tensor,
        sh0: torch.Tensor,
        shN: torch.Tensor,
        requires_grad: bool = False,
    ) -> None:
        """
        Set the state of the GaussianSplat3d instance with the provided parameters.

        Note: If `requires_grad` is True, this call will reset the gradient state (see `reset_grad_state`).

        Args:
            means (torch.Tensor): Tensor of shape (N, 3) representing the means of the Gaussians.
                N is the number of Gaussians (see `num_gaussians`).
            quats (torch.Tensor): Tensor of shape (N, 4) representing the quaternions of the Gaussians.
                N is the number of Gaussians (see `num_gaussians`).
            log_scales (torch.Tensor): Tensor of shape (N, 3) representing the log scales of the Gaussians.
                N is the number of Gaussians (see `num_gaussians`).
            logit_opacities (torch.Tensor): Tensor of shape (N,) representing the logit opacities of the Gaussians.
                N is the number of Gaussians (see `num_gaussians`).
            sh0 (torch.Tensor): Tensor of shape (N, 1, D) representing the diffuse SH coefficients where D is the number of channels (see `num_channels`).
            shN (torch.Tensor): Tensor of shape (N, K-1, D) representing the directionally varying SH coefficients where D is the number of channels (see `num_channels`),
                and K is the number of spherical harmonic bases (see `num_sh_bases`).
            requires_grad (bool): If True, the GaussianSplat3d instance will track gradients for the parameters.
        """
        self._impl.set_state(
            means=means,
            quats=quats,
            log_scales=log_scales,
            logit_opacities=logit_opacities,
            sh0=sh0,
            shN=shN,
            requires_grad=requires_grad,
        )

    def state_dict(self) -> dict[str, torch.Tensor]:
        """
        Return the state dictionary of the GaussianSplat3d instance.

        Returns:
            state_dict (dict[str, torch.Tensor]): A dictionary containing the state of the GaussianSplat3d instance. Here N denotes the number of Gaussians (see `num_gaussians`).
                The keys should match the expected state variables of the GaussianSplat3d. At the minimum, it should contain:
                - 'means': Tensor of shape (N, 3) representing the means of the Gaussians.
                - 'quats': Tensor of shape (N, 4) representing the quaternions of the Gaussians.
                - 'log_scales': Tensor of shape (N, 3) representing the log scales of the Gaussians.
                - 'logit_opacities': Tensor of shape (N,) representing the logit opacities of the Gaussians.
                - 'sh0': Tensor of shape (N, 1, D) representing the diffuse SH coefficients where D is the number of channels (see `num_channels`).
                - 'shN': Tensor of shape (N, K-1, D) representing the directionally varying SH coefficients where D is the number of channels (see `num_channels`), and K is the number of spherical harmonic bases (see `num_sh_bases`).
                - 'requires_grad': bool Tensor with a single element indicating whether gradients should be computed for these parameters.
                - 'track_max_2d_radii_for_grad': bool Tensor with a single element indicating whether to track the maximum 2D radii for gradients.
                - 'track_max_2d_radii_for_grad': bool Tensor with a single element indicating whether to track the average norm of the gradient of projected means for each Gaussian.
                It can optionally contain:
                - 'accumulated_gradient_step_counts_for_grad': Tensor of shape (N,) representing the accumulated gradient step counts for each Gaussian.
                - 'accumulated_max_2d_radii_for_grad': Tensor of shape (N,) representing the maximum 2D projected radius for each Gaussian across every iteration of optimization.
                - 'accumulated_mean_2d_gradient_norms_for_grad': Tensor of shape (N,) representing the average norm of the gradient of projected means for each Gaussian across every iteration of optimization.

        """
        return self._impl.state_dict()
