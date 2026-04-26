# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Python torch.autograd.Function wrappers for volume rendering functions."""

from __future__ import annotations

from typing import Any, cast

import torch

from . import _fvdb_cpp


class _VolumeRenderFn(torch.autograd.Function):
    """Autograd Function binding the volume-render forward / backward kernels."""

    @staticmethod
    def forward(
        ctx: Any,
        sigmas: torch.Tensor,
        rgbs: torch.Tensor,
        delta_ts: torch.Tensor,
        ts: torch.Tensor,
        pack_info: torch.Tensor,
        transmittance_thresh: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the forward kernel in backward-aware mode.

        Always dispatches ``_fvdb_cpp.volume_render_fwd`` with
        ``needs_backward=True`` so the backward-only outputs (``out_ws``,
        ``out_depth``, ``out_total``) are fully materialized and available
        to :meth:`backward`. The ray inputs and the composited per-ray
        outputs are stashed on ``ctx`` via :meth:`ctx.save_for_backward`,
        and ``transmittance_thresh`` is stored as a Python scalar on
        ``ctx`` (not a tensor, so it does not participate in autograd).
        """
        out_rgb, out_depth, out_opacity, out_ws, out_total = _fvdb_cpp.volume_render_fwd(
            sigmas,
            rgbs,
            delta_ts,
            ts,
            pack_info,
            float(transmittance_thresh),
            True,
        )
        ctx.save_for_backward(
            sigmas,
            rgbs,
            delta_ts,
            ts,
            pack_info,
            out_opacity,
            out_depth,
            out_rgb,
            out_ws,
        )
        ctx.transmittance_thresh = float(transmittance_thresh)
        return out_rgb, out_depth, out_opacity, out_ws, out_total

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor | None):
        """Propagate gradients through the compositing integral.

        Consumes ``dL/drgb``, ``dL/ddepth``, ``dL/dopacity`` and
        ``dL/dws`` from ``grad_outputs`` (the incoming gradient on
        ``total_samples`` is unused — that output is an integer counter
        and is intentionally non-differentiable).

        Any of the four consumed grad outputs may be ``None`` — autograd
        passes ``None`` for any forward output that the downstream loss
        does not depend on (e.g. a loss on ``rgb`` alone yields ``None``
        for ``dL/ddepth``, ``dL/dopacity`` and ``dL/dws``). The C++
        backward kernel requires all four as real tensors, so any
        missing grad is materialized as a zero tensor with the shape /
        dtype / device of the corresponding saved forward output. All
        grads are also made contiguous before being handed to C++.

        Returns a 6-tuple of gradient slots matching :meth:`forward`'s
        positional arguments: gradients flow only into ``sigmas`` and
        ``rgbs``; the slots for ``delta_ts``, ``ts``, ``pack_info`` and
        ``transmittance_thresh`` are ``None``.
        """
        dL_drgb, dL_ddepth, dL_dopacity, dL_dws, _dL_dtotal = grad_outputs
        (
            sigmas,
            rgbs,
            delta_ts,
            ts,
            pack_info,
            out_opacity,
            out_depth,
            out_rgb,
            out_ws,
        ) = ctx.saved_tensors

        def _coerce(grad: torch.Tensor | None, like: torch.Tensor) -> torch.Tensor:
            if grad is None:
                return torch.zeros_like(like, memory_format=torch.contiguous_format)
            return grad.contiguous()

        dL_drgb = _coerce(dL_drgb, out_rgb)
        dL_ddepth = _coerce(dL_ddepth, out_depth)
        dL_dopacity = _coerce(dL_dopacity, out_opacity)
        dL_dws = _coerce(dL_dws, out_ws)

        dL_dsigmas, dL_drgbs = _fvdb_cpp.volume_render_bwd(
            dL_dopacity,
            dL_ddepth,
            dL_drgb,
            dL_dws,
            sigmas,
            rgbs,
            out_ws,
            delta_ts,
            ts,
            pack_info,
            out_opacity,
            out_depth,
            out_rgb,
            ctx.transmittance_thresh,
        )
        return dL_dsigmas, dL_drgbs, None, None, None, None


def volume_render(
    sigmas: torch.Tensor,
    rgbs: torch.Tensor,
    delta_ts: torch.Tensor,
    ts: torch.Tensor,
    pack_info: torch.Tensor,
    transmittance_thresh: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Volume-render per-sample ``sigmas`` / ``rgbs`` into per-ray outputs.

    Performs front-to-back Beer-Lambert compositing along each ray. For a
    sample ``s`` with extinction ``sigma_s`` and step length ``delta_s``,
    the alpha is ``a_s = 1 - exp(-sigma_s * delta_s)``, the per-sample
    weight is ``w_s = a_s * T_s`` (where ``T_s`` is the accumulated
    transmittance up to ``s``), and the per-ray outputs are ``rgb = sum_s
    w_s * rgb_s``, ``depth = sum_s w_s * t_s``, ``opacity = sum_s w_s``.
    Each ray early-terminates once ``T <= transmittance_thresh``.

    All samples for all rays are packed contiguously into the ``N``-dim
    input tensors; ``pack_info`` tells the kernel which contiguous slice
    belongs to each of the ``R`` rays (see below).

    Execution paths:
      * Training / graph-capture path. If autograd is globally enabled
        **and** ``sigmas`` or ``rgbs`` requires grad, this routes through
        :class:`_VolumeRenderFn` which runs the kernel with
        ``needsBackward=True``. The backward-only outputs (``depth``,
        ``ws``, ``total_samples``) are fully materialized and saved on
        the autograd graph, and gradients flow into ``sigmas`` and
        ``rgbs``. ``delta_ts`` and ``ts`` are non-differentiable inputs
        even when their ``requires_grad`` is set, so they do not on
        their own select this path.
      * Inference fast path. Otherwise this calls the C++ forward directly
        with ``needsBackward=False``. The kernel skips the per-sample
        ``ws`` store (the dominant global-memory traffic savings) and the per-ray
        ``depth`` / ``total_samples`` stores, and the host skips the
        zero-init of the size-``N`` ``ws`` buffer. Those three outputs are
        returned as size-0 placeholder tensors with matching dtype /
        device. Callers that only consume ``rgb`` / ``opacity`` (e.g. a
        pure renderer) see no behavioral change.

    Args:
        sigmas (torch.Tensor): Per-sample extinction (density) values,
            floating-point tensor of shape ``[N]``, where ``N`` is the
            total number of samples packed across all ``R`` rays. Must be
            on the same device and dtype as ``rgbs``, ``delta_ts``, ``ts``.
            Differentiable input.
        rgbs (torch.Tensor): Per-sample emitted color, floating-point
            tensor of shape ``[N, C]`` where
            ``1 <= C <= MAX_VOLUME_RENDER_CHANNELS`` (currently 16).
            Differentiable input.
        delta_ts (torch.Tensor): Per-sample step length along the ray
            (``delta_s = t_{s+1} - t_s``), floating-point tensor of shape
            ``[N]``. Not differentiable.
        ts (torch.Tensor): Per-sample parametric distance along the ray
            (typically the interval midpoint) used for depth compositing,
            floating-point tensor of shape ``[N]``. Not differentiable.
        pack_info (torch.Tensor): CSR-style per-ray offsets into the
            sample-packed tensors. ``int64`` tensor of shape ``[R + 1]``
            where ``R`` is the number of rays; ray ``r`` owns samples in
            ``[pack_info[r], pack_info[r + 1])`` and
            ``pack_info[R] == N``. Not differentiable.
        transmittance_thresh (float): Early-termination threshold on the
            accumulated transmittance ``T``. The inner compositing loop
            breaks for a ray as soon as ``T <= transmittance_thresh``.
            Typical values are in the range ``[1e-4, 1e-2]``.

    Returns:
        A 5-tuple ``(rgb, depth, opacity, ws, total_samples)``:

          * ``rgb`` (``torch.Tensor``): Composited per-ray color,
            floating-point tensor of shape ``[R, C]``. Always
            materialized.
          * ``depth`` (``torch.Tensor``): Composited per-ray depth,
            floating-point tensor of shape ``[R]`` on the training path
            or ``[0]`` on the inference fast path.
          * ``opacity`` (``torch.Tensor``): Accumulated per-ray opacity
            ``1 - T_final``, floating-point tensor of shape ``[R]``.
            Always materialized.
          * ``ws`` (``torch.Tensor``): Per-sample compositing weight
            ``a_s * T_s`` consumed by the backward pass, floating-point
            tensor of shape ``[N]`` on the training path or ``[0]`` on
            the inference fast path. When materialized, entries past a
            ray's early-termination point remain zero (the buffer is
            zero-initialized on the host).
          * ``total_samples`` (``torch.Tensor``): Number of samples
            actually composited for each ray before early termination,
            ``int64`` tensor of shape ``[R]`` on the training path or
            ``[0]`` on the inference fast path.

    Note:
        Callers that need the full set of outputs under no-grad (for
        example, a diagnostic that wants ``total_samples``) should
        either run under :func:`torch.enable_grad` with
        ``sigmas.requires_grad_(True)`` or ``rgbs.requires_grad_(True)``,
        or call :func:`_fvdb_cpp.volume_render_fwd` directly with
        ``needsBackward=True``. Setting ``requires_grad`` on
        ``delta_ts`` or ``ts`` does not select this path because no
        gradient is propagated into those inputs.
    """
    needs_grad = torch.is_grad_enabled() and any(
        isinstance(t, torch.Tensor) and t.requires_grad for t in (sigmas, rgbs)
    )
    if needs_grad:
        return cast(
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            _VolumeRenderFn.apply(sigmas, rgbs, delta_ts, ts, pack_info, float(transmittance_thresh)),
        )
    return _fvdb_cpp.volume_render_fwd(
        sigmas,
        rgbs,
        delta_ts,
        ts,
        pack_info,
        float(transmittance_thresh),
        False,
    )
