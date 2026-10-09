# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from .jagged_tensor import JaggedTensor

_SDP_FLASH = SDPBackend.FLASH_ATTENTION.value
_SDP_MATH = SDPBackend.MATH.value


def _make_nested_view(jt: JaggedTensor) -> torch.Tensor:
    """Create a zero-copy nested tensor view from a JaggedTensor.

    Constructs a PyTorch nested tensor that shares storage with the JaggedTensor's
    underlying ``jdata`` buffer. This avoids any data copy and is intended for
    use with attention backends that support nested tensors (for example,
    the memory-efficient and cuDNN scaled dot product attention backends).

    Args:
        jt: A JaggedTensor with ``ldim == 1`` and ``jdata`` of shape ``(Total, H, D)``.

    Returns:
        A nested tensor whose *i*-th component has shape ``(L_i, H, D)``.
    """
    # The view reads the flat buffer, so the data must be contiguous.
    data = jt.jdata.contiguous()  # (Total, H, D)
    H = data.size(1)
    D = data.size(2)
    stride_L = H * D
    num_tensors = jt.num_tensors
    lsizes = cast(list[int], jt.lshape)

    lengths = torch.tensor(lsizes, dtype=torch.long)

    # nested_size: (N, 3) -> [L_i, H, D]
    nested_size = torch.empty(num_tensors, 3, dtype=torch.long)
    nested_size[:, 0] = lengths
    nested_size[:, 1] = H
    nested_size[:, 2] = D

    # nested_strides: (N, 3) -> [H*D, D, 1]
    nested_strides = torch.empty(num_tensors, 3, dtype=torch.long)
    nested_strides[:, 0] = stride_L
    nested_strides[:, 1] = D
    nested_strides[:, 2] = 1

    # storage_offsets from cumulative sum of lengths * stride
    offsets = torch.zeros(num_tensors, dtype=torch.long)
    if num_tensors > 1:
        offsets[1:] = torch.cumsum(lengths[:-1], dim=0) * stride_L

    return torch._nested_view_from_buffer(data.view(-1), nested_size, nested_strides, offsets)


def _make_nested_tensor(jt: JaggedTensor) -> torch.Tensor:
    """Create a nested tensor by copying slices from a JaggedTensor.

    Each sub-tensor is sliced from ``jdata`` and made contiguous, then packed
    into a PyTorch nested tensor.  This is required by the flash-attention and
    math backends, which cannot operate on the zero-copy buffer view.

    Args:
        jt: A JaggedTensor with ``ldim == 1`` and ``jdata`` of shape ``(Total, H, D)``.

    Returns:
        A nested tensor whose *i*-th component has shape ``(L_i, H, D)``.
    """
    data = jt.jdata
    lsizes = cast(list[int], jt.lshape)

    tensor_list = []
    start = 0
    for length in lsizes:
        tensor_list.append(data[start : start + length])
        start += length

    return torch._nested_tensor_from_tensor_list(tensor_list)


_HALF_DTYPES = (torch.float16, torch.bfloat16)
_INT32_MAX = 2**31 - 1


@dataclass(frozen=True)
class _PatchLayout:
    """Index maps from a serialized jagged sequence to padded, patch-aligned sequences."""

    gather_idx: torch.Tensor  # (Tp,) padded position -> source row
    unpad_idx: torch.Tensor  # (T,) source row -> padded position
    cu_seqlens: torch.Tensor  # (num_patches + 1,) int32
    max_seqlen: int


def _patch_layout(joffsets: torch.Tensor, lshape: list[int], patch_size: int) -> _PatchLayout:
    """Split each sequence into patches of ``patch_size`` tokens, padding as Pointcept does.

    A sequence of length ``L <= patch_size`` forms a single patch without padding. A longer
    sequence is padded up to a multiple of ``patch_size``; padded position ``p`` repeats the
    token at ``p - patch_size``, so the last patch is filled from the tail of the one before it.

    Device tensors are built from ``joffsets`` and host sizes from ``lshape``, with no device sync.
    """
    p = patch_size
    padded_host = [n if n <= p else -(-n // p) * p for n in lshape]
    total = sum(lshape)
    total_padded = sum(padded_host)
    num_patches = sum(-(-n // p) for n in padded_host)
    if total_padded > _INT32_MAX:
        raise ValueError(f"Padded token count {total_padded} exceeds the int32 range of cu_seqlens")

    device = joffsets.device
    grid_ids = torch.arange(len(lshape), device=device)
    lengths = joffsets[1:] - joffsets[:-1]
    padded = torch.where(lengths > p, (lengths + p - 1) // p * p, lengths)
    padded_offsets = F.pad(torch.cumsum(padded, 0), (1, 0))
    patches = (padded + p - 1) // p
    patch_offsets = F.pad(torch.cumsum(patches, 0), (1, 0))

    # Padded position -> source row. Positions past the sequence end read one patch earlier.
    grid_of_padded = torch.repeat_interleave(grid_ids, padded, output_size=total_padded)
    local = torch.arange(total_padded, device=device) - padded_offsets[grid_of_padded]
    local = torch.where(local < lengths[grid_of_padded], local, local - p)
    gather_idx = joffsets[grid_of_padded] + local

    # Source row -> padded position.
    grid_of_row = torch.repeat_interleave(grid_ids, lengths, output_size=total)
    unpad_idx = torch.arange(total, device=device) + (padded_offsets - joffsets)[grid_of_row]

    # Patch start offsets across all sequences.
    grid_of_patch = torch.repeat_interleave(grid_ids, patches, output_size=num_patches)
    local_patch = torch.arange(num_patches, device=device) - patch_offsets[grid_of_patch]
    starts = padded_offsets[grid_of_patch] + local_patch * p
    cu_seqlens = torch.cat([starts, padded_offsets[-1:]]).to(torch.int32)

    return _PatchLayout(gather_idx, unpad_idx, cu_seqlens, min(p, max(lshape)))


def _window_bounds(window_size: int | tuple[int, int]) -> tuple[int, int]:
    """Convert ``window_size`` to flash-attention ``(left, right)`` bounds."""
    if isinstance(window_size, int):
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        return (window_size // 2, window_size // 2)
    left, right = (int(w) for w in window_size)
    if left < -1 or right < -1:
        raise ValueError(f"window_size bounds must be >= -1, got {(left, right)}")
    return (left, right)


def _require_varlen(device: torch.device):
    """Return ``torch.nn.attention.varlen.varlen_attn``, or raise if it cannot run here."""
    try:
        from torch.nn.attention.varlen import varlen_attn
    except ImportError:
        raise RuntimeError(
            "Patch and window attention require torch.nn.attention.varlen.varlen_attn, which needs "
            f"PyTorch >= 2.11 (found {torch.__version__}). Use patch_size=0 and window_size=0 for "
            "global attention, which works on older PyTorch."
        ) from None
    if device.type != "cuda":
        raise RuntimeError(f"Patch and window attention require a CUDA device, got {device}")
    if torch.cuda.get_device_capability(device)[0] < 8:
        raise RuntimeError("Patch and window attention require an SM80 (Ampere) or newer GPU")
    return varlen_attn


def _compute_dtype(dtype: torch.dtype, compute_dtype: torch.dtype | None) -> torch.dtype:
    if compute_dtype is not None:
        if compute_dtype not in _HALF_DTYPES:
            raise TypeError(f"compute_dtype must be float16 or bfloat16, got {compute_dtype}")
        return compute_dtype
    if dtype in _HALF_DTYPES:
        return dtype
    if dtype == torch.float32:
        return torch.bfloat16
    raise TypeError(f"Patch and window attention support float16, bfloat16 and float32 inputs, got {dtype}")


def _varlen_attention(
    query: JaggedTensor,
    key: JaggedTensor,
    value: JaggedTensor,
    scale: float,
    patch_size: int,
    window_size: int | tuple[int, int],
    compute_dtype: torch.dtype | None,
) -> JaggedTensor:
    """Patch or window attention over each sequence with ``varlen_attn``."""
    varlen_attn = _require_varlen(query.device)
    out_dtype = query.jdata.dtype
    dtype = _compute_dtype(out_dtype, compute_dtype)
    lshape = cast(list[int], query.lshape)
    total = sum(lshape)
    if total == 0:
        empty = value.jdata.new_empty((0, value.jdata.size(1), value.jdata.size(2)))
        return query.jagged_like(empty.to(out_dtype))
    if total > _INT32_MAX:
        raise ValueError(f"Token count {total} exceeds the int32 range of cu_seqlens")

    q, k, v = (t.jdata.to(dtype) for t in (query, key, value))
    layout = None
    if patch_size > 0 and max(lshape) > patch_size:
        layout = _patch_layout(query.joffsets, lshape, patch_size)
        q, k, v = (t.index_select(0, layout.gather_idx) for t in (q, k, v))
        cu_seqlens, max_seqlen, bounds = layout.cu_seqlens, layout.max_seqlen, (-1, -1)
    else:
        # One sequence per grid: every grid fits in one patch, or window mode.
        cu_seqlens, max_seqlen = query.joffsets.to(torch.int32), max(lshape)
        bounds = (-1, -1) if patch_size > 0 else _window_bounds(window_size)

    out = varlen_attn(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, scale=scale, window_size=bounds)
    out = cast(torch.Tensor, out)  # a tensor unless return_aux is requested
    if layout is not None:
        out = out.index_select(0, layout.unpad_idx)
    return JaggedTensor.from_data_and_offsets(out.to(out_dtype), query.joffsets)


def scaled_dot_product_attention(
    query: JaggedTensor,
    key: JaggedTensor,
    value: JaggedTensor,
    scale: float,
    *,
    patch_size: int = 0,
    window_size: int | tuple[int, int] = 0,
    compute_dtype: torch.dtype | None = None,
) -> JaggedTensor:
    """Compute scaled dot-product attention over jagged sequences.

    Each batch element of the jagged inputs is an independent sequence. By default every token
    attends to every token in its own sequence (global attention). Two local modes restrict the
    keys each token sees, as used by serialized attention in Point Transformer V3:

    - **Patch attention** (``patch_size > 0``): each sequence is cut into consecutive patches of
      ``patch_size`` tokens and attention runs within each patch. A sequence no longer than
      ``patch_size`` forms one patch. A longer sequence is padded to a multiple of
      ``patch_size`` by repeating the tokens one patch earlier, matching Pointcept.
    - **Window attention** (``window_size != 0``): token ``i`` attends to tokens ``j`` with
      ``i - left <= j <= i + right`` in its sequence. An ``int`` ``W`` means
      ``(W // 2, W // 2)``; a tuple gives ``(left, right)`` directly, and ``-1`` is unbounded.

    For the local modes the inputs should already be ordered so that neighboring tokens are close
    in space, for example with :func:`fvdb.serialize` and :func:`fvdb.permute_jagged`.

    Global attention wraps :func:`torch.nn.functional.scaled_dot_product_attention` on PyTorch
    nested tensors. The backend (flash, memory-efficient, math, or cuDNN) follows
    :func:`torch.nn.attention.sdpa_kernel`. The local modes call
    ``torch.nn.attention.varlen.varlen_attn``, which needs PyTorch 2.11 or newer, a CUDA GPU
    of SM80 or newer, and computes in float16 or bfloat16.

    See `PyTorch SDPA documentation
    <https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html>`_
    for details on the underlying attention computation.

    Args:
        query (JaggedTensor): Query of shape ``[B, -1, H, E]`` where *B* is the number of
            sequences, *H* the number of heads, and *E* the embedding dimension.
        key (JaggedTensor): Key of shape ``[B, -1, H, E]``.
        value (JaggedTensor): Value of shape ``[B, -1, H, V]`` where *V* is the value dimension.
            Key and value must share the same sequence lengths.
        scale (float): Scaling factor applied to the dot products before softmax.
        patch_size (int): Patch length for patch attention. ``0`` disables it. Defaults to ``0``.
        window_size (int | tuple[int, int]): Window for window attention. ``0`` disables it.
            Defaults to ``0``.
        compute_dtype (torch.dtype | None): Kernel dtype for the local modes. ``None`` keeps
            float16 and bfloat16 inputs and computes float32 inputs in bfloat16. The output has
            the input dtype. Defaults to ``None``.

    Returns:
        JaggedTensor: Attention output of shape ``[B, -1, H, V]`` with the same jagged structure
        as the query.
    """

    for name, jt in [("query", query), ("key", key), ("value", value)]:
        if jt.ldim != 1:
            raise ValueError(
                f"{name} must have ldim == 1 (a flat list of tensors), got ldim == {jt.ldim}. "
                f"Nested jagged structures (list of lists) are not supported by SDPA."
            )
        if jt.jdata.ndim != 3:
            raise ValueError(f"{name}.jdata must be 3-dimensional (Total, H, D), got shape {tuple(jt.jdata.shape)}")

    if query.num_tensors != key.num_tensors or query.num_tensors != value.num_tensors:
        raise ValueError(
            f"query, key, and value must have the same batch size (num_tensors), "
            f"got {query.num_tensors}, {key.num_tensors}, {value.num_tensors}"
        )
    if not torch.equal(key.joffsets, value.joffsets):
        raise ValueError("key and value must have matching sequence lengths (joffsets differ)")

    if patch_size < 0:
        raise ValueError(f"patch_size must be non-negative, got {patch_size}")
    if patch_size > 0 or window_size != 0:
        if patch_size > 0 and window_size != 0:
            raise ValueError("Set at most one of patch_size and window_size")
        if not torch.equal(query.joffsets, key.joffsets):
            raise ValueError("Patch and window attention require query and key with matching sequence lengths")
        return _varlen_attention(query, key, value, scale, patch_size, window_size, compute_dtype)

    # Build zero-copy views first (cheap metadata, no data copy), then probe
    # which backend PyTorch will actually select for the given inputs.
    q_nested = _make_nested_view(query).transpose(1, 2)
    k_nested = _make_nested_view(key).transpose(1, 2)
    v_nested = _make_nested_view(value).transpose(1, 2)

    backend = torch._fused_sdp_choice(q_nested, k_nested, v_nested, None, 0.0, False, scale=scale)

    # Only the flash and math backends implement backward for these nested tensors.
    needs_grad = torch.is_grad_enabled() and any(t.jdata.requires_grad for t in (query, key, value))
    if needs_grad and backend not in (_SDP_FLASH, _SDP_MATH):
        backend = _SDP_MATH

    if backend == _SDP_FLASH:
        q_nested = _make_nested_tensor(query).transpose(1, 2)
        k_nested = _make_nested_tensor(key).transpose(1, 2)
        v_nested = _make_nested_tensor(value).transpose(1, 2)
    elif backend == _SDP_MATH:
        q_nested = _make_nested_tensor(query).transpose(1, 2).contiguous()
        k_nested = _make_nested_tensor(key).transpose(1, 2).contiguous()
        v_nested = _make_nested_tensor(value).transpose(1, 2).contiguous()

    if backend == _SDP_MATH:
        with sdpa_kernel(SDPBackend.MATH):
            out_nested = F.scaled_dot_product_attention(q_nested, k_nested, v_nested, scale=scale)
    else:
        out_nested = F.scaled_dot_product_attention(q_nested, k_nested, v_nested, scale=scale)

    # out_nested components have shape (H, L_i, D) -- convert back to (L_i, H, D)
    out_data = torch.cat([t.permute(1, 0, 2) for t in out_nested.unbind()], dim=0)

    return JaggedTensor.from_data_and_offsets(out_data, query.joffsets)
