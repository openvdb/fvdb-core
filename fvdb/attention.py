# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import cast

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend

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
    data = jt.jdata  # (Total, H, D)
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


def scaled_dot_product_attention(
    query: JaggedTensor, key: JaggedTensor, value: JaggedTensor, scale: float
) -> JaggedTensor:
    """Compute scaled dot-product attention over jagged sequences.

    Wraps :func:`torch.nn.functional.scaled_dot_product_attention` so that it
    operates on :class:`JaggedTensor` inputs, where each batch element may have
    a different sequence length.  Internally the jagged tensors are converted to
    PyTorch nested tensors, attention is computed, and the result is converted
    back to a :class:`JaggedTensor`.

    The SDP backend (flash, memory-efficient, math, or cuDNN) is chosen
    automatically based on the context set by
    :func:`torch.nn.attention.sdpa_kernel`.  Each backend has different nested
    tensor requirements, so this function probes the actual backend selection
    via ``_fused_sdp_choice`` and constructs the nested tensors accordingly:

    - **Efficient / cuDNN**: zero-copy view into the JaggedTensor buffer
    - **Flash**: copy-based nested tensor (required by the kernel)
    - **Math**: copy-based nested tensor, made contiguous after transpose

    See `PyTorch SDPA documentation
    <https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html>`_
    for details on the underlying attention computation.

    Args:
        query: Query JaggedTensor of shape ``[B, -1, H, E]`` where *B* is the
            batch size (number of jagged sequences), *H* is the number of
            attention heads, and *E* is the embedding dimension.
        key: Key JaggedTensor of shape ``[B, -1, H, E]``.
        value: Value JaggedTensor of shape ``[B, -1, H, V]`` where *V* is the
            value dimension.  Key and value must share the same sequence
            lengths.
        scale: Scaling factor applied to the dot products before softmax.

    Returns:
        A JaggedTensor of shape ``[B, -1, H, V]`` containing the attention
        output, with the same jagged structure as the query.
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

    # Build zero-copy views first (cheap metadata, no data copy), then probe
    # which backend PyTorch will actually select for the given inputs.
    q_nested = _make_nested_view(query).transpose(1, 2)
    k_nested = _make_nested_view(key).transpose(1, 2)
    v_nested = _make_nested_view(value).transpose(1, 2)

    backend = torch._fused_sdp_choice(q_nested, k_nested, v_nested, None, 0.0, False, scale=scale)

    if backend == _SDP_FLASH:
        q_nested = _make_nested_tensor(query).transpose(1, 2)
        k_nested = _make_nested_tensor(key).transpose(1, 2)
        v_nested = _make_nested_tensor(value).transpose(1, 2)
    elif backend == _SDP_MATH:
        q_nested = _make_nested_tensor(query).transpose(1, 2).contiguous()
        k_nested = _make_nested_tensor(key).transpose(1, 2).contiguous()
        v_nested = _make_nested_tensor(value).transpose(1, 2).contiguous()

    out_nested = F.scaled_dot_product_attention(q_nested, k_nested, v_nested, scale=scale)

    # out_nested components have shape (H, L_i, D) -- convert back to (L_i, H, D)
    out_data = torch.cat([t.permute(1, 0, 2) for t in out_nested.unbind()], dim=0)

    return JaggedTensor.from_data_and_offsets(out_data, query.joffsets)
