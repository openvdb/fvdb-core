# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as nnf

from fvdb.utils.metrics import fused_l1_loss, ssim

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize(
    "shape",
    [
        (1, 3, 1, 1),  # A single pixel in a partial tile.
        (1, 3, 16, 19),  # One complete tile row, with a partial width tile.
        (1, 3, 33, 19),  # Multiple tile rows with a partial final row.
        (3, 4, 17, 23),  # Multiple images and channels.
        (2, 1, 32, 16),
    ],
)
@pytest.mark.parametrize("needs_grad", [(True, False), (False, True), (True, True)])
def test_fused_l1_forward_backward(shape, needs_grad):
    generator = torch.Generator().manual_seed(17)
    img1 = torch.randn(shape, generator=generator)
    img2 = torch.randn(shape, generator=generator)
    img2.flatten()[::7] = img1.flatten()[::7]  # Include the zero subgradient.
    reference = [value.clone().requires_grad_(need) for value, need in zip((img1, img2), needs_grad)]
    actual = [value.to("cuda").requires_grad_(need) for value, need in zip((img1, img2), needs_grad)]

    expected_loss = nnf.l1_loss(*reference)
    actual_loss = fused_l1_loss(*actual)
    torch.testing.assert_close(actual_loss.cpu(), expected_loss.detach(), rtol=2e-6, atol=1e-6)
    expected_loss.backward(torch.tensor(-2.75))
    actual_loss.backward(torch.tensor(-2.75, device="cuda"))
    for expected, result, need in zip(reference, actual, needs_grad):
        if need:
            torch.testing.assert_close(result.grad.cpu(), expected.grad)
        else:
            assert result.grad is None


@pytest.mark.parametrize("layout", ["strided", "offset"])
def test_fused_l1_input_views(layout):
    generator = torch.Generator().manual_seed(23)
    values = [torch.randn(1, 3, 33, 38, generator=generator) for _ in range(2)]
    reference = [value.clone().requires_grad_() for value in values]
    actual = [value.to("cuda").requires_grad_() for value in values]

    def view(value):
        if layout == "strided":
            return value[..., 1::2]
        # A contiguous image view with a nonzero storage offset.
        return value.flatten()[1 : 1 + 3 * 17 * 19].view(1, 3, 17, 19)

    expected = nnf.l1_loss(*(view(value) for value in reference))
    result = fused_l1_loss(*(view(value) for value in actual))
    torch.testing.assert_close(result.cpu(), expected.detach(), rtol=2e-6, atol=1e-6)
    expected.backward()
    result.backward()
    for expected_input, actual_input in zip(reference, actual):
        torch.testing.assert_close(actual_input.grad.cpu(), expected_input.grad)


@pytest.mark.parametrize("shape", [(0, 3, 17, 19), (1, 0, 17, 19), (1, 3, 0, 19), (1, 3, 17, 0)])
def test_fused_l1_empty(shape):
    inputs = [torch.empty(shape, device="cuda", requires_grad=True) for _ in range(2)]
    result = fused_l1_loss(*inputs)
    assert result.ndim == 0
    assert torch.isnan(result.cpu())
    result.backward()
    for value in inputs:
        assert value.grad is not None
        assert value.grad.shape == value.shape


def test_fused_l1_nonfinite_residuals():
    img1 = torch.tensor([0.0, 2.0, -2.0, float("inf"), -float("inf"), float("nan"), float("inf")])
    img2 = torch.tensor([0.0, 1.0, -1.0, 0.0, 0.0, 1.0, float("inf")])
    reference = [value.view(1, 1, 1, -1).requires_grad_() for value in (img1, img2)]
    actual = [value.detach().to("cuda").requires_grad_() for value in reference]
    expected = nnf.l1_loss(*reference)
    result = fused_l1_loss(*actual)
    torch.testing.assert_close(result.cpu(), expected.detach(), equal_nan=True)
    expected.backward()
    result.backward()
    for expected_input, actual_input in zip(reference, actual):
        torch.testing.assert_close(actual_input.grad.cpu(), expected_input.grad, equal_nan=True)


@pytest.mark.parametrize(
    "case,message",
    [("dimension", "NCHW"), ("shape", "identical image shapes"), ("dtype", "float32")],
)
def test_fused_l1_invalid_inputs(case, message):
    img1 = torch.empty((1, 3, 17, 19), device="cuda")
    img2 = torch.empty_like(img1)
    if case == "dimension":
        img1, img2 = img1[0], img2[0]
    elif case == "shape":
        img2 = img2[:, :1]
    else:
        img2 = img2.double()
    with pytest.raises(ValueError, match=message):
        fused_l1_loss(img1, img2)


def test_fused_l1_nondefault_streams():
    stream = torch.cuda.Stream()
    values = [torch.randn(3, 3, 17, 19) for _ in range(2)]
    reference = [value.clone().requires_grad_() for value in values]
    expected = nnf.l1_loss(*reference)
    expected.backward()

    with torch.cuda.stream(stream):
        actual = [value.to("cuda").requires_grad_() for value in values]
        current_device = torch.cuda.current_device()
        result = fused_l1_loss(*actual)
        assert torch.cuda.current_device() == current_device
        result.backward()

    torch.cuda.current_stream(stream.device).wait_stream(stream)
    torch.testing.assert_close(result.cpu(), expected.detach(), rtol=2e-6, atol=1e-6)
    for expected_input, actual_input in zip(reference, actual):
        torch.testing.assert_close(actual_input.grad.cpu(), expected_input.grad)


def test_fused_l1_with_ssim():
    generator = torch.Generator().manual_seed(41)
    img1 = torch.rand(1, 3, 33, 47, generator=generator)
    img2 = torch.rand(1, 3, 33, 47, generator=generator)
    reference = img1.to("cuda").requires_grad_()
    target = img2.to("cuda")
    actual = img1.to("cuda").requires_grad_()
    actual_target = img2.to("cuda")

    expected = torch.lerp(nnf.l1_loss(reference, target), 1.0 - ssim(reference, target), 0.2)
    result = torch.lerp(fused_l1_loss(actual, actual_target), 1.0 - ssim(actual, actual_target), 0.2)
    expected.backward()
    result.backward()
    torch.testing.assert_close(result.cpu(), expected.detach().cpu(), rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(actual.grad.cpu(), reference.grad.cpu(), rtol=2e-4, atol=2e-6)
