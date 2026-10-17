# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

from contextlib import ExitStack

import pytest
import torch
import torch.nn.functional as nnf

from fvdb.utils.metrics import fused_l1_ssim_loss, ssim


@pytest.fixture(params=["cuda", "dgx"])
def image_device(request):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if request.param == "dgx":
        pytest.importorskip("torch_dgx")
    return request.param


def make_images(shape):
    generator = torch.Generator().manual_seed(41)
    img1 = torch.rand(shape, generator=generator)
    img2 = torch.rand(shape, generator=generator)
    img2.flatten()[::7] = img1.flatten()[::7]  # Exercise the zero L1 subgradient.
    return img1, img2


@pytest.mark.parametrize("shape", [(1, 3, 1, 1), (1, 3, 16, 19), (1, 3, 33, 19), (3, 4, 17, 23)])
@pytest.mark.parametrize("weight", [0.0, 0.2, 0.8, 1.0])
def test_fused_l1_ssim_forward_backward(image_device, shape, weight):
    img1, img2 = make_images(shape)
    reference = img1.to("cuda").requires_grad_()
    reference_target = img2.to("cuda")
    actual = img1.to(image_device).requires_grad_()
    actual_target = img2.to(image_device)

    expected_l1 = nnf.l1_loss(reference, reference_target)
    expected_ssim = 1.0 - ssim(reference, reference_target)
    expected = torch.lerp(expected_l1, expected_ssim, weight)
    result = fused_l1_ssim_loss(actual, actual_target, weight)
    assert result.ndim == 0
    torch.testing.assert_close(result.cpu(), expected.detach().cpu(), rtol=2e-5, atol=2e-6)

    expected.backward(torch.tensor(-2.75, device="cuda"))
    result.backward(torch.tensor(-2.75, device=image_device))
    torch.testing.assert_close(actual.grad.cpu(), reference.grad.cpu(), rtol=2e-4, atol=2e-6)


def test_fused_l1_ssim_noncontiguous_inputs(image_device):
    img1, img2 = make_images((2, 3, 33, 38))
    reference = img1.to("cuda").requires_grad_()
    actual = img1.to(image_device).requires_grad_()
    reference_target = img2.to("cuda")[..., 1::2]
    actual_target = img2.to(image_device)[..., 1::2]
    expected = torch.lerp(
        nnf.l1_loss(reference[..., 1::2], reference_target),
        1.0 - ssim(reference[..., 1::2], reference_target),
        0.2,
    )
    result = fused_l1_ssim_loss(actual[..., 1::2], actual_target, 0.2)
    expected.backward()
    result.backward()
    torch.testing.assert_close(result.cpu(), expected.detach().cpu(), rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(actual.grad.cpu(), reference.grad.cpu(), rtol=2e-4, atol=2e-6)


def test_fused_l1_ssim_nondefault_streams(image_device):
    img1, img2 = make_images((3, 3, 17, 19))
    reference = img1.to("cuda").requires_grad_()
    target = img2.to("cuda")
    expected = torch.lerp(nnf.l1_loss(reference, target), 1.0 - ssim(reference, target), 0.2)
    expected.backward()

    device_ids = range(torch.cuda.device_count()) if image_device == "dgx" else [torch.cuda.current_device()]
    streams = [torch.cuda.Stream(device=device) for device in device_ids]
    with ExitStack() as stack:
        for stream in streams:
            stack.enter_context(torch.cuda.stream(stream))
        actual = img1.to(image_device).requires_grad_()
        actual_target = img2.to(image_device)
        current_device = torch.cuda.current_device()
        result = fused_l1_ssim_loss(actual, actual_target, 0.2)
        if image_device == "cuda":
            assert torch.cuda.current_device() == current_device
        result.backward()
        if image_device == "cuda":
            assert torch.cuda.current_device() == current_device

    for stream in streams:
        torch.cuda.current_stream(stream.device).wait_stream(stream)
    torch.testing.assert_close(result.cpu(), expected.detach().cpu(), rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(actual.grad.cpu(), reference.grad.cpu(), rtol=2e-4, atol=2e-6)


def test_fused_l1_ssim_without_gradients(image_device):
    img1, img2 = [value.to(image_device) for value in make_images((1, 3, 17, 19))]
    img1.requires_grad_()
    with torch.no_grad():
        result = fused_l1_ssim_loss(img1, img2, 0.2)
        expected = torch.lerp(nnf.l1_loss(img1, img2), 1.0 - ssim(img1, img2, train=False), 0.2)
    assert not result.requires_grad
    torch.testing.assert_close(result.cpu(), expected.cpu(), rtol=2e-5, atol=2e-6)


def test_fused_l1_ssim_rejects_target_gradients():
    img1 = torch.zeros(1, 3, 17, 19)
    img2 = torch.zeros_like(img1, requires_grad=True)
    with pytest.raises(ValueError, match="target gradients"):
        fused_l1_ssim_loss(img1, img2, 0.2)


def test_fused_l1_ssim_rejects_empty_images():
    empty = torch.empty(1, 3, 0, 19)
    with pytest.raises(ValueError, match="nonempty"):
        fused_l1_ssim_loss(empty, empty, 0.2)


def test_fused_l1_ssim_native_parameters(image_device):
    from fvdb.utils.metrics.ssim import FusedSSIMMap

    img1, img2 = make_images((2, 3, 17, 19))
    reference = img1.to("cuda").requires_grad_()
    target = img2.to("cuda")
    C1, C2, weight = 0.02**2, 0.06**2, 0.35
    expected_l1 = nnf.l1_loss(reference, target)
    expected_ssim = 1.0 - FusedSSIMMap.apply(C1, C2, reference, target, "same", True).mean()
    expected = torch.lerp(expected_l1, expected_ssim, weight)
    expected.backward(torch.tensor(-1.5, device="cuda"))

    actual = img1.to(image_device)
    actual_target = img2.to(image_device)
    result, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = torch.ops.fvdb._fused_l1_ssim.default(
        actual, actual_target, weight, C1, C2, True
    )
    grad = torch.ops.fvdb._fused_l1_ssim_backward.default(
        actual,
        actual_target,
        torch.tensor(-1.5, device=image_device),
        dm_dmu1,
        dm_dsigma1_sq,
        dm_dsigma12,
        weight,
        C1,
        C2,
    )
    assert result.ndim == 0
    torch.testing.assert_close(result.cpu(), expected.detach().cpu(), rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(grad.cpu(), reference.grad.cpu(), rtol=2e-4, atol=2e-6)
