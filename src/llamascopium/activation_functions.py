"""Activation functions for sparse autoencoders.

This module contains custom activation functions used in sparse autoencoder implementations,
including JumpReLU and its associated STEFunction.
"""

import torch
import torch.distributed.tensor
from torch.distributed.device_mesh import DeviceMesh
from typing_extensions import cast

from llamascopium.utils.distributed import DimMap


class STEFunction(torch.autograd.Function):
    """
    STE function for the jumprelu activation function.
    """

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        jumprelu_threshold: torch.Tensor,
        jumprelu_threshold_window: float,
        dims_to_keep_in_bwd: tuple[int, ...],
    ):
        mask = input.gt(jumprelu_threshold)
        ctx.save_for_backward(
            input,
            jumprelu_threshold,
            torch.tensor(jumprelu_threshold_window, dtype=input.dtype, device=input.device),
            mask,
        )
        ctx.dims_to_keep_in_bwd = dims_to_keep_in_bwd
        return input * mask

    @staticmethod
    def backward(ctx, *grad_outputs: torch.Tensor, **args):
        assert len(grad_outputs) == 1
        grad_output = grad_outputs[0]

        input, jumprelu_threshold, jumprelu_threshold_window, mask = ctx.saved_tensors

        grad_jumprelu_threshold = (
            torch.where(
                ((input - jumprelu_threshold).abs() < jumprelu_threshold_window * 0.5) * (input > 0.0),
                -jumprelu_threshold / jumprelu_threshold_window,
                0.0,
            )
            * grad_output
        )

        x_grad = grad_output * mask

        grad_jumprelu_threshold = grad_jumprelu_threshold.sum(
            dim=tuple(
                i
                for i in range(grad_jumprelu_threshold.ndim)
                if i not in ctx.dims_to_keep_in_bwd and i - grad_jumprelu_threshold.ndim not in ctx.dims_to_keep_in_bwd
            )
        )

        return x_grad, grad_jumprelu_threshold, None, None


class JumpReLUL0Function(torch.autograd.Function):
    """
    STE approximation for the JumpReLU L0 indicator.
    """

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        jumprelu_threshold: torch.Tensor,
        jumprelu_threshold_window: float,
        dims_to_keep_in_bwd: tuple[int, ...],
    ):
        active = input.gt(jumprelu_threshold)
        ctx.save_for_backward(
            input,
            jumprelu_threshold,
            torch.tensor(jumprelu_threshold_window, dtype=input.dtype, device=input.device),
        )
        ctx.dims_to_keep_in_bwd = dims_to_keep_in_bwd
        return active.to(input.dtype)

    @staticmethod
    def backward(ctx, *grad_outputs: torch.Tensor, **args):
        assert len(grad_outputs) == 1
        grad_output = grad_outputs[0]

        input, jumprelu_threshold, jumprelu_threshold_window = ctx.saved_tensors
        threshold_window = (input - jumprelu_threshold).abs() < jumprelu_threshold_window * 0.5
        grad_jumprelu_threshold = (
            torch.where(
                threshold_window * (input > 0.0),
                -jumprelu_threshold / jumprelu_threshold_window,
                0.0,
            )
            * grad_output
        )
        grad_jumprelu_threshold = grad_jumprelu_threshold.sum(
            dim=tuple(
                i
                for i in range(grad_jumprelu_threshold.ndim)
                if i not in ctx.dims_to_keep_in_bwd and i - grad_jumprelu_threshold.ndim not in ctx.dims_to_keep_in_bwd
            )
        )

        return torch.zeros_like(input), grad_jumprelu_threshold, None, None


class JumpReLU(torch.nn.Module):
    """
    JumpReLU activation function.
    """

    def __init__(
        self,
        jumprelu_threshold_window: float,
        *,
        shape: tuple[int, ...],
        dims_to_keep_in_bwd: tuple[int, ...] = (-1,),
        device: torch.device | str | None = None,
        dtype: torch.dtype,
        device_mesh: DeviceMesh | None = None,
    ):
        """
        Args:
            jumprelu_threshold_window: The window size for the jumprelu threshold.
            shape: The shape of the input tensor.
            dims_to_keep_in_bwd: The dimensions to keep in the backward pass.
                We want to keep gradients in the backward pass the same shape as JumpReLU threshold.
                For example in CLT, we have (n_layers, d_sae) as the shape of the threshold.
                For SAEs, we have (d_sae,).
            device: The device to use.
            dtype: The dtype to use.
            device_mesh: The device mesh to use.
        """
        super(JumpReLU, self).__init__()
        self.jumprelu_threshold_window = jumprelu_threshold_window
        self.shape = shape
        self.dims_to_keep_in_bwd = dims_to_keep_in_bwd
        self.device_mesh = device_mesh
        self.dtype = dtype
        if device_mesh is None:
            self.log_jumprelu_threshold = torch.nn.Parameter(torch.empty(shape, device=device, dtype=dtype))
        else:
            self.log_jumprelu_threshold = torch.nn.Parameter(
                torch.distributed.tensor.empty(
                    shape,
                    dtype=dtype,
                    device_mesh=device_mesh,
                    placements=self.dim_maps()["log_jumprelu_threshold"].placements(device_mesh),
                )
            )
        self._check_dims_to_keep_in_bwd()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return cast(
            torch.Tensor,
            STEFunction.apply(
                input.to(self.dtype),
                self.log_jumprelu_threshold.exp(),
                self.jumprelu_threshold_window,
                self.dims_to_keep_in_bwd,
            ),
        ).to(input.dtype)

    def get_jumprelu_threshold(self) -> torch.Tensor:
        return self.log_jumprelu_threshold.exp()

    def l0(self, input: torch.Tensor) -> torch.Tensor:
        return cast(
            torch.Tensor,
            JumpReLUL0Function.apply(
                input.to(self.dtype),
                self.log_jumprelu_threshold.exp(),
                self.jumprelu_threshold_window,
                self.dims_to_keep_in_bwd,
            ),
        ).to(input.dtype)

    def dim_maps(self) -> dict[str, DimMap]:
        return {
            "log_jumprelu_threshold": DimMap({"model": len(self.shape) - 1}),
        }

    def override_dtypes(self) -> dict[str, torch.dtype]:
        return {
            "log_jumprelu_threshold": self.dtype,
        }

    def _check_dims_to_keep_in_bwd(self):
        assert len(self.dims_to_keep_in_bwd) == len(self.shape), (
            f"dims_to_keep_in_bwd must have the same length as shape, got {self.dims_to_keep_in_bwd} and {self.shape}"
        )
