import torch
import triton
import triton.language as tl
from torch import Tensor
from typing import Optional

@triton.jit
def _layer_norm_forward_kernel(
    X, Y, W, B, Mean, RSTD,
    user_shape_0: int, user_shape_1: int,
    stride_x_row: int, stride_y_row: int,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for forward layer norm.
    Normalizes a normalized layer across all but the last dimension.
    Args:
        X (Tensor): Input tensor.
        Y (Tensor): Output tensor.
        W (Tensor): Weight tensor.
        B (Tensor): Bias tensor.
        Mean (Tensor): Mean tensor.
        RSTD (Tensor): RSTD tensor.
        user_shape_0 (int): User input shape.
        user_shape_1 (int): User input shape.
        stride_x_row (int): Stride for X rows.
        stride_y_row (int): Stride for Y rows.
        BLOCK_SIZE (tl.constexpr): Size of each block.
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < user_shape_1

    X += row * stride_x_row
    Y += row * stride_y_row

    x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0).to(tl.float32)

    mean = tl.sum(x, axis=0) / user_shape_1
    var = tl.sum((x - mean) * (x - mean), axis=0) / user_shape_1
    rstd = 1 / tl.sqrt(var + 1e-5)

    tl.store(Mean + row, mean)
    tl.store(RSTD + row, rstd)

    x_hat = (x - mean) * rstd
    y = x_hat * w + b

    tl.store(Y + cols, y, mask=mask)


def layer_norm_forward(X: Tensor, Weight: Tensor, Bias: Tensor):
    """
    Wrapper function for forward kernel.
    Args:
        X (Tensor): Input tensor.
        Weight (Tensor): Weight tensor.
        Bias (Tensor): Bias tensor.
    Returns:
        (Tensor, Tensor, Tensor): Output tensor, mean tensor, rstd tensor.
    """
    assert X.shape[-1] == Weight.shape[0] == Bias.shape[0], "Incompatible shapes"
    user_shape = X.shape
    X = X.view(-1, X.shape[-1])
    Y = torch.empty_like(X)
    Mean = torch.empty((X.shape[0],), dtype=torch.float32, device=X.device)
    RSTD = torch.empty((X.shape[0],), dtype=torch.float32, device=X.device)

    sm_count = torch.cuda.get_device_properties(X.device).multi_processor_count
    BLOCK_SIZE, num_warps = calculate_settings(Weight.shape[0])

    _layer_norm_forward_kernel[(X.shape[0],)](
        X, Y, Weight, Bias, Mean, RSTD,
        user_shape[0], user_shape[1],
        X.stride(0), Y.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        sm_count=sm_count,
    )
    return Y.view(*user_shape), Mean, RSTD


@triton.jit
def _layer_norm_backward_kernel(
    DX, DY, DW, DB,
    X, W, B, Mean, RSTD,
    user_shape_0, user_shape_1,
    stride_dx_row: int, stride_dy_row: int,
    BLOCK_SIZE: tl.constexpr,
    sm_count: tl.constexpr,
):
    """
    Triton kernel for backward layer norm.
    Args:
        DX (Tensor): Gradient tensor.
        DY (Tensor): Output gradient tensor.
        DW (Tensor): Weight gradient tensor.
        DB (Tensor): Bias gradient tensor.
        X (Tensor): Input tensor.
        W (Tensor): Weight tensor.
        B (Tensor): Bias tensor.
        Mean (Tensor): Mean tensor.
        RSTD (Tensor): RSTD tensor.
        user_shape_0 (int): User input shape.
        user_shape_1 (int): User input shape.
        stride_dx_row (int): Stride for DX rows.
        stride_dy_row (int): Stride for DY rows.
        BLOCK_SIZE (tl.constexpr): Size of each block.
        sm_count (tl.constexpr): Number of SMs.
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < user_shape_1

    DY += row * stride_dy_row
    DX += row * stride_dx_row

    dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
    x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0).to(tl.float32)
    mean = tl.load(Mean + row)
    rstd = tl.load(RSTD + row)

    xhat = (x - mean) * rstd
    wdy = w * dy
    c1 = tl.sum(xhat * wdy, axis=0) / user_shape_1
    c2 = tl.sum(wdy, axis=0) / user_shape_1
    dx = (wdy - (xhat * c1 + c2)) * rstd

    tl.store(DX + cols, dx, mask=mask)

    if sm_count == 1:
        tl.debug_barrier()

    tl.store(DW + cols * user_shape_0 + row, tl.sum(dx * xhat, axis=0), mask=mask)
    tl.store(DB + cols * user_shape_0 + row, tl.sum(dx, axis=0), mask=mask)


def layer_norm_backward(DY: Tensor, X: Tensor, Weight: Tensor, Bias: Tensor, Mean: Tensor, RSTD: Tensor):
    """
    Wrapper function for backward kernel.
    Args:
        DY (Tensor): Output gradient tensor.
        X (Tensor): Input tensor.
        Weight (Tensor): Weight tensor.
        Bias (Tensor): Bias tensor.
        Mean (Tensor): Mean tensor.
        RSTD (Tensor): RSTD tensor.
    Returns:
        (Tensor, Tensor, Tensor): Gradient tensor, weight gradient tensor, bias gradient tensor.
    """
    assert DY.shape[-1] == Weight.shape[0] == Bias.shape[0], "Incompatible shapes"
    user_shape = DY.shape
    DY = DY.view(-1, DY.shape[-1])
    DX = torch.empty_like(DY)
    sm_count = torch.cuda.get_device_properties(DY.device).multi_processor_count
    BLOCK_SIZE, num_warps = calculate_settings(DY.shape[1])

    grid = (DY.shape[0],)
    _layer_norm_backward_kernel[grid](
        DX, DY, Weight, Bias, Mean, RSTD,
        user_shape[0], user_shape[1],
        X.stride(0), DY.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        sm_count=sm_count,
    )
    DW = torch.zeros((user_shape[1], user_shape[0]), dtype=X.dtype, device=X.device)
    DB = torch.zeros((user_shape[1], user_shape[0]), dtype=X.dtype, device=X.device)
    _layer_norm_backward_kernel[grid](
        DX, DY, DW, DB,
        X, Weight, Bias, Mean, RSTD,
        user_shape[0], user_shape[1],
        X.stride(0), DY.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        sm_count=sm_count,
    )
    return DX.view(*user_shape), DW.sum(1), DB.sum(1)


class LigerLayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, normalized_shape, weight, bias):
        """
        Args:
            ctx (ContextManager): A context manager for storing backpropagation information.
            x (Tensor): Input tensor.
            normalized_shape (Tensor): Normalized shape.
            weight (Tensor): Weight tensor.
            bias (Tensor): Bias tensor.
        Returns:
            Tensor: Output tensor.
        """
        y, mean, rstd = layer_norm_forward(x, weight, bias)
        ctx.save_for_backward(x, weight, bias, mean, rstd)
        return y

    @staticmethod
    def backward(ctx, dy):
        """
        Args:
            ctx (ContextManager): A context manager for storing backpropagation information.
            dy (Tensor): Output gradient tensor.
        Returns:
            Tensor: Gradient tensor.
            None: Gradient tensor is not used for normalized_shape.
            Tensor: Weight gradient tensor.
            Tensor: Bias gradient tensor.
        """
        x, w, b, m, v = ctx.saved_tensors
        dx, dw, db = layer_norm_backward(dy, x, w, b, m, v)
        return dx, None, dw, db


def liger_layer_norm(x, normalized_shape, weight, bias):
    """
    Args:
        x (Tensor): Input tensor.
        normalized_shape (Tensor): Normalized shape.
        weight (Tensor): Weight tensor.
        bias (Tensor): Bias tensor.
    Returns:
