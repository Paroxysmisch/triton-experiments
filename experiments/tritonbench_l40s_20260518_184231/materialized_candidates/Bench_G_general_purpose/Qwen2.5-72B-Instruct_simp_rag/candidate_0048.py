import triton
import triton.language as tl

@triton.jit
def _l2_norm_fwd_1pass_kernel(
    X,
    Y,
    stride_m,
    stride_n,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Implements a forward kernel for L2 normalization.
    
    Parameters:
    X (tl.tensor): Input tensor where each row represents a feature vector.
    Y (tl.tensor): Output tensor for normalized features.
    stride_m (int): Stride to access elements along the row dimension.
    stride_n (int): Stride to access elements along the column dimension.
    N (int): Number of features per row.
    eps (float): Small epsilon value for numerical stability in division.
    BLOCK_SIZE (tl.constexpr): Block size used for partitioning computations.
    """
    row = tl.program_id(0)
    X += row * stride_m
    Y += row * stride_m

    _sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols * stride_n, mask=cols < N, other=0.0).to(tl.float32)
        _sum += a * a
    norm = tl.sqrt(tl.sum(_sum) + eps)

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols * stride_n, mask=mask, other=0.0).to(tl.float32)
        y = x / norm
        tl.store(Y + cols * stride_n, y, mask=mask)

@triton.jit
def _l2_norm_bwd_kernel(
    X,
    DY,
    DX,
    stride_m,
    stride_n,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Implements a backward kernel for L2 normalization.
    
    Parameters:
    X (tl.tensor): Input tensor where each row represents a feature vector.
    DY (tl.tensor): Gradient of the loss with respect to the output tensor.
    DX (tl.tensor): Gradient of the loss with respect to the input tensor.
    stride_m (int): Stride to access elements along the row dimension.
    stride_n (int): Stride to access elements along the column dimension.
    N (int): Number of features per row.
    eps (float): Small epsilon value for numerical stability in division.
    BLOCK_SIZE (tl.constexpr): Block size used for partitioning computations.
    """
    row = tl.program_id(0)
    X += row * stride_m
    DY += row * stride_m
    DX += row * stride_m

    _sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols * stride_n, mask=cols < N, other=0.0).to(tl.float32)
        _sum += a * a
    norm = tl.sqrt(tl.sum(_sum) + eps)

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols * stride_n, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(DY + cols * stride_n, mask=mask, other=0.0).to(tl.float32)
        dx = dy / norm - (tl.sum(dy * x) / (norm * norm * norm)) * x
        tl.store(DX + cols * stride_n, dx, mask=mask)

import torch

def _l2_norm_fwd(X, Y, eps=1e-6):
    """
    Wrapper function for the L2 normalization forward pass.
    
    Parameters:
    X (torch.Tensor): Input tensor of shape (M, N).
    Y (torch.Tensor): Output tensor of shape (M, N).
    eps (float): Small epsilon value for numerical stability.
    """
    M, N = X.shape
    assert Y.shape == X.shape, "Output tensor must have the same shape as input tensor"
    
    # Reshape tensors to 1D for Triton kernel
    X = X.view(-1)
    Y = Y.view(-1)
    
    # Launch the Triton kernel
    grid = (M, )
    _l2_norm_fwd_1pass_kernel[grid](
        X, Y, N, eps, BLOCK_SIZE=128
    )

def _l2_norm_bwd(X, DY, DX, eps=1e-6):
    """
    Wrapper function for the L2 normalization backward pass.
    
    Parameters:
    X (torch.Tensor): Input tensor of shape (M, N).
    DY (torch.Tensor): Gradient of the loss with respect to the output tensor of shape (M, N).
    DX (torch.Tensor): Gradient of the loss with respect to the input tensor of shape (M, N).
    eps (float): Small epsilon value for numerical stability.
    """
    M, N = X.shape
    assert DY.shape == X.shape, "Gradient tensor must have the same shape as input tensor"
    assert DX.shape == X.shape, "Output gradient tensor must have the same shape as input tensor"
    
    # Reshape tensors to 1D for Triton kernel
    X = X.view(-1)
    DY = DY.view(-1)
    DX = DX.view(-1)
    
    # Launch the Triton kernel
    grid = (M, )
    _l2_norm_bwd_kernel[grid](
        X, DY, DX, N, eps, BLOCK_SIZE=128
    )

import torch

# Create input tensor
X = torch.randn(1024, 512, device='cuda')
Y = torch.empty_like(X)
DX = torch.empty_like(X)
DY = torch.randn_like(X)

# Forward pass
_l2_norm_fwd(X, Y)

# Backward pass
_l2_norm_bwd(X, DY, DX)

# Print results
print("Normalized output (Y):", Y)
print("Gradient w.r.t. input (DX):", DX)
