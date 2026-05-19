import triton
import triton.language as tl

@triton.jit
def _layer_norm_fwd_fused(X, Y, W, B, Mean, Rstd, stride_x_row, stride_x_col, stride_y_row, stride_y_col, stride_w, stride_b, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    row = pid * BLOCK_SIZE

    # Initialize offsets
    offsets_x = row + tl.arange(0, BLOCK_SIZE)
    offsets_y = offsets_x
    mask = offsets_x < N

    # Load data
    x = tl.load(X + offsets_x, mask=mask, other=0.0)

    # Compute mean
    mean = tl.sum(x, axis=0) / N
    tl.store(Mean + row, mean)

    # Compute variance
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    tl.store(Rstd + row, rstd)

    # Normalize and apply scale and shift
    y = (x - mean) * rstd * tl.load(W + offsets_x, mask=mask, other=0.0) + tl.load(B + offsets_x, mask=mask, other=0.0)
    tl.store(Y + offsets_y, y, mask=mask)

@triton.jit
def _layer_norm_bwd_dx_fused(DY, DX, W, DW, DB, Mean, Rstd, stride_dy_row, stride_dy_col, stride_dx_row, stride_dx_col, stride_w, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    row = pid * BLOCK_SIZE

    # Initialize offsets
    offsets_dy = row + tl.arange(0, BLOCK_SIZE)
    offsets_dx = offsets_dy
    mask = offsets_dy < N

    # Load data
    dy = tl.load(DY + offsets_dy, mask=mask, other=0.0)
    w = tl.load(W + offsets_dy, mask=mask, other=0.0)
    mean = tl.load(Mean + row)
    rstd = tl.load(Rstd + row)

    # Compute gradients
    dx = dy * w * rstd
    dmean = -tl.sum(dx, axis=0)
    dvar = -0.5 * tl.sum(dx * (dy * w * rstd * rstd * rstd), axis=0)

    # Adjust for mean and variance
    dx += dmean / N
    dx += 2 * dvar * (dy * w * rstd * rstd) / N

    # Store gradients
    tl.store(DX + offsets_dx, dx, mask=mask)

    # Compute partial gradients for weights and biases
    partial_dw = dy * dx
    partial_db = dy

    # Reduce partial gradients
    partial_dw = tl.sum(partial_dw, axis=0)
    partial_db = tl.sum(partial_db, axis=0)

    # Store partial gradients
    tl.atomic_add(DW + row, partial_dw)
    tl.atomic_add(DB + row, partial_db)

@triton.jit
def _layer_norm_bwd_dwdb(DW, DB, FINAL_DW, FINAL_DB, stride_dw, stride_db, stride_final_dw, stride_final_db, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    row = pid * BLOCK_SIZE

    # Initialize offsets
    offsets_dw = row + tl.arange(0, BLOCK_SIZE)
    offsets_db = offsets_dw
    mask = offsets_dw < N

    # Load partial gradients
    partial_dw = tl.load(DW + offsets_dw, mask=mask, other=0.0)
    partial_db = tl.load(DB + offsets_db, mask=mask, other=0.0)

    # Reduce partial gradients
    final_dw = tl.sum(partial_dw, axis=0)
    final_db = tl.sum(partial_db, axis=0)

    # Store final gradients
    tl.store(FINAL_DW + row, final_dw)
    tl.store(FINAL_DB + row, final_db)

import torch
from torch.autograd import Function

class LayerNorm(Function):
    @staticmethod
    def forward(ctx, X, W, B):
        # Reshape inputs
        X = X.contiguous()
        W = W.contiguous()
        B = B.contiguous()
        N = X.shape[-1]
        M = X.numel() // N

        # Allocate buffers
        Y = torch.empty_like(X)
        Mean = torch.empty((M,), device=X.device, dtype=X.dtype)
        Rstd = torch.empty((M,), device=X.device, dtype=X.dtype)

        # Launch kernel
        BLOCK_SIZE = 256
        grid = (M // BLOCK_SIZE + 1,)
        _layer_norm_fwd_fused[grid](X, Y, W, B, Mean, Rstd, X.stride(0), X.stride(1), Y.stride(0), Y.stride(1), W.stride(0), B.stride(0), N, BLOCK_SIZE)

        # Save for backward
        ctx.save_for_backward(X, W, B, Mean, Rstd)
        ctx.N = N

        return Y

    @staticmethod
    def backward(ctx, grad_output):
        X, W, B, Mean, Rstd = ctx.saved_tensors
        N = ctx.N

        # Allocate buffers
        grad_input = torch.empty_like(X)
        grad_W = torch.empty_like(W)
        grad_B = torch.empty_like(B)
        partial_grad_W = torch.zeros_like(grad_W)
        partial_grad_B = torch.zeros_like(grad_B)

        # Launch kernel for DX
        BLOCK_SIZE = 256
        grid = (X.numel() // (N * BLOCK_SIZE) + 1,)
        _layer_norm_bwd_dx_fused[grid](grad_output, grad_input, W, partial_grad_W, partial_grad_B, Mean, Rstd, grad_output.stride(0), grad_output.stride(1), grad_input.stride(0), grad_input.stride(1), W.stride(0), N, BLOCK_SIZE)

        # Launch kernel for DW and DB
        grid = (W.numel() // BLOCK_SIZE + 1,)
        _layer_norm_bwd_dwdb[grid](partial_grad_W, partial_grad_B, grad_W, grad_B, partial_grad_W.stride(0), partial_grad_B.stride(0), grad_W.stride(0), grad_B.stride(0), N, BLOCK_SIZE)

        return grad_input, grad_W, grad_B

# Example usage
if __name__ == "__main__":
    X = torch.randn(1024, 1024, device='cuda')
    W = torch.randn(1024, device='cuda', requires_grad=True)
    B = torch.randn(1024, device='cuda', requires_grad=True)

    Y = LayerNorm.apply(X, W, B)
    loss = Y.sum()
    loss.backward()

    print(W.grad)
    print(B.grad)
