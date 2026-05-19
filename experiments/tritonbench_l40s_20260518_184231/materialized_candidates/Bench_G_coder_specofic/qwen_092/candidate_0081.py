import torch
import triton
import triton.language as tl

class LayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, B):
        # Get input dimensions
        m, n = X.shape
        BLOCK_SIZE = 128
        GROUP_SIZE_M = 8
        GROUP_SIZE_N = BLOCK_SIZE // GROUP_SIZE_M

        # Allocate output and buffers
        Y = torch.zeros_like(X)
        Mean = torch.zeros((m,), device=X.device)
        Rstd = torch.zeros((m,), device=X.device)

        # Forward pass
        grid = (triton.cdiv(m, GROUP_SIZE_M), n)
        _layer_norm_fwd_fused[grid](X, Y, W, B, Mean, Rstd, BLOCK_SIZE, GROUP_SIZE_M)

        ctx.save_for_backward(X, W, B, Mean, Rstd)
        return Y

    @staticmethod
    def backward(ctx, DY):
        X, W, B, Mean, Rstd = ctx.saved_tensors
        m, n = X.shape
        BLOCK_SIZE = 128
        GROUP_SIZE_M = 8
        GROUP_SIZE_N = BLOCK_SIZE // GROUP_SIZE_M

        # Allocate gradients
        DX = torch.zeros_like(X)
        DW = torch.zeros_like(W)
        DB = torch.zeros_like(B)
        FINAL_DW = torch.zeros_like(W)
        FINAL_DB = torch.zeros_like(B)

        # Backward pass
        grid = (triton.cdiv(m, GROUP_SIZE_M), n)
        _layer_norm_bwd_dx_fused[grid](DY, X, W, Mean, Rstd, DX, DW, DB, GROUP_SIZE_M, GROUP_SIZE_N)
        _layer_norm_bwd_dwdb[grid](DW, DB, FINAL_DW, FINAL_DB, GROUP_SIZE_M, GROUP_SIZE_N)

        return DX, FINAL_DW, FINAL_DB

# Register the custom function
layer_norm = LayerNorm.apply
