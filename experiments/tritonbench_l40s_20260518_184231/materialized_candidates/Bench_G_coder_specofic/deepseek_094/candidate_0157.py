import triton
import torch
from torch import nn

BLOCK_SIZE = 1024

@triton.jit
def rms_norm_fwd_fused(X_ptr, Y_ptr, W_ptr, stride, N, eps, out_opt):
    row = tl.program_id(0)
    col = tl.program_id(1)
    batch = tl.program_id(2)

    # Pointer to the start of the current row in X, Y, and W
    X_ptrs = X_ptr + row * stride
    Y_ptrs = Y_ptr + row * stride
    W_ptrs = W_ptr + row

    # Load elements from X, W
    x = tl.load(X_ptrs + col)
    w = tl.load(W_ptrs + col)

    # Calculate variance
    var = tl.sum(x * x) / N

    # Calculate rstd
    rstd = 1 / tl.sqrt(var + eps)

    # Apply learned weight and store in Y
    y = x * rstd * w
    tl.store(Y_ptrs + col, y)

class TritonLlamaRMSNorm(nn.Module):
    def __init__(self, weight, eps=1e-5):
        super().__init__()
        self.weight = weight
        self.eps = eps

    def forward(self, X):
        # Reshape X into 2D
        X = X.reshape(-1, X.shape[-1])

        # Calculate optimal block size
        N = X.shape[1]
        grid = lambda meta: (triton.cdiv(N, BLOCK_SIZE), meta['serial_dep'])

        # Allocate output tensor
        Y = torch.empty_like(X)

        # Enqueue kernel
        rms_norm_fwd_fused[grid](X, Y, self.weight, X.stride(0), N, self.eps, out_opt=Y)

        return Y
