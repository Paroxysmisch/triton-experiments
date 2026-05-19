import triton
import triton.language as tl

@triton.jit
def rms_norm_fwd_fused(X, Y, W, stride, N, eps, BLOCK_SIZE: tl.constexpr):
    # Get the row index for the current thread block
    row_idx = tl.program_id(0)
    
    # Initialize the sum of squares for the current row
    sum_of_squares = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Iterate over the elements in the current row
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + row_idx * stride + cols, mask=mask)
        x = x.to(tl.float32)
        sum_of_squares += x * x
    
    # Reduce the sum of squares across the block
    sum_of_squares = tl.sum(sum_of_squares, axis=0)
    sum_of_squares = tl.sum(tl.to(tl.float32, sum_of_squares), axis=0)
    
    # Compute the variance and the inverse of the standard deviation
    variance = sum_of_squares / N
    rstd = 1.0 / tl.sqrt(variance + eps)
    
    # Normalize the elements in the current row
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + row_idx * stride + cols, mask=mask)
        x = x.to(tl.float32)
        y = (x - tl.zeros_like(x)) * rstd
        w = tl.load(W + cols, mask=mask)
        y = y * w
        tl.store(Y + row_idx * stride + cols, y, mask=mask)

import torch
import torch.nn as nn
import triton
import triton.language as tl

class TritonLlamaRMSNorm(nn.Module):
    def __init__(self, weight, eps=1e-6):
        super(TritonLlamaRMSNorm, self).__init__()
        self.weight = weight
        self.eps = eps

    def forward(self, x):
        # Reshape the input tensor to 2D format
        B, N = x.shape[0], x.numel() // x.shape[0]
        x = x.view(B, N)
        
        # Allocate the output tensor
        y = torch.empty_like(x)
        
        # Calculate the optimal block size
        BLOCK_SIZE = triton.next_power_of_2(min(1024, N))
        
        # Enqueue the Triton kernel for execution
        rms_norm_fwd_fused[(B,)](x, y, self.weight, x.stride(0), N, self.eps, BLOCK_SIZE)
        
        return y
