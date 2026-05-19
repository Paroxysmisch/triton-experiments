import torch
import triton
import triton.language as tl

@triton.jit
def _rms_norm_fwd_fused(
    X, Y, W,
    stride, N, eps,
    BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK_SIZE)
    
    # Compute the row offset
    row_offset = row * stride
    
    # Load input elements
    x = tl.load(X + row_offset + col, mask=col < N, other=0.0)
    
    # Compute variance
    x_sq = x * x
    var = tl.sum(x_sq, axis=0) / N
    
    # Compute reciprocal standard deviation (rstd)
    rstd = 1 / tl.sqrt(var + eps)
    
    # Normalize and apply weight
    w = tl.load(W + col, mask=col < N, other=1.0)
    y = x * rstd * w
    
    # Store the result
    tl.store(Y + row_offset + col, y, mask=col < N)

# PyTorch wrapper function
def rmsnorm_forward(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    assert x.is_cuda and weight.is_cuda, "Input tensors must be on GPU"
    assert weight.shape == (x.shape[-1],), "Weight shape must match last dimension of input"
    
    output = torch.empty_like(x)
    n_rows, n_cols = x.shape[0], x.shape[-1]
    
    # Reshape input if it's not 2D
    if x.dim() > 2:
        x = x.view(-1, n_cols)
        output = output.view(-1, n_cols)
    
    # Compute launch grid
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    BLOCK_SIZE = min(BLOCK_SIZE, 1024)  # Ensure block size doesn't exceed 1024
    grid = (n_rows,)
    
    # Determine number of warps based on feature size
    num_warps = 4
    if n_cols >= 2048:
        num_warps = 8
    elif n_cols >= 4096:
        num_warps = 16
    
    # Launch kernel
    _rms_norm_fwd_fused[grid](
        x, output, weight,
        x.stride(0), n_cols, eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )
    
    # Reshape output if input wasn't 2D
    if x.dim() > 2:
        output = output.view(x.shape)
    
    return output

# Example usage
if __name__ == "__main__":
    # Set up example inputs
    batch_size, seq_len, hidden_size = 32, 128, 512
    x = torch.randn(batch_size, seq_len, hidden_size, device='cuda')
    weight = torch.randn(hidden_size, device='cuda')
    
    # Run the RMSNorm
    output = rmsnorm_forward(x, weight)
    print(f"Output shape: {output.shape}")
