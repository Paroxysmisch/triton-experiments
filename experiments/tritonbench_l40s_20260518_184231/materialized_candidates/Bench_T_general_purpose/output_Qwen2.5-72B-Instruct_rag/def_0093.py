import triton
import triton.language as tl
import torch

# Triton kernel for softmax_log
@triton.jit
def _softmax_log_kernel(X, OUT, stride_xm, stride_xn, stride_om, stride_on, N, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_N
    offsets = block_start + tl.arange(0, BLOCK_N)
    mask = offsets < N
    x_ptrs = X + offsets * stride_xn
    x = tl.load(x_ptrs, mask=mask, other=-float('inf'))
    
    # Compute log(x)
    log_x = tl.log(x)
    
    # Compute max(log_x) for numerical stability
    max_log_x = tl.max(log_x, axis=0)
    
    # Compute exp(log_x - max_log_x)
    exp_log_x = tl.exp(log_x - max_log_x)
    
    # Compute sum(exp(log_x - max_log_x))
    sum_exp_log_x = tl.sum(exp_log_x, axis=0)
    
    # Compute softmax(log(x))
    softmax_log_x = exp_log_x / sum_exp_log_x
    
    # Store the result
    out_ptrs = OUT + offsets * stride_on
    tl.store(out_ptrs, softmax_log_x, mask=mask)

# Wrapper function for softmax_log
def softmax_log(input, dim=-1, dtype=None) -> torch.Tensor:
    if dtype is not None:
        input = input.to(dtype)
    
    *batch, N = input.shape
    input = input.view(-1, N)
    out = input.new_empty(input.shape)
    
    M = input.shape[0]
    _softmax_log_kernel[(M,)](input, out, input.stride(0), input.stride(1), out.stride(0), out.stride(1), N, BLOCK_N=1024, num_warps=4)
    
    return out.view(*batch, N)

# Example usage
if __name__ == "__main__":
    import torch
    import torch.nn.functional as F
    
    # Define input tensor
    input = torch.rand(3, 4) * 10
    
    # Apply softmax_log along the last dimension
    result = softmax_log(input, dim=1)
    print(result)
    
    # Apply softmax_log along a different dimension
    result = softmax_log(input, dim=0)
    print(result)
