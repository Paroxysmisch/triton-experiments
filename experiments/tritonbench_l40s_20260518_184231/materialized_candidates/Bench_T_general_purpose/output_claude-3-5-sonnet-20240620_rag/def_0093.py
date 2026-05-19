import triton
import triton.language as tl
import torch
from torch import Tensor

# Triton kernel for applying log and softmax
@triton.jit
def _softmax_log(X, OUT, xm_stride, xn_stride, out_stride, N, dim: tl.constexpr):
    rm = tl.program_id(0)
    # Load input tensor
    x = tl.load(X + rm * xm_stride)
    # Apply natural logarithm
    log_x = tl.log(x)
    
    # Compute softmax
    max_log_x = tl.max(log_x, axis=dim)
    exp_log_x = tl.exp(log_x - max_log_x)
    sum_exp_log_x = tl.sum(exp_log_x, axis=dim)
    softmax_result = exp_log_x / sum_exp_log_x
    
    # Store the result
    tl.store(OUT + rm * out_stride, softmax_result)

# Wrapper function for softmax_log
def softmax_log(input: Tensor, dim: int = -1, dtype: torch.dtype = None) -> Tensor:
    if dtype is not None:
        input = input.to(dtype)
    
    assert input.is_cuda
    *dims, N = input.shape
    input = input.view(-1, N)
    out = input.new_empty(*dims).view(-1)
    M = input.shape[0]
    
    # Call the Triton kernel
    _softmax_log[(M,)](input, out, input.stride(0), input.stride(1), out.stride(0), N, dim=dim)
    
    return out.view(*dims)

# Example usage
if __name__ == "__main__":
    input = torch.rand(3, 4) * 10
    result = softmax_log(input, dim=1)
    print(result)
