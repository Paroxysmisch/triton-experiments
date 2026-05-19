import triton
import triton.language as tl
import torch

@triton.jit
def _log_softmax_kernel(X, OUT, dim_size, stride_x, stride_out, BLOCK_SIZE: tl.constexpr):
    # Get program ID for block-level parallelism
    pid = tl.program_id(0)
    # Calculate offsets
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Load input data
    x = tl.load(X + offsets * stride_x, mask=offsets < dim_size, other=-float('inf'))
    # Apply logarithm
    x_log = tl.log(x)
    # Compute max for numerical stability
    max_x = tl.max(x_log, axis=0)
    # Compute softmax
    exp_x = tl.exp(x_log - max_x)
    sum_exp_x = tl.sum(exp_x, axis=0)
    softmax_log = exp_x / sum_exp_x
    # Store the result
    tl.store(OUT + offsets * stride_out, softmax_log, mask=offsets < dim_size)

def softmax_log(input, dim=-1, dtype=None):
    if not input.is_cuda:
        raise ValueError("Input tensor must be a CUDA tensor")
    
    if dtype is not None:
        input = input.to(dtype)

    # Move the dimension to the last for easier computation
    input = input.transpose(dim, -1).contiguous()
    *other_dims, dim_size = input.shape

    # Allocate output tensor
    output = torch.empty_like(input)
    
    # Launch Triton kernel
    grid = lambda META: (triton.cdiv(dim_size, META['BLOCK_SIZE']),)
    _log_softmax_kernel[grid](input, output, dim_size, input.stride(-1), output.stride(-1), BLOCK_SIZE=1024)
    
    # Move the dimension back to its original position
    output = output.transpose(dim, -1)
    
    return output

# Example usage
input_tensor = torch.rand(3, 4, device='cuda') * 10
result = softmax_log(input_tensor, dim=1)
print(result)
