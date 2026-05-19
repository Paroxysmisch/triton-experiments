import torch
import triton
import triton.language as tl

@triton.jit
def logsumexp_kernel(
    input_ptr,
    output_ptr,
    reduction_size,
    input_row_stride,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    row_start = row_idx * input_row_stride

    # Compute max value in the row
    max_val = -tl.infinity(tl.float32)
    for offset in range(0, reduction_size, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < reduction_size
        x = tl.load(input_ptr + row_start + cols, mask=mask, other=-tl.inf)
        curr_max = tl.max(x, axis=0)
        max_val = tl.maximum(max_val, curr_max)
    
    # Compute sum of exp(x - max_val)
    sum_exp = 0.0
    for offset in range(0, reduction_size, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < reduction_size
        x = tl.load(input_ptr + row_start + cols, mask=mask, other=0.0)
        x = x - max_val
        exp_x = tl.exp(x)
        sum_exp += tl.sum(exp_x, axis=0)
    
    # Compute log(sum_exp) + max_val and store the result
    log_sum_exp = tl.log(sum_exp) + max_val
    tl.store(output_ptr + row_idx, log_sum_exp)

def logsumexp(input, dim, keepdim=False, *, out=None) -> torch.Tensor:
    # Check if the dimension is valid
    if dim < 0:
        dim += input.dim()
    assert dim >= 0 and dim < input.dim(), "dim out of range"
    
    # Flatten all dimensions except the reduction dim to 2D (n_rows, reduction_size)
    input_flat = input.transpose(dim, -1)
    orig_shape = input_flat.shape
    reduction_size = orig_shape[-1]
    input_flat = input_flat.contiguous().view(-1, reduction_size)
    n_rows = input_flat.shape[0]
    
    # Allocate output tensor
    output = torch.empty(n_rows, dtype=input.dtype, device=input.device)
    
    # Launch kernel
    BLOCK_SIZE = 1024  # tunable depending on hardware
    logsumexp_kernel[(n_rows,)](input_flat, output, reduction_size, input_flat.stride(0), BLOCK_SIZE=BLOCK_SIZE)
    
    # Reshape output to the original shape with appropriate dimensions
    output_shape = list(input.shape)
    if keepdim:
        output_shape[dim] = 1
    else:
        output_shape.pop(dim)
    output = output.view(output_shape)
    
    # Handle out parameter
    if out is not None:
        out.copy_(output)
        return out
    return output

# Verify that the wrapper matches the func_inputs
import inspect
sig = inspect.signature(logsumexp)
assert str(sig) == "(input, dim, keepdim=False, *, out=None) -> Tensor", "Wrapper signature does not match"
print("Verification passed: Wrapper signature matches the provided func_inputs.")
