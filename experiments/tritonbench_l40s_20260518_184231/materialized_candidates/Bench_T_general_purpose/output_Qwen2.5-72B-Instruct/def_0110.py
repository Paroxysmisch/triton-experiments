import triton
import triton.language as tl

@triton.jit
def exp_mean_kernel(
    input_ptr,  # Pointer to the input tensor
    output_ptr,  # Pointer to the output tensor
    n_elements,  # Number of elements in the input tensor
    block_size: tl.constexpr,  # Block size for parallelization
):
    pid = tl.program_id(axis=0)
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)
    mask = offsets < n_elements
    input_vec = tl.load(input_ptr + offsets, mask=mask)
    exp_vec = tl.exp(input_vec)
    sum_exp = tl.sum(exp_vec, axis=0)
    count = tl.sum(mask, axis=0)
    mean_exp = sum_exp / count
    tl.store(output_ptr + pid, mean_exp)

import torch
import triton
import triton.language as tl

def exp_mean(input, dim=None, keepdim=False, dtype=None, out=None) -> torch.Tensor:
    if dtype is not None:
        input = input.to(dtype)
    
    if dim is None:
        # Compute the mean over all elements
        input_flat = input.flatten()
        n_elements = input_flat.numel()
        output = torch.empty(1, device=input.device, dtype=input.dtype)
        grid = (1,)
    else:
        # Compute the mean along the specified dimension
        dim = dim if dim >= 0 else input.dim() + dim
        input_size = input.size()
        n_elements = input.size(dim)
        reduced_shape = list(input_size)
        reduced_shape[dim] = 1
        output = torch.empty(reduced_shape, device=input.device, dtype=input.dtype)
        grid = (input.numel() // n_elements,)
    
    # Launch the Triton kernel
    exp_mean_kernel[grid](
        input.data_ptr(),
        output.data_ptr(),
        n_elements,
        triton.next_power_of_2(n_elements),
    )
    
    if not keepdim and dim is not None:
        output = output.squeeze(dim)
    
    if out is not None:
        out.copy_(output)
        return out
    else:
        return output
