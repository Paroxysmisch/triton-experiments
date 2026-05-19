import torch
import triton
import triton.language as tl

@triton.jit
def batch_norm_kernel(x_ptr, mean_ptr, var_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Identify the program ID
    pid = tl.program_id(axis=0)

    # Calculate offsets and create a mask for out-of-bounds access
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load x from DRAM, masking out any extra elements
    x = tl.load(x_ptr + offsets, mask=mask)

    # Load mean and variance from DRAM
    mean = tl.load(mean_ptr)
    var = tl.load(var_ptr)

    # Normalize x using the mean and variance
    normalized_x = (x - mean) / tl.sqrt(var + 1e-5)

    # Write the normalized x back to DRAM
    tl.store(output_ptr + offsets, normalized_x, mask=mask)

def batch_norm(x: torch.Tensor, running_mean: torch.Tensor, running_var: torch.Tensor):
    # Calculate the mean and variance of x
    mean = x.mean()
    var = x.var()

    # Update the running mean and variance
    running_mean += (mean - running_mean)
    running_var += (var - running_var)

    # Preallocate the output
    output = torch.empty_like(x)
    n_elements = output.numel()

    # Define the grid size
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )

    # Launch the kernel
    batch_norm_kernel[grid](x, mean, var, output, n_elements, BLOCK_SIZE=1024)

    return output
