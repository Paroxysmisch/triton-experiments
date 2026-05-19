import triton
import triton.language as tl

@triton.jit
def sub_gelu_kernel(
    input_ptr,  # Pointer to the input tensor
    other_ptr,  # Pointer to the other tensor or number
    out_ptr,    # Pointer to the output tensor
    n_elements, # Number of elements in the tensors
    alpha,      # Scaling factor for other
    approximate, # Approximation method for GELU
    BLOCK_SIZE: tl.constexpr, # Block size for parallelization
):
    pid = tl.program_id(axis=0)  # Get the program ID
    block_start = pid * BLOCK_SIZE  # Start index of the block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # Offsets for the block
    mask = offsets < n_elements  # Mask to handle blocks that are not fully filled

    # Load input and other tensors
    input_block = tl.load(input_ptr + offsets, mask=mask)
    other_block = tl.load(other_ptr + offsets, mask=mask)

    # Perform the subtraction and scaling
    result = input_block - alpha * other_block

    # Apply GELU activation
    if approximate == 0:  # Exact GELU
        result = result * 0.5 * (1 + tl.erf(result / tl.sqrt(2.0)))
    elif approximate == 1:  # Approximate GELU using tanh
        cdf = 0.5 * (1 + tl.tanh(0.0356774 * result + 0.797885 * result * result * result))
        result = result * cdf

    # Store the result
    tl.store(out_ptr + offsets, result, mask=mask)

import torch
import triton
import triton.language as tl

def sub_gelu(input, other, alpha=1, approximate='none', out=None):
    # Ensure input and other are tensors
    if not isinstance(other, torch.Tensor):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)

    # Ensure input and other have the same shape
    if input.shape != other.shape:
        raise ValueError("input and other must have the same shape")

    # Determine the number of elements
    n_elements = input.numel()

    # Determine the approximation method
    if approximate == 'none':
        approximate = 0
    elif approximate == 'tanh':
        approximate = 1
    else:
        raise ValueError("approximate must be 'none' or 'tanh'")

    # Allocate output tensor if not provided
    if out is None:
        out = torch.empty_like(input)

    # Define the grid and block sizes
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    # Launch the Triton kernel
    sub_gelu_kernel[grid](
        input_ptr=input.data_ptr(),
        other_ptr=other.data_ptr(),
        out_ptr=out.data_ptr(),
        n_elements=n_elements,
        alpha=alpha,
        approximate=approximate,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out
