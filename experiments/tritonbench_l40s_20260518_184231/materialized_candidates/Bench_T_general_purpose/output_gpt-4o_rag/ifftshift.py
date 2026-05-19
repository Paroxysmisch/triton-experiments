import triton
import triton.language as tl
import torch

@triton.jit
def ifftshift_kernel(input_ptr, output_ptr, n_elements, dim_offset, dim_size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Compute the shift amount
    shift = dim_size // 2

    # Compute the new position after ifftshift
    new_offsets = (offsets + shift) % dim_size + dim_offset

    # Load the data from input
    input_data = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    # Store the data to output in the shifted position
    tl.store(output_ptr + new_offsets, input_data, mask=mask)

def ifftshift(input, dim=None):
    if dim is None:
        # Flatten the input and apply ifftshift on all dimensions
        input = input.flatten()
        dim = 0

    # Ensure dim is a tuple
    if isinstance(dim, int):
        dim = (dim,)

    # Create output tensor
    output = torch.empty_like(input)

    # Calculate number of elements and dimension sizes
    n_elements = input.numel()
    dim_size = input.size(dim[0])

    # Launch Triton kernel
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    ifftshift_kernel[grid](input, output, n_elements, 0, dim_size, BLOCK_SIZE=1024)

    return output
