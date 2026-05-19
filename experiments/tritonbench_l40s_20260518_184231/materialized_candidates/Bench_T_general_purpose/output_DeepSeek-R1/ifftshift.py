import torch
import triton
import triton.language as tl

@triton.jit
def ifftshift_kernel(
    input_ptr,
    output_ptr,
    num_dims,
    sizes_ptr,
    strides_ptr,
    shifts_ptr,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    # Load sizes, strides, and shifts for each dimension
    sizes = tl.load(sizes_ptr + tl.arange(0, num_dims), mask=(tl.arange(0, num_dims) < num_dims))
    strides = tl.load(strides_ptr + tl.arange(0, num_dims), mask=(tl.arange(0, num_dims) < num_dims))
    shifts = tl.load(shifts_ptr + tl.arange(0, num_dims), mask=(tl.arange(0, num_dims) < num_dims))

    original_offsets = tl.zeros_like(offsets)
    remaining = offsets
    for d in range(num_dims):
        stride = strides[d]
        size = sizes[d]
        shift = shifts[d]
        # Compute coordinate in this dimension
        coord = (remaining // stride) % size
        # Apply inverse shift
        new_coord = (coord - shift) % size
        # Update the original offset with the shifted coordinate
        original_offsets += new_coord * stride
        # Update remaining for next dimension
        remaining = remaining % stride

    input_vals = tl.load(input_ptr + original_offsets, mask=mask)
    tl.store(output_ptr + offsets, input_vals, mask=mask)

def ifftshift(input, dim=None):
    if dim is None:
        dim = list(range(input.dim()))
    else:
        if isinstance(dim, int):
            dim = [dim]
        dim = [d if d >= 0 else d + input.dim() for d in dim]
    
    shifts = [0] * input.dim()
    for d in dim:
        size = input.size(d)
        shifts[d] = (size + 1) // 2
    
    # Prepare tensor parameters for the kernel
    sizes = torch.tensor(input.shape, device=input.device, dtype=torch.int32)
    strides = torch.tensor(input.stride(), device=input.device, dtype=torch.int32)
    shifts_tensor = torch.tensor(shifts, device=input.device, dtype=torch.int32)
    
    output = torch.empty_like(input)
    total_elements = input.numel()
    grid = lambda meta: (triton.cdiv(total_elements, meta['BLOCK_SIZE']),)
    
    ifftshift_kernel[grid](
        input,
        output,
        input.dim(),
        sizes,
        strides,
        shifts_tensor,
        total_elements,
        BLOCK_SIZE=1024,
    )
    return output
