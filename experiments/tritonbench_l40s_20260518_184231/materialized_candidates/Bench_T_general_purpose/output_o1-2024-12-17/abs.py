import triton
import triton.language as tl

@triton.jit
def abs_kernel(
    input_ptr,  # *Pointer to input
    output_ptr,  # *Pointer to output
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr  # Block size
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask)
    x_abs = tl.abs(x)
    tl.store(output_ptr + offsets, x_abs, mask=mask)

def abs(input, *, out=None):
    # Prepare output
    if out is None:
        import torch
        out = torch.empty_like(input)

    # Launch kernel
    n_elements = input.numel()
    # We'll define a block size; can be tuned
    BLOCK_SIZE = 1024
    grid = ( (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE, )

    abs_kernel[grid](
        input_ptr=input.data_ptr(),
        output_ptr=out.data_ptr(),
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out
