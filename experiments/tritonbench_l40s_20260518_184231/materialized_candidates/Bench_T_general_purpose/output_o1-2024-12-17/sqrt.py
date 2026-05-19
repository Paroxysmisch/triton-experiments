import triton
import triton.language as tl

@triton.jit
def _sqrt_kernel(
    input_ptr, 
    output_ptr, 
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask)
    y = tl.sqrt(x)
    tl.store(output_ptr + offsets, y, mask=mask)

def sqrt(input, *, out=None):
    # Flatten input to 1D for simplicity
    in_data = input.flatten()
    n_elements = in_data.shape[0]

    # Allocate out if not provided
    if out is None:
        import torch
        out = torch.empty_like(input)

    out_data = out.flatten()

    # Launch kernel
    grid = lambda meta: ( (n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'], )
    _sqrt_kernel[grid](
        in_data, 
        out_data,
        n_elements,
        BLOCK_SIZE=1024
    )

    return out
