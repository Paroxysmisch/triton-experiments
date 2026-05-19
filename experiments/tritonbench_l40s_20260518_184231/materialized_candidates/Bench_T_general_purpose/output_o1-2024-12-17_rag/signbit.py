import torch
import triton
import triton.language as tl

@triton.jit
def _signbit_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    # Cast from float to int to inspect the sign bit
    x_int = tl.bitcast_to_int32(x)
    # Extract the sign bit (bit 31 for float32)
    sbit = (x_int >> 31) & 1
    tl.store(output_ptr + offsets, sbit, mask=mask)


def signbit(input: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    if out is not None and out.shape != input.shape:
        raise ValueError("Expected out tensor to have the same shape as input.")
    # Allocate output if none is provided
    if out is None:
        out = torch.empty_like(input, dtype=torch.bool)

    input_contig = input.contiguous()
    out_contig = out.contiguous()

    n_elements = input_contig.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Launch Triton kernel
    _signbit_kernel[grid](
        input_contig, 
        out_contig, 
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )

    # Reshape 'out_contig' to match original 'out' shape (if 'out' was provided)
    if out is not out_contig:
        out.copy_(out_contig.view_as(out))
    else:
        out = out_contig.view_as(input)

    return out
