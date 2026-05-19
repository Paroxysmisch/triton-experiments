import torch
import triton
import triton.language as tl

@triton.jit
def gammaln_kernel(
    output_ptr,
    input_ptr,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    abs_x = tl.abs(x)
    y = tl.lgamma(abs_x)
    tl.store(output_ptr + offsets, y, mask=mask)

def gammaln(input, *, out=None):
    # Ensure input is contiguous
    input_contig = input.contiguous()
    num_elements = input_contig.numel()
    
    # Handle output tensor
    if out is None:
        out = torch.empty_like(input_contig)
    else:
        if out.shape != input_contig.shape:
            raise ValueError("Output tensor shape must match input tensor shape")
        if not out.is_contiguous():
            raise ValueError("Output tensor must be contiguous")
        out = out.contiguous()  # Ensure it's contiguous
    
    # Kernel configuration
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(num_elements, BLOCK_SIZE), )
    num_warps = BLOCK_SIZE // 32
    
    # Launch kernel
    gammaln_kernel[grid](
        out.data_ptr(),
        input_contig.data_ptr(),
        num_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    
    return out
