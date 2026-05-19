import torch
import triton
import triton.language as tl

@triton.jit
def sqrt_tanh_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input = tl.load(input_ptr + offsets, mask=mask)
    sqrt_val = tl.sqrt(input)
    output = tl.tanh(sqrt_val)
    tl.store(output_ptr + offsets, output, mask=mask)

def sqrt_tanh(input, out=None):
    if not input.is_cuda:
        raise RuntimeError("Input tensor must be on a CUDA device")
    
    input_contig = input.contiguous()
    n_elements = input_contig.numel()
    
    if out is None:
        output_contig = torch.empty_like(input_contig)
    else:
        if out.shape != input_contig.shape:
            raise RuntimeError("Output tensor shape must match input tensor")
        if out.device != input_contig.device:
            raise RuntimeError("Output tensor device must match input tensor")
        if out.is_contiguous():
            output_contig = out
        else:
            output_contig = torch.empty_like(input_contig)
    
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    sqrt_tanh_kernel[grid](
        input_contig.data_ptr(),
        output_contig.data_ptr(),
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    if out is not None and not out.is_contiguous():
        out.copy_(output_contig)
        return out
    else:
        return output_contig if out is None else out
