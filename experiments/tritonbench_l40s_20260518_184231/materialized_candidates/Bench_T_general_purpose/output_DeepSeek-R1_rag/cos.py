import torch
import triton
import triton.language as tl

device = 'cuda:0'

@triton.jit
def cos_kernel(
    input_ptr,
    output_ptr,
    N,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(input_ptr + offsets, mask=mask)
    y = tl.cos(x)
    tl.store(output_ptr + offsets, y, mask=mask)

def cos(input: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    assert input.is_cuda, "Input tensor must be on CUDA"
    input_contig = input.contiguous()
    N = input_contig.numel()
    
    if out is None:
        output = torch.empty_like(input_contig)
    else:
        assert out.is_cuda, "Output tensor must be on CUDA"
        assert out.shape == input.shape, "Output shape must match input"
        assert out.is_contiguous(), "Output tensor must be contiguous"
        output = out
    
    input_flat = input_contig.view(-1)
    output_flat = output.view(-1)
    
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    cos_kernel[grid](input_flat, output_flat, N, BLOCK_SIZE=BLOCK_SIZE)
    
    return output
