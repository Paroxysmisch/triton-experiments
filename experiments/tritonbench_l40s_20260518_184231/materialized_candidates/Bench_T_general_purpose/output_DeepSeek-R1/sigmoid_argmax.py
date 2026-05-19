import torch
import triton
import triton.language as tl

@triton.jit
def sigmoid_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    output = 1.0 / (1.0 + tl.exp(-x))
    tl.store(output_ptr + offsets, output, mask=mask)

def sigmoid_argmax(input: torch.Tensor, dim: int = None, keepdim: bool = False) -> torch.LongTensor:
    # Ensure the input tensor is contiguous
    input_contig = input.contiguous()
    output = torch.empty_like(input_contig)
    n_elements = input_contig.numel()
    
    # Launch the Triton kernel with appropriate grid and block size
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    sigmoid_kernel[grid](input_contig, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    # Compute the argmax based on the specified dimension
    if dim is not None:
        return torch.argmax(output, dim=dim, keepdim=keepdim)
    else:
        return torch.argmax(output.view(-1), dim=0)
