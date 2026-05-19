import torch
import triton
import triton.language as tl

@triton.jit
def sigmoid_kernel(output_ptr, input_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input = tl.load(input_ptr + offsets, mask=mask)
    output = 1.0 / (1.0 + tl.exp(-input))
    tl.store(output_ptr + offsets, output, mask=mask)

def sigmoid_argmax(input: torch.Tensor, dim=None, keepdim=False) -> torch.LongTensor:
    # Ensure input is contiguous and on the correct device
    input_contig = input.contiguous()
    n_elements = input_contig.numel()
    output = torch.empty_like(input_contig)
    
    # Determine kernel launch parameters
    BLOCK_SIZE = triton.next_power_of_2(n_elements)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Launch sigmoid kernel
    sigmoid_kernel[grid](output, input_contig, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    # Compute argmax based on the specified dimension
    if dim is None:
        return torch.argmax(output.view(-1), dim=0).long()
    else:
        return torch.argmax(output, dim=dim, keepdim=keepdim).long()
