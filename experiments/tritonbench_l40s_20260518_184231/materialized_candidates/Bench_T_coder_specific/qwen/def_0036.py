import triton
import torch

@torch.jit.script
def add_gelu(input, other, alpha=1, approximate='none', out=None):
    if out is None:
        out = torch.empty_like(input)

    n_elements = input.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    gelu_kernel[grid](input.data_ptr(), other.data_ptr(), alpha.data_ptr(), out.data_ptr(),
                       n_elements=n_elements, BLOCK_SIZE=triton.next_power_of_two(64))

    return out
