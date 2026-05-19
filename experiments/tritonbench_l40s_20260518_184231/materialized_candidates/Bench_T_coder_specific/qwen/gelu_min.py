import triton
import triton.language as tl
import torch
from typing import Tuple, Optional

@triton.jit
def gelu_kernel(
    output_ptr,
    input_ptr,
    n_elements,
    block_size: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * block_size + tl.arange(0, block_size)
    valid_mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=valid_mask)

    # Approximate GELU using tanh
    cdf_x = 0.5 * (1 + tl.tanh(tl.sqrt(2 / 3.14159) * (x + 0.044715 * x * x * x)))
    gelu_x = x * cdf_x

    tl.store(output_ptr + offsets, gelu_x, mask=valid_mask)

def gelu_min(
    input: torch.Tensor,
    approximate: str = 'none',
    dim: Optional[int] = None,
    keepdim: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert approximate in ['none', 'tanh'], "approximate must be either 'none' or 'tanh'"
    
    if dim is not None:
        # Apply GELU and then compute min along the specified dimension
        n_elements = input.size(dim)
        output = torch.empty_like(input)
        
        grid = lambda meta: (triton.cdiv(n_elements, meta['block_size']),)
        
        gelu_kernel[(grid)](output, input, n_elements, block_size=64)
        
        result, indices = torch.min(output, dim=dim, keepdim=keepdim)
        return result, indices
    
    else:
        # Flatten the tensor and apply GELU
        flat_input = input.flatten()
        flat_output = torch.empty_like(flat_input)
        
        grid = lambda meta: (triton.cdiv(len(flat_input), meta['block_size']),)
        
        gelu_kernel[(grid)](flat_output, flat_input, len(flat_input), block_size=64)
        
        return torch.min(flat_output, dim=0)[0]

# Example usage
input_tensor = torch.tensor([-3.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float32)
result, indices = gelu_min(input_tensor, approximate='tanh', dim=0)
print("Result:", result)
print("Indices:", indices)
