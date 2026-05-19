import torch
import triton
import triton.language as tl

@triton.jit
def _seeded_dropout(
    x_ptr,
    output_ptr,
    n_elements,
    p,
    seed,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate the offset for this program instance
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask for valid elements
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Generate random numbers
    random = tl.rand(seed, offsets)
    
    # Create dropout mask
    keep_mask = random > p
    
    # Apply dropout and scaling
    output = tl.where(keep_mask, x / (1.0 - p), 0.0)
    
    # Store the result
    tl.store(output_ptr + offsets, output, mask=mask)

def seeded_dropout(x: torch.Tensor, p: float, seed: int):
    output = torch.empty_like(x)
    assert x.is_contiguous(), "Input tensor must be contiguous"
    
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    _seeded_dropout[grid](
        x_ptr=x.data_ptr(),
        output_ptr=output.data_ptr(),
        n_elements=n_elements,
        p=p,
        seed=seed,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output
