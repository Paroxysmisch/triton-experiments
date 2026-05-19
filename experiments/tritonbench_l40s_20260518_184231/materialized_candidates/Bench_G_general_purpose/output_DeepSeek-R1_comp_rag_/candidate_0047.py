import triton
import triton.language as tl
import torch

@triton.jit
def relu_kernel(
    in_ptr,    # Input tensor pointer
    out_ptr,   # Output tensor pointer
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,  # Block size (compile-time constant)
):
    # Get 1D program ID for current block
    pid = tl.program_id(axis=0)
    
    # Calculate block's starting offset and valid offsets
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask to prevent out-of-bounds memory accesses
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(in_ptr + offsets, mask=mask)
    
    # Compute ReLU activation (x if x>0 else 0)
    output = tl.where(x >= 0, x, 0.0)
    
    # Store result conditionally using mask
    tl.store(out_ptr + offsets, output, mask=mask)

def relu(x: torch.Tensor) -> torch.Tensor:
    # Ensure input is on GPU and contiguous
    assert x.is_cuda, "Input tensor must be on GPU"
    assert x.is_contiguous(), "Input tensor must be contiguous"
    
    # Create output tensor with same shape/device
    output = torch.empty_like(x)
    
    # Calculate total elements and grid size
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Launch kernel with optimal block size
    relu_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    
    return output
