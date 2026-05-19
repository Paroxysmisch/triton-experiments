import triton
import triton.language as tl

@triton.jit
def kldivergence_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Unique program ID
    pid = tl.program_id(axis=0)
    
    # Calculate starting index for this block
    block_start = pid * BLOCK_SIZE
    
    # Create offset tensor for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask to handle boundary conditions
    mask = offsets < n_elements
    
    # Load x and y values
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # Compute KL divergence
    output = x * tl.log(x / y)
    
    # Store the result
    tl.store(output_ptr + offsets, output, mask=mask)

def kldivergence(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and y.is_cuda, "Input tensors must be on GPU"
    assert x.shape == y.shape, "Input tensors must have the same shape"
    
    output = torch.empty_like(x)
    n_elements = output.numel()
    
    # Define block size (can be tuned for performance)
    BLOCK_SIZE = 1024
    
    # Compute grid
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Launch kernel
    kldivergence_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return output
