import math
import torch
import triton
import triton.language as tl

@triton.jit
def _dequantize_rowwise(
    x_ptr,
    state_x,
    output_ptr,
    inv_127,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    P2: tl.constexpr,
):
    # Get program ID for current row
    pid = tl.program_id(axis=0)
    
    # Calculate starting index for this block
    block_start = pid * BLOCK_SIZE
    
    # Create index range for vectorized operations
    arange = tl.arange(0, P2)
    
    # Compute memory offsets for this block
    offsets = block_start + arange
    
    # Create mask to avoid out-of-bounds accesses
    row_mask = arange < BLOCK_SIZE
    
    # Load quantized values from memory
    x = tl.load(x_ptr + offsets, mask=row_mask)
    
    # Load maximum value for this row
    max_val = tl.load(state_x + pid)
    
    # Dequantize using vectorized operations
    output = max_val * x * inv_127
    
    # Store dequantized values back to memory
    tl.store(output_ptr + offsets, output, mask=row_mask)

def dequantize_rowwise(x: torch.Tensor, state_x: torch.Tensor) -> torch.Tensor:
    # Create output tensor with same shape as input
    output = torch.empty(*x.shape, device=x.device, dtype=torch.float16)
    
    # Calculate nearest power of two for columns
    P2 = int(2 ** math.ceil(math.log2(x.shape[1])))
    
    # Verify tensors are on CUDA device
    assert x.is_cuda and output.is_cuda
    
    # Calculate total number of elements
    n_elements = output.numel()
    
    # Define kernel grid layout (one block per row)
    grid = lambda meta: (x.shape[0],)
    
    # Launch Triton kernel with optimized parameters
    _dequantize_rowwise[grid](
        x, state_x, output, 
        1.0 / 127,  # Precomputed inverse of 127
        n_elements,
        BLOCK_SIZE=x.shape[1],  # Actual number of columns
        P2=P2                   # Power-of-two padded size
    )
    
    return output
