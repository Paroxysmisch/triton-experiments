import triton
import triton.language as tl
import torch

@triton.jit
def isfinite_func_kernel_rank_1(
    output_ptr,  # Pointer to output tensor
    input_ptr,   # Pointer to input tensor
    n_elements,  # Number of elements in tensor
    BLOCK_SIZE: tl.constexpr,  # Size of block to process
    one_tile_per_cta: tl.constexpr,  # Whether to use monolithic or grid-stride approach
):
    # Get program ID
    pid = tl.program_id(axis=0)
    
    # Calculate block start and offsets
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Handle grid-stride loop if not using one tile per CTA
    while True:
        # Create mask for valid elements
        mask = offsets < n_elements
        
        # Load input data with boundary checking
        x = tl.load(input_ptr + offsets, mask=mask)
        
        # Check for finite values using appropriate function based on dtype
        is_finite = tl.where(
            mask,
            ~(tl.isnan(x) | tl.isinf(x)),
            False
        )
        
        # Store results
        tl.store(output_ptr + offsets, is_finite, mask=mask)
        
        # Exit if using monolithic approach or processed all elements
        if one_tile_per_cta:
            break
            
        # Update offsets for next iteration
        block_start += tl.grid(0) * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        if tl.all(block_start >= n_elements):
            break

def heuristics_for_tile_size(n_elements):
    """Determine optimal tile size based on input size"""
    if n_elements < 1024:
        return 128
    elif n_elements < 4096:
        return 256
    else:
        return 512

def heuristics_for_num_warps(tile_size):
    """Determine optimal number of warps based on tile size"""
    if tile_size <= 128:
        return 4
    elif tile_size <= 256:
        return 8
    else:
        return 16

def isfinite_func_wrapper_rank_1(input_tensor):
    """Wrapper function for isfinite kernel"""
    # Input validation
    assert input_tensor.dim() == 1, "Input tensor must be rank-1"
    
    # Create output tensor
    output = torch.empty_like(input_tensor, dtype=torch.bool)
    
    # Get tensor properties
    n_elements = input_tensor.numel()
    
    # Determine execution parameters
    tile_size = heuristics_for_tile_size(n_elements)
    num_warps = heuristics_for_num_warps(tile_size)
    
    # Calculate grid parameters
    one_tile_per_cta = n_elements <= 65536
    if one_tile_per_cta:
        num_ctas = triton.cdiv(n_elements, tile_size)
    else:
        num_ctas = min(
            triton.cdiv(n_elements, tile_size),
            1024  # Maximum number of CTAs
        )
    
    # Launch kernel
    isfinite_func_kernel_rank_1[(num_ctas,)](
        output,
        input_tensor,
        n_elements,
        BLOCK_SIZE=tile_size,
        one_tile_per_cta=one_tile_per_cta,
        num_warps=num_warps
    )
    
    return output
