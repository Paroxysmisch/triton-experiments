import triton
import triton.language as tl
import torch

@triton.jit
def isfinite_func_kernel_rank_1(
    input_ptr,  # Pointer to input tensor
    output_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,  # Size of parallel processing block
    input_dtype: tl.constexpr,  # Input data type
    one_tile_per_cta: tl.constexpr,  # Whether to use monolithic or grid-stride approach
):
    # Get program ID
    pid = tl.program_id(axis=0)
    
    # Calculate block start and offsets
    if one_tile_per_cta:
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
    else:
        # Grid-stride loop
        num_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
        for block_start in range(pid * BLOCK_SIZE, n_elements, BLOCK_SIZE * num_blocks):
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            
            # Load input data
            x = tl.load(input_ptr + offsets, mask=mask)
            
            # Check for finite values based on data type
            if input_dtype == tl.float64:
                is_finite = tl._isfinited(x)
            else:
                is_finite = tl._finitef(x)
                
            # Store results
            tl.store(output_ptr + offsets, is_finite, mask=mask)

def heuristics_for_tile_size(n_elements):
    """Determine optimal tile size based on input size"""
    if n_elements < 1024:
        return 128
    elif n_elements < 8192:
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

def isfinite_func_wrapper_rank_1(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function for the isfinite Triton kernel
    Args:
        input_tensor: Input tensor to check for finite values
    Returns:
        output_tensor: Boolean tensor indicating finite values
    """
    # Input validation
    assert input_tensor.dim() == 1, "Input tensor must be rank 1"
    
    # Create output tensor
    output_tensor = torch.empty_like(input_tensor, dtype=torch.bool)
    
    # Get tensor properties
    n_elements = input_tensor.numel()
    
    # Determine kernel parameters
    tile_size = heuristics_for_tile_size(n_elements)
    num_warps = heuristics_for_num_warps(tile_size)
    
    # Calculate grid parameters
    one_tile_per_cta = n_elements <= 1024
    if one_tile_per_cta:
        num_ctas = triton.cdiv(n_elements, tile_size)
    else:
        num_ctas = min(256, triton.cdiv(n_elements, tile_size))
    
    # Launch kernel
    isfinite_func_kernel_rank_1[(num_ctas,)](
        input_ptr=input_tensor,
        output_ptr=output_tensor,
        n_elements=n_elements,
        BLOCK_SIZE=tile_size,
        input_dtype=input_tensor.dtype,
        one_tile_per_cta=one_tile_per_cta,
        num_warps=num_warps
    )
    
    return output_tensor
