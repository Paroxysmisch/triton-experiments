import torch
import triton
import triton.language as tl

def f8_to_f16(x, dtype):
    """Convert float8 tensor to float16 tensor using Triton kernel.
    
    Args:
        x: Input tensor in float8 format (stored as int8)
        dtype: String specifying float8 format ('float8_e4m3' or 'float8_e5m2')
    
    Returns:
        Tensor in float16 format
    """
    @triton.jit
    def _f8_to_f16_kernel(
        Y,  # Output pointer (float16)
        X,  # Input pointer (float8 reinterpreted as specified dtype)
        N,  # Total number of elements
        BLOCK_SIZE: tl.constexpr,  # Number of elements per block
    ):
        # Calculate block ID and offsets
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        
        # Create mask for valid elements
        mask = offsets < N
        
        # Load float8 values (reinterpreted as specified dtype)
        x = tl.load(X + offsets, mask=mask)
        
        # Store converted float16 values
        tl.store(Y + offsets, x, mask=mask)
    
    # Create output tensor
    output = torch.empty(x.shape, dtype=torch.float16, device=x.device)
    
    # Calculate grid size based on input size and block size
    grid = lambda META: (triton.cdiv(x.numel(), META['BLOCK_SIZE']),)
    
    # Convert dtype string to Triton dtype
    triton_dtype = getattr(tl, dtype)
    
    # Launch kernel
    _f8_to_f16_kernel[grid](
        output,
        triton.reinterpret(x, triton_dtype),
        output.numel(),
        BLOCK_SIZE=1024,
    )
    
    return output
