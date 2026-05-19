import torch
import triton
import triton.language as tl

@triton.jit
def kernel_f8_to_f16(
    X,  # Input pointer (int8)
    Y,  # Output pointer (float16)
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,  # Number of elements per block
):
    # Compute program ID and corresponding offsets
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements
    mask = offs < n_elements
    
    # Load data with masking
    x = tl.load(X + offs, mask=mask)
    
    # Reinterpret int8 as float8 and convert to float16
    x = tl.float8_to_float16(x)
    
    # Store results
    tl.store(Y + offs, x, mask=mask)

def f8_to_f16(x: torch.Tensor) -> torch.Tensor:
    """Convert float8 (stored as int8) tensor to float16."""
    assert x.dtype == torch.int8, f"Input tensor must be int8, got {x.dtype}"
    assert x.is_cuda, "Input tensor must be on CUDA device"
    
    # Create output tensor
    y = torch.empty_like(x, dtype=torch.float16, device=x.device)
    
    # Calculate grid size
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(x.numel(), BLOCK_SIZE),)
    
    # Launch kernel
    kernel_f8_to_f16[grid](
        x, y,
        x.numel(),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y

@triton.jit
def kernel_f16_to_f8(
    X,  # Input pointer (float16/float32)
    Y,  # Output pointer (int8)
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,  # Number of elements per block
):
    # Compute program ID and corresponding offsets
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements
    mask = offs < n_elements
    
    # Load data with masking
    x = tl.load(X + offs, mask=mask)
    
    # Convert to float8 and reinterpret as int8
    x = tl.float16_to_float8(x)
    
    # Store results
    tl.store(Y + offs, x, mask=mask)

def f16_to_f8(x: torch.Tensor) -> torch.Tensor:
    """Convert float16/float32 tensor to float8 (stored as int8)."""
    assert x.dtype in [torch.float16, torch.float32], f"Input tensor must be float16 or float32, got {x.dtype}"
    assert x.is_cuda, "Input tensor must be on CUDA device"
    
    # Create output tensor
    y = torch.empty_like(x, dtype=torch.int8, device=x.device)
    
    # Calculate grid size
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(x.numel(), BLOCK_SIZE),)
    
    # Launch kernel
    kernel_f16_to_f8[grid](
        x, y,
        x.numel(),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y
