import torch
import triton
import triton.language as tl

@triton.jit
def kernel_f8_to_f16(Y, X, N, BLOCK_SIZE: tl.constexpr):
    # Get program ID and compute offsets
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid indices
    mask = offs < N
    
    # Load data with masking
    x = tl.load(X + offs, mask=mask)
    
    # Store converted data
    tl.store(Y + offs, x, mask=mask)

def f8_to_f16(x):
    # Input validation
    assert x.dtype == torch.int8, f"Input tensor must be int8, got {x.dtype}"
    assert x.device.type == "cuda", "Input tensor must be on CUDA device"
    
    # Create output tensor
    out = torch.empty(x.shape, dtype=torch.float16, device=x.device)
    
    # Calculate grid size
    grid = lambda META: (triton.cdiv(x.numel(), META['BLOCK_SIZE']),)
    
    # Launch kernel
    kernel_f8_to_f16[grid](
        out, 
        triton.reinterpret(x, tl.float8), 
        x.numel(), 
        BLOCK_SIZE=1024
    )
    return out

@triton.jit
def kernel_f16_to_f8(Y, X, N, BLOCK_SIZE: tl.constexpr):
    # Get program ID and compute offsets
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid indices
    mask = offs < N
    
    # Load data with masking
    x = tl.load(X + offs, mask=mask)
    
    # Store converted data
    tl.store(Y + offs, x, mask=mask)

def f16_to_f8(x):
    # Input validation
    assert x.dtype in [torch.float16, torch.float32], f"Input tensor must be float16 or float32, got {x.dtype}"
    assert x.device.type == "cuda", "Input tensor must be on CUDA device"
    
    # Create output tensor
    out = torch.empty(x.shape, dtype=torch.int8, device=x.device)
    
    # Calculate grid size
    grid = lambda META: (triton.cdiv(x.numel(), META['BLOCK_SIZE']),)
    
    # Launch kernel
    kernel_f16_to_f8[grid](
        triton.reinterpret(out, tl.float8),
        x,
        x.numel(),
        BLOCK_SIZE=1024
    )
    return out
