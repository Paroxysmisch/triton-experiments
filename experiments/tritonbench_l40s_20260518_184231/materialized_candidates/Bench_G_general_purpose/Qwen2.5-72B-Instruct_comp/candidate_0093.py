import triton
import triton.language as tl
import torch

# Triton kernel for converting float8 (stored as int8) to float16
@triton.jit
def kernel_f8_to_f16(X, Y, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(X + offsets, mask=mask)
    y = x.to(tl.float16)
    tl.store(Y + offsets, y, mask=mask)

# Wrapper function for f8_to_f16
def f8_to_f16(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.int8, "Input tensor must be of type torch.int8"
    assert x.device.type == 'cuda', "Input tensor must be on a CUDA device"
    
    y = torch.empty_like(x, dtype=torch.float16, device=x.device)
    n_elements = x.numel()
    grid = (triton.cdiv(n_elements, 1024),)
    kernel_f8_to_f16[grid](x, y, n_elements, BLOCK_SIZE=1024)
    return y

# Triton kernel for converting float16 or float32 to float8 (stored as int8)
@triton.jit
def kernel_f16_to_f8(X, Y, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(X + offsets, mask=mask)
    y = x.to(tl.int8)
    tl.store(Y + offsets, y, mask=mask)

# Wrapper function for f16_to_f8
def f16_to_f8(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype in [torch.float16, torch.float32], "Input tensor must be of type torch.float16 or torch.float32"
    assert x.device.type == 'cuda', "Input tensor must be on a CUDA device"
    
    y = torch.empty_like(x, dtype=torch.int8, device=x.device)
    n_elements = x.numel()
    grid = (triton.cdiv(n_elements, 1024),)
    kernel_f16_to_f8[grid](x, y, n_elements, BLOCK_SIZE=1024)
    return y
