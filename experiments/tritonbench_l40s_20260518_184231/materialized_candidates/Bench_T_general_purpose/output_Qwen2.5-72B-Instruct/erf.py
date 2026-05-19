import triton
import triton.language as tl

@triton.jit
def erf_kernel(
    input_ptr,  # Pointer to the input tensor
    output_ptr,  # Pointer to the output tensor
    n_elements,  # Number of elements in the tensor
    BLOCK_SIZE: tl.constexpr,  # Block size
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # Compute the error function
    t = x * x
    t = -t
    t = tl.exp(t)
    t = t * (2 / tl.sqrt(tl.float32(3.141592653589793)))
    t = t * x
    t = t * (1 + t * (0.5 + t * (0.125 + t * (0.0625 + t * 0.03125))))
    t = t * (1 - t * (0.5 + t * (0.125 + t * (0.0625 + t * 0.03125))))
    
    tl.store(output_ptr + offsets, t, mask=mask)

import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
    ],
    key=['n_elements'],
)
@triton.jit
def erf_kernel(
    input_ptr,  # Pointer to the input tensor
    output_ptr,  # Pointer to the output tensor
    n_elements,  # Number of elements in the tensor
    BLOCK_SIZE: tl.constexpr,  # Block size
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # Compute the error function
    t = x * x
    t = -t
    t = tl.exp(t)
    t = t * (2 / tl.sqrt(tl.float32(3.141592653589793)))
    t = t * x
    t = t * (1 + t * (0.5 + t * (0.125 + t * (0.0625 + t * 0.03125))))
    t = t * (1 - t * (0.5 + t * (0.125 + t * (0.0625 + t * 0.03125))))
    
    tl.store(output_ptr + offsets, t, mask=mask)

def erf(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    
    n_elements = input.numel()
    grid = (triton.cdiv(n_elements, 1024),)
    
    erf_kernel[grid](
        input.data_ptr(),
        out.data_ptr(),
        n_elements,
        BLOCK_SIZE=1024,
    )
    
    return out
