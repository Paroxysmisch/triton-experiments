import torch
import triton
import triton.language as tl
from typing import Optional

device = 'cuda:0'

@triton.jit
def zeta_kernel(
    input_ptr,
    other_ptr,
    output_ptr,
    num_elements,
    K_MAX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    x = tl.load(input_ptr + offsets, mask=mask)
    q = tl.load(other_ptr + offsets, mask=mask)

    sum_zeta = tl.zeros_like(x)
    for k in range(0, K_MAX):
        denom = k + q
        log_denom = tl.log(denom)
        pow_denom = tl.exp(x * log_denom)
        term = 1.0 / pow_denom
        sum_zeta += term

    tl.store(output_ptr + offsets, sum_zeta, mask=mask)

def zeta(input: torch.Tensor, other: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    assert input.is_cuda and other.is_cuda, "Input tensors must be on GPU"
    
    # Broadcast input and other to a common shape
    broadcasted_shape = torch.broadcast_shapes(input.shape, other.shape)
    input_bc = input.expand(broadcasted_shape).contiguous()
    other_bc = other.expand(broadcasted_shape).contiguous()
    
    # Prepare output tensor
    if out is None:
        out = torch.empty_like(input_bc)
    else:
        assert out.is_cuda, "Output tensor must be on GPU"
        assert out.shape == input_bc.shape, "Output shape does not match broadcasted input shape"
    
    num_elements = input_bc.numel()
    
    # Kernel configuration
    BLOCK_SIZE = 1024  # Tunable block size
    K_MAX = 1000       # Tunable maximum terms for convergence
    grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
    
    zeta_kernel[grid](
        input_bc, other_bc, out,
        num_elements,
        K_MAX=K_MAX,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out
