import torch
import triton
import triton.language as tl

device = 'cuda:0'

@triton.jit
def rsqrt_kernel(
    input_ptr,
    output_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    # Load input elements
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Compute reciprocal square root, handling negative values
    x_safe = tl.where(x >= 0, x, float('nan'))
    rsqrt_val = 1.0 / tl.sqrt(x_safe)
    
    # Store the result
    tl.store(output_ptr + offsets, rsqrt_val, mask=mask)

def rsqrt(input: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    # Ensure input is on CUDA
    assert input.is_cuda, "Input tensor must be on CUDA"
    
    # Create output tensor if not provided
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.shape == input.shape, "Output tensor shape mismatch"
        assert out.is_cuda, "Output tensor must be on CUDA"
    
    numel = input.numel()
    if numel == 0:
        return out  # Handle empty tensor
    
    # Set block size, using a reasonable power of two
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(numel, BLOCK_SIZE),)
    
    # Launch kernel
    rsqrt_kernel[grid](input, out, numel, BLOCK_SIZE=BLOCK_SIZE)
    
    return out
