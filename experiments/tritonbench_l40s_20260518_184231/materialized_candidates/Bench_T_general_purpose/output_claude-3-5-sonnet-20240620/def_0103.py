import torch
import triton
import triton.language as tl
import math

@triton.jit
def bitwise_and_binomial_kernel(
    output_ptr, input_ptr, other_ptr, total_count_ptr, probs_ptr, 
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Compute block start/end
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    x = tl.load(input_ptr + offsets, mask=mask)
    y = tl.load(other_ptr + offsets, mask=mask)
    n = tl.load(total_count_ptr + offsets, mask=mask)
    p = tl.load(probs_ptr + offsets, mask=mask)
    
    # Compute bitwise AND
    and_result = x & y
    
    # Binomial sampling using inverse transform sampling
    # We'll approximate it using normal distribution when n is large
    mean = and_result * n * p
    std = tl.sqrt(mean * (1 - p))
    
    # Generate random values using Triton's RNG
    rand = tl.rand(tl.zeros([BLOCK_SIZE], tl.float32))
    
    # Approximate binomial using normal distribution
    result = mean + std * tl.sqrt(-2 * tl.log(rand)) * tl.cos(2 * math.pi * rand)
    result = tl.round(result)
    
    # Clamp results between 0 and n
    result = tl.maximum(0, tl.minimum(n, result))
    
    # Store results
    tl.store(output_ptr + offsets, result, mask=mask)

def bitwise_and_binomial(
    input: torch.Tensor,
    other: torch.Tensor,
    total_count: torch.Tensor,
    probs: torch.Tensor = None,
    logits: torch.Tensor = None
) -> torch.Tensor:
    # Input validation
    if (probs is None) == (logits is None):
        raise ValueError("Either probs or logits must be specified, but not both")
    
    # Convert logits to probs if necessary
    if logits is not None:
        probs = torch.sigmoid(logits)
    
    # Ensure inputs are on the same device
    device = input.device
    input = input.to(device)
    other = other.to(device)
    total_count = total_count.to(device)
    probs = probs.to(device)
    
    # Get tensor shape
    n_elements = input.numel()
    
    # Create output tensor
    output = torch.empty_like(input, dtype=torch.float32)
    
    # Configure grid and block sizes
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    bitwise_and_binomial_kernel[grid](
        output.data_ptr(),
        input.data_ptr(),
        other.data_ptr(),
        total_count.data_ptr(),
        probs.data_ptr(),
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output
