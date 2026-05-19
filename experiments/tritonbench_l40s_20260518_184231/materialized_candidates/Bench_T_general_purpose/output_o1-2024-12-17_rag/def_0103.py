import triton
import triton.language as tl
import torch
import math

@triton.jit
def bitwise_and_kernel(
    input_ptr,
    other_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input_vals = tl.load(input_ptr + offsets, mask=mask)
    other_vals = tl.load(other_ptr + offsets, mask=mask)
    and_result = input_vals & other_vals
    tl.store(output_ptr + offsets, and_result, mask=mask)

def bitwise_and_triton(input: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(input)
    n_elements = output.numel()
    block_size = triton.next_power_of_2(1024)
    grid_size = triton.cdiv(n_elements, block_size)
    bitwise_and_kernel[(grid_size, 1, 1)](
        input, other, output, n_elements, BLOCK_SIZE=block_size
    )
    return output

def bitwise_and_binomial(
    input: torch.Tensor,
    other: torch.Tensor,
    total_count: torch.Tensor,
    probs: torch.Tensor = None,
    logits: torch.Tensor = None,
) -> torch.Tensor:
    if (probs is None) == (logits is None):
        raise ValueError("Exactly one of probs or logits must be provided")
    
    # Compute bitwise AND
    and_result = bitwise_and_triton(input, other)
    
    # Broadcast and_result to match total_count's shape if necessary
    # Here, we assume and_result is compatible for broadcasting with total_count
    # This may require additional checks in a production environment
    total_count = total_count.expand_as(and_result)
    
    # Use the bitwise_and result as the number of trials for each element
    # Note: This assumes that total_count is not needed, which conflicts with the wrapper's parameters
    # The following line is a conceptual step, but requires adjustment to fit the wrapper's parameters
    # In this case, we prioritize the functional description's requirement
    binomial_total_count = and_result
    
    # Perform binomial sampling
    if probs is not None:
        probs = probs.expand_as(binomial_total_count).to(binomial_total_count.dtype)
        return torch.binomial(binomial_total_count, probs=probs)
    else:
        logits = logits.expand_as(binomial_total_count).to(binomial_total_count.dtype)
        return torch.binomial(binomial_total_count, logits=logits)
