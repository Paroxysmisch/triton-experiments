import triton
import triton.language as tl
import torch
from binomial import binomial # Assume that the Binomial op is already compiled and available

@triton.jit
def bitwise_and_binomial_kernel(input, other, total_count, probs, logits, output, 
                               n_elements,
                               BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs into SRAM
    input_vals = tl.load(input + offsets, mask=mask)
    other_vals = tl.load(other + offsets, mask=mask)
    total_count_vals = tl.load(total_count + offsets, mask=mask)
    
    # Bitwise AND operation
    and_result = input_vals & other_vals
    
    # Use AND result as Binomial parameter and sample from Binomial distribution
    # Note: only one of probs or logits should be passed in, so we ignore probs here
    sampled_vals = binomial(total_count_vals, static_cast<fp64>(and_result), logits=logits)
    
    # Store result in DRAM
    tl.store(output + offsets, sampled_vals, mask=mask)

def bitwise_and_binomial(input: torch.Tensor, other: torch.Tensor, total_count: torch.Tensor, probs: torch.Tensor = None, logits: torch.Tensor = None) -> torch.Tensor:
    """
    Element-wise operation that performs a bitwise AND on input and other, then samples from a binomial distribution using the result as the success rate.

    Args:
        input (torch.Tensor): Input tensor for AND operation.
        other (torch.Tensor): Other tensor for AND operation.
        total_count (torch.Tensor): Number of Bernoulli trials for binomial distribution.
        probs (torch.Tensor, optional): Tensor of probabilities for binomial distribution. Defaults to None.
        logits (torch.Tensor, optional): Tensor of logits for binomial distribution. Defaults to None.

    Returns:
        torch.Tensor: Resultant tensor from binomial sampling.
    """
    assert input.is_integral_type() | input.is_boolean(), "Invalid input type, must be integral or boolean"
    assert other.is_integral_type() | other.is_boolean(), "Invalid other type, must be integral or boolean"
    assert total_count.ndim == 0 or total_count.shape == broadcasted_shape, "Total count shape must match broadcasting of input and other shapes"
    assert (probs is None) ^ (logits is None), "Must provide exactly one of probs or logits"
    
    broadcasted_shape = broadcast_shapes([input.shape, other.shape, total_count.shape])
    output = torch.empty(broadcasted_shape, device=input.device, dtype=torch.int64)
    n_elements = output.numel()
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    
    # Run the Triton kernel
    bitwise_and_binomial_kernel[grid](input, other, total_count, probs, logits, output, n_elements, BLOCK_SIZE=1024)
    
    return output
