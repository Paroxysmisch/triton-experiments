import triton
import triton.language as tl
import torch
import math
from torch.distributions import Binomial

# Reuse the bitwise_and function from the provided document
def bitwise_and(A, B):
    C = torch.empty_like(A)
    n_elements = C.numel()
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    grid_size = triton.cdiv(n_elements, block_size)
    
    if isinstance(B, torch.Tensor):
        bitwise_and_func_tensor[(grid_size, 1, 1)](A, B, C, n_elements, block_size)
    else:
        bitwise_and_func_scalar[(grid_size, 1, 1)](A, B, C, n_elements, block_size)   
    return C

@triton.jit
def bitwise_and_func_tensor(A_ptr, B_ptr, C_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    A = tl.load(A_ptr + offsets, mask=mask)
    B = tl.load(B_ptr + offsets, mask=mask)
    C = A & B
    tl.store(C_ptr + offsets, C, mask=mask)

@triton.jit
def bitwise_and_func_scalar(A_ptr, B: tl.constexpr, C_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    A = tl.load(A_ptr + offsets, mask=mask)
    C = A & B
    tl.store(C_ptr + offsets, C, mask=mask)

def bitwise_and_binomial(
    input: torch.Tensor,
    other: torch.Tensor,
    total_count: torch.Tensor,
    probs: torch.Tensor = None,
    logits: torch.Tensor = None
) -> torch.Tensor:
    # Compute the bitwise AND of input and other using Triton
    bit_result = bitwise_and(input, other)
    
    # Adjust the total_count by the bitwise result (element-wise multiplication with broadcasting)
    adjusted_total_count = bit_result * total_count
    
    # Ensure only one of probs or logits is provided
    if (probs is not None) and (logits is not None):
        raise ValueError("Only one of probs or logits should be specified.")
    if probs is None and logits is None:
        raise ValueError("Either probs or logits must be specified.")
    
    # Create the Binomial distribution
    if probs is not None:
        dist = Binomial(total_count=adjusted_total_count, probs=probs)
    else:
        dist = Binomial(total_count=adjusted_total_count, logits=logits)
    
    # Sample from the distribution
    return dist.sample()
