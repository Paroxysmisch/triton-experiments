import triton
import triton.language as tl

@triton.jit
def bitwise_and_binomial_kernel(
    input_ptr, other_ptr, total_count_ptr, probs_ptr, logits_ptr, output_ptr,
    n_elements, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    input_vals = tl.load(input_ptr + offsets, mask=mask)
    other_vals = tl.load(other_ptr + offsets, mask=mask)
    total_count_vals = tl.load(total_count_ptr + offsets, mask=mask)
    probs_vals = tl.load(probs_ptr + offsets, mask=mask) if probs_ptr != 0 else 0.0
    logits_vals = tl.load(logits_ptr + offsets, mask=mask) if logits_ptr != 0 else 0.0

    # Compute bitwise AND
    and_result = input_vals & other_vals

    # Convert logits to probs if logits are provided
    if logits_ptr != 0:
        probs_vals = 1 / (1 + tl.exp(-logits_vals))

    # Sample from Binomial distribution
    output_vals = tl.random.binomial(and_result, total_count_vals, probs_vals, seed=tl.program_id(axis=1))

    tl.store(output_ptr + offsets, output_vals, mask=mask)

import torch
import triton
import triton.language as tl

def bitwise_and_binomial(input: torch.Tensor, other: torch.Tensor, total_count: torch.Tensor, probs: torch.Tensor = None, logits: torch.Tensor = None) -> torch.Tensor:
    assert input.dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.bool], "input must be of integral or Boolean type"
    assert other.dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.bool], "other must be of integral or Boolean type"
    assert probs is not None or logits is not None, "Either probs or logits must be provided"
    assert probs is None or logits is None, "Only one of probs or logits should be provided"
    
    # Ensure total_count is broadcastable with probs or logits
    if probs is not None:
        assert total_count.shape == probs.shape, "total_count and probs must have the same shape"
    if logits is not None:
        assert total_count.shape == logits.shape, "total_count and logits must have the same shape"

    # Ensure input and other are broadcastable
    assert input.shape == other.shape, "input and other must have the same shape"

    # Flatten the tensors for the kernel
    input_flat = input.flatten()
    other_flat = other.flatten()
    total_count_flat = total_count.flatten()
    probs_flat = probs.flatten() if probs is not None else None
    logits_flat = logits.flatten() if logits is not None else None

    # Allocate output tensor
    output = torch.empty_like(input_flat, dtype=torch.int32)

    # Launch the kernel
    n_elements = input_flat.numel()
    grid = (triton.cdiv(n_elements, 1024), 1)
    bitwise_and_binomial_kernel[grid](
        input_flat, other_flat, total_count_flat, probs_flat, logits_flat, output,
        n_elements, BLOCK_SIZE=1024
    )

    return output.reshape(input.shape)

import torch

# Sample data
input = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
other = torch.tensor([1, 1, 1, 1], dtype=torch.int32)
total_count = torch.tensor([2, 3, 4, 5], dtype=torch.int32)
probs = torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.float32)

# Call the function
output = bitwise_and_binomial(input, other, total_count, probs=probs)

print(output)
