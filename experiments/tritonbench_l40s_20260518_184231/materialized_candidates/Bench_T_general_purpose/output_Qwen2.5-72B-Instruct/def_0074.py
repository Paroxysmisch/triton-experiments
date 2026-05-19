import triton
import triton.language as tl

@triton.jit
def normalized_cosine_similarity_kernel(
    x1_ptr, x2_ptr, output_ptr, 
    x1_size, x2_size, output_size, 
    dim, p_norm, eps_norm, eps_similarity,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < output_size

    # Load the input tensors
    x1 = tl.load(x1_ptr + offsets, mask=mask, other=0.0)
    x2 = tl.load(x2_ptr + offsets, mask=mask, other=0.0)

    # Normalize x1 and x2 along the specified dimension
    norm_x1 = tl.norm(x1, p_norm, dim, eps_norm)
    norm_x2 = tl.norm(x2, p_norm, dim, eps_norm)

    x1_normalized = x1 / tl.max(norm_x1, eps_norm)
    x2_normalized = x2 / tl.max(norm_x2, eps_norm)

    # Compute the dot product
    dot_product = tl.sum(x1_normalized * x2_normalized, dim)

    # Compute the cosine similarity
    norm_x1_normalized = tl.norm(x1_normalized, 2, dim, eps_similarity)
    norm_x2_normalized = tl.norm(x2_normalized, 2, dim, eps_similarity)
    similarity = dot_product / (tl.max(norm_x1_normalized, eps_similarity) * tl.max(norm_x2_normalized, eps_similarity))

    # Store the result
    tl.store(output_ptr + offsets, similarity, mask=mask)

import torch
import triton
import triton.language as tl

def normalized_cosine_similarity(x1: torch.Tensor, x2: torch.Tensor, dim: int = 1, eps_similarity: float = 1e-8, p_norm: float = 2, eps_norm: float = 1e-12) -> torch.Tensor:
    # Ensure x2 is broadcasted to match x1's shape
    x2 = x2.expand_as(x1)

    # Get the sizes of the tensors
    x1_size = x1.numel()
    x2_size = x2.numel()
    output_size = x1.size(dim)

    # Allocate output tensor
    output = torch.empty_like(x1)

    # Define the grid and block sizes
    BLOCK_SIZE = 128
    grid = (output_size + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the Triton kernel
    normalized_cosine_similarity_kernel[grid, BLOCK_SIZE](
        x1, x2, output,
        x1_size, x2_size, output_size,
        dim, p_norm, eps_norm, eps_similarity
    )

    return output

import torch

# Example inputs
x1 = torch.randn(4, 5, 6)
x2 = torch.randn(1, 5, 6)

# Call the function
output = normalized_cosine_similarity(x1, x2, dim=1, eps_similarity=1e-8, p_norm=2, eps_norm=1e-12)

print(output)
