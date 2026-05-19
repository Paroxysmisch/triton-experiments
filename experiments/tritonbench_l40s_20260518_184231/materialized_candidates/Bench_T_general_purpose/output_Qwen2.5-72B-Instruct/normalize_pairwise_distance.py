import triton
import triton.language as tl

@triton.jit
def pairwise_distance_kernel(
    x1_ptr,  # Pointer to the first input tensor
    x2_ptr,  # Pointer to the second input tensor
    out_ptr,  # Pointer to the output tensor
    n_elements,  # Number of elements in each vector
    n_vectors,  # Number of vectors
    p,  # Norm degree
    eps,  # Small value to avoid division by zero
    BLOCK_SIZE: tl.constexpr  # Block size for parallelization
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_vectors

    x1_offsets = offsets[:, None] * n_elements + tl.arange(0, n_elements)[None, :]
    x2_offsets = offsets[:, None] * n_elements + tl.arange(0, n_elements)[None, :]

    x1 = tl.load(x1_ptr + x1_offsets, mask=mask[:, None], other=0.0)
    x2 = tl.load(x2_ptr + x2_offsets, mask=mask[:, None], other=0.0)

    diff = x1 - x2
    distance = tl.sum(tl.abs(diff) ** p, axis=1) ** (1.0 / p)
    distance = tl.where(distance < eps, eps, distance)

    tl.store(out_ptr + offsets, distance, mask=mask)

import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
    ],
    key=['n_vectors', 'n_elements']
)
@triton.jit
def normalize_pairwise_distance(
    x1,  # The first input tensor
    x2,  # The second input tensor, must have the same shape as `x1`
    p_distance=2.0,  # The norm degree for computing the pairwise distance
    eps_distance=1e-6,  # Small value to avoid division by zero in pairwise distance calculation
    keepdim=False,  # Whether to keep the reduced dimensions in the output
    p_norm=2,  # The exponent value in the norm formulation for normalization
    dim_norm=1,  # The dimension along which normalization is applied
    eps_norm=1e-12  # Small value to avoid division by zero in normalization
):
    # Compute pairwise distance
    n_vectors, n_elements = x1.shape
    pairwise_distance = torch.empty((n_vectors,), dtype=x1.dtype, device=x1.device)
    pairwise_distance_kernel[(n_vectors,)](x1, x2, pairwise_distance, n_elements, n_vectors, p_distance, eps_distance, BLOCK_SIZE=128)

    # Compute norm of pairwise distances
    norm = torch.norm(pairwise_distance, p=p_norm, dim=dim_norm, keepdim=keepdim)
    norm = torch.max(norm, torch.tensor(eps_norm, device=x1.device, dtype=x1.dtype))

    # Normalize distances
    normalized_distance = pairwise_distance / norm

    return normalized_distance

import torch

# Test data
x1 = torch.randn((10, 5), device='cuda')
x2 = torch.randn((10, 5), device='cuda')

# Call the function
result = normalize_pairwise_distance(x1, x2, p_distance=2.0, eps_distance=1e-6, keepdim=False, p_norm=2, dim_norm=1, eps_norm=1e-12)

# Print the result
print(result)
