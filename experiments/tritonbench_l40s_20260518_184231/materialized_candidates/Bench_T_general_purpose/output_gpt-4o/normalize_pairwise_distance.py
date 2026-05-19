import triton
import triton.language as tl

@triton.jit
def pairwise_distance_kernel(x1_ptr, x2_ptr, output_ptr, n_elements, p_distance, eps_distance, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x1 = tl.load(x1_ptr + offsets, mask=mask, other=0.0)
    x2 = tl.load(x2_ptr + offsets, mask=mask, other=0.0)

    diff = x1 - x2
    abs_diff_p = tl.abs(diff) ** p_distance
    sum_abs_diff_p = tl.sum(abs_diff_p, axis=0)
    distance = (sum_abs_diff_p + eps_distance) ** (1.0 / p_distance)

    tl.store(output_ptr + offsets, distance, mask=mask)

import torch

def normalize_pairwise_distance(x1, x2, p_distance=2.0, eps_distance=1e-6, keepdim=False, p_norm=2, dim_norm=1, eps_norm=1e-12):
    assert x1.shape == x2.shape, "x1 and x2 must have the same shape"
    n_elements = x1.numel()
    
    # Allocate output tensor
    output = torch.empty_like(x1)
    
    # Launch Triton kernel
    BLOCK_SIZE = 1024  # Choose a suitable block size
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    pairwise_distance_kernel[grid](x1, x2, output, n_elements, p_distance, eps_distance, BLOCK_SIZE=BLOCK_SIZE)
    
    # Normalize the pairwise distances
    norm = torch.norm(output, p=p_norm, dim=dim_norm, keepdim=keepdim)
    norm = torch.maximum(norm, torch.tensor(eps_norm, device=norm.device))
    normalized_output = output / norm
    
    return normalized_output
