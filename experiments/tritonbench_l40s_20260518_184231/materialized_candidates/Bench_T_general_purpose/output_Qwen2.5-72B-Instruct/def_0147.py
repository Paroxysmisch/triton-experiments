import triton
import triton.language as tl

@triton.jit
def pairwise_distance_normalize_kernel(
    x1_ptr,  # *Pointer* to first input tensor
    x2_ptr,  # *Pointer* to second input tensor
    out_ptr,  # *Pointer* to output tensor
    n_elements,  # Number of elements in the input tensors
    n_features,  # Number of features (dimension to normalize along)
    p_norm,  # Norm exponent
    eps_norm,  # Small value for normalization
    eps_distance,  # Small value for distance calculation
    keepdim,  # Whether to keep the last dimension
    BLOCK_SIZE: tl.constexpr  # Block size for parallelization
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load the data
    x1 = tl.load(x1_ptr + offsets, mask=mask)
    x2 = tl.load(x2_ptr + offsets, mask=mask)

    # Normalize x1 and x2
    norm_x1 = tl.sum(tl.abs(x1) ** p_norm, axis=1) ** (1.0 / p_norm)
    norm_x2 = tl.sum(tl.abs(x2) ** p_norm, axis=1) ** (1.0 / p_norm)
    x1_normalized = x1 / (norm_x1 + eps_norm)
    x2_normalized = x2 / (norm_x2 + eps_norm)

    # Compute pairwise distance
    diff = x1_normalized - x2_normalized
    distance = tl.sum(tl.abs(diff) ** p_norm, axis=1) ** (1.0 / p_norm) + eps_distance

    # Store the result
    tl.store(out_ptr + offsets, distance, mask=mask)

import torch
import triton
import triton.language as tl

def fused_pairwise_distance_normalize(x1: torch.Tensor, x2: torch.Tensor, p_norm: float = 2.0, eps_norm: float = 1e-12, eps_distance: float = 1e-6, keepdim: bool = False) -> torch.Tensor:
    # Ensure x1 and x2 have the same shape
    assert x1.shape == x2.shape, "Input tensors must have the same shape"

    # Flatten the tensors to 2D for normalization
    x1_flattened = x1.view(-1, x1.shape[-1])
    x2_flattened = x2.view(-1, x2.shape[-1])

    # Allocate output tensor
    out_shape = list(x1.shape[:-1]) + ([1] if keepdim else [])
    out = torch.empty(out_shape, device=x1.device, dtype=x1.dtype)

    # Launch the Triton kernel
    n_elements = x1_flattened.numel()
    n_features = x1_flattened.shape[1]
    grid = (triton.cdiv(n_elements, 1024),)
    pairwise_distance_normalize_kernel[grid](
        x1_flattened, x2_flattened, out, n_elements, n_features, p_norm, eps_norm, eps_distance, keepdim, BLOCK_SIZE=1024
    )

    return out
