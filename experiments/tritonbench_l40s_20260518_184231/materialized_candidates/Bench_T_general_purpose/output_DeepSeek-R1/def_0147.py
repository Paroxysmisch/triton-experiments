import torch
import triton
import triton.language as tl

@triton.jit
def fused_pairwise_distance_normalize_kernel(
    x1_ptr, x2_ptr, output_ptr,
    p_norm, eps_norm, eps_distance,
    x1_stride_row, x1_stride_col,
    x2_stride_row, x2_stride_col,
    output_stride_row, output_stride_col,
    feature_dim, M, N,
    BLOCK_SIZE: tl.constexpr
):
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    # Compute x1's norm
    x1_row = pid_i
    x1_norm_sum = 0.0
    for offset in range(0, feature_dim, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < feature_dim
        x1_ptrs = x1_ptr + x1_row * x1_stride_row + cols * x1_stride_col
        x1 = tl.load(x1_ptrs, mask=mask, other=0.0)
        x1_abs = tl.math.abs(x1)
        x1_pow = tl.math.pow(x1_abs, p_norm)
        x1_norm_sum += tl.sum(x1_pow, axis=0)
    x1_norm_val = tl.math.pow(x1_norm_sum, 1.0 / p_norm) + eps_norm
    
    # Compute x2's norm
    x2_row = pid_j
    x2_norm_sum = 0.0
    for offset in range(0, feature_dim, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < feature_dim
        x2_ptrs = x2_ptr + x2_row * x2_stride_row + cols * x2_stride_col
        x2 = tl.load(x2_ptrs, mask=mask, other=0.0)
        x2_abs = tl.math.abs(x2)
        x2_pow = tl.math.pow(x2_abs, p_norm)
        x2_norm_sum += tl.sum(x2_pow, axis=0)
    x2_norm_val = tl.math.pow(x2_norm_sum, 1.0 / p_norm) + eps_norm
    
    # Compute distance
    distance_sum = 0.0
    for offset in range(0, feature_dim, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < feature_dim
        x1_ptrs = x1_ptr + x1_row * x1_stride_row + cols * x1_stride_col
        x2_ptrs = x2_ptr + x2_row * x2_stride_row + cols * x2_stride_col
        x1 = tl.load(x1_ptrs, mask=mask, other=0.0)
        x2 = tl.load(x2_ptrs, mask=mask, other=0.0)
        x1_normalized = x1 / x1_norm_val
        x2_normalized = x2 / x2_norm_val
        diff = x1_normalized - x2_normalized
        diff_abs = tl.math.abs(diff)
        diff_pow = tl.math.pow(diff_abs, p_norm)
        distance_sum += tl.sum(diff_pow, axis=0)
    distance_sum += eps_distance
    distance = tl.math.pow(distance_sum, 1.0 / p_norm)
    
    # Store the result
    output_ptrs = output_ptr + pid_i * output_stride_row + pid_j * output_stride_col
    tl.store(output_ptrs, distance)

def fused_pairwise_distance_normalize(x1: torch.Tensor, x2: torch.Tensor, p_norm: float = 2.0, eps_norm: float = 1e-12, eps_distance: float = 1e-6, keepdim: bool = False) -> torch.Tensor:
    assert x1.dim() == 2 and x2.dim() == 2, "Inputs must be 2D tensors"
    M, D = x1.shape
    N, D2 = x2.shape
    assert D == D2, "Feature dimensions must match"
    
    output = torch.empty((M, N), device=x1.device, dtype=x1.dtype)
    
    BLOCK_SIZE = 128  # Tune this based on your GPU's capabilities
    
    grid = (M, N)
    
    fused_pairwise_distance_normalize_kernel[grid](
        x1, x2, output,
        p_norm, eps_norm, eps_distance,
        x1.stride(0), x1.stride(1),
        x2.stride(0), x2.stride(1),
        output.stride(0), output.stride(1),
        D, M, N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    if not keepdim:
        output = output.squeeze(-1) if output.dim() > 2 else output
    return output
