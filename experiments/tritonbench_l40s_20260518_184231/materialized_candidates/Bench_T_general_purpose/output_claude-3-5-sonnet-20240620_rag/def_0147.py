import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.runtime.jit import get_cuda_stream

@triton.jit
def fused_pairwise_distance_normalize_kernel(
    x1_ptr, x2_ptr, output_ptr,
    stride_x1_row, stride_x2_row,
    n_cols, p_norm, eps_norm, eps_distance,
    N_COLS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Get program ID for the current instance
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1)
    
    # Create offsets for the block
    offsets = tl.arange(0, BLOCK_N)
    
    # Load data for x1
    x1_row_ptr = x1_ptr + row_idx * stride_x1_row
    x1_data = tl.load(x1_row_ptr + offsets, mask=offsets < n_cols)
    x1_float = x1_data.to(tl.float32)
    
    # Load data for x2
    x2_row_ptr = x2_ptr + col_idx * stride_x2_row
    x2_data = tl.load(x2_row_ptr + offsets, mask=offsets < n_cols)
    x2_float = x2_data.to(tl.float32)
    
    # Normalize x1
    x1_norm = tl.sum(x1_float * x1_float, 0)
    x1_norm = tl.sqrt(x1_norm + eps_norm)
    x1_normalized = x1_float / x1_norm
    
    # Normalize x2
    x2_norm = tl.sum(x2_float * x2_float, 0)
    x2_norm = tl.sqrt(x2_norm + eps_norm)
    x2_normalized = x2_float / x2_norm
    
    # Compute pairwise distance
    diff = x1_normalized - x2_normalized
    if p_norm == 2.0:
        distance = tl.sum(diff * diff, 0)
        distance = tl.sqrt(distance + eps_distance)
    else:
        distance = tl.sum(tl.abs(diff) ** p_norm, 0)
        distance = distance ** (1.0 / p_norm)
    
    # Store the result
    output_idx = row_idx * x2_data.shape[0] + col_idx
    tl.store(output_ptr + output_idx, distance)

@torch.inference_mode()
def fused_pairwise_distance_normalize(
    x1: torch.Tensor,
    x2: torch.Tensor,
    p_norm: float = 2.0,
    eps_norm: float = 1e-12,
    eps_distance: float = 1e-6,
    keepdim: bool = False
) -> torch.Tensor:
    """
    Computes the pairwise distance between two input tensors after normalization.
    
    Args:
        x1 (Tensor): First input tensor
        x2 (Tensor): Second input tensor
        p_norm (float, optional): The exponent value in the norm for normalization. Default: 2.0
        eps_norm (float, optional): Small value to avoid division by zero during normalization. Default: 1e-12
        eps_distance (float, optional): Small value to avoid division by zero in distance calculation. Default: 1e-6
        keepdim (bool, optional): If True, retains the last dimension in the output. Default: False
        
    Returns:
        Tensor: Pairwise distances between normalized vectors
    """
    # Input validation
    assert x1.is_cuda and x2.is_cuda, "Inputs must be CUDA tensors"
    assert x1.dim() == 2 and x2.dim() == 2, "Inputs must be 2-dimensional"
    assert x1.size(1) == x2.size(1), "Feature dimensions must match"
    
    # Get dimensions
    n_x1, feat_size = x1.shape
    n_x2 = x2.shape[0]
    
    # Prepare kernel parameters
    BLOCK_N = triton.next_power_of_2(feat_size)
    
    # Prepare output tensor
    output = torch.empty((n_x1, n_x2), device=x1.device, dtype=x1.dtype)
    
    # Get kernel metadata
    device = x1.device
    device_idx = device.index
    device_type = device.type
    stream = get_cuda_stream(device_idx)
    kernel_meta = dict(device=device, device_type=device_type, stream=stream)
    
    # Launch kernel
    grid = (n_x1, n_x2)
    fused_pairwise_distance_normalize_kernel[grid](
        x1, x2, output,
        x1.stride(0), x2.stride(0),
        feat_size, p_norm, eps_norm, eps_distance,
        feat_size, BLOCK_N,
        num_warps=4,
        num_stages=2,
        **kernel_meta
    )
    
    if keepdim:
        output = output.unsqueeze(-1)
    
    return output
