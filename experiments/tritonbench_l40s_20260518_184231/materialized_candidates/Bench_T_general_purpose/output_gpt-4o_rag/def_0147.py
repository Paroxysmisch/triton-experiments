import torch
import triton
import triton.language as tl

@triton.jit
def normalize_and_pairwise_distance_kernel(
    x1_ptr, x2_ptr, output_ptr,
    stride_x1, stride_x2, stride_out,
    n_cols, eps_norm, eps_distance,
    p_norm: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    
    x1 = tl.load(x1_ptr + pid * stride_x1 + offsets, mask=offsets < n_cols, other=0.0)
    x2 = tl.load(x2_ptr + pid * stride_x2 + offsets, mask=offsets < n_cols, other=0.0)
    
    # Normalize x1 and x2
    norm_x1 = tl.norm(x1, p_norm) + eps_norm
    norm_x2 = tl.norm(x2, p_norm) + eps_norm
    x1_normalized = x1 / norm_x1
    x2_normalized = x2 / norm_x2
    
    # Compute pairwise distance
    diff = x1_normalized - x2_normalized
    distance = tl.norm(diff, p_norm) + eps_distance
    
    tl.store(output_ptr + pid * stride_out + offsets, distance, mask=offsets < n_cols)

@torch.inference_mode()
def fused_pairwise_distance_normalize(x1: torch.Tensor, x2: torch.Tensor, p_norm: float = 2.0, eps_norm: float = 1e-12, eps_distance: float = 1e-6, keepdim: bool = False) -> torch.Tensor:
    """
    Computes the pairwise distance between two input tensors `x1` and `x2` after normalizing both tensors.
    
    Args:
        x1 (Tensor): First input tensor.
        x2 (Tensor): Second input tensor.
        p_norm (float, optional): The exponent value in the norm for normalization. Default: 2.
        eps_norm (float, optional): Small value to avoid division by zero during normalization. Default: 1e-12.
        eps_distance (float, optional): Small value to avoid division by zero in distance calculation. Default: 1e-6.
        keepdim (bool, optional): If `True`, retains the last dimension in the output. Default: `False`.
    
    Returns:
        Tensor: The output tensor containing pairwise distances.
    """
    assert x1.shape == x2.shape, "Input tensors must have the same shape"
    assert x1.ndim == 2, "Input tensors must be 2D"

    n_rows, n_cols = x1.shape
    output = torch.empty((n_rows, 1 if keepdim else n_cols), device=x1.device, dtype=x1.dtype)

    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    grid = (n_rows,)

    normalize_and_pairwise_distance_kernel[grid](
        x1, x2, output,
        x1.stride(0), x2.stride(0), output.stride(0),
        n_cols, eps_norm, eps_distance,
        p_norm=p_norm, BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4
    )

    return output

# Example usage
x1 = torch.randn(128, 256, device='cuda')
x2 = torch.randn(128, 256, device='cuda')
output = fused_pairwise_distance_normalize(x1, x2)
print(output)
