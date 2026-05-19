import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.runtime.jit import get_cuda_stream

# Triton kernel for fused pairwise distance and normalization
@triton.jit
def fused_pairwise_distance_normalize_kernel(
    x1_ptr, x2_ptr, output_ptr, x1_stride, x2_stride, output_stride,
    n_cols, p_norm, eps_norm, eps_distance, N_COLS: tl.constexpr, BLOCK_N: tl.constexpr
):
    prog_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_N)

    # Load x1 and x2
    x1 = tl.load(x1_ptr + prog_id * x1_stride + offsets, mask=offsets < n_cols)
    x2 = tl.load(x2_ptr + prog_id * x2_stride + offsets, mask=offsets < n_cols)

    # Normalize x1 and x2
    norm1 = tl.sum(tl.abs(x1) ** p_norm, axis=0) ** (1.0 / p_norm)
    norm2 = tl.sum(tl.abs(x2) ** p_norm, axis=0) ** (1.0 / p_norm)
    x1_normalized = x1 / (norm1 + eps_norm)
    x2_normalized = x2 / (norm2 + eps_norm)

    # Compute pairwise distance
    diff = x1_normalized - x2_normalized
    distance = tl.sum(tl.abs(diff) ** p_norm, axis=0) ** (1.0 / p_norm)
    distance = tl.where(tl.isnan(distance), 0.0, distance) + eps_distance

    # Store the result
    tl.store(output_ptr + prog_id * output_stride + offsets, distance, mask=offsets < n_cols)

# Wrapper function
@torch.inference_mode()
def fused_pairwise_distance_normalize(x1: Tensor, x2: Tensor, p_norm: float = 2.0, eps_norm: float = 1e-12, eps_distance: float = 1e-6, keepdim: bool = False) -> Tensor:
    """
    Computes the pairwise distance between two input tensors `x1` and `x2` after normalizing both tensors.
    Normalization is performed along the specified dimension, followed by pairwise distance calculation.

    Args:
        x1 (Tensor): First input tensor.
        x2 (Tensor): Second input tensor.
        p_norm (float, optional): The exponent value in the norm for normalization. Default: 2.
        eps_norm (float, optional): Small value to avoid division by zero during normalization. Default: 1e-12.
        eps_distance (float, optional): Small value to avoid division by zero in distance calculation. Default: 1e-6.
        keepdim (bool, optional): If `True`, retains the last dimension in the output. Default: `False`.

    Returns:
        Tensor: The output tensor after applying the fused pairwise distance and normalization.
    """

    def _kernel_meta():
        device = x1.device
        device_idx = device.index
        device_type = device.type
        stream = get_cuda_stream(device_idx)
        return dict(device=device, device_type=device_type, stream=stream)

    feat_size = x1.size(-1)
    seq_len = x1.numel() // feat_size
    x1_stride = x1.stride(-2)
    x2_stride = x2.stride(-2)

    BLOCK_N = triton.next_power_of_2(feat_size)
    output_shape = list(x1.shape)
    if not keepdim:
        output_shape[-1] = 1
    output = torch.empty(output_shape, dtype=x1.dtype, device=x1.device)

    kernel_meta = _kernel_meta()
    grid = (seq_len,)
    fused_pairwise_distance_normalize_kernel[grid](
        x1, x2, output, x1_stride, x2_stride, output.stride(-2),
        feat_size, p_norm, eps_norm, eps_distance, feat_size, BLOCK_N,
        num_warps=4, num_stages=2, **kernel_meta
    )

    return output
