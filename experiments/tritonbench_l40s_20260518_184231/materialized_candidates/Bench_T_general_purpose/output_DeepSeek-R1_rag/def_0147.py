import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.runtime.jit import get_cuda_stream

@triton.jit
def normalize_kernel(
    input_ptr,
    output_ptr,
    input_row_stride,
    n_cols,
    p_norm,
    eps_norm,
    N_COLS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_N)
    mask = offsets < N_COLS
    
    row_start = input_ptr + row_idx * input_row_stride
    x = tl.load(row_start + offsets, mask=mask, other=0.0)
    
    x_abs = tl.abs(x)
    x_p = tl.pow(x_abs, p_norm)
    sum_p = tl.sum(x_p, axis=0)
    norm = tl.pow(sum_p, 1.0 / p_norm)
    denominator = norm + eps_norm
    normalized_x = x / denominator
    
    output_row_start = output_ptr + row_idx * input_row_stride
    tl.store(output_row_start + offsets, normalized_x, mask=mask)

@triton.jit
def pairwise_distance_kernel(
    x1_ptr,
    x1_row_stride,
    x2_ptr,
    x2_row_stride,
    output_ptr,
    output_row_stride,
    n_cols,
    p_norm,
    eps_distance,
    N_COLS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    i = tl.program_id(0)
    j = tl.program_id(1)
    
    offsets = tl.arange(0, BLOCK_N)
    mask = offsets < N_COLS
    
    x1_row = tl.load(x1_ptr + i * x1_row_stride + offsets, mask=mask, other=0.0)
    x2_row = tl.load(x2_ptr + j * x2_row_stride + offsets, mask=mask, other=0.0)
    
    diff = x1_row - x2_row
    abs_diff = tl.abs(diff)
    diff_p = tl.pow(abs_diff, p_norm)
    sum_p = tl.sum(diff_p, axis=0)
    distance = tl.pow(sum_p + eps_distance, 1.0 / p_norm)
    
    output_index = i * output_row_stride + j
    tl.store(output_ptr + output_index, distance)

@torch.inference_mode()
def fused_pairwise_distance_normalize(
    x1: torch.Tensor,
    x2: torch.Tensor,
    p_norm: float = 2.0,
    eps_norm: float = 1e-12,
    eps_distance: float = 1e-6,
    keepdim: bool = False,
) -> torch.Tensor:
    assert x1.shape[-1] == x2.shape[-1], "Feature dimension must be the same for x1 and x2"
    
    normalized_x1 = _normalize(x1, p_norm, eps_norm)
    normalized_x2 = _normalize(x2, p_norm, eps_norm)
    
    output = _pairwise_distance(normalized_x1, normalized_x2, p_norm, eps_distance)
    
    if keepdim:
        output = output.unsqueeze(-1)
    
    return output

def _normalize(input_tensor, p_norm, eps_norm):
    original_shape = input_tensor.shape
    input_flat = input_tensor.view(-1, original_shape[-1])
    n_rows, n_cols = input_flat.shape
    
    output = torch.empty_like(input_flat)
    BLOCK_N = triton.next_power_of_two(n_cols)
    grid = (n_rows,)
    
    def _kernel_meta():
        device = input_tensor.device
        device_idx = device.index
        stream = get_cuda_stream(device_idx)
        return dict(device=device, device_type='cuda', stream=stream)
    
    kernel_meta = _kernel_meta()
    
    normalize_kernel[grid](
        input_flat,
        output,
        input_flat.stride(0),
        n_cols,
        p_norm,
        eps_norm,
        n_cols,
        BLOCK_N,
        num_warps=4,
        num_stages=2,
        **kernel_meta,
    )
    
    return output.view(original_shape)

def _pairwise_distance(x1, x2, p_norm, eps_distance):
    original_shape_x1 = x1.shape
    original_shape_x2 = x2.shape
    x1_flat = x1.view(-1, original_shape_x1[-1])
    x2_flat = x2.view(-1, original_shape_x2[-1])
    N, D = x1_flat.shape
    M, D2 = x2_flat.shape
    assert D == D2, "Feature dimension mismatch"
    
    output = torch.empty((N, M), device=x1.device, dtype=x1.dtype)
    BLOCK_N = triton.next_power_of_two(D)
    grid = (N, M)
    
    def _kernel_meta():
        device = x1.device
        device_idx = device.index
        stream = get_cuda_stream(device_idx)
        return dict(device=device, device_type='cuda', stream=stream)
    
    kernel_meta = _kernel_meta()
    
    pairwise_distance_kernel[grid](
        x1_flat,
        x1_flat.stride(0),
        x2_flat,
        x2_flat.stride(0),
        output,
        output.stride(0),
        D,
        p_norm,
        eps_distance,
        D,
        BLOCK_N,
        num_warps=4,
        num_stages=2,
        **kernel_meta,
    )
    
    leading_dims = original_shape_x1[:-1]
    output_shape = leading_dims + (original_shape_x2[-2],)
    return output.view(output_shape)
