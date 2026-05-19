import torch
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def _pairwise_distance_kernel(
    diff_ptr,
    output_ptr,
    p,
    eps,
    flattened_size,
    B,
    stride_diff_row,
    stride_diff_col,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    if row_idx >= B:
        return
    
    row_start = row_idx * stride_diff_row
    offsets = row_start + tl.arange(0, BLOCK_SIZE) * stride_diff_col
    
    acc = tl.zeros((1,), dtype=tl.float32)
    
    for col_idx in range(0, flattened_size, BLOCK_SIZE):
        cols = col_idx + tl.arange(0, BLOCK_SIZE)
        mask = cols < flattened_size
        
        diff = tl.load(diff_ptr + row_start + cols, mask=mask, other=0.0)
        abs_diff_p = tl.abs(diff) ** p
        acc += tl.sum(abs_diff_p, axis=0)
    
    sum_abs_diff_p = acc + eps
    distance = sum_abs_diff_p ** (1.0 / p)
    
    tl.store(output_ptr + row_idx, distance)

def fused_pairwise_distance_adaptive_avg_pool2d(
    x1: torch.Tensor,
    x2: torch.Tensor,
    output_size: int or tuple,
    p: float = 2.0,
    eps: float = 1e-6,
    keepdim: bool = False
) -> torch.Tensor:
    assert x1.dim() == 4 and x2.dim() == 4, "Input tensors must be 4D"
    assert x1.size(0) == x2.size(0), "Batch sizes of x1 and x2 must match"
    
    x1_pool = F.adaptive_avg_pool2d(x1, output_size)
    x2_pool = F.adaptive_avg_pool2d(x2, output_size)
    
    diff = x1_pool - x2_pool
    B = diff.size(0)
    flattened_diff = diff.contiguous().view(B, -1)
    flattened_size = flattened_diff.size(1)
    
    if p <= 0:
        raise ValueError("p must be greater than 0")
    
    output = torch.empty(B, device=diff.device, dtype=diff.dtype)
    
    BLOCK_SIZE = triton.next_power_of_2(flattened_size)
    BLOCK_SIZE = min(BLOCK_SIZE, 4096)
    
    grid = lambda meta: (B,)
    _pairwise_distance_kernel[grid](
        flattened_diff,
        output,
        p,
        eps,
        flattened_size,
        B,
        flattened_diff.stride(0),
        flattened_diff.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    if keepdim:
        output = output.view(B, *((1,) * (x1_pool.dim() - 1)))
    
    return output
