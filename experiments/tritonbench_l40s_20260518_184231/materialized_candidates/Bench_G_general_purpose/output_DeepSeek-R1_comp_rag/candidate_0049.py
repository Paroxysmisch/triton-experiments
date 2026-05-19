import triton
import triton.language as tl
import torch

@triton.jit
def isfinite_func_kernel_rank_1(
    input_ptr,
    output_ptr,
    n_elements,
    input_stride,
    output_stride,
    TILE_SIZE: tl.constexpr,
    one_tile_per_cta: tl.constexpr,
):
    pid = tl.program_id(0)
    if one_tile_per_cta:
        start_idx = pid * TILE_SIZE
    else:
        grid_size = tl.num_programs(0)
        start_idx = pid * TILE_SIZE
        start_idx += tl.multiple_of(start_idx, grid_size * TILE_SIZE)
    
    offsets = start_idx + tl.arange(0, TILE_SIZE)
    mask = offsets < n_elements
    
    input_ptrs = input_ptr + offsets * input_stride
    elements = tl.load(input_ptrs, mask=mask, other=0)
    is_finite = tl.math.isfinite(elements)
    
    output_ptrs = output_ptr + offsets * output_stride
    tl.store(output_ptrs, is_finite, mask=mask)

def heuristics_for_tile_size(n_elements):
    if n_elements < 1024:
        return 128
    elif n_elements < 1048576:
        return 512
    else:
        return 1024

def heuristics_for_num_warps(n_elements):
    return 8 if n_elements >= 1048576 else 4

def isfinite_func_wrapper_rank_1(input_tensor: torch.Tensor) -> torch.Tensor:
    assert input_tensor.dim() == 1, "Input tensor must be of rank 1"
    n_elements = input_tensor.numel()
    output_tensor = torch.empty_like(input_tensor, dtype=torch.bool)
    
    tile_size = heuristics_for_tile_size(n_elements)
    num_warps = heuristics_for_num_warps(n_elements)
    
    grid_size = (triton.cdiv(n_elements, tile_size),)
    
    max_grid_size = 65535  # Typical maximum grid size per dimension
    one_tile_per_cta = grid_size[0] <= max_grid_size
    
    isfinite_func_kernel_rank_1[grid_size](
        input_tensor,
        output_tensor,
        n_elements,
        input_tensor.stride(0),
        output_tensor.stride(0),
        TILE_SIZE=tile_size,
        one_tile_per_cta=one_tile_per_cta,
        num_warps=num_warps,
    )
    return output_tensor
