import torch
import triton
import triton.language as tl

def heuristics_for_tile_size(size):
    if size < 4096:
        return 128
    elif size < 131072:
        return 512
    return 1024

def heuristics_for_num_warps(tile_size):
    return 8 if tile_size >= 512 else 4

def isfinite_func_wrapper_rank_1(input_tensor, output_tensor):
    # Validate inputs
    assert input_tensor.is_cuda and output_tensor.is_cuda, "Tensors must reside on CUDA device"
    assert input_tensor.shape == output_tensor.shape, "Input and output shapes must match"
    assert input_tensor.is_contiguous(), "Input tensor must be contiguous"
    assert output_tensor.is_contiguous(), "Output tensor must be contiguous"
    
    size = input_tensor.numel()
    if size == 0:
        return

    # Determine kernel parameters
    tile_size = heuristics_for_tile_size(size)
    num_warps = heuristics_for_num_warps(tile_size)
    one_tile_per_cta = size <= (tile_size * 2048)
    
    # Calculate launch grid
    num_ctas = triton.cdiv(size, tile_size) if one_tile_per_cta else 1
    grid = (num_ctas, 1, 1)

    # Map torch dtype to Triton dtype
    dtype_map = {
        torch.float64: tl.float64,
        torch.float32: tl.float32,
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16
    }
    triton_dtype = dtype_map.get(input_tensor.dtype, tl.float32)

    # Launch kernel
    isfinite_func_kernel_rank_1[grid](
        input_tensor.data_ptr(),
        output_tensor.data_ptr(),
        size,
        dtype=triton_dtype,
        tile_size=tile_size,
        one_tile_per_cta=one_tile_per_cta,
        num_warps=num_warps
    )

@triton.jit
def isfinite_func_kernel_rank_1(
    in_ptr, out_ptr, size,
    dtype: tl.constexpr,
    tile_size: tl.constexpr,
    one_tile_per_cta: tl.constexpr,
):
    pid = tl.program_id(0)
    
    if one_tile_per_cta:
        # Monolithic block processing
        offset = pid * tile_size
        offsets = offset + tl.arange(0, tile_size)
        mask = offsets < size
        
        # Load and process elements
        x = tl.load(in_ptr + offsets, mask=mask)
        is_finite = _compute_isfinite(x, dtype)
        tl.store(out_ptr + offsets, is_finite, mask=mask)
    else:
        # Grid-stride loop processing
        grid_size = tl.num_programs(0)
        total_tiles = tl.cdiv(size, tile_size)
        
        for tile_idx in range(pid, total_tiles, grid_size):
            offset = tile_idx * tile_size
            offsets = offset + tl.arange(0, tile_size)
            mask = offsets < size
            
            # Load and process elements
            x = tl.load(in_ptr + offsets, mask=mask)
            is_finite = _compute_isfinite(x, dtype)
            tl.store(out_ptr + offsets, is_finite, mask=mask)

@triton.jit
def _compute_isfinite(x, dtype):
    if dtype == tl.float64:
        return tl.math.isfinite(x)
    else:
        # Cast non-fp64 types to float32 for finite check
        x_f32 = tl.math.semantic_cast(x, tl.float32)
        return tl.math.isfinite(x_f32)

input_tensor = torch.randn(1_000_000, device='cuda', dtype=torch.float32)
output_tensor = torch.empty_like(input_tensor, dtype=torch.bool)

isfinite_func_wrapper_rank_1(input_tensor, output_tensor)
