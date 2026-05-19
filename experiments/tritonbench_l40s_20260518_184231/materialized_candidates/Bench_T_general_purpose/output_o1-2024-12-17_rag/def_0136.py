import torch
import triton
import triton.language as tl
from triton.runtime import driver

@triton.jit
def softmax_kernel(
    output_ptr, 
    input_ptr, 
    input_row_stride, 
    output_row_stride, 
    n_rows, 
    n_cols, 
    BLOCK_SIZE: tl.constexpr, 
    num_stages: tl.constexpr
):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        row_vals = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float('inf'))
        max_val = tl.max(row_vals, axis=0)
        row_minus_max = row_vals - max_val
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        tl.store(output_row_start_ptr + col_offsets, softmax_output, mask=mask)

device = torch.cuda.current_device()
properties = driver.active.utils.get_device_properties(device)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
kernels_cache = {}

def _softmax_2d(x: torch.Tensor) -> torch.Tensor:
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 8
    num_stages = 4 if SIZE_SMEM > 200000 else 2

    y = torch.empty_like(x)

    cached = kernels_cache.get(BLOCK_SIZE, None)
    if cached is not None:
        kernel, num_programs = cached
    else:
        kernel = softmax_kernel.warmup(
            y, x, x.stride(0), y.stride(0), n_rows, n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
            num_stages=num_stages,
            num_warps=num_warps,
            grid=(1,)
        )
        kernel._init_handles()
        n_regs = kernel.n_regs
        size_smem = kernel.metadata.shared

        def is_hip():
            return hasattr(torch.cuda, 'is_available') and torch.version.hip is not None

        def is_cdna():
            arch = torch.cuda.get_device_name().lower()
            return 'gfx9' in arch or 'gfx10' in arch or 'gfx11' in arch

        if is_hip():
            if is_cdna():
                NUM_GPRS = NUM_REGS * 2
            else:
                NUM_GPRS = NUM_REGS
            MAX_NUM_THREADS = properties["max_threads_per_sm"]
            max_num_waves = MAX_NUM_THREADS // WARP_SIZE
            occupancy = min(NUM_GPRS // WARP_SIZE // n_regs, max_num_waves) // num_warps
        else:
            occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)

        occupancy = min(occupancy, SIZE_SMEM // size_smem)
        num_programs = NUM_SM * occupancy
        kernels_cache[BLOCK_SIZE] = (kernel, num_programs)

    num_programs = min(num_programs, n_rows)
    kernel[(num_programs, 1, 1)](
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_rows,
        n_cols
    )
    return y

def softmax(input, dim, dtype=None):
    if dtype is not None:
        input = input.to(dtype)

    original_shape = input.shape
    dim = dim if dim >= 0 else dim + input.dim()
    # Move the dimension along which softmax is computed to the end
    perm = list(range(input.dim()))
    perm[dim], perm[-1] = perm[-1], perm[dim]
    transposed = input.permute(perm)
    # Flatten all but the last dimension
    n_cols = transposed.shape[-1]
    transposed_2d = transposed.reshape(-1, n_cols)
    out_2d = _softmax_2d(transposed_2d)
    # Reshape and permute back
    out_transposed = out_2d.reshape(list(transposed.shape))
    # Swap back the dimension
    perm[dim], perm[-1] = perm[-1], perm[dim]
    return out_transposed.permute(perm)
