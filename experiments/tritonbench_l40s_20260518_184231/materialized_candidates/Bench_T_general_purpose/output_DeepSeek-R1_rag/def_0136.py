import torch
import triton
import triton.language as tl
from triton.runtime import driver

@triton.jit
def softmax_kernel(
    output_ptr, input_ptr, input_row_stride, output_row_stride,
    n_rows, n_cols, BLOCK_SIZE: tl.constexpr, num_stages: tl.constexpr
):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)

device = torch.cuda.current_device()
properties = driver.active.utils.get_device_properties(device)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
kernels = {}

def is_hip():
    return triton.runtime.driver.active.get_current_target()[0] == 'hip'

def is_cdna():
    return properties.get("gcn_arch_name", "").startswith("gfx90a")

def softmax(input, dim, dtype=None):
    if dtype is not None:
        input = input.to(dtype)
    input = input.contiguous()
    original_shape = input.shape
    dim = dim if dim >= 0 else dim + input.dim()
    assert 0 <= dim < input.dim(), "dim out of range"
    size_dim = input.size(dim)
    input_flat = input.view(-1, size_dim)
    n_rows, n_cols = input_flat.shape

    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 8
    num_stages = 4 if SIZE_SMEM > 200000 else 2
    y = torch.empty_like(input_flat)

    kernel_key = (BLOCK_SIZE, input_flat.dtype, y.dtype)
    kernel, num_programs = kernels.get(kernel_key, (None, 0))
    if kernel is None:
        kernel = softmax_kernel.warmup(
            y, input_flat, input_flat.stride(0), y.stride(0),
            n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE, num_stages=num_stages,
            num_warps=num_warps, grid=(1,)
        )
        kernel._init_handles()
        n_regs = kernel.n_regs
        size_smem = kernel.metadata.shared
        if is_hip():
            if is_cdna():
                NUM_GPRS = NUM_REGS * 2
            else:
                NUM_GPRS = NUM_REGS
            MAX_NUM_THREADS = properties["max_threads_per_multi_processor"]
            max_num_waves = MAX_NUM_THREADS // WARP_SIZE
            occupancy = min(NUM_GPRS // (n_regs * WARP_SIZE * num_warps), max_num_waves)
        else:
            occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
        occupancy = min(occupancy, SIZE_SMEM // size_smem)
        num_programs = NUM_SM * occupancy
        kernels[kernel_key] = (kernel, num_programs)
    num_programs = min(num_programs, n_rows)
    kernel[(num_programs, 1, 1)](
        y, input_flat, input_flat.stride(0), y.stride(0), n_rows, n_cols
    )
    return y.view(original_shape)
