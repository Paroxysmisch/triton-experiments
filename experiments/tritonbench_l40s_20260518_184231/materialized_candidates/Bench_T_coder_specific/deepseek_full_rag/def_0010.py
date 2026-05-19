s, 256 of each type.
            # Therefore, divide NUM_REGS by 2 to get the number of VGPRs.
            n_regs //= 2
        # occupancy is the number of CTAs that we can run on the GPU with the given number of registers and shared memory
        occupancy = occupancy_calculator(NUM_SM, NUM_REGS, size_smem, WARP_SIZE, target)
        # We can fit at most `occupancy` many kernels into a single kernel package
        num_programs = occupancy // n_regs

    # Launch kernel
    grid = lambda meta: (triton.cdiv(n_rows, meta["BLOCK_SIZE"]), )
    with torch.cuda.device(x.device.index):
        kernel[(num_programs, )](y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE,
                                num_stages=num_stages, grid=grid)
    return y

kernels[BLOCK_SIZE] = (kernel, num_programs)
|<system|>import torch
import triton
import triton.language as tl
from triton.runtime import driver

@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr,
                   num_stages: tl.constexpr):
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
target = triton.runtime.driver.active.get_current_target()
kernels = {}

def softmax(x):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 8
    num_stages = 4 if SIZE_SMEM > 200000 else 2
    y = torch.empty_like(x)
    kernel, num_programs = kernels.get(BLOCK_SIZE, (None, 0))
    if kernel is None:
        kernel = softmax_kernel.warmup(y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE,
                                       num_stages=num_stages, num_warps=num_warps, grid=(1, ))
        kernel._init_handles()
        n_regs = kernel.n_regs
        size_smem = kernel.metadata.shared
        if is_hip():
            n_regs //= 2
        occupancy = occupancy_calculator(NUM_SM, NUM_REGS, size_smem, WARP_SIZE, target)
        num_programs = occupancy // n_regs
    grid = lambda meta: (triton.cdiv(n_rows, meta["BLOCK_SIZE"]), )
    with torch.cuda.device(x.device.index):
        kernel[(num_programs, )](y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE,
                                num_stages=num_stages, grid=grid)
    return y

kernels[BLOCK_SIZE] = (kernel, num_programs)
