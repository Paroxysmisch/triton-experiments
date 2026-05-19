import triton
import triton.language as tl
import torch

@triton.jit
def mean_reduce_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr,
                       num_stages: tl.constexpr):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        row_sum = tl.dot(tl.load(input_ptrs, mask=mask, other=0), tl.ones((BLOCK_SIZE,), dtype=tl.float32), mask=mask)
        row_mean = row_sum / n_cols
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, row_mean, mask=mask)

def mean_reduce(input_tensor, dim, keepdim=False, dtype=None, out=None):
    n_rows, n_cols = input_tensor.shape

    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 8
    num_stages = 4 if triton.runtime.driver.active.get_device_properties(torch.cuda.current_device())['max_shared_mem'] > 200000 else 2

    y = torch.empty((n_rows, 1 if keepdim else 1), device=input_tensor.device, dtype=dtype if dtype else input_tensor.dtype)

    kernel, _ = triton.jit(mean_reduce_kernel).compile(
        BLOCK_SIZE=BLOCK_SIZE, 
        num_stages=num_stages, 
        num_warps=num_warps
    )

    kernel[(n_rows, 1, 1)](
        y,
        input_tensor,
        input_tensor.stride(0),
        y.stride(0),
        n_rows,
        n_cols,
    )

    return y.squeeze() if not keepdim else y
