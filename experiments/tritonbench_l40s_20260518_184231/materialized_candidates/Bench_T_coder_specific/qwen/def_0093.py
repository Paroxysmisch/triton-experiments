import triton
from triton.language import *
import torch

@triton.jit
def softmax_log_kernel(output_ptr, input_ptr, batch_size, num_elements_per_batch, dim):
    pid = tl.program_id(axis=0)
    tid = tl.program_id(axis=1)
    bid = tl.program_id(axis=2)

    if dim == 1:
        input_idx = bid * num_elements_per_batch + tid
        output_idx = bid * num_elements_per_batch + tid
    elif dim == 0:
        input_idx = bid + tid * batch_size
        output_idx = bid + tid * batch_size

    input_val = tl.load(input_ptr[input_idx])
    log_val = tl.math.log(input_val)
    max_val = tl.max(log_val, axis=0)
    log_val -= max_val
    exp_val = tl.math.exp(log_val)
    sum_val = tl.sum(exp_val, axis=0)
    softmax_val = exp_val / sum_val

    tl.store(output_ptr[output_idx], softmax_val)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_X': 256}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_SIZE_X': 128}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_SIZE_X': 64}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_SIZE_X': 32}, num_stages=1, num_warps=4),
    ],
    key=['batch_size', 'num_elements_per_batch']
)
def softmax_log(input, dim=-1, dtype=None):
    batch_size = input.shape[0]
    num_elements_per_batch = input.shape[1]

    if dtype is not None:
        input = input.to(dtype)

    output = torch.empty_like(input)

    grid = (batch_size, 1, 1)
    block = (num_elements_per_batch, 1, 1)

    softmax_log_kernel[grid](output, input, batch_size, num_elements_per_batch, dim)

    return output
