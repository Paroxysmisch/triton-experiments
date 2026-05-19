import torch
import triton
import triton.language as tl
from torch import Tensor

@triton.jit
def softmax_kernel(input, input_exp, output, sum_exp,
                   n_elements, n_dim,
                   dim,
                   BLOCK_SIZE_N: tl.constexpr,
                   BLOCK_SIZE_M: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offset_m = pid_m * BLOCK_SIZE_M
    offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    n_mask = offsets_n < n_dim
    mask = n_mask

    if dim == 0:
        offsets = offsets_n * n_elements + offset_m
        mask = n_mask and mask

    elif dim == 1:
        offsets = offset_m * n_dim + offsets_n
        mask = n_mask and mask

    values = tl.load(input + offsets, mask=mask, other=0).to(tl.float32)
    tl.store(input_exp + offsets, values, mask=mask)

    exp_values = tl.exp(values)
    tl.store(input_exp + offsets, exp_values, mask=mask)

    if dim == 0:
        sum_ = tl.sum(exp_values, axis=0)
        tl.store(sum_exp + offset_m, sum_, mask=True)

    elif dim == 1:
        sum_ = tl.sum(exp_values, axis=1)
        tl.store(sum_exp + offsets_n, sum_, mask=n_mask)

    if dim == 0:
        sum_ = tl.load(sum_exp + offset_m)
        normalized_output = exp_values / sum_

    elif dim == 1:
        sum_ = tl.load(sum_exp + offsets_n, mask=n_mask)
        normalized_output = exp_values / sum_

    tl.store(output + offsets, normalized_output, mask=mask)

def softmax(input, dim, dtype=None):
    if dtype is None:
        dtype = input.dtype

    n_dim = input.shape[dim]
    n_elements = n_dim

    if dim == 0:
        n_elements = 1

    shape = list(input.shape)
    shape[dim] = 1

    input_exp = torch.empty(shape, dtype=dtype, device=input.device)
    sum_exp = torch.empty(shape, dtype=dtype, device=input.device)

    output = torch.empty(input.shape, dtype=dtype, device=input.device)

    if dim == 0:
        instance_dim = 1
        instance_size = n_elements
        n_dim = input.size(0)

    elif dim == 1:
        instance_dim = input.size(0)
        instance_size = n_elements
        n_dim = input.size(1)

    grid = lambda meta: (
        triton.cdiv(n_dim, meta["BLOCK_SIZE_M"]),
        triton.cdiv(n_dim, meta["BLOCK_SIZE_N"])
    )

    softmax_kernel[grid](input, input_exp, output, sum_exp,
                         n_elements, n_dim,
                         dim)

    return output
