import torch
import triton
import triton.language as tl

@triton.jit
def softmax_log_kernel(input, output, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input_ptrs = input + offsets
    out_ptrs = output + offsets
    x = tl.load(input_ptrs, mask=mask)
    c = tl.max(x, axis=0)
    x = x - c
    exp_x = tl.exp(x)
    sum_x = tl.sum(exp_x, axis=0)
    y = exp_x / sum_x
    tl.store(out_ptrs, y, mask=mask)

def softmax_log(input, dim=-1, dtype=None) -> torch.Tensor:
    if dtype is None:
        dtype = input.dtype
    if dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError(f"Unsupported dtype: {dtype}")
    input = input.contiguous()
    if dim < -input.ndim or dim >= input.ndim:
        raise IndexError(
            "Dimension out of range (expected to be in range of [{}, {}], but got {})".format(
                -input.ndim, input.ndim - 1, dim
            )
        )
    dim = dim % input.ndim
    shape = input.shape
    n_elements = input.numel()
    sorted_dim = dim if dim >= 0 else input.ndim + dim
    sorted_shape = sorted_shape = list(input.shape)
    sorted_shape[sorted_dim] = 1
    n_elements_in_sorted_dim = sorted_shape[sorted_dim]
    output = torch.empty_like(input, dtype=dtype)
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    softmax_log_kernel[grid](input, output, n_elements, BLOCK_SIZE=n_elements_in_sorted_dim)
    return output
