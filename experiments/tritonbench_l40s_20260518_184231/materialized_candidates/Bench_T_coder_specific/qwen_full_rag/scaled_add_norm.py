import torch
import triton
import triton.language as tl

@triton.jit
def _scaled_add_norm(y, x, n_elements, alpha):
    pid = tl.program_id(axis=0)
    offset = pid * n_elements
    x = tl.load(x + offset + tl.arange(0, n_elements))
    y = tl.load(y + offset + tl.arange(0, n_elements))
    y += alpha * x
    norm = tl.math.sqrt(tl.sum(y * y))
    return y, norm

def scaled_add_norm(y, x, alpha):
    assert (
        y.shape == x.shape
    ), "Shape must be same for all input tensors. Got {} and {}".format(
        y.shape, x.shape
    )
    assert (
        y.is_contiguous() and x.is_contiguous()
    ), "Input tensors must be contiguous. Got {} and {}".format(
        y.is_contiguous(), x.is_contiguous()
    )
    n_elements = y.numel()
    block_size = triton.next_power_of_2(n_elements)
    num_warps = 8
    if block_size >= 32768:
        num_warps = 32
    elif block_size >= 16384:
        num_warps = 16

    y, norm = _scaled_add_norm[(1,)](
        y, x, n_elements.to(tl.int64), alpha, num_warps=num_warps
    )
    return norm
