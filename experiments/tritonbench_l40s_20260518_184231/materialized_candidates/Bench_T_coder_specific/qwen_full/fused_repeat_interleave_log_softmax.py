import torch
import triton
import triton.language as tl
from torch import Tensor
from flag_gems.utils.shape_utils import volume

def cfggen():
    block_m = [1, 2, 4, 8]
    configs = [
        triton.Config({"BLOCK_M": m}, num_warps=4) for m in block_m
    ]
    return configs

@triton.jit
def logsumexp(x, y):
    m = tl.maximum(x, y)
    s = tl.where(x == m, y - m, tl.log(tl.exp(x - m) + tl.exp(y - m)))
    return m + s

@triton.jit
def fused_repeat_interleave_log_softmax_kernel(
    input_ptr,
    output_ptr,
    repeats_ptr,
    input_n_elements,
    output_n_elements,
    repeats_size,
    input_stride,
    output_stride,
    repeats_stride,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    if pid * BLOCK_M >= repeats_size:
        return
    offset = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = offset < repeats_size

    input_block_ptr = tl.make_block_ptr(
        base=input_ptr,
        shape=(input_n_elements, ),
        strides=(input_stride, ),
        offsets=(0, ),
        block_shape=(1, ),
        order=(0, ),
    )
    output_block_ptr = tl.make_block_ptr(
        base=output_ptr,
        shape=(output_n_elements, ),
        strides=(output_stride, ),
        offsets=(0, ),
        block_shape=(1, ),
        order=(0, ),
    )
    repeats = tl.load(repeats_ptr + offset, mask=mask)

    for i in range(repeats):
        x = tl.load(input_block_ptr, boundary_check=(0, ))
        x = x.to(tl.float32)
        x = tl.where(repeats_ptr + offset == 0, x, -float("inf"))
        x = logsumexp(x, tl.load(output_block_ptr, boundary_check=(0, )))
        tl.store(output_block_ptr, x.to(output_ptr.dtype.element_ty),
                 boundary_check=(0, ))
        input_block_ptr = tl.advance(input_block_ptr, (1, ))
        output_block_ptr = tl.advance(output_block_ptr, (1, ))

def fused_repeat_interleave_log_softmax(
    input: Tensor,
    repeats: Tensor,
    dim: int = None,
    output_size: int = None,
    dtype: None = None,
    out: None = None
) -> Tensor:
    if dtype is None:
        dtype = input.dtype
    if out is None:
        out = torch.empty_like(input, dtype=dtype)
    else:
        assert out.dtype == dtype

    if dim is None or input.ndim == 1:
        input = input.flatten()
        dim = 0
    else:
        assert 0 <= dim < input.ndim, "Invalid dim"
    
    assert repeats.ndim == 1, "Repeats should be a 1D vector"
    assert input.shape[dim] == repeats.shape[0], "Input and repeats shape should align"

    input_stride = list(input.stride())
    repeats_size = repeats.numel()
    output_stride = list(out.stride())
    input_n_elements = volume(input.shape) // input.shape[dim]
    output_n_elements = volume(out.shape) // out.shape[dim]

    grid = lambda meta: (triton.cdiv(repeats_size, meta["BLOCK_M"]), )

    fused_repeat_interleave_log_softmax_kernel[grid](
        input,
        out,
        repeats,
        input_n_elements,
        output_n_elements,
        repeats_size,
        input_stride[dim],
        output_stride[dim],
        repeats.stride(0),
    )
    return out
