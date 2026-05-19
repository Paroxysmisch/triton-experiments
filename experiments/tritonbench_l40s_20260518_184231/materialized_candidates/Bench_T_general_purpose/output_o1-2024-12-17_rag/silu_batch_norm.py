import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.runtime.jit import get_cuda_stream

@triton.jit
def _silu_batch_norm_kernel(
    input_ptr, 
    output_ptr, 
    mean_ptr, 
    var_ptr,
    weight_ptr, 
    bias_ptr,
    weight_flag, 
    bias_flag,
    total_size,
    channels,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_size

    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    c = offsets % channels
    m = tl.load(mean_ptr + c)
    v = tl.load(var_ptr + c)
    scale = 1.0 / tl.sqrt(v + eps)
    y = (x - m) * scale

    if weight_flag:
        w = tl.load(weight_ptr + c)
