import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid
from torch._inductor.utils import instance_descriptor
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.triton_helpers import (
    inline_block,
    num_warps,
    configuration_of,
)
from torch._inductor.triton_heuristics import grid
from torch._inductor.triton_kernels import triton_helpers

from torch._inductor.triton_heuristics import (
    max_power_of_two_less_than,
    num_threads_per_warp,
    block_shape_from_int,
    grid_size_for_threads,
    prefer_cc,
)

@triton.jit
def pow_func_scalar_tensor_kernel_rank_1(
    instance_descriptor,
    num_warps: tl.constexpr,
    num_threads: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    arange,
    pow_scalar,
    input_tensor_ptr, 
    input_tensor_batch_stride, 
    input_tensor_stride, 
):
    # kernel code


def pow_func_scalar_tensor_wrapper_rank_1(
    instance: instance_descriptor,
    inp,
    pow_scalar,
):
    if inp.is_floating_point() and inp.stride(-1) != 1:
        inp = inp.contiguous()
    output = inp.clone()
    if isinstance(inp, torch.StridedBuffer):
        inp_desc = inp
        inp = torch.empty_like(inp, dtype=torch.float32, device=inp.device)
    else:
        inp_desc = torch.arange(inp.numel(), device=inp.device)

    NUM_WARPS = 4
    if inp.dtype == torch.float16:
        NUM_WARPS = 8
    grid = (inp.numel() + TILE_SIZE - 1) // TILE_SIZE
    pow_func_scalar_tensor_kernel_rank_1[grid](
        instance,
        num_warps=NUM_WARPS,
        num_threads=TILE_SIZE,
        TILE_SIZE=TILE_SIZE,
        arange=inp_desc,
        pow_scalar=pow_scalar,
        input_tensor_ptr=inp,
        input_tensor_batch_stride=inp.stride(0),
        input_tensor_stride=inp.stride(1),
        num_warps=4,
    )
    return output

def extern_kernels(
    var_size_0,
    contiguous: bool,
):
    # extern_kernel function code

! UNUSED CODE intended to be documented or hidden by indentation pragma

def pow_func(inp, pow_scalar):
    # pow_func function code
