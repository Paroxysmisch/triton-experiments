import torch
import triton
import triton.language as tl
from torch._inductor.triton_heuristics import reduction
from torch._inductor import triton_helpers
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_heuristics

@reduction(
    size_hints=[8192, 512],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={
        'signature': {0: ('pointer', 'float32')},
        'device': 0,
        'device_type': 'cuda',
        'constants': {},
        'mutated_arg_names': [],
        'autotune_hints': set(),
        'kernel_name': 'max',
        'configs': [instance_descriptor()]
    }
)
@triton.jit
def triton_max(output_ptr, 
               numel, 
               XBLOCK: tl.constexpr, 
               RBLOCK: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * RBLOCK
    offset = block_start + tl.arange(0, RBLOCK)
    mask = offset < numel
    ptrs = output_ptr + offset
    x = tl.load(ptrs, mask=mask, other=-float('inf'))
    current_max = tl.max(x, axis=0)
    tl.store(output_ptr + pid, current_max)

def wrapper(input, dim, keepdim=False, *, out=None):
    if input.is_floating():
        dtype = input.dtype
    else:
        dtype = torch.float32
    if out is None:
        max_values = torch.empty(input.shape[demote_to_scalar(input.dim() - 1)],
                                 dtype=dtype,
                                 device=input.device)
    else:
        max_values = out[0]
    if input.dim() > 1:
        input_shape = input.shape
        input_stride = input.stride()
        dim = dim % input.dim()
        if dim < 0:
            dim = dim + len(input_shape)
        dim_size = input_shape[dim]
        dim_stride = input_stride[dim]
        if dim_stride != 1:
            input = input.contiguous()
            input_stride = input.stride()
            dim_stride = input_stride[dim]
        shape = [-1, dim_size]
        strides = [shape[1] * dim_stride, 1]
        view = StridedBuffer(input, shape, strides)
        triton_max(max_values, numels[0])
        if not keepdim:
            max_values = max_values.squeeze(dim)
        return (max_values, )
    else:
        triton_max(max_values, numels[0])
        return (max_values, )
