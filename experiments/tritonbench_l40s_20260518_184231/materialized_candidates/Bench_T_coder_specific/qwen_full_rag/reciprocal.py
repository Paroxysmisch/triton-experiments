import torch
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_heuristics import reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[8192, 512],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={
        "signature": {
            0: (tl.pointer_type(tl.float32), 1),
            1: (tl.float32, 1),
        },
        "device": 0,
        "device_type": "cuda",
        "constants": {},
        "mutated_arg_names": ["out"],
        "autotune_hints": set(),
        "kernel_name": "triton_helpers.promoted_reciprocal_kernel",
        "configs": [
            triton.Config(
                instance_descriptor=instance_descriptor(),
                num_warps=8,
            ),
        ],
    },
)
@triton.jit
def promoted_reciprocal_kernel(out, N: tl.constexpr, **meta):
    # Compute thread ID
    pid = tl.program_id(0)
    # Calculate offsets for the current chunk of data
    offsets = pid * N
    # Load input data and compute the reciprocal
    x = tl.load(out + offsets, mask=N)
    y = 1.0 / x.to(tl.float32)
    # Store the result
    tl.store(out + offsets, y, mask=N)

def reciprocal(inp, *, out=None):
    # Determine if inplace operation is allowed
    allow_inplace = out is not None and inp.storage().data_ptr() == out.storage().data_ptr()
    if not allow_inplace:
        inp = inp.contiguous()
    # Call the Triton kernel for reciprocal calculation
    promoted_reciprocal_kernel[(1,)](inp, inp.numel())
    return inp
