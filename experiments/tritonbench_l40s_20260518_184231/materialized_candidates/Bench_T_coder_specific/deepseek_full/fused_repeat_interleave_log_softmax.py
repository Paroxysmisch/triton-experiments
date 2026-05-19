import torch
import triton
import triton.language as tl
from torch import Tensor
from torch._inductor.triton_heuristics import (
    pointwise,
    reduction,
)
from torch._inductor.utils import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.select_algorithm import register_algorithm
from torch._inductor.select_algorithm import select_cudablas
from torch._inductor.select_algorithm import select_algorithm
from torch._inductor.select_algorithm.triton_helpers import (
    instance_descriptor,
    grid_descriptor,
)

@extern_kernels(
    {
        "fused_repeat_interleave_log_softmax": (
            "torchinductor_kernel_fused_repeat_interleave_log_softmax"
        )
    }
)
@register_algorithm(
    signature={0: "*i8", 1: "*i8", 2: "*i8"},
    constants={},
    attributes=[],
    extern_module=__name__,
    extern_name="fused_repeat_interleave_log_softmax",
)
@select_algorithm(
    "torchinductor",
    ["fused", "fused_repeat_interleave_log_softmax", "log_softmax"],
)
@select_cudablas(func_name="fused_repeat_interleave_log_softmax")
@pointwise(
    size_hints=[4096, 256],
    filename=__file__,
    triton_meta={
        "signature": {0: "*i8", 1: "*i8", 2: "*i8"},
        "constants": {},
        "mutated_arg_names": [],
        "configs": [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())],
    },
)
@reduction(
    size_hints=[4096, 256],
    filename=__file__,
    triton_meta={
        "signature": {0: "*i8", 1: "*i8", 2: "*i8"},
        "constants": {},
        "mutated_arg_names": [],
        "configs": [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())],
    },
)
def fused_repeat_interleave_log_softmax(input, repeats, dim=None, *, output_size=None, dtype=None, out=None):
    # The actual Triton kernel is defined in a separate file.
    pass

def call_fused_repeat_interleave_log_softmax(input, repeats, dim=None, *, output_size=None, dtype=None, out=None):
    # The actual Triton kernel is invoked here.
    pass
