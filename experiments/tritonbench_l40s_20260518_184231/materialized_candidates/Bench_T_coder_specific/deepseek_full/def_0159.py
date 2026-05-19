import torch
import triton
import triton.language as tl
from torch._inductor.triton_heuristics import inline_external_call
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers


@triton.jit
def cholesky_kernel(A, stride_za,
                    BLOCK_SIZE: tl.constexpr,
                    upper: tl.constexpr,
                    IS_SCHEDULED_ON_DEVICE: tl.constexpr):
    # Triton kernel for Cholesky decomposition.
    pass


def _cholesky(A, *, upper=False, out=None):
    # Wrapper function for Cholesky decomposition.
    pass


# This instance descriptor is used to key this function and should be unique.
_cholesky_instance_descriptor = instance_descriptor(
    _cholesky,
    attr_names=["upper"],
    device_tags=["cuda"],
    call_state=None,
)


def cholesky_wrapper(A, *, upper=False, out=None):
    # Wrapper function for Triton kernel.
    pass


def init_cholesky_kernels():
    # Initialization function to add Triton kernels.
    triton_helpers.add_triton_kernel(
        _cholesky_instance_descriptor,
        cholesky_wrapper,
        [
            inline_external_call(torch.linalg.cholesky, {"dtype_policy": "promote"}),
        ],
    )
