import torch
import triton
import triton.language as tl
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@triton.jit
def linalg_cholesky_kernel(
    a_ptr,
    a_strides,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    a_ptr += pid * a_strides[0]
    a_stride_1 = a_strides[1]
    a_stride_2 = a_strides[2]

    if BLOCK_SIZE == 128:
        a1 = tl.load(a_ptr + 0 * a_stride_1 + 0 * a_stride_2)
        a2 = tl.load(a_ptr + 0 * a_stride_1 + 1 * a_stride_2)
        a3 = tl.load(a_ptr + 0 * a_stride_1 + 2 * a_stride_2)
        a4 = tl.load(a_ptr + 0 * a_stride_1 + 3 * a_stride_2)
        a5 = tl.load(a_ptr + 0 * a_stride_1 + 4 * a_stride_2)
        a6 = tl.load(a_ptr + 0 * a_stride_1 + 5 * a_stride_2)
        a7 = tl.load(a_ptr + 0 * a_stride_1 + 6 * a_stride_2)
        a8 = tl.load(a_ptr + 0 * a_stride_1 + 7 * a_stride_2)
        tl.store(a_ptr + 0 * a_stride_1 + 0 * a_stride_2, a1)
        tl.store(a_ptr + 1 * a_stride_1 + 1 * a_stride_2, a2)
        tl.store(a_ptr + 2 * a_stride_1 + 2 * a_stride_2, a3)
        tl.store(a_ptr + 3 * a_stride_1 + 3 * a_stride_2, a4)
        tl.store(a_ptr + 4 * a_stride_1 + 4 * a_stride_2, a5)
        tl.store(a_ptr + 5 * a_stride_1 + 5 * a_stride_2, a6)
        tl.store(a_ptr + 6 * a_stride_1 + 6 * a_stride_2, a7)
        tl.store(a_ptr + 7 * a_stride_1 + 7 * a_stride_2, a8)
    elif BLOCK_SIZE == 64:
        a1 = tl.load(a_ptr + 0 * a_stride_1 + 0 * a_stride_2)
        a2 = tl.load(a_ptr + 0 * a_stride_1 + 1 * a_stride_2)
        a3 = tl.load(a_ptr + 0 * a_stride_1 + 2 * a_stride_2)
        a4 = tl.load(a_ptr + 0 * a_stride_1 + 3 * a_stride_2)
        a5 = tl.load(a_ptr + 0 * a_stride_1 + 4 * a_stride_2)
        a6 = tl.load(a_ptr + 0 * a_stride_1 + 5 * a_stride_2)
        a7 = tl.load(a_ptr + 0 * a_stride_1 + 6 * a_stride_2)
        tl.store(a_ptr + 0 * a_stride_1 + 0 * a_stride_2, a1)
        tl.store(a_ptr + 1 * a_stride_1 + 1 * a_stride_2, a2)
        tl.store(a_ptr + 2 * a_stride_1 + 2 * a_stride_2, a3)
        tl.store(a_ptr + 3 * a_stride_1 + 3 * a_stride_2, a4)
        tl.store(a_ptr + 4 * a_stride_1 + 4 * a_stride_2, a5)
        tl.store(a_ptr + 5 * a_stride_1 + 5 * a_stride_2, a6)
        tl.store(a_ptr + 6 * a_stride_1 + 6 * a_stride_2, a7)
    elif BLOCK_SIZE == 32:
        a1 = tl.load(a_ptr + 0 * a_stride_1 + 0 * a_stride_2)
        a2 = tl.load(a_ptr + 1 * a_stride_1 + 1 * a_stride_2)
        a3 = tl.load(a_ptr + 2 * a_stride_1 + 2 * a_stride_2)
        a4 = tl.load(a_ptr + 3 * a_stride_1 + 3 * a_stride_2)
        tl.store(a_ptr + 0 * a_stride_1 + 0 * a_stride_2, a1)
        tl.store(a_ptr + 1 * a_stride_1 + 1 * a_stride_2, a2)
        tl.store(a_ptr + 2 * a_stride_1 + 2 * a_stride_2, a3)
        tl.store(a_ptr + 3 * a_stride_1 + 3 * a_stride_2, a4)
    elif BLOCK_SIZE == 16:
        a1 = tl.load(a_ptr + 0 * a_stride_1 + 0 * a_stride_2)
        a2 = tl.load(a_ptr + 1 * a_stride_1 + 1 * a_stride_2)
        a3 = tl.load(a_ptr + 2 * a_stride_1 + 2 * a_stride_2)
        tl.store(a_ptr + 0 * a_stride_1 + 0 * a_stride_2, a1)
        tl.store(a_ptr + 1 * a_stride_1 + 1 * a_stride_2, a2)
        tl.store(a_ptr + 2 * a_stride_1 + 2 * a_stride_2, a3)
    elif BLOCK_SIZE == 8:
        a1 = tl.load(a_ptr + 0 * a_stride_1 + 0 * a_stride_2)
        a2 = tl.load(a_ptr + 1 * a_stride_1 + 1 * a_stride_2)
        tl.store(a_ptr + 0 * a_stride_1 + 0 * a_stride_2, a1)
        tl.store(a_ptr + 1 * a_stride_1 + 1 * a_stride_2, a2)
    elif BLOCK_SIZE == 4:
        a1 = tl.load(a_ptr + 0 * a_stride_1 + 0 * a_stride_2)
        tl.store(a_ptr + 0 * a_stride_1 + 0 * a_stride_2, a1)

def linalg_cholesky(a, *, upper=False, out=None):
    a = a.contiguous()
    assert a.ndim >= 2
    n = a.shape[-1]
    assert n <= 128, "GEMS CHOL only supports n <= 128"
    batch_shape = a.shape[:-2]
    if out is None:
        out = torch.empty_like(a)
    else:
        out = out.contiguous()
    assert out.shape == a.shape
    full_batch_stride = out.stride()[:-2]
    for bpid in range(len(full_batch_stride)):
        a_stride_0 = out.stride(-2)
        a_stride_1 = out.stride(-1)
        a1 = triton_helpers.promote_to_tensor(a_stride_0)
        a2 = triton_helpers.promote_to_tensor(a_stride_1)
        BLOCK_SIZE = triton.cdiv(n, a1)
        if a2 != 0:
            BLOCK_SIZE = min(BLOCK_SIZE, triton.cdiv(n, a2))
        if BLOCK_SIZE >= 1:
            linalg_cholesky_kernel[(n,)](
                out,
                (a_stride_0, a_stride_1),
                n,
                BLOCK_SIZE=BLOCK_SIZE,
            )
    return out
