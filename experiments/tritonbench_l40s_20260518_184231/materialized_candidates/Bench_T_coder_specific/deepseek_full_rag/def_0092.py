import torch
import triton
import triton.language as tl
from torch.functional import Tensor

@triton.jit
def tensordot_rsqrt_kernel(
    a_ptr,
    b_ptr,
    output_ptr,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    a_tile_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(M, K),
        strides=(1, 1),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    b_tile_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(K, N),
        strides=(1, 1),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )
    output_tile_ptr = tl.make_block_ptr(
        base=output_ptr,
        shape=(M, N),
        strides=(1, 1),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    a = tl.load(a_tile_ptr)
    b = tl.load(b_tile_ptr)
    c = tl.dot(a, b)
    result = tl.rsqrt(c)
    tl.store(output_tile_ptr, result)

def tensordot_rsqrt(a: Tensor, b: Tensor, dims) -> Tensor:
    M = a.shape[0]
    N = b.shape[1]
    K = a.shape[-1]
    output = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )
    with torch.cuda.device(a.device):
        tensordot_rsqrt_kernel[grid](a, b, output, M, N, K)
    return output
