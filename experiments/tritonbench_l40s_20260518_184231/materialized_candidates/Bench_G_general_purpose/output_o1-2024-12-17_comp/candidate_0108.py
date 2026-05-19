import torch
import triton
import triton.language as tl

@triton.jit
def kernel(
    M, Out,
    matrix_stridex, matrix_stridey,
    out_stridex, out_stridey,
    SIZE_M, D_HEAD,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = offsets % SIZE_M
    y = offsets // SIZE_M
    mask = offsets < (SIZE_M * D_HEAD)
    m_ptrs = M + x * matrix_stridex + y * matrix_stridey
    out_ptrs = Out + y * out_stridex + x * out_stridey
    val = tl.load(m_ptrs, mask=mask, other=0.0)
    tl.store(out_ptrs, val, mask=mask)

def wrapper(SIZE_M, D_HEAD):
    matrix = torch.randn((SIZE_M, D_HEAD), device='cuda', dtype=torch.float16)
    out = torch.zeros((D_HEAD, SIZE_M), device='cuda', dtype=torch.float16)
    BLOCK_SIZE = 1024
    grid = ((SIZE_M * D_HEAD + BLOCK_SIZE - 1) // BLOCK_SIZE, )
    kernel[grid](
        matrix, out,
        matrix.stride(0), matrix.stride(1),
        out.stride(0), out.stride(1),
        SIZE_M, D_HEAD,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out
