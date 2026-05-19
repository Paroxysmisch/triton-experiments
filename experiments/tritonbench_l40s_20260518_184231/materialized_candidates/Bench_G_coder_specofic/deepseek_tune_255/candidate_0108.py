import triton
import triton.language as tl
import torch

@triton.jit
def kernel(M, Out, stride_m, stride_out, SIZE_M: tl.constexpr, D_HEAD: tl.constexpr):
    pid = tl.program_id(0)
    matrix_idx = pid
    out_idx = pid

    matrix_stridex = stride_m * SIZE_M
    matrix_stridey = stride_m

    out_stridex = stride_out * D_HEAD
    out_stridey = stride_out

    for i in range(SIZE_M):
        for j in range(D_HEAD):
            matrix_ptr = M + matrix_idx + i * matrix_stridex + j * matrix_stridey
            out_ptr = Out + out_idx + i * out_stridex + j * out_stridey
            tl.store(out_ptr, tl.load(matrix_ptr))

def wrapper(matrix, out, SIZE_M, D_HEAD):
    matrix = torch.randn((SIZE_M, D_HEAD), device="cuda", dtype=torch.float16)
    out = torch.zeros((SIZE_M, D_HEAD), device="cuda", dtype=torch.float16)
    grid = (1,)
    kernel[grid](matrix, out, SIZE_M, D_HEAD)
    return out

SIZE_M = 1024
D_HEAD = 128
ret = wrapper(None, None, SIZE_M, D_HEAD)
print(ret)
