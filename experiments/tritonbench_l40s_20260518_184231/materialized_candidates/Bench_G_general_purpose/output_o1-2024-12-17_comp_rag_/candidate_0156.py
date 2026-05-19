import torch
import triton
import triton.language as tl
from triton.runtime.driver import CPUDriver

@triton.jit
def nested3(in_ptr, out_ptr, stride_m, stride_n):
    offs_am = tl.arange(0, 2)
    offs_an = tl.arange(0, 2)
    a_ptrs = in_ptr + (offs_am[:, None] * stride_m + offs_an[None, :] * stride_n)

    offs_cm = tl.arange(0, 2)
    offs_cn = tl.arange(0, 2)
    c_ptrs = out_ptr + stride_m * offs_cm[:, None] + stride_n * offs_cn[None, :]

    for i in range(0, 2):
        a1 = tl.load(a_ptrs)
        for j in range(0, 2):
            a_ptrs += 2 * stride_n
            a2 = tl.load(a_ptrs)
            for k in range(0, 2):
                a_ptrs += 2 * stride_n
                a3 = tl.load(a_ptrs)
                tl.store(c_ptrs, a1)
                c_ptrs += 2 * stride_n
                tl.store(c_ptrs, a2)
                c_ptrs += 2 * stride_n
                tl.store(c_ptrs, a3)
                c_ptrs += 2 * stride_n
        a_ptrs += 2 * stride_n

def test_nested3():
    n_rows = 4
    n_cols = 48
    expected = torch.tensor([
        [ 0,  1,  2,  3,  4,  5,  0,  1,  2,  3,  6,  7,  0,  1,
          8,  9, 10, 11,  0,  1,  8,  9, 12, 13, 14, 15, 16, 17,
         18, 19, 14, 15, 16, 17, 20, 21, 14, 15, 22, 23, 24, 25,
         14, 15, 22, 23, 26, 27],
        [48, 49, 50, 51, 52, 53, 48, 49, 50, 51, 54, 55, 48, 49,
         56, 57, 58, 59, 48, 49, 56, 57, 60, 61, 62, 63, 64, 65,
         66, 67, 62, 63, 64, 65, 68, 69, 62, 63, 70, 71, 72, 73,
         62, 63, 70, 71, 74, 75],
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
          0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
          0,  0,
