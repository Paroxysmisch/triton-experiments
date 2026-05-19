import torch
import triton
import triton.language as tl

@triton.jit
def nested3(in_ptr, out_ptr, stride_m, stride_n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    col_offset = pid * BLOCK_SIZE
    for i in range(2):
        for j in range(2):
            for k in range(2):
                offset = i * stride_m + (col_offset + j) * stride_n + k
                a_ptrs = in_ptr + offset
                c_ptrs = out_ptr + offset
                val = tl.load(a_ptrs)
                tl.store(c_ptrs, val)

def wrapper_nested3(n_rows, n_cols):
    x = torch.arange(n_rows * n_cols, dtype=torch.float32, device='cuda').reshape(n_rows, n_cols)
    output = torch.zeros_like(x)
    BLOCK_SIZE = 4
    grid = (n_cols // BLOCK_SIZE,)
    nested3[grid](x, output, x.stride(0), x.stride(1), BLOCK_SIZE)
    print(output)
