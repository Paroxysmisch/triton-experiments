import triton
import triton.language as tl
import torch

@triton.jit
def nested3(in_ptr, out_ptr, stride_m, stride_n, **meta):
    NT = meta["NT"]
    i = tl.program_id(0)
    j = tl.program_id(1)
    k = tl.arange(0, NT)

    a_ptrs = in_ptr + i * stride_m + (j * 2 + 0) * stride_n + k
    b_ptrs = in_ptr + i * stride_m + (j * 2 + 1) * stride_n + k
    c_ptrs = out_ptr + i * stride_m + (j * 2 + 0) * stride_n + k
    d_ptrs = out_ptr + i * stride_m + (j * 2 + 1) * stride_n + k

    a = tl.load(a_ptrs)
    b = tl.load(b_ptrs)
    c = tl.load(c_ptrs)
    d = tl.load(d_ptrs)

    tl.store(c_ptrs, a)
    tl.store(d_ptrs, b)


def wrapper_nested3(n_rows, n_cols):
    x = torch.randn((n_rows, n_cols), device="cuda", dtype=torch.long)
    output = torch.zeros((n_rows, n_cols), device="cuda", dtype=torch.long)

    def grid(meta):
        return (triton.cdiv(n_rows, 1), triton.cdiv(n_cols, 2))

    nested3[grid](x, output, x.stride(0), x.stride(1), NT=4)
    print(output)
    return output
