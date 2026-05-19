import torch
import triton
import triton.language as tl

# Triton kernel
@triton.jit
def nested3(in_ptr0, out_ptr0, xnumel, rnumel, stride_m: tl.constexpr, stride_n: tl.constexpr):
    for x in range(0, xnumel):
        for r in range(0, rnumel):
            for k in range(0, 1):
                a_ptrs = in_ptr0 + (x + k * stride_m)
                c_ptrs = out_ptr0 + (r + k * stride_n)
                tl.store(c_ptrs, tl.load(a_ptrs))

# Wrapper function
def wrapper_nested3(x, output, n_rows, n_cols):
    x = x.cuda()
    output = output.cuda()
    xnumel = n_rows
    rnumel = n_cols
    grid = lambda meta: (triton.cdiv(n_cols, 4),)
    nested3[grid](x, output, xnumel, rnumel, stride_m=1, stride_n=1)
    print(output)

# Example usage
x = torch.randn(2, 2)
output = torch.empty(2, 2)
wrapper_nested3(x, output, 2, 2)
