import torch
import triton
import triton.language as tl

@triton.jit
def nested3(in_ptr, out_ptr, stride_m, stride_n):
    offs_am = tl.arange(0, 2)
    offs_an = tl.arange(0, 2)
    a_ptrs = in_ptr + (offs_am[:, None] * stride_m + offs_an[None, :] * stride_n)

    offs_cm = tl.arange(0, 2)
    offs_cn = tl.arange(0, 2)
    c_ptrs = out_ptr + (offs_cm[:, None] * stride_m + offs_cn[None, :] * stride_n)

    for i in range(2):
        a1 = tl.load(a_ptrs)

        for j in range(2):
            a_ptrs += 2 * stride_n
            a2 = tl.load(a_ptrs)

            for k in range(2):
                a_ptrs += 2 * stride_n
                a3 = tl.load(a_ptrs)
                tl.store(c_ptrs, a1)
                c_ptrs += 2 * stride_n

                tl.store(c_ptrs, a2)
                c_ptrs += 2 * stride_n
                tl.store(c_ptrs, a3)
                c_ptrs += 2 * stride_n

        a_ptrs += 2 * stride_n

def wrapper_nested3(n_rows, n_cols):
    x = torch.arange(0, n_rows * n_cols, device="cuda", dtype=torch.int32).reshape([n_rows, n_cols])
    output = torch.zeros([n_rows, n_cols], device=x.device, dtype=x.dtype)
    grid = lambda meta: (n_cols // 4,)

    print('Input tensor:')
    print(x)
    print('Output tensor before kernel execution:')
    print(output)

    nested3[grid](x, output, x.stride(0), x.stride(1))

    print('Output tensor after kernel execution:')
    print(output)

# Example usage
n_rows = 4
n_cols = 48
wrapper_nested3(n_rows, n_cols)
