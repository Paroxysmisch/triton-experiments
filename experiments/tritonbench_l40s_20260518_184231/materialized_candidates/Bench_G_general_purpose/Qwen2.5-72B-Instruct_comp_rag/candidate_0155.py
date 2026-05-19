import torch
import triton
import triton.language as tl

@triton.jit
def nested3(in_ptr, out_ptr, stride_m, stride_n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_cols = BLOCK_SIZE * 4
    col_start = pid * num_cols

    for i in range(0, 2):
        for j in range(0, 2):
            for k in range(0, 2):
                a_ptrs = in_ptr + (i * 2 * stride_m + j * 2 * stride_n + k * 2 * stride_n)
                c_ptrs = out_ptr + (i * 2 * stride_m + j * 2 * stride_n + k * 2 * stride_n + col_start)

                a1 = tl.load(a_ptrs)
                a2 = tl.load(a_ptrs + 2 * stride_n)
                a3 = tl.load(a_ptrs + 4 * stride_n)

                tl.store(c_ptrs, a1)
                tl.store(c_ptrs + 2 * stride_n, a2)
                tl.store(c_ptrs + 4 * stride_n, a3)

def test_nested3():
    n_rows = 4
    n_cols = 48
    expected = torch.tensor([
        [ 0,  1,  2,  3,  4,  5,  0,  1,  2,  3,  6,  7,  0,  1,  8,  9, 10, 11,  0,  1,  8,  9, 12, 13, 14, 15, 16, 17, 18, 19, 14, 15, 16, 17, 20, 21, 14, 15, 22, 23, 24, 25, 14, 15, 22, 23, 26, 27],
        [48, 49, 50, 51, 52, 53, 48, 49, 50, 51, 54, 55, 48, 49, 56, 57, 58, 59, 48, 49, 56, 57, 60, 61, 62, 63, 64, 65, 66, 67, 62, 63, 64, 65, 68, 69, 62, 63, 70, 71, 72, 73, 62, 63, 70, 71, 74, 75],
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]
    ], dtype=torch.int32, device='cpu')

    x = torch.arange(0, n_rows * n_cols, device="cpu", dtype=torch.int32).reshape([n_rows, n_cols])
    output = torch.zeros([n_rows, n_cols], device=x.device, dtype=x.dtype)
    grid = lambda meta: (n_cols // 4,)

    print('before:')
    print(x)
    print(output)

    nested3[grid](x, output, x.stride(0), x.stride(1), BLOCK_SIZE=2)
    print(output)
    torch.testing.assert_close(output, expected, rtol=0.001, atol=1e-5)
    print("Pass!")

    src = triton.compiler.ASTSource(
        fn=nested3,
        signature="*fp32,*fp32,i32,i32,i32",
    )
    ret = triton.compile(
        src,
    )
    print(ret.asm["ttir"])
    print('Pass')

test_nested3()
