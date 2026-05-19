import torch
import triton
import triton.language as tl

@triton.jit
def nested3(in_ptr, out_ptr, stride_m, stride_n):
    # Define offsets for 2x2 tiles
    offs_am = tl.arange(0, 2)
    offs_an = tl.arange(0, 2)
    a_ptrs = in_ptr + (offs_am[:, None] * stride_m +
                       offs_an[None, :] * stride_n)

    offs_cm = tl.arange(0, 2)
    offs_cn = tl.arange(0, 2)
    c_ptrs = out_ptr + stride_m * offs_cm[:, None] + stride_n * offs_cn[None, :]

    # Nested loops to iterate through the tiles
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

def wrapper_nested3():
    # Define input tensor dimensions
    n_rows = 4
    n_cols = 48
    # Initialize input tensor
    x = torch.arange(0, n_rows * n_cols, device="cuda", dtype=torch.int32).reshape([n_rows, n_cols])
    # Initialize output tensor
    output = torch.zeros([n_rows, n_cols], device=x.device, dtype=x.dtype)

    # Grid configuration for kernel launch
    grid = lambda meta: (n_cols // 4,)

    print('Input Tensor:')
    print(x)
    print('Output Tensor (before):')
    print(output)

    # Launch the kernel
    nested3[grid](x, output, x.stride(0), x.stride(1))

    print('Output Tensor (after):')
    print(output)

    # Expected output for verification
    expected = torch.tensor([[ 0,  1,  2,  3,  4,  5,  0,  1,  2,  3,  6,  7,  0,  1,
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
                               0,  0,  0,  0,  0,  0]], dtype=torch.int32, device='cuda')

    # Verify the output
    torch.testing.assert_close(output, expected, rtol=0.001, atol=1e-5)
    print("Output matches expected tensor. Pass!")

# Run the wrapper function
wrapper_nested3()
