import triton
import triton.language as tl

@triton.jit
def rotary_kernel(
    X, OUT, COS, SIN, CU_SEQLENS, 
    M, N, H, 
    stride_xm, stride_xh, stride_xn, 
    stride_cosm, stride_cosn, 
    stride_sinn, 
    stride_outm, stride_outh, stride_outn, 
    CONJUGATE: tl.constexpr, 
    INTERLEAVED: tl.constexpr, 
    BLOCK_SIZE_M: tl.constexpr, 
    BLOCK_SIZE_N: tl.constexpr
):
    pid_batch = tl.program_id(axis=0)
    pid_head = tl.program_id(axis=1)
    pid_m = tl.program_id(axis=2)

    # Compute the start and end indices for the current batch and head
    start_m = pid_m * BLOCK_SIZE_M
    end_m = min(start_m + BLOCK_SIZE_M, M)
    start_n = 0
    end_n = N

    # Compute the start and end indices for the current sequence
    if CU_SEQLENS is not None:
        start_m = tl.load(CU_SEQLENS + pid_batch * H + pid_head + pid_m)
        end_m = tl.load(CU_SEQLENS + pid_batch * H + pid_head + pid_m + 1)

    # Load the data blocks
    x_block = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    cos_block = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    sin_block = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for m in range(start_m, end_m):
        for n in range(start_n, end_n):
            x_block[m - start_m, n - start_n] = tl.load(X + pid_batch * stride_xm + pid_head * stride_xh + m * stride_xn + n)
            cos_block[m - start_m, n - start_n] = tl.load(COS + m * stride_cosm + n * stride_cosn)
            sin_block[m - start_m, n - start_n] = tl.load(SIN + m * stride_sinn + n * stride_sinn)

    # Apply rotary position encoding
    for m in range(start_m, end_m):
        for n in range(start_n, end_n):
            if INTERLEAVED:
                x0 = x_block[m - start_m, n - start_n]
                x1 = x_block[m - start_m, n - start_n + 1]
                out0 = x0 * cos_block[m - start_m, n - start_n] - x1 * sin_block[m - start_m, n - start_n]
                out1 = x0 * sin_block[m - start_m, n - start_n] + x1 * cos_block[m - start_m, n - start_n]
                if CONJUGATE:
                    out1 = -out1
                tl.store(OUT + pid_batch * stride_outm + pid_head * stride_outh + m * stride_outn + n, out0)
                tl.store(OUT + pid_batch * stride_outm + pid_head * stride_outh + m * stride_outn + n + 1, out1)
            else:
                out = x_block[m - start_m, n - start_n] * cos_block[m - start_m, n - start_n] - x_block[m - start_m, n - start_n + 1] * sin_block[m - start_m, n - start_n]
                if CONJUGATE:
                    out = -out
                tl.store(OUT + pid_batch * stride_outm + pid_head * stride_outh + m * stride_outn + n, out)

import torch
import triton
import triton.language as tl

def apply_rotary(x, cos, sin, cu_seqlens=None, conjugate=False, interleaved=False):
    # Get the shape of the input tensor
    B, H, M, N = x.shape

    # Determine the grid and block sizes
    grid = (B, H, (M + 127) // 128)
    block = (128, 128)

    # Initialize the output tensor
    out = torch.empty_like(x)

    # Determine the strides
    stride_xm, stride_xh, stride_xn = x.stride(2), x.stride(1), x.stride(3)
    stride_cosm, stride_cosn = cos.stride(0), cos.stride(1)
    stride_sinn = sin.stride(1)
    stride_outm, stride_outh, stride_outn = out.stride(2), out.stride(1), out.stride(3)

    # Copy non-rotary parts of x to out if required
    if not interleaved:
        out[:, :, :, 0::2] = x[:, :, :, 0::2]

    # Call the Triton kernel
    rotary_kernel[grid, block](
        x, out, cos, sin, cu_seqlens,
        M, N, H,
        stride_xm, stride_xh, stride_xn,
        stride_cosm, stride_cosn,
        stride_sinn,
        stride_outm, stride_outh, stride_outn,
        conjugate, interleaved, 128, 128
    )

    return out
