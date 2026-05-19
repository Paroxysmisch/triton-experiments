import triton
import triton.language as tl

# Define block sizes for tiling
BLOCK_SIZE_M = 128
BLOCK_SIZE_N = 128
BLOCK_SIZE_K = 32

@triton.jit
def ff_llama_kernel(
    x_ptr, w1_ptr, w3_ptr, rms_w_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_w1k, stride_w1n,
    stride_w3k, stride_w3n,
    stride_outm, stride_outn,
    **meta
):
    # Program ID and offsets
    pid = tl.program_id(axis=0)
    offs_m = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Pointers to input and output data
    x = tl.load(x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk, mask=offs_m[:, None] < M)
    w1 = tl.load(w1_ptr + offs_k[:, None] * stride_w1k + offs_n[None, :] * stride_w1n)
    w3 = tl.load(w3_ptr + offs_k[:, None] * stride_w3k + offs_n[None, :] * stride_w3n)

    # Matrix multiplication
    x_w1 = tl.dot(x, w1)
    x_w3 = tl.dot(x, w3)

    # Element-wise SILU activation
    silu_x_w1 = x_w1 * tl.sigmoid(x_w1)

    # Root Mean Square (RMS) scaling
    rms_scale = tl.sqrt(tl.sum(x * x, axis=1) / K)
    scaled_x = x / rms_scale[:, None]

    # Combine results
    result = silu_x_w1 + x_w3

    # Store result
    tl.store(out_ptr + offs_m[:, None] * stride_outm + offs_n[None, :] * stride_outn, result, mask=offs_m[:, None] < M)

def kernel_ff(x, w1, w3, rms_w):
    # Ensure input types are correct
    assert x.dtype in [triton.float16, triton.int8], "x must be float16 or int8"
    assert w1.dtype in [triton.float16, triton.int8], "w1 must be float16 or int8"
    assert w3.dtype in [triton.float16, triton.int8], "w3 must be float16 or int8"

    # Get shapes and strides
    M, K = x.shape
    _, N = w1.shape
    x_strides = x.strides
    w1_strides = w1.strides
    w3_strides = w3.strides
    out = triton.empty((M, N), dtype=triton.float16)

    # Launch kernel
    grid = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    ff_llama_kernel[grid](
        x, w1, w3, rms_w, out,
        M, N, K,
        x_strides[0], x_strides[1],
        w1_strides[0], w1_strides[1],
        w3_strides[0], w3_strides[1],
        out.strides[0], out.strides[1]
    )

    return out
