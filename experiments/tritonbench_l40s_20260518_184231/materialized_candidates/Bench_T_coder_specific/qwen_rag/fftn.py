import triton
import triton.language as tl

@triton.jit
def dft_kernel(
    x_ptr, 
    y_ptr, 
    stride_x, 
    stride_y, 
    N, 
    M, 
    dtype: tl.constexpr,
    fp16: tl.constexpr,
    tf32: tl.constexpr,
    norm: tl.constexpr
):
    pid = tl.program_id(axis=0)
    grid_size = tl.cdiv(N * M, 1024)

    i = pid // M
    j = pid % M

    # Load input element
    if dtype == tl.float16:
        x = tl.load(x_ptr + i * stride_x + j, dtype=tl.float16)
    else:
        x = tl.load(x_ptr + i * stride_x + j, dtype=tl.complex64)

    # Initialize output element
    y = tl.zeros_like(x)

    # Compute DFT
    k = tl.arange(M)
    exp_term = tl.exp(-2j * tl.pi * k * j / M)

    # Apply normalization
    if norm == 'forward':
        exp_term /= tl.sqrt(M)
    elif norm == 'ortho':
        exp_term /= M

    # Perform the summation
    for l in range(M):
        if dtype == tl.float16:
            y += x[l] * exp_term[l].to(tl.float16)
        else:
            y += x[l] * exp_term[l]

    # Store the result
    tl.store(y_ptr + i * stride_y + j, y)
