@triton.jit
def kernel_f8_to_f16(Y, X, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs<N
    x = tl.load(X + offs, mask=mask)
    x = triton.float8(x, ordered=True)  # Convert int8 to float8
    y = triton.float16(x)  # Convert float8 to float16
    tl.store(Y + offs, y, mask=mask)
