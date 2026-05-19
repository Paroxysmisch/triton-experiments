import triton
import triton.language as tl

@triton.jit
def fused_mv_sigmoid_sub_kernel(
    A_ptr, v_ptr, other_ptr, out_ptr,
    n, m, alpha,
    stride_am, stride_an, stride_v, stride_out,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_M: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    # Create pointers for the block
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

    A = tl.load(A_ptr + offs_n[:, None] * stride_an + offs_m[None, :] * stride_am, mask=(offs_n[:, None] < n) & (offs_m[None, :] < m), other=0)
    v = tl.load(v_ptr + offs_m * stride_v, mask=offs_m < m, other=0)

    # Matrix-vector multiplication
    z = tl.dot(A, v)

    # Sigmoid activation
    s = 1 / (1 + tl.exp(-z))

    # Load 'other' and perform subtraction
    other = tl.load(other_ptr + offs_n, mask=offs_n < n, other=0)
    y = s - alpha * other

    # Store result
    tl.store(out_ptr + offs_n, y, mask=offs_n < n)

def fused_mv_sigmoid_sub(input, vec, other, alpha=1, *, out=None):
    n, m = input.shape
    assert vec.shape[0] == m, "The shape of `vec` must match the second dimension of `input`."

    # Allocate output tensor if not provided
    if out is None:
        out = torch.empty((n,), device=input.device, dtype=input.dtype)

    # Determine grid size
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_M = 128
    grid = (triton.cdiv(n, BLOCK_SIZE_N), triton.cdiv(m, BLOCK_SIZE_M))

    # Launch kernel
    fused_mv_sigmoid_sub_kernel[grid](
        input, vec, other, out,
        n, m, alpha,
        input.stride(0), input.stride(1), vec.stride(0), out.stride(0),
        BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_M=BLOCK_SIZE_M
    )

    return out
