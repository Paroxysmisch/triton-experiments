import triton
import triton.language as tl

@triton.jit
def _int8_matmul_rowwise_dequantize_kernel(
    A_ptr, B_ptr, C_ptr, 
    state_x_ptr, state_w_ptr, bias_ptr, 
    M, N, K, 
    stride_am, stride_ak, 
    stride_bn, stride_bk, 
    stride_cm, stride_cn, 
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    # Calculate offsets
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    # Initialize accumulation buffer
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    # Load scale factors for dequantization
    scale_x = tl.load(state_x_ptr + offs_am)
    scale_w = tl.load(state_w_ptr + offs_bn)

    # Perform the matrix multiplication
    for k in range(0, K, BLOCK_K):
        a = tl.load(A_ptr + offs_am[:, None] * stride_am + (offs_k + k) * stride_ak, mask=offs_k + k < K, other=0).to(tl.int32)
        b = tl.load(B_ptr + offs_k[:, None] * stride_bk + offs_bn * stride_bn, mask=offs_k + k < K, other=0).to(tl.int32)
        acc += tl.dot(a, b)

    # Apply dequantization
    acc = acc.to(tl.float32) * (scale_x[:, None] * scale_w[None, :])

    # Add bias if present
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + offs_bn)
        acc += bias[None, :]

    # Write back results
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if SPLIT_K > 1:
        # Atomic add if SPLIT_K > 1
        tl.atomic_add(C_ptr + offs_cm[:, None] * stride_cm + offs_cn * stride_cn, acc, mask=(offs_cm[:, None] < M) & (offs_cn < N))
    else:
        # Direct store
        tl.store(C_ptr + offs_cm[:, None] * stride_cm + offs_cn * stride_cn, acc, mask=(offs_cm[:, None] < M) & (offs_cn < N))


def int8_matmul_rowwise_dequantize(A, B, state_x, state_w, bias=None, BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, SPLIT_K=1):
    # Get input shapes
    M, K = A.shape
    _, N = B.shape

    # Allocate output tensor
    C = torch.zeros((M, N), dtype=torch.float32, device='cuda')

    # Define grid size
    grid = lambda META: (
        (M + META['BLOCK_M'] - 1) // META['BLOCK_M'],
        (N + META['BLOCK_N'] - 1) // META['BLOCK_N'],
        SPLIT_K
    )

    # Launch kernel
    _int8_matmul_rowwise_dequantize_kernel[grid](
        A, B, C, 
        state_x, state_w, bias, 
        M, N, K, 
        A.stride(0), A.stride(1), 
        B.stride(1), B.stride(0), 
        C.stride(0), C.stride(1), 
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, SPLIT_K=SPLIT_K
    )

    return C
