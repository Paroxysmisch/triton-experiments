import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'SPLIT_K': 2}, num_warps=8),
        # Add more configurations as needed
    ],
    key=['M', 'N', 'K']
)
@triton.jit
def _int8_matmul_rowwise_dequantize(A_ptr, B_ptr, state_x_ptr, state_w_ptr, C_ptr, bias_ptr, 
                                    M, N, K, stride_am, stride_ak, stride_bn, stride_bk, stride_cm, stride_cn,
                                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_m = (M + BLOCK_M - 1) // BLOCK_M
    num_pid_n = (N + BLOCK_N - 1) // BLOCK_N
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    A_tile = tl.load(A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak), mask=offs_am[:, None] < M)
    B_tile = tl.load(B_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn), mask=offs_bn[None, :] < N)

    state_x = tl.load(state_x_ptr + offs_am, mask=offs_am < M)
    state_w = tl.load(state_w_ptr + offs_bn, mask=offs_bn < N)

    C_tile = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        A_sub = A_tile.to(tl.int32)
        B_sub = B_tile.to(tl.int32)
        C_tile += tl.dot(A_sub, B_sub)

    C_tile = C_tile * (state_x[:, None].to(tl.float32) * state_w[None, :].to(tl.float32))

    if bias_ptr is not None:
        bias = tl.load(bias_ptr + offs_bn, mask=offs_bn < N)
        C_tile += bias[None, :]

    C_tile = C_tile.to(tl.float16)
    tl.store(C_ptr + (offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn), C_tile, mask=offs_am[:, None] < M)

def int8_matmul_rowwise_dequantize(A, B, state_x, state_w, bias=None):
    M, K = A.shape
    K, N = B.shape

    C = torch.empty((M, N), dtype=torch.float16, device=A.device)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    _int8_matmul_rowwise_dequantize[grid](
        A, B, state_x, state_w, C, bias,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(1), B.stride(0),
        C.stride(0), C.stride(1),
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, SPLIT_K=1  # Default values, autotuner will choose best
    )
    return C
