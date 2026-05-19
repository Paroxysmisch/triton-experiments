import triton
import triton.language as tl
import torch

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'SPLIT_K': 1}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'SPLIT_K': 2}),
        # Add more configurations as needed for tuning
    ],
    key=['M', 'N', 'K']
)
@triton.jit
def _int8_matmul_rowwise_dequantize(
    A_ptr, B_ptr, C_ptr,
    state_x_ptr, state_w_ptr,
    bias_ptr, M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    stride_sx, stride_sw,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_k = tl.cdiv(K, BLOCK_K)

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    A_tile_ptrs = A_ptr + (offs_am[:, None] * stride_am + tl.arange(0, BLOCK_K)[None, :] * stride_ak)
    B_tile_ptrs = B_ptr + (tl.arange(0, BLOCK_K)[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, num_pid_k):
        A_tile = tl.load(A_tile_ptrs + k * BLOCK_K * stride_ak)
        B_tile = tl.load(B_tile_ptrs + k * BLOCK_K * stride_bk)

        acc += tl.dot(A_tile.to(tl.float32), B_tile.to(tl.float32))

    scale_x = tl.load(state_x_ptr + offs_am * stride_sx)
    scale_w = tl.load(state_w_ptr + offs_bn * stride_sw)

    acc *= scale_x[:, None] * scale_w[None, :]

    if bias_ptr is not None:
        bias = tl.load(bias_ptr + offs_bn * stride_cn)
        acc += bias[None, :]

    C_tile_ptrs = C_ptr + (offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn)
    tl.store(C_tile_ptrs, acc.to(tl.float16))

def int8_matmul_rowwise_dequantize(
    A: torch.Tensor, B: torch.Tensor, state_x: torch.Tensor, state_w: torch.Tensor,
    bias: Optional[torch.Tensor] = None
) -> torch.Tensor:
    assert A.dtype == torch.int8 and B.dtype == torch.int8
    assert state_x.dtype == torch.float32 and state_w.dtype == torch.float32

    M, K = A.shape
    _, N = B.shape

    C = torch.empty((M, N), dtype=torch.float16, device=A.device)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)

    _int8_matmul_rowwise_dequantize[grid](
        A, B, C, state_x, state_w, bias,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        state_x.stride(0), state_w.stride(0)
    )

    return C
