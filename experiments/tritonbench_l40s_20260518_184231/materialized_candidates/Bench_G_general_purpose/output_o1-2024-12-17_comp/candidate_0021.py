import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config(
            {
                'TILE_M': 64,
                'TILE_N': 64,
                'TILE_K': 32,
                'GROUP_M': 4,
                'DIVISIBLE_M': True,
                'DIVISIBLE_N': True,
                'DIVISIBLE_K': True
            },
            num_warps=4,
            num_stages=2
        ),
        triton.Config(
            {
                'TILE_M': 128,
                'TILE_N': 64,
                'TILE_K': 32,
                'GROUP_M': 4,
                'DIVISIBLE_M': False,
                'DIVISIBLE_N': False,
                'DIVISIBLE_K': False
            },
            num_warps=4,
            num_stages=2
        ),
    ],
    key=['A_ptr', 'B_ptr', 'O_ptr', 'M', 'N', 'K']
)
@triton.jit
def bmm_kernel(
    A_ptr, B_ptr, O_ptr,
    M, N, K,
    stride_am, stride_ak, stride_ab,
    stride_bk, stride_bn, stride_bb,
    stride_om, stride_on, stride_ob,
    GROUP_M: tl.constexpr,
    TILE_M: tl.constexpr, TILE_N: tl.constexpr, TILE_K: tl.constexpr,
    DIVISIBLE_M: tl.constexpr, DIVISIBLE_N: tl.constexpr, DIVISIBLE_K: tl.constexpr
):
    pid = tl.program_id(0)
    batch_id = tl.program_id(1)

    num_pid_m = (M + TILE_M - 1) // TILE_M
    group_id = pid // GROUP_M
    group_size = num_pid_m // GROUP_M + int(group_id < (num_pid_m % GROUP_M))
    pid_m = group_id * group_size + (pid % GROUP_M)
    pid_n = pid // (GROUP_M * group_size)

    m_start = pid_m * TILE_M
    n_start = pid_n * TILE_N
    b_id = batch_id

    off_m = m_start + tl.arange(0, TILE_M)
    off_n = n_start + tl.arange(0, TILE_N)
    off_k = tl.arange(0, TILE_K)

    A_tile_ptr = A_ptr + b_id * stride_ab + off_m[:, None] * stride_am + off_k[None, :] * stride_ak
    B_tile_ptr = B_ptr + b_id * stride_bb + off_k[:, None] * stride_bk + off_n[None, :] * stride_bn

    acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)

    # k-loop
    for k_off in range(0, K, TILE_K):
        if not DIVISIBLE_K:
            k_remaining = K - k_off
            off_k = tl.arange(0, TILE_K)
            off_k = tl.where(off_k < k_remaining, off_k, 0)
        A_in = tl.load(A_tile_ptr, mask=(off_m[:, None] < M) & (off_k[None, :] < K) if not DIVISIBLE_M or not DIVISIBLE_K else None, other=0.0)
        B_in = tl.load(B_tile_ptr, mask=(off_k[:, None] < K) & (off_n[None, :] < N) if not DIVISIBLE_N or not DIVISIBLE_K else None, other=0.0)
        acc += tl.dot(A_in, B_in)
        A_tile_ptr += TILE_K * stride_ak
        B_tile_ptr += TILE_K * stride_bk

    O_ptrs = O_ptr + b_id * stride_ob + off_m[:, None] * stride_om + off_n[None, :] * stride_on
    mask_o = (off_m[:, None] < M) & (off_n[None, :] < N)
    tl.store(O_ptrs, acc, mask=mask_o if not (DIVISIBLE_M and DIVISIBLE_N) else None)

def bmm(A: torch.Tensor, B: torch.Tensor):
    assert A.is_cuda and B.is_cuda
    assert A.ndim == 3 and B.ndim == 3
    BATCH, M, K = A.shape
    BK, KN = B.shape[1], B.shape[2]
    assert BK == K
    N = KN
    O = torch.empty((BATCH, M, N), device=A.device, dtype=A.dtype)

    stride_am = A.stride(1)
    stride_ak = A.stride(2)
    stride_ab = A.stride(0)
    stride_bk = B.stride(1)
    stride_bn = B.stride(2)
    stride_bb = B.stride(0)
    stride_om = O.stride(1)
    stride_on = O.stride(2)
    stride_ob = O.stride(0)

    grid = ( ( (M + 63) // 64 ) * ( (N + 63) // 64 ), BATCH )
    bmm_kernel[grid](
        A, B, O,
        M, N, K,
        stride_am, stride_ak, stride_ab,
        stride_bk, stride_bn, stride_bb,
        stride_om, stride_on, stride_ob
    )
    return O
