import triton
import triton.language as tl
from typing import Optional, Tuple

@triton.jit
def bmm_kernel(
    A,
    B,
    O,
    M,
    N,
    K,
    TILE_M,
    TILE_N,
    TILE_K,
    GROUP_M,
    DIVISIBLE_M,
    DIVISIBLE_N,
    DIVISIBLE_K,
):
    pid = tl.program_id(axis=0)
    n_group_x = tl.cdiv(M, GROUP_M)
    assert n_group_x <= tl.cdiv(M, TILE_M)  # M has to be divisible by GROUP_M
    group_x = pid % n_group_x
    group_offset_x = group_x * GROUP_M
    assert group_offset_x % TILE_M == 0  # GROUP_M has to be divisible by TILE_M
    group_start_x = group_offset_x
    group_end_x = group_start_x + GROUP_M
    grid_x = pid // n_group_x

    # Prepare pointers
    # Matrices A and B are manually strided. Matrix O strides across groups.
    A_ptrs = [A + grid_x * K * M + group_start_x * K + (group_offset_x % TILE_M) * K]
    B_ptrs = [B + grid_x * K * N + (group_offset_x % TILE_M) * N + group_start_x * N]
    O_ptrs = [tl.all_codes(O)]
    O_ptrs += [O_ptrs[0] + grid_x * M * N]
    O_ptrs += [O_ptrs[1] + (group_start_x % TILE_M) * N + group_start_x * M * N]

    # Iterate over M and K dimensions
    for kk in range(K):
        O_ptrs[0] += M
        O_ptrs[1] += M

        # Iterate over N dimension
        for nn in range(N):
            O_ptrs[0]++
            O_ptrs[1] += M
            O_wise = tl.load(O_ptrs, mask=((group_start_x % TILE_M) < 
                                           (M - TILE_M)) & ((group_start_x + TILE_M) <= M))
            A_wise = tl.load(A_ptrs, mask=((group_start_x % TILE_M) < 
                                           (M - TILE_M)) & ((group_start_x + TILE_M) <= M))
            B_wise = tl.load(B_ptrs, mask=((group_start_x % TILE_M) < 
                                           (M - TILE_M)) & ((group_start_x + TILE_M) <= M))
            partial = tl.dot(A_wise, B_wise)
            O_wise += partial
            tl.store(O_ptrs, O_wise)

def bmm(
    A,
    B,
    *,
    out: Optional[torch.Tensor] = None,
    skip_checks: bool = False,
    max_grid: Optional[Tuple[Optional[int], Optional[int], Optional[int]]] = None,
):
    if out is None:
        out = torch.empty((A.shape[0], A.shape[1], B.shape[2]), dtype=A.dtype, device=A.device)

    M, K = A.shape[1], A.shape[2]
    N = B.shape[2]
    GROUP_M = 16
    TILE_M = 32
    TILE_N = N
    TILE_K = K
    DIVISIBLE_M = M % TILE_M == 0
    DIVISIBLE_N = N % TILE_N == 0
    DIVISIBLE_K = K % TILE_K == 0

    grid = lambda M: ((M + GROUP_M - 1) // GROUP_M, 1, 1)

    triton.autotune(
        lambda TILE_M, TILE_N, TILE_K: bmm_kernel[grid(M), (TILE_M, TILE_N, TILE_K)]
        (A, B, out, M, N, K, TILE_M, TILE_N, TILE_K, GROUP_M, DIVISIBLE_M, DIVISIBLE_N, DIVISIBLE_K),
        # search space for hyper-parameter tuning
        configs=[
            dict(TILE_M=32, TILE_N=32, TILE_K=32),
            dict(TILE_M=64, TILE_N=64, TILE_K=64),
            dict(TILE_M=128, TILE_N=128, TILE_K=128),
        ],
        # stop criteria
        max_num_trials=8, min_delta_loss=0.1,
        max_grid=max_grid, skip_checks=skip_checks
    )

    return out
