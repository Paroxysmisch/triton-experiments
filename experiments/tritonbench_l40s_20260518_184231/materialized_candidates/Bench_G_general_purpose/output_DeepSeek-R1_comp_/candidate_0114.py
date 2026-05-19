import triton
import triton.language as tl
import torch

@triton.jit
def spinning_lock_kernel(
    P_ptr, C_ptr, locks_ptr,
    num_sms, k, M, N,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_m = pid // num_sms
    pid_n = pid % num_sms

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for _ in range(9):
        if pid % k == 0:
            # Leader block: accumulate data from other blocks in the group
            for i in tl.static_range(1, k):
                other_pid = pid + i
                other_pid_m = other_pid // num_sms
                other_pid_n = other_pid % num_sms
                other_offs_m = other_pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                other_offs_n = other_pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                other_mask = (other_offs_m[:, None] < M) & (other_offs_n[None, :] < N)
                p_ptrs = P_ptr + other_offs_m[:, None] * stride_cm + other_offs_n[None, :] * stride_cn
                lock_addr = locks_ptr + other_pid
                lock_val = tl.atomic_cas(lock_addr, 0, 1)
                if lock_val == 0:
                    p = tl.load(p_ptrs, mask=other_mask, other=0.0)
                    acc += p
        else:
            # Non-leader block: store data to P and release lock
            p_ptrs = P_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
            tl.store(p_ptrs, acc, mask=mask)
            lock_addr = locks_ptr + pid
            tl.atomic_xchg(lock_addr, 0)

    if pid % k == 0:
        c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, acc, mask=mask)

def spinning_lock(P: torch.Tensor, C: torch.Tensor, locks: torch.Tensor, num_sms: int, k: int):
    assert P.is_cuda and C.is_cuda and locks.is_cuda
    M, N = C.shape
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    grid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    total_blocks = grid_m * num_sms
    grid = (total_blocks,)
    spinning_lock_kernel[grid](
        P, C, locks,
        num_sms, k, M, N,
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
