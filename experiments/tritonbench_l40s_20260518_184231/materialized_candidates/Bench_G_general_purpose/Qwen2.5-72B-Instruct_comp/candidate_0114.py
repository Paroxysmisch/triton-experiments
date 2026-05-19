import triton
import triton.language as tl

@triton.jit
def spinning_lock_kernel(P, C, locks, num_sms, k, M, N, stride_cm, stride_cn, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    pid = tl.program_id(axis=0)
    pid_m = pid // N
    pid_n = pid % N

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for _ in range(9):
        if pid % k == 0:
            for i in range(k):
                if tl.atomic_cas(locks + pid, 0, 1) == 0:
                    acc += tl.load(P + pid_m * stride_cm + (pid_n + i) * stride_cn)
                    tl.atomic_xchg(locks + pid, 0)
        else:
            tl.store(P + pid_m * stride_cm + pid_n * stride_cn, acc)
            tl.atomic_xchg(locks + pid, 0)

    mask = (pid_m < M) & (pid_n < N)
    tl.store(C + pid_m * stride_cm + pid_n * stride_cn, acc, mask=mask)

import triton
import triton.runtime

def spinning_lock(P, C, locks, num_sms, k, M, N, stride_cm, stride_cn, BLOCK_SIZE_M, BLOCK_SIZE_N):
    grid = (num_sms * k, 1, 1)
    num_warps = 4  # Adjust based on your GPU architecture and problem size

    spinning_lock_kernel[grid, num_warps](
        P, C, locks, num_sms, k, M, N, stride_cm, stride_cn, BLOCK_SIZE_M, BLOCK_SIZE_N
    )
