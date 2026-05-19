import triton
import triton.language as tl

@triton.jit
def spinning_lock_kernel(
    P_ptr, C_ptr, locks_ptr,
    num_sms, k, M, N, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    pid = tl.program_id(0)
    pid_m = pid // num_sms
    pid_n = pid % num_sms

    # Compute offsets
    off_m = pid_m * BLOCK_SIZE_M
    off_n = pid_n * BLOCK_SIZE_N

    # Create tl.tensor for accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Spinning lock and reduction
    for _ in range(9):
        if (pid % k) == 0:
            # Acquire locks from other participating threads
            for thread_id in range(1, k):
                other_pid = pid + thread_id
                # Attempt to acquire lock if valid
                locked = True
                while locked:
                    prev = tl.atomic_cas(locks_ptr + other_pid, 0, 1)
                    locked = prev != 0
                # Accumulate partial result from the other thread
                partial_offset = other_pid * BLOCK_SIZE_M * BLOCK_SIZE_N
                partial_data = tl.load(P_ptr + partial_offset, mask=True)
                acc += partial_data
                # Release lock
                tl.atomic_xchg(locks_ptr + other_pid, 0)
        else:
            # Acquire lock
            locked = True
            while locked:
                prev = tl.atomic_cas(locks_ptr + pid, 0, 1)
                locked = prev != 0
            # Store partial and release
            offset = pid * BLOCK_SIZE_M * BLOCK_SIZE_N
            tl.store(P_ptr + offset, acc, mask=True)
            tl.atomic_xchg(locks_ptr + pid, 0)

    # final write
    row_offsets = off_m + tl.arange(0, BLOCK_SIZE_M)
    col_offsets = off_n + tl.arange(0, BLOCK_SIZE_N)
    mask = (row_offsets[:, None] < M) & (col_offsets[None, :] < N)
    c_offset = row_offsets[:, None] * stride_cm + col_offsets[None, :] * stride_cn
    tl.store(C_ptr + c_offset, acc, mask=mask)


def spinning_lock(P, C, locks, num_sms, k, M, N, stride_cm, stride_cn,
                  BLOCK_SIZE_M=64, BLOCK_SIZE_N=64):
    grid = ( (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M * \
             (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N, )
    spinning_lock_kernel[grid](
        P, C, locks, num_sms, k, M, N, stride_cm, stride_cn,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N
    )
