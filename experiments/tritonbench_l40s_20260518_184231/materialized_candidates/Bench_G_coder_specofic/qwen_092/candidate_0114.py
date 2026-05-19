import triton
import triton.language as tl

@triton.jit
def spinning_lock_kernel(
    P: tl.tensor, C: tl.tensor, locks: tl.tensor,
    num_sms: tl.int32, k: tl.int32, M: tl.int32, N: tl.int32, stride_cm: tl.int32, stride_cn: tl.int32,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    # Each block is identified by a unique program id
    pid = tl.program_id(0)
    pid_m = pid // (N // BLOCK_SIZE_N)
    pid_n = pid % (N // BLOCK_SIZE_N)

    # Initialize the accumulator to zero
    acc = tl.zeros([], dtype=tl.float32)

    # Loop up to 9 times
    for _ in range(9):
        # Only the first thread in the block performs accumulation
        if pid % k == 0:
            for tid in range(BLOCK_SIZE_M):
                for t in range(BLOCK_SIZE_N):
                    # Compute the global indices
                    global_idx = pid_m * stride_cm * BLOCK_SIZE_M + tid * stride_cm + pid_n * stride_cn * BLOCK_SIZE_N + t * stride_cn
                    # Attempt to acquire the lock
                    if tl.atomic_cas(locks[global_idx], 0, 1) == 0:
                        # If the lock is acquired, accumulate the value
                        acc += P[global_idx]
                        # Release the lock
                        tl.atomic_xchg(locks[global_idx], 0)

    # Write the result to C
    C[pid] = acc
