import triton
import triton.language as tl

@triton.jit
def spinning_lock_kernel(P, C, locks, num_sms, k, M, N, stride_cm, stride_cn, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    pid = tl.program_id(0)
    pid_m = pid // (N // BLOCK_SIZE_N)
    pid_n = pid % (N // BLOCK_SIZE_N)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Calculate memory offsets
    offset_c = pid_m * BLOCK_SIZE_M * stride_cm + pid_n * BLOCK_SIZE_N * stride_cn
    
    # Loop for up to 9 iterations
    for _ in range(9):
        # Check if this thread should participate in the reduction
        if pid % k == 0:
            # Attempt to acquire lock and accumulate
            for i in range(k):
                other_pid = pid + i
                lock_acquired = tl.atomic_cas(locks + other_pid, 0, 1)
                if lock_acquired:
                    # Perform accumulation
                    acc += tl.load(P + offset_c)
                    # Release lock
                    tl.atomic_xchg(locks + other_pid, 0)
    
    # Store result and release lock for non-participating threads
    if pid % k != 0:
        tl.store(P + offset_c, acc)
        tl.atomic_xchg(locks + pid, 0)
    
    # Write results to C with masking for valid memory access
    mask = (pid_m * BLOCK_SIZE_M < M) & (pid_n * BLOCK_SIZE_N < N)
    tl.store(C + offset_c, acc, mask=mask)

def spinning_lock(P, C, locks, num_sms, k, M, N, stride_cm, stride_cn, BLOCK_SIZE_M, BLOCK_SIZE_N):
    # Calculate grid size
    grid = (num_sms * k, )
    
    # Launch kernel
    spinning_lock_kernel[grid](
        P, C, locks, num_sms, k, M, N, stride_cm, stride_cn,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N
    )
