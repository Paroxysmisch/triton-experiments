import torch
import triton
import triton.language as tl

@triton.jit
def spinning_lock_kernel(
    P, C, locks,
    num_sms, k, M, N,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    pid = tl.program_id(0)
    pid_m = pid // num_sms
    pid_n = pid % num_sms

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Reduction logic for participating threads
    if pid % k == 0:
        # Reducer thread: accumulate from next k-1 threads
        for offset in tl.range(1, k):
            next_pid = pid + offset
            if next_pid >= num_sms:
                break
            
            # Attempt to acquire lock up to 9 times
            acquired = False
            for _ in tl.range(9):
                lock_val = tl.atomic_cas(locks + next_pid, 0, 1)
                if lock_val == 1:  # Lock was available
                    acquired = True
                    break
            
            if acquired:
                # Load and accumulate
                offs_m = tl.arange(0, BLOCK_SIZE_M)
                offs_n = tl.arange(0, BLOCK_SIZE_N)
                p_ptr = P + next_pid * BLOCK_SIZE_M * BLOCK_SIZE_N + offs_m[:, None] * BLOCK_SIZE_N + offs_n[None, :]
                acc += tl.load(p_ptr)
    else:
        # Contributing thread: store data and release lock
        offs_m = tl.arange(0, BLOCK_SIZE_M)
        offs_n = tl.arange(0, BLOCK_SIZE_N)
        p_ptr = P + pid * BLOCK_SIZE_M * BLOCK_SIZE_N + offs_m[:, None] * BLOCK_SIZE_N + offs_n[None, :]
        tl.store(p_ptr, acc)
        tl.atomic_xchg(locks + pid, 1)  # Release lock

    # Write final result
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptr = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m < M)[:, None] & (offs_n < N)[None, :]
    tl.store(c_ptr, acc, mask=mask)

def spinning_lock(
    P: torch.Tensor, C: torch.Tensor,
    locks: torch.Tensor, num_sms: int,
    k: int, BLOCK_SIZE_M: int, BLOCK_SIZE_N: int
):
    grid = (num_sms,)
    spinning_lock_kernel[grid](
        P, C, locks,
        num_sms, k,
        C.shape[0], C.shape[1],
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )

# Example usage
if __name__ == "__main__":
    BLOCK_SIZE_M, BLOCK_SIZE_N = 128, 128
    M, N = 1024, 1024
    num_sms = 304
    k = 3

    # Initialize tensors
    P = torch.randn((num_sms * BLOCK_SIZE_M, BLOCK_SIZE_N), device='cuda')
    C = torch.zeros((M, N), device='cuda')
    locks = torch.zeros(num_sms, dtype=torch.int32, device='cuda')

    # Execute kernel
    spinning_lock(P, C, locks, num_sms, k, BLOCK_SIZE_M, BLOCK_SIZE_N)
    print("Reduction result:", C)
