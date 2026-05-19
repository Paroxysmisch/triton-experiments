import torch
import triton
import triton.language as tl

@triton.jit
def spinning_lock_kernel(
    P, C, locks,
    num_sms, k, M, N,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr
):
    # Get program ID and calculate position in grid
    pid = tl.program_id(0)
    pid_m = pid // num_sms
    pid_n = pid % num_sms
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Main reduction loop
    for _ in range(9):
        if pid % k == 0:
            # This thread is responsible for reduction
            for i in range(1, k):
                next_pid = pid + i
                if next_pid >= num_sms:
                    break
                    
                # Spin until lock is acquired
                while tl.atomic_cas(locks + next_pid, 0, 1) != 0:
                    pass
                
                # Load and accumulate data from other thread
                offset = next_pid * BLOCK_SIZE_M * BLOCK_SIZE_N
                rm = tl.arange(0, BLOCK_SIZE_M)
                rn = tl.arange(0, BLOCK_SIZE_N)
                P_ptr = P + offset + rm[:, None] * BLOCK_SIZE_N + rn[None, :]
                acc += tl.load(P_ptr)
        else:
            # Store intermediate results and signal completion
            offset = pid * BLOCK_SIZE_M * BLOCK_SIZE_N
            rm = tl.arange(0, BLOCK_SIZE_M)
            rn = tl.arange(0, BLOCK_SIZE_N)
            P_ptr = P + offset + rm[:, None] * BLOCK_SIZE_N + rn[None, :]
            tl.store(P_ptr, acc)
            tl.atomic_xchg(locks + pid, 0)
    
    # Write final results to output tensor C
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    C_ptr = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(C_ptr, acc, mask=mask)

def spinning_lock(P, C, num_sms, k, BLOCK_SIZE_M=128, BLOCK_SIZE_N=128):
    """
    Wrapper function to launch the spinning lock kernel
    
    Args:
        P: Input tensor for intermediate storage
        C: Output tensor for final results
        num_sms: Number of streaming multiprocessors
        k: Reduction factor
        BLOCK_SIZE_M: Block size for M dimension
        BLOCK_SIZE_N: Block size for N dimension
    """
    M, N = C.shape
    # Initialize locks tensor
    locks = torch.zeros(num_sms, dtype=torch.int32, device='cuda')
    
    # Get strides for the output tensor
    stride_cm = C.stride(0)
    stride_cn = C.stride(1)
    
    # Launch kernel with grid size of num_sms
    grid = (num_sms,)
    spinning_lock_kernel[grid](
        P, C, locks,
        num_sms, k, M, N,
        stride_cm, stride_cn,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )
