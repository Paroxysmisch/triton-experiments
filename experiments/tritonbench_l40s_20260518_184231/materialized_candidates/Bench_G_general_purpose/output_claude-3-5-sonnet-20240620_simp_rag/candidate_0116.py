import torch
import triton
import triton.language as tl

@triton.jit
def spinning_lock_kernel(
    P_ptr,  # Pointer to temporary storage
    C_ptr,  # Pointer to output
    locks_ptr,  # Pointer to locks array
    num_sms,  # Number of SMs
    k,  # Reduction factor
    M, N,  # Output dimensions
    stride_cm,  # Stride for C rows
    stride_cn,  # Stride for C columns
    BLOCK_SIZE_M: tl.constexpr,  # Block size for rows
    BLOCK_SIZE_N: tl.constexpr,  # Block size for columns
):
    # Get program ID and calculate position
    pid = tl.program_id(0)
    pid_m = pid // N
    pid_n = pid % N
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Main reduction loop
    for _ in range(9):
        if pid % k == 0:
            # Leader thread performs reduction
            for offset in range(1, k):
                next_pid = pid + offset
                if next_pid >= num_sms:
                    break
                    
                # Spin until lock is acquired
                while tl.atomic_cas(locks_ptr + next_pid, 0, 1) != 0:
                    pass
                
                # Load and accumulate from temporary storage
                offsets = (
                    next_pid * BLOCK_SIZE_M * BLOCK_SIZE_N + 
                    tl.arange(0, BLOCK_SIZE_M)[:, None] * BLOCK_SIZE_N + 
                    tl.arange(0, BLOCK_SIZE_N)[None, :]
                )
                temp = tl.load(P_ptr + offsets)
                acc += temp
                
                # Release lock
                tl.atomic_xchg(locks_ptr + next_pid, 0)
        else:
            # Worker threads store to temporary storage
            offsets = (
                pid * BLOCK_SIZE_M * BLOCK_SIZE_N + 
                tl.arange(0, BLOCK_SIZE_M)[:, None] * BLOCK_SIZE_N + 
                tl.arange(0, BLOCK_SIZE_N)[None, :]
            )
            tl.store(P_ptr + offsets, acc)
            
            # Signal completion
            tl.atomic_xchg(locks_ptr + pid, 0)
            
        # Barrier synchronization
        tl.debug_barrier()
    
    # Store final result
    row_idx = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    col_idx = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = (row_idx[:, None] < M) & (col_idx[None, :] < N)
    C_offsets = row_idx[:, None] * stride_cm + col_idx[None, :] * stride_cn
    tl.store(C_ptr + C_offsets, acc, mask=mask)

def spinning_lock(P, C, num_sms, k):
    """
    Wrapper function to launch the spinning lock kernel
    
    Args:
        P: Temporary storage tensor
        C: Output tensor
        num_sms: Number of streaming multiprocessors
        k: Reduction factor
    """
    M, N = C.shape
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    
    # Initialize locks
    locks = torch.zeros(num_sms, dtype=torch.int32, device='cuda')
    
    # Launch kernel
    grid = (num_sms,)
    spinning_lock_kernel[grid](
        P, C, locks,
        num_sms, k,
        M, N,
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
