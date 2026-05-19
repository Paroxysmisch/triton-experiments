import triton
import triton.language as tl

@triton.jit
def spinning_lock_kernel(P, C, locks, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get the thread ID and block ID
    pid = tl.program_id(axis=0)
    tid = tl.program_id(axis=1)
    
    # Initialize the accumulator
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Each thread processes a segment of the input
    for i in range(tid * BLOCK_SIZE, (tid + 1) * BLOCK_SIZE):
        if i < n_elements:
            acc[tid] += P[i]
    
    # Perform the reduction within the block
    for offset in range(BLOCK_SIZE // 2, 0, -1):
        if tid < offset:
            acc[tid] += acc[tid + offset]
    
    # Only the first thread in the block writes the result to the shared memory
    if tid == 0:
        tl.atomic_cas(locks + pid, 0, 1)  # Lock the block
        for _ in range(9):  # Perform 9 cycles
            while tl.atomic_cas(locks + pid, 1, 1) == 1:  # Spin lock
                pass
            C[pid] += acc[0]  # Perform the addition
            tl.atomic_xchg(locks + pid, 0)  # Unlock the block

import triton
import triton.runtime

def spinning_lock(P, C, locks, block_size, grid_size):
    # Convert inputs to Triton tensors
    P_t = triton.runtime.Tensor(P, dtype=triton.float32)
    C_t = triton.runtime.Tensor(C, dtype=triton.float32)
    locks_t = triton.runtime.Tensor(locks, dtype=triton.int32)
    
    # Launch the kernel
    spinning_lock_kernel[grid_size, block_size](P_t, C_t, locks_t, P_t.shape[0], BLOCK_SIZE=block_size)
    
    # Synchronize to ensure the kernel has finished
    triton.runtime.synchronize()
    
    # Convert the output back to a NumPy array
    return C_t.numpy()

# Example usage
import numpy as np

# Input data
P = np.random.rand(1024).astype(np.float32)
C = np.zeros(32).astype(np.float32)
locks = np.zeros(32).astype(np.int32)

# Parameters
block_size = 32
grid_size = (32, 1)  # 32 blocks, 1 thread per block

# Run the kernel
C_result = spinning_lock(P, C, locks, block_size, grid_size)

print(C_result)
