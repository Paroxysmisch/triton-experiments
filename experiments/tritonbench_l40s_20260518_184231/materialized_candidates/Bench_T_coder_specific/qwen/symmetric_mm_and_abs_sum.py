import triton
import triton.language as tl

@triton.jit
def symmetric_mm_and_abs_sum_kernel(
    A_ptr, A_stride_n, A_stride_m,
    C_ptr, C_stride_n, C_stride_m,
    alpha, beta,
    n, m, BLOCK_SIZE=32):
    # Define shared memory pointers
    s_A = tl.shared.array(shape=(BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    s_C = tl.shared.array(shape=(BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    # Get the current block indices
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)

    # Compute the global indices within the block
    i = pid_i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    j = pid_j * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Initialize accumulators for A and C
    acc_A = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    acc_C = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    # Load data into shared memory
    if i < n and j < m:
        a = tl.load(A_ptr + i * A_stride_n + j * A_stride_m)
        s_A[i % BLOCK_SIZE, j % BLOCK_SIZE] = a
        if pid_i == pid_j:
            c = tl.load(C_ptr + i * C_stride_n + j * C_stride_m)
            s_C[i % BLOCK_SIZE, j % BLOCK_SIZE] = c

    # Synchronize threads to ensure all loads are completed
    tl.barrier()

    # Perform the matrix multiplication and accumulation
    for k in range(m // BLOCK_SIZE):
        a = s_A[:, :]
        b = tl.load(A_ptr + i * A_stride_n + (k * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)) * A_stride_m)
        acc_A += a @ b.T
        tl.barrier()
        if pid_i == pid_j:
            c = s_C[:, :]
            d = tl.load(C_ptr + i * C_stride_n + (k * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)) * C_stride_m)
            acc_C += c @ d.T
        tl.barrier()

    # Write back to global memory
    if i < n and j < m:
        if pid_i == pid_j:
            tl.store(C_ptr + i * C_stride_n + j * C_stride_m, alpha * acc_A + beta * acc_C)
        else:
            tl.store(C_ptr + i * C_stride_n + j * C_stride_m, alpha * acc_A)

# Function to calculate the sum of absolute values
@triton.jit
def sum_of_abs_values_kernel(
    C_ptr, C_stride_n, C_stride_m,
    n, m, BLOCK_SIZE=32):
    # Define shared memory pointer
    s_C = tl.shared.array(shape=(BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    # Get the current block index
    pid = tl.program_id(0)

    # Compute the global index within the block
    i = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Initialize accumulator for the sum of absolute values
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Load data into shared memory
    if i < n:
        c = tl.load(C_ptr + i * C_stride_n)
        s_C[i % BLOCK_SIZE] = c

    # Synchronize threads to ensure all loads are completed
    tl.barrier()

    # Accumulate the sum of absolute values
    for j in range(n // BLOCK_SIZE):
        c = s_C[:]
        acc += tl.abs(c)
        tl.barrier()

    # Write back to global memory
    if pid == 0:
        tl.store(C_ptr, acc[0])
