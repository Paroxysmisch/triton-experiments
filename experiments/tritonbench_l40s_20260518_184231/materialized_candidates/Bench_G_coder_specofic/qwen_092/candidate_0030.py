import triton
import triton.language as tl

@triton.jit
def triton_red_fused_native_layer_norm_no_welford(
    x_ptr: tl.tensor,  # Input data
    in_out_ptr0: tl.tensor,  # Output mean
    in_out_ptr1: tl.tensor,  # Output inverse standard deviation
    in_ptr1: tl.tensor,  # Scale
    in_ptr2: tl.tensor,  # Shift
    xnumel: int,  # Number of elements in x_ptr
    rnumel: int,  # Number of elements in in_ptr1 and in_ptr2
    n: int,  # Number of elements to normalize
    block_size: int,  # Block size for parallel computation
):
    # Define constants
    XBLOCK = block_size
    RBLOCK = block_size

    # Get thread index
    pid = tl.program_id(axis=0)
    tid = tl.program_id(axis=1)
    xnumel_per_block = XBLOCK * RBLOCK
    xnumel_per_thread = xnumel_per_block // XBLOCK

    # Initialize accumulators
    mean_acc = tl.zeros((1,), dtype=tl.float32)
    variance_acc = tl.zeros((1,), dtype=tl.float32)

    # Load data into shared memory
    x_shared = tl.zeros((XBLOCK, RBLOCK), dtype=tl.float32)
    x_shared[tid // RBLOCK, tid % RBLOCK] = tl.load(x_ptr + pid * xnumel + tid)

    # Synchronize threads in the block
    tl.barrier()

    # Compute mean and variance
    for i in range(xnumel_per_thread):
        mean_acc[0] += x_shared[i // RBLOCK, i % RBLOCK]
        variance_acc[0] += x_shared[i // RBLOCK, i % RBLOCK] ** 2

    # Reduce mean and variance across the block
    mean_acc = tl.sum(mean_acc, axis=0)
    variance_acc = tl.sum(variance_acc, axis=0)

    # Normalize and apply scale and shift
    if tid < xnumel:
        x_val = tl.load(x_ptr + pid * xnumel + tid)
        mean = mean_acc / xnumel_per_block
        variance = variance_acc / xnumel_per_block - mean ** 2
        inv_std = tl.rsqrt(variance + 1e-5)
        normalized_val = (x_val - mean) * inv_std * tl.load(in_ptr1 + pid) + tl.load(in_ptr2 + pid)
        tl.store(out_ptr0 + pid * xnumel + tid, normalized_val)

    # Store mean and inverse standard deviation
    if tid < xnumel:
        tl.store(in_out_ptr0 + pid, mean)
        tl.store(in_out_ptr1 + pid, inv_std)
