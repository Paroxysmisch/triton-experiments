import triton
import triton.language as tl
import torch
import math

# -----------------------------
# Triton Kernel: Partial Sum
# -----------------------------
# This kernel applies GELU to each element, then accumulates the partial sum and
# partial sum of squares in block-local reductions. The final partial sums are
# stored in partial_sum_ptr and partial_sq_sum_ptr, to be combined later in Python.
@triton.jit
def _gelu_std_partial_kernel(
    input_ptr,                 # *float32, input data
    partial_sum_ptr,           # *float32, partial sum output
    partial_sq_sum_ptr,        # *float32, partial sum of squares output
    N,                         # total number of elements to process
    CORRECTION,                # correction factor for later adjustment
    USE_TANH_APPROX: tl.constexpr,  # bool-like for approximate='tanh'
    BLOCK_SIZE: tl.constexpr
):
    # Program ID for 1D grid
    pid = tl.program_id(0)
    # Compute the block start
    block_start = pid * BLOCK_SIZE
    # Offsets for each thread within the block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to guard memory operations
    mask = offsets < N

    # Load input under the mask. Elements beyond N are set to 0.
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    # GELU activation
    # exact: gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))         (approximate='none')
    # tanh:  gelu(x) = 0.5 * x * [1 + tanh( sqrt(2/pi)*(x + 0.044715*x^3) )] (approximate='tanh')
    if USE_TANH_APPROX:
        # tanh-based approximation
        # factor = sqrt(2/pi)
        factor = 0.79788456  # approximate sqrt(2.0 / math.pi)
        y = x + 0.044715 * (x * x * x)
        y = factor * y
        gelu_x = 0.5 * x * (1.0 + tl.tanh(y))
    else:
        # "none" - exact form using erf
        # erf is approximated in Triton as well, but we use tl.erf
        inv_sqrt2 = 0.70710678  # approximate 1 / sqrt(2)
        y = x * inv_sqrt2
        gelu_x = 0.5 * x * (1.0 + tl.erf(y))

    # Each thread stores local values for sum and sum of squares
    local_sum = gelu_x
    local_sq_sum = gelu_x * gelu_x

    # -----------------------------
    # Block Reduction: local_sum and local_sq_sum
    # -----------------------------
    # We'll do a simple parallel reduction within the block. The block size is
    # assumed to be a power of two for simplicity.

    # Step 1: local buffers
    # Occupy registers with local_sum and local_sq_sum for each thread
    # We'll reduce in a tree fashion. Each step halves the number of active threads.

    def block_reduce(value):
        # Typical Triton 1D block reduce pattern for BLOCK_SIZE power-of-two
        # We'll unroll manually for simplicity
        # e.g. if 1024, reduce in log2(1024)=10 steps
        # "value" is a scalar per thread
        # We'll do warp-synch with "tl.multiple_of" or just rely on correct hardware behavior
        # Each half of the threads adds to the other half
        # We'll do repeated merges: 512 merges 512, 256 merges 256, etc.
        size = BLOCK_SIZE
        stride = size // 2
        while stride > 0:
            # Butterfly add
            other = tl.shift_right(value, stride)
            value = value + other
            stride //= 2
        return value

    # Because we hold local_sum in each lane, we need to ensure we shift gather from "value" but within same warp.
    # However, for simplicity in a demonstration, let's store them in a local array and do repeated half merges.

    # We'll do a fixed approach only if BLOCK_SIZE <= 1024 and is a power of two.
    # Let's do it in code steps for clarity:

    # Merge step 1: index i merges with i + 512
    # Merge step 2: i merges with i + 256
    # ...
    # We'll do it iteratively. We'll rely on each lane's thread idx.

    # For the shift, we need an index. We can do so by:
    lane_id = tl.arange(0, BLOCK_SIZE)
    # We only do merges if lane_id < stride at each step
    # We'll define a local function to do half merges:

    # We'll define a small function that merges at a given stride
    def merge_for_stride(val, stride):
        # threads that are in the lower half (lane_id < stride) add the top half's data
        mask_merge = lane_id < stride
        top = tl.shift_right(val, stride)
        new_val = tl.where(mask_merge, val + top, val)
        return new_val

    # We'll do the loop for local_sum and local_sq_sum
    for step in [512, 256, 128, 64, 32, 16, 8, 4, 2, 1]:
        if BLOCK_SIZE >= step * 2:
            local_sum = merge_for_stride(local_sum, step)
            local_sq_sum = merge_for_stride(local_sq_sum, step)

    # Now local_sum, local_sq_sum in lane_id=0 (lowest lane) contain the block's partial sums
    # We'll have only lane 0 in each block store the result
    if tl.thread_id_x() == 0:
        tl.store(partial_sum_ptr + pid, local_sum, mask=True)
        tl.store(partial_sq_sum_ptr + pid, local_sq_sum, mask=True)

# -----------------------------------------
# Python Wrapper: gelu_std(...)
# -----------------------------------------
def gelu_std
