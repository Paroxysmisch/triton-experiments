triton
import triton
import triton.language as tl

@triton.jit
def _quantize_global_transpose_kernel(
    A_ptr: tl.tensor, B_ptr: tl.tensor,
    stride_am: tl.int32, stride_an: tl.int32,
    stride_bm: tl.int32, stride_bn: tl.int32,
    absmax_inv: tl.float32,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
    M: tl.constexpr, N: tl.constexpr
):
    # Compute the 2D grid indices
    pid = tl.program_id(axis=0)
    pid_m = pid // (M // GROUP_M)
    pid_n = pid % (M // GROUP_M)

    # Compute the 2D block indices
    bid = tl.block_id(axis=0)
    bidx_m = bid // (BLOCK_M // GROUP_M)
    bidx_n = bid % (BLOCK_M // GROUP_M)

    # Compute the 2D thread indices
    tid_m = tl.thread_id(axis=0)
    tid_n = tl.thread_id(axis=1)

    # Compute the global indices
    i = pid_m * BLOCK_M + bidx_m * GROUP_M + tid_m
    j = pid_n * BLOCK_N + bidx_n * GROUP_M + tid_n

    # Ensure the indices are within bounds
    if i < M and j < N:
        # Load the value from A
        a = tl.load(A_ptr + i * stride_am + j * stride_an)

        # Quantize and scale to int8
        q = tl.round(a * absmax_inv)
        q = tl.clamp(q, -128, 127)

        # Store the quantized value in B, transposed
        tl.store(B_ptr + j * stride_bm + i * stride_bn, q)
