import triton
import triton.language as tl

@triton.jit
def _quantize_global_transpose(A, B, absmax_inv_ptr, stride_am, stride_an, stride_bm, stride_bn, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Program ID
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Create offsets for memory access
    offsets_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_an = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_bm = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_bn = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    # Load data from A
    a = tl.load(A + offsets_am[:, None] * stride_am + offsets_an[None, :] * stride_an)

    # Load the inverse of the maximum absolute value
    absmax_inv = tl.load(absmax_inv_ptr)

    # Quantize A
    quantized_a = tl.cast(a * absmax_inv, tl.int8)

    # Transpose and store the result in B
    tl.store(B + offsets_bm[:, None] * stride_bm + offsets_bn[None, :] * stride_bn, quantized_a)

def quantize_global_transpose(a, b, absmax_inv, stride_am, stride_an, stride_bm, stride_bn):
    # Define constants
    BLOCK_M = 128
    BLOCK_N = 128

    # Calculate grid dimensions
    grid_m = (a.shape[0] + BLOCK_M - 1) // BLOCK_M
    grid_n = (a.shape[1] + BLOCK_N - 1) // BLOCK_N

    # Launch the Triton kernel
    _quantize_global_transpose[(grid_m, grid_n)](a, b, absmax_inv, stride_am, stride_an, stride_bm, stride_bn, BLOCK_M, BLOCK_N)
