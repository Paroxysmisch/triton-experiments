import triton
import triton.language as tl

@triton.jit
def _quantize_global_transpose(A, stride_am, stride_an, B, stride_bm, stride_bn, absmax_inv, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
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

    # Quantize the data
    quantized = a * absmax_inv
    quantized = tl.libdevice.rint(quantized)  # Round to nearest integer
    quantized = tl.clamp(quantized, -128, 127).to(tl.int8)  # Clamp to int8 range

    # Store transposed result in B
    tl.store(B + offsets_bm[:, None] * stride_bm + offsets_bn[None, :] * stride_bn, quantized)

def quantize_global_transpose(A, B, stride_am, stride_an, stride_bm, stride_bn):
    # Define constants
    BLOCK_M = 128
    BLOCK_N = 128

    # Calculate absmax and its inverse
    absmax = tl.abs(A).max()
    absmax_inv = 1.0 / absmax

    # Calculate grid dimensions
    grid_m = (A.shape[0] + BLOCK_M - 1) // BLOCK_M
    grid_n = (A.shape[1] + BLOCK_N - 1) // BLOCK_N

    # Launch the Triton kernel
    _quantize_global_transpose[(grid_m, grid_n)](A, stride_am, stride_an, B, stride_bm, stride_bn, absmax_inv, BLOCK_M, BLOCK_N)
