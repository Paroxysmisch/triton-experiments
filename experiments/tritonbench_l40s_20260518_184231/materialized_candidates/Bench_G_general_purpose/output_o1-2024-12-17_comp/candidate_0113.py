import triton
import triton.language as tl


# -----------------------------------------
# Triton Kernel
# -----------------------------------------
@triton.jit
def triton_red_fused_native_layer_norm_0(
    primals_3_ptr,        # [S, D], float32
    primals_1_ptr,        # [D],     float32 (scale) or None
    primals_2_ptr,        # [D],     float32 (bias)  or None
    out_ptr0,             # [S, 3],  float32 (stores: mean, var, count)
    out_ptr1,             # [S, D],  float32 (final normalized output)
    S,                    # total rows
    D,                    # total cols
    stride_in,            # leading stride for primals_3
    stride_scale,         # stride for scale
    stride_bias,          # stride for bias
    stride_out0,          # stride for out_ptr0
    stride_out1,          # stride for out_ptr1
    BLOCK_SIZE: tl.constexpr,  # size of block in the D dimension
    EPSILON: tl.constexpr      # small constant for numerical stability
):
    # Each program_id(0) corresponds to one row
    row_id = tl.program_id(0)
    # Row pointer offset
    row_offset_in   = row_id * stride_in
    row_offset_out0 = row_id * stride_out0
    row_offset_out1 = row_id * stride_out1

    # Welford accumulation variables
    mean = tl.zeros([1], dtype=tl.float32)
    m2   = tl.zeros([1], dtype=tl.float32)
    wgt  = tl.zeros([1], dtype=tl.float32)

    # ----------------------------------------------------------------
    # PASS 1: Compute mean and variance with Welford in chunks of BLOCK_SIZE
    # ----------------------------------------------------------------
    num_full = (D + BLOCK_SIZE - 1) // BLOCK_SIZE
    for i in range(num_full):
        col_start = i * BLOCK_SIZE
        offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < D

        # Load input chunk
        x_ptrs = primals_3_ptr + row_offset_in + offsets
        x = tl.where(mask, tl.load(x_ptrs, mask=mask, other=0.0), 0.0)

        # Update Welford (per-element)
        # welford algorithm: combine partial sums
        # new_count = old_count + 1
        # delta     = x - old_mean
        # new_mean  = old_mean + delta / new_count
        # m2        = m2 + delta * (x - new_mean)
        for j in range(BLOCK_SIZE):
            xj = x[j]
