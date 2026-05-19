import torch
import triton
import triton.language as tl

# -------------------------------------
# Kernel: Batched Matrix Multiplication
# -------------------------------------
@triton.jit
def _bmm_kernel(
    A_ptr, B_ptr, C_ptr,
    B, N, M, P,
    stride_a_b, stride_a_n, stride_a_m,
    stride_b_b, stride_b_m, stride_b_p,
    stride_c_b, stride_c_n, stride_c_p,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    """
    Computes C = A @ B for each batch in [B].
    A is [B, N, M], B is [B, M, P], C is [B, N, P].
    """
    pid = tl.program_id(0)
    # batch index
    bid = tl.program_id(1)

    # block indices along output dimensions
    block_n = pid // tl.num_programs(2)
    block_m = pid % tl.num_programs(2)

    # Starting indices for each block
    row = block_n * BLOCK_N
    col = block_m * BLOCK_M

    # Pointer offsets for each batch
    A_batch = A_ptr + bid * stride_a_b
    B_batch = B_ptr + bid * stride_b_b
    C_batch = C_ptr + bid * stride_c_b

    # Accumulator for partial sums
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)

    # Loop over K dimension in chunks of BLOCK_K
    for k in range(0, M, BLOCK_K):
        # Load a block of A
        a_row = row + tl.arange(0, BLOCK_N)
        a_col = k + tl.arange(0, BLOCK_K)
        a_mask = (a_row < N)[:, None] & (a_col < M)[None, :]
        A_tile = tl.load(
            A_batch + a_row[:, None] * stride_a_n + a_col[None, :] * stride_a_m,
            mask=a_mask,
            other=0.0
        )

        # Load a block of B
        b_row = k + tl.arange(0, BLOCK_K)
        b_col = col + tl.arange(0, BLOCK_M)
        b_mask = (b_row < M)[:, None] & (b_col < P)[None, :]
        B_tile = tl.load(
            B_batch + b_row[:, None] * stride_b_m + b_col[None, :] * stride_b_p,
            mask=b_mask,
            other=0.0
        )

        # Accumulate
        acc += tl.dot(A_tile, B_tile)

    # Write out result
    c_row = row + tl.arange(0, BLOCK_N)
    c_col = col + tl.arange(0, BLOCK_M)
    c_mask = (c_row < N)[:, None] & (c_col < P)[None, :]
    tl.store(
        C_batch + c_row[:, None] * stride_c_n + c_col[None, :] * stride_c_p,
        acc,
        mask=c_mask
    )


# --------------------------------------------------
# Kernel: RMSNorm + GELU + (optional) Dropout in one
# --------------------------------------------------
@triton.jit
def _rmsnorm_gelu_dropout_kernel(
    X_ptr,  # [B, N, P] after bmm
    Y_ptr,  # output pointer of same shape
    B, N, P,
    eps, p, training, approximate,
    stride_x_b, stride_x_n, stride_x_p,
    stride_y_b, stride_y_n, stride_y_p,
    BLOCK_SIZE: tl.constexpr
):
    """
    For each row in [B, N], compute RMS norm across P dimension,
    then apply GELU, then apply dropout if training=True.
    """
    b_idx = tl.program_id(0)
    n_idx = tl.program_id(1)

    row_offsets = b_idx * stride_x_b + n_idx * stride_x_n
    out_offsets = b_idx * stride_y_b + n_idx * stride_y_n

    # Each program processes a slice of length BLOCK_SIZE in the last dimension
    offset = tl.arange(0, BLOCK_SIZE)
    x_ptrs = X_ptr + row_offsets + offset * stride_x_p
