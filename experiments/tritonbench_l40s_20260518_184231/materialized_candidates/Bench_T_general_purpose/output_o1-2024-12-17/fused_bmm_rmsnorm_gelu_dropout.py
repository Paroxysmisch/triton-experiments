import math
import torch
import triton
import triton.language as tl


# --------------------------------------------------------------------------------
# Kernel 1: Batched Matrix Multiplication
# --------------------------------------------------------------------------------
@triton.jit
def _bmm_kernel(
    # Pointers
    A_ptr, B_ptr, C_ptr,
    # Dimensions
    B, N, M, P,
    stride_ab, stride_an, stride_am,
    stride_bb, stride_bm, stride_bp,
    stride_cb, stride_cn, stride_cp,
    # Meta-parameters
    BLOCK_M: tl.constexpr,  # rows of output tile (accumulator)
    BLOCK_N: tl.constexpr,  # cols of output tile (accumulator)
    BLOCK_K: tl.constexpr   # shared dimension
):
    """
    Each program instance computes a [BLOCK_M x BLOCK_N] tile of the 
    batch-matrix multiplication: C = A x B.
    Shapes:
      A: [B, N, M]
      B: [B, M, P]
      C: [B, N, P]
    """
    # Program ID: we have (grid_b, grid_m, grid_n).
    bid = tl.program_id(0)  # batch dimension
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    # Compute the row/col offsets for the block
    # in the output C (and in A, B).
    row_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    b_offset = bid

    # Create pointers for the output block
    # We use a mask to guard out-of-bounds.
    mask_m = row_offs < N
    mask_n = col_offs < P

    # Output pointer
    C_ptrs = C_ptr + (b_offset * stride_cb
                      + row_offs[:, None] * stride_cn
                      + col_offs[None, :] * stride_cp)

    # Initialize accumulator 
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # k-range: we loop over the "M" dimension for partial sums
    # We go in steps of BLOCK_K
    for k_start in range(0, M, BLOCK_K):
        # Offsets for A
        k_offs = k_start + tl.arange(0, BLOCK_K)
        mask_k = k_offs < M

        A_ptrs = A_ptr + (b_offset * stride_ab
                          + row_offs[:, None] * stride_an
                          + k_offs[None, :] * stride_am)

        # Offsets for B
        B_ptrs = B_ptr + (b_offset * stride_bb
                          + k_offs[:, None] * stride_bm
                          + col_offs[None, :] * stride_bp)

        # Load A and B tiles
        a = tl.load(A_ptrs, mask=(mask_m[:, None] & mask_k[None, :]), other=0.0)
        b = tl.load(B_ptrs, mask=(mask_k[:, None] & mask_n[None, :]), other=0.0)
        # Accumulate
        acc += tl.dot(a, b)

    # Write output tile
    c = acc.to(tl.float16)
    tl.store(C_ptrs, c, mask=(mask_m[:, None] & mask_n[None, :]))


# --------------------------------------------------------------------------------
# Kernel 2: RMSNorm + GELU + Dropout
# --------------------------------------------------------------------------------
@triton.jit
def _rmsnorm_gelu_dropout_kernel(
    # Pointers
    C_ptr,  # input: (B, N, P)
    O_ptr,  # output: (B, N, P)
    # Leading strides
    stride_cb, stride_cn, stride_cp,
    stride_ob, stride_on, stride_op,
    # Normalization + dropout parameters
    eps, p, training, seed,
    N, P,
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr
):
    """
    Applies, in-place for each b:
      1) RMSNorm
      2) GELU
      3) Dropout
    across the last two dimensions (N, P).
    """
    bid = tl.program_id(0)
    # Flatten the (N, P) dimension into a single dimension for simpler block coverage:
    idx = tl.arange(0, BLOCK_SIZE)
    offset = bid * (N * P) + idx

    # Out-of-bounds mask
    mask = offset < (N * P)

    # Load input
    c_ptrs = C_ptr + (offset // P) * stride_cn + (offset % P) * stride_cp + (bid * stride_cb * 0)
    vals = tl.load(c_ptrs, mask=mask, other=0.0).to(tl.float32)

    # Compute sum of squares across entire (N * P) for each batch in parallel
    # We can do a blockwise partial reduction, but each block sees only BLOCK_SIZE elements.
    # We'll use atomic or group indexing. Simpler approach: do a two-pass approach.
    # This kernel is for demonstration, so we'll do a naive approach with partial sums.

    # 1) compute partial sum of squares in registers
    sq = vals * vals
    partial_sq_sum = tl.sum(sq, axis=0)

    # Let one thread store the partial result in a buffer
    # We'll store partial sums in a buffer sized by the grid dimension if needed
    # For simplicity, let's just local reduce if BLOCK_SIZE == N*P on a single program_id(0).
    # This won't scale to large N*P but is simpler for demonstration.
    # A real-world scenario would require a parallel segmented reduction or a second pass.
    # We'll assume one block covers the entire (N*P) dimension for demonstration.
    # This means BLOCK_SIZE >= N*P in practice.

    # broadcast partial sum to all
    sq_sum = tl.sum(tl.broadcast_to(sq, [BLOCK_SIZE]), axis=0)
    # We'll do that only once if program_id(0) = 0, but here we assume one block covers everything.
    # Then compute the norm
    mean_sq = sq_sum / float(N * P)
    denom = tl.sqrt(mean_sq + eps)

    # RMSNorm
    vals = vals / denom

    # GELU
    # approximate='none' or approximate='tanh'
    # We'll assume 'none' if approximate=0, 'tanh' if approximate=1
    # We'll interpret 'training' high bits to choose approximation or not, just for demonstration.
    # Real code would pass an additional bool or so. Here we keep it simple.
    # We'll treat 'training' only for dropout. For approximate, let's do 'none
