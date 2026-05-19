import triton
import triton.language as tl

# ------------------------------------------------------------------------------
# Triton kernel for computing the backward pass of a batched matrix multiplication
# operation with chunking. The result is written into db_ptr.
# ------------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config(
            {
                'BLOCK_SIZE_M': 64,
                'BLOCK_SIZE_N': 64,
                'BLOCK_SIZE_CS': 32
            },
            num_stages=2,
            num_warps=4
        ),
        triton.Config(
            {
                'BLOCK_SIZE_M': 128,
                'BLOCK_SIZE_N': 64,
                'BLOCK_SIZE_CS': 32
            },
            num_stages=2,
            num_warps=8
        ),
        triton.Config(
            {
                'BLOCK_SIZE_M': 64,
                'BLOCK_SIZE_N': 128,
                'BLOCK_SIZE_CS': 32
            },
            num_stages=3,
            num_warps=8
        ),
    ],
    key=['M', 'N', 'CS']
)
@triton.jit
def _bmm_chunk_bwd_kernel(
    a_ptr, dout_ptr, res_ptr, db_ptr,
    stride_a_batch, stride_a_m, stride_a_k,
    stride_dout_batch, stride_dout_csize_m, stride_dout_csize_n,
    stride_res_batch, stride_res_m, stride_res_n,
    stride_db_batch, stride_db_m0, stride_db_m1,
    BATCH: tl.constexpr,
    M: tl.constexpr,  # total rows of A (and DB)
    N: tl.constexpr,  # total columns of DOut (and DB)
    K: tl.constexpr,  # total columns of A / rows of DOut
    CS: tl.constexpr,  # chunk size in the K dimension
    BLOCK_SIZE_M: tl.constexpr,  # block size along M dimension
    BLOCK_SIZE_N: tl.constexpr,  # block size along N dimension
    BLOCK_SIZE_CS: tl.constexpr, # block size along chunked dimension
    HAS_RESIDUAL: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    batch_id = tl.program_id(2)

    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * BLOCK_SIZE_N
    # Each chunk processes a subset of the total K dimension
    chunk_start = 0

    # Create a 0-initialized accumulator for partial sums
    accum = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over the chunked K dimension in steps of BLOCK_SIZE_CS
    while chunk_start < CS:
        k_offset = chunk_start * BLOCK_SIZE_CS
        k_size = tl.min(BLOCK_SIZE_CS, CS - chunk_start * BLOCK_SIZE_CS)

        # Offsets for loads
        a_offset = batch_id * stride_a_batch \
                   + (m_start + tl.arange(0, BLOCK_SIZE_M))[:, None] * stride_a_m \
                   + (k_offset + tl.arange(0, k_size))[None, :] * stride_a_k

        dout_offset = batch_id * stride_dout_batch \
                      + (m_start + tl.arange(0, BLOCK_SIZE_M))[:, None] * stride_dout_csize_m \
                      + (n_start + tl.arange(0, BLOCK_SIZE_N))[None, :] * stride_dout_csize_n

        # Load data; mask out-of-bounds
        a_mask = (m_start + tl.arange(0, BLOCK_SIZE_M)) < M
        a_mask &= (k_offset + tl.arange(0, k_size)) < K
        dout_mask = (m_start + tl.arange(0, BLOCK_SIZE_M)) < M
        dout_mask &= (n_start + tl.arange(0, BLOCK_SIZE_N)) < N

        a_curr = tl.load(a_ptr + a_offset, mask=a_mask[:, None], other=0.0)
        dout_curr = tl.load(dout_ptr + dout_offset, mask=dout_mask[:, None], other=0.0)

        # Dot product
        accum += tl.dot(a_curr, dout_curr)

        chunk_start += 1

    # Optionally add residual
    if HAS_RESIDUAL:
        res_offset = batch_id * stride_res_batch \
                     + (m_start + tl.arange(0, BLOCK_SIZE_M))[:, None] * stride_res_m \
                     + (n_start + tl.arange(0, BLOCK_SIZE_N))[None, :] * stride_res_n
        res_mask = (m_start + tl.arange(0, BLOCK_SIZE_M)) < M
        res_mask &= (n_start + tl.arange(0, BLOCK_SIZE_N)) < N
        residual = tl.load(res_ptr + res_offset, mask=res_mask, other=0.0)
        accum += residual

    # Store the result
    db_offset = batch_id * stride_db_batch \
                + (m_start + tl.arange(0, BLOCK_SIZE_M))[:, None] * stride_db_m0 \
                + (n_start + tl.arange(0, BLOCK_SIZE_N))[None, :] * stride_db_m1
    db_mask = (m_start + tl.arange(0, BLOCK_SIZE_M)) < M
    db_mask &= (n_start + tl.arange(0, BLOCK_SIZE_N)) < N
    tl.store(db_ptr + db_offset, accum, mask=db_mask)


def _bmm_chunk_bwd(a, dout, res=None):
    """
    Wrapper function preparing tensors and launching _bmm_chunk_bwd_kernel.
    """
    import math
    # Determine shapes
    B, M, K = a.shape
    _, M_, N = dout.shape
    assert M == M_, "M dimension mismatch between 'a' and 'dout'"
    if res is not None:
        assert res.shape == dout.shape, "Residual must match the shape of 'dout'"

    # For demonstration, assume chunk_size = K
    chunk_size = K
    grid_m = math.ceil(M / 64)  # heuristics
    grid_n = math.ceil(N / 64)  # heuristics
    # We launch one CTA per batch in the third dimension
    grid = (grid_m, grid_n, B)

    # Strides
    a_contiguous = a.contiguous()
    dout_contiguous = dout.contiguous()
    if res is not None:
        res_contiguous = res.contiguous()
    else:
        # Placeholder empty tensor if no residual
        import torch
        res_contiguous = torch.zeros_like(dout_contiguous)
    db = a.new_empty(B, M, N)

    stride_a_batch = a_contiguous.stride(0)
    stride_a_m = a_contiguous.stride(1)
    stride_a_k = a_contiguous.stride(2)

    stride_dout_batch = dout_contiguous.stride(0)
    stride_dout_csize_m = dout_contiguous.stride(1)
    stride_dout_csize_n = dout_contiguous.stride(2)

    stride_res_batch = res_contiguous.stride(0)
    stride_res_m = res_contiguous.stride(1)
    stride_res_n = res_contiguous.stride(2)

    stride_db_batch = db.stride(0)
    stride_db_m0 = db.stride(1)
    stride_db_m1 = db.stride(2)

    HAS_RESIDUAL = (res is not None)

    _bmm_chunk_bwd_kernel[grid](
        a_contiguous, dout_contiguous, res_contiguous if HAS_RESIDUAL else res_contiguous,
        db,
        stride_a_batch, stride_a_m, stride_a_k,
        stride_dout_batch, stride_dout_csize_m, stride_dout_csize_n,
        stride_res_batch, stride_res_m, stride_res_n,
        stride_db_batch, stride_db_m0, stride_db_m1,
        B, M, N, K, chunk_size,
        BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_CS=32,
        HAS_RESIDUAL=HAS_RESIDUAL
    )
    return db
