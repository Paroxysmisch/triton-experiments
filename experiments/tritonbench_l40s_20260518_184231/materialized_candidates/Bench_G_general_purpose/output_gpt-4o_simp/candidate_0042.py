import triton
import triton.language as tl

# Define constants for block sizes
BLOCK_SIZE_M = 128
BLOCK_SIZE_N = 128
BLOCK_SIZE_K = 32

@triton.jit
def _bmm_chunk_bwd_kernel(a_ptr, dout_ptr, db_ptr, res_ptr,
                          M, N, K, CS,
                          stride_am, stride_ak,
                          stride_do_n, stride_do_k,
                          stride_dbm, stride_dbn,
                          stride_resm, stride_resn,
                          HAS_RESIDUAL: tl.constexpr):
    # Program ID for indexing
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute block indices
    m_block_start = pid_m * BLOCK_SIZE_M
    n_block_start = pid_n * BLOCK_SIZE_N

    # Initialize accumulators
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over the K dimension in chunks
    for k in range(0, K, BLOCK_SIZE_K):
        # Load a chunk of matrix 'a'
        a_offset = m_block_start * stride_am + k * stride_ak
        a = tl.load(a_ptr + a_offset + tl.arange(0, BLOCK_SIZE_M)[:, None] * stride_am + tl.arange(0, BLOCK_SIZE_K)[None, :])

        # Load a chunk of matrix 'dout'
        dout_offset = n_block_start * stride_do_n + k * stride_do_k
        dout = tl.load(dout_ptr + dout_offset + tl.arange(0, BLOCK_SIZE_N)[:, None] * stride_do_n + tl.arange(0, BLOCK_SIZE_K)[None, :])

        # Perform dot product and accumulate
        acc += tl.dot(a, dout)

    # Store result in db
    db_offset = m_block_start * stride_dbm + n_block_start * stride_dbn
    tl.store(db_ptr + db_offset + tl.arange(0, BLOCK_SIZE_M)[:, None] * stride_dbm + tl.arange(0, BLOCK_SIZE_N)[None, :], acc)

    # Optionally add residual
    if HAS_RESIDUAL:
        res_offset = m_block_start * stride_resm + n_block_start * stride_resn
        res = tl.load(res_ptr + res_offset + tl.arange(0, BLOCK_SIZE_M)[:, None] * stride_resm + tl.arange(0, BLOCK_SIZE_N)[None, :])
        acc += res
        tl.store(db_ptr + db_offset + tl.arange(0, BLOCK_SIZE_M)[:, None] * stride_dbm + tl.arange(0, BLOCK_SIZE_N)[None, :], acc)

def _bmm_chunk_bwd(a, dout, db, res=None, has_residual=False):
    # Get shapes and strides
    M, K = a.shape
    _, N = dout.shape

    stride_am, stride_ak = a.stride()
    stride_do_n, stride_do_k = dout.stride()
    stride_dbm, stride_dbn = db.stride()

    if has_residual:
        stride_resm, stride_resn = res.stride()
    else:
        stride_resm, stride_resn = 0, 0

    # Define grid size
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))

    # Launch the Triton kernel
    _bmm_chunk_bwd_kernel[grid](
        a, dout, db, res,
        M, N, K, 0,
        stride_am, stride_ak,
        stride_do_n, stride_do_k,
        stride_dbm, stride_dbn,
        stride_resm, stride_resn,
        HAS_RESIDUAL=has_residual
    )
