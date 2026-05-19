import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_CS': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_CS': 16}, num_stages=4, num_warps=8),
        # Add more configurations as needed
    ],
    key=['M', 'N', 'CS']
)
@triton.jit
def _bmm_chunk_bwd_kernel(a_ptr, dout_ptr, db_ptr, res_ptr,
                          stride_a_batch, stride_a_csize_k, stride_a_csize_m,
                          stride_dout_batch, stride_dout_csize_n, stride_dout_csize_m,
                          stride_db_batch, stride_db_csize_n, stride_db_csize_m,
                          stride_res_batch, stride_res_csize_n, stride_res_csize_m,
                          M, N, CS, HAS_RESIDUAL: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Define block indices
    block_m = pid_m * BLOCK_SIZE_M
    block_n = pid_n * BLOCK_SIZE_N

    # Initialize accumulators
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over CS dimension
    for cs in range(0, CS, BLOCK_SIZE_CS):
        a_offset = block_m * stride_a_csize_m + cs * stride_a_csize_k
        dout_offset = block_n * stride_dout_csize_n + cs * stride_dout_csize_m

        a = tl.load(a_ptr + a_offset)
        dout = tl.load(dout_ptr + dout_offset)

        # Compute partial dot product
        acc += tl.dot(a, dout)

    # Add residual if required
    if HAS_RESIDUAL:
        res_offset = block_m * stride_res_csize_m + block_n * stride_res_csize_n
        res = tl.load(res_ptr + res_offset)
        acc += res

    # Store result
    db_offset = block_m * stride_db_csize_m + block_n * stride_db_csize_n
    tl.store(db_ptr + db_offset, acc)


def _bmm_chunk_bwd(a, dout, res=None):
    # Ensure tensors are contiguous
    a = a.contiguous()
    dout = dout.contiguous()
    if res is not None:
        res = res.contiguous()

    # Extract dimensions
    B, M, K = a.shape
    _, N, _ = dout.shape

    # Determine grid size
    grid = (triton.cdiv(M, 128), triton.cdiv(N, 128))

    # Launch kernel
    _bmm_chunk_bwd_kernel[grid](
        a, dout, res,
        a.stride(0), a.stride(1), a.stride(2),
        dout.stride(0), dout.stride(1), dout.stride(2),
        res.stride(0) if res is not None else 0,
        res.stride(1) if res is not None else 0,
        res.stride(2) if res is not None else 0,
        M, N, K,
        HAS_RESIDUAL=(res is not None)
    )

# Example usage
# a = torch.randn(B, M, K, device='cuda')
# dout = torch.randn(B, N, K, device='cuda')
# res = torch.randn(B, M, N, device='cuda')  # If needed
# _bmm_chunk_bwd(a, dout, res)
