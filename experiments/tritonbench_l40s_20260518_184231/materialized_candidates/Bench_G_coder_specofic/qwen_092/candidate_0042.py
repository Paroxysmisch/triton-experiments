import triton
import triton.language as tl

# Triton kernel for backward pass of batched matrix multiplication with chunking
@triton.jit(
    config={
        "stages": 2,
        "warps": 4,
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 32,
        "BLOCK_SIZE_CS": 16,
    },
    num_warps=4,
    num_stages=2,
)
def _bmm_chunk_bwd_kernel(
    a_ptr, dout_ptr, db_ptr, res_ptr, stride_a_batch, stride_a_m, stride_a_n,
    stride_dout_csize_m, stride_dout_csize_n, stride_dout_batch, stride_db_batch,
    stride_db_csize_m, stride_db_csize_n, batch_size, m, n, k, has_residual,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_CS: tl.constexpr,
):
    # Matrix indices
    pid = tl.program_id(axis=0)
    grid_size = batch_size * m * n
    pid = pid % grid_size
    b = pid // (m * n)
    i = (pid // n) % m
    j = pid % n

    # Load a, dout, and initialize db
    a = tl.load(a_ptr + b * stride_a_batch + i * stride_a_m + j * stride_a_n)
    dout = tl.load(dout_ptr + b * stride_dout_batch + i * stride_dout_csize_m + j * stride_dout_csize_n)
    db = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Load residual if available
    if has_residual:
        res = tl.load(res_ptr + b * stride_a_batch + i * stride_a_m + j * stride_a_n)
        a = a + res

    # Compute partial dot product
    for s in range(0, k, BLOCK_SIZE_CS):
        a_block = tl.load(a_ptr + b * stride_a_batch + i * stride_a_m + s * stride_a_n, mask=s < k, other=0.0)
        dout_block = tl.load(dout_ptr + b * stride_dout_batch + i * stride_dout_csize_m + s * stride_dout_csize_n, mask=s < k, other=0.0)
        db += tl.dot(a_block, dout_block.T)

    # Store db
    tl.store(db_ptr + b * stride_db_batch + i * stride_db_csize_m + j * stride_db_csize_n, db, mask=True)

# Auxiliary function to prepare tensors and invoke the Triton kernel
def _bmm_chunk_bwd(a, dout, db, res=None, has_residual=False):
    # Ensure tensors are contiguous and appropriately shaped
    a = a.contiguous()
    dout = dout.contiguous()
    db = db.contiguous()

    # Determine grid size
    batch_size, m, n, k = a.shape
    grid_size = batch_size * m * n

    # Set strides
    stride_a_batch = a.stride(0)
    stride_a_m = a.stride(1)
    stride_a_n = a.stride(2)
    stride_dout_csize_m = dout.stride(1)
    stride_dout_csize_n = dout.stride(2)
    stride_dout_batch = dout.stride(0)
    stride_db_batch = db.stride(0)
    stride_db_csize_m = db.stride(1)
    stride_db_csize_n = db.stride(2)

    # Ensure tensors are of the same type
    if res is not None:
        res = res.contiguous().type_as(a)
        stride_res_batch = res.stride(0)
        stride_res_m = res.stride(1)
        stride_res_n = res.stride(2)

    # Invoke the Triton kernel
    if has_residual:
        _bmm_chunk_bwd_kernel[grid_size, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_CS](
            a_ptr=a.data_ptr(), dout_ptr=dout.data_ptr(), db_ptr=db.data_ptr(), res_ptr=res.data_ptr(),
            stride_a_batch=stride_a_batch, stride_a_m=stride_a_m, stride_a_n=stride_a_n,
            stride_dout_csize_m=stride_dout_csize_m, stride_dout_csize_n=stride_dout_csize_n,
            stride_dout_batch=stride_dout_batch, stride_db_batch=stride_db_batch,
            stride_db_csize_m=stride_db_csize_m, stride_db_csize_n=stride_db_csize_n,
            batch_size=batch_size, m=m, n=n, k=k, has_residual=True,
        )
    else:
        _bmm_chunk_bwd_kernel[grid_size, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_CS](
            a_ptr=a.data_ptr(), dout_ptr=dout.data_ptr(), db_ptr=db.data_ptr(), res_ptr=None,
            stride_a_batch=stride_a_batch, stride_a_m=stride_a_m, stride_a_n=stride_a_n,
            stride_dout_csize_m=stride_dout_csize_m, stride_dout_csize_n=stride_dout_csize_n,
            stride_dout_batch=stride_dout_batch, stride_db_batch=stride_db_batch,
            stride_db_csize_m=stride_db_csize_m, stride_db_csize_n=stride_db_csize_n,
            batch_size=batch_size, m=m, n=n, k=k, has_residual=False,
        )

    return db
