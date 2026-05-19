@triton.jit
def _bmm_chunk_bwd_kernel(
    a_ptr,
    dout_ptr,
    db_ptr,
    res_ptr,
    stride_a_batch,
    stride_dout_csize_m,
    stride_db_batch,
    stride_res_batch,
    batch_size,
    chunk_size,
    grid_size,
    block_size,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    BLOCK_SIZE_CS,
    DTYPE,
    ptx=False
):
    # Kernel code here
    pass

def _bmm_chunk_bwd(
    a,
    dout,
    db,
    res,
    stride_a_batch,
    stride_dout_csize_m,
    stride_db_batch,
    stride_res_batch,
    batch_size,
    chunk_size,
    grid_size,
    block_size,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    BLOCK_SIZE_CS,
    DTYPE
):
    # Wrapper code here
    pass
