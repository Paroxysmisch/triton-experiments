import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_CS': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_CS': 32}, num_stages=4, num_warps=4),
        # ... other configs ...
    ],
    key=['chunk_size', 'K'],
)
@triton.jit
def _bmm_chunk_bwd_kernel(
    a_ptr, dout_ptr, db_ptr, res_ptr,
    seqlen, chunk_size, K, ngroups,
    stride_a_batch, stride_a_seqlen, stride_a_head, stride_ak,
    stride_dout_batch, stride_dout_chunk, stride_dout_head, stride_dout_csize_m, stride_dout_csize_n,
    stride_db_batch, stride_db_seqlen, stride_db_head, stride_db_k,
    stride_res_batch, stride_res_seqlen, stride_res_head, stride_res_k,
    dot_dtype: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_CS: tl.constexpr,
):
    # Get program IDs for the current block
    pid_b = tl.program_id(axis=1)  # Batch dimension
    pid_ch = tl.program_id(axis=2)  # Combined chunk and head dimension
    pid_c = pid_ch // ngroups      # Chunk dimension
    pid_h = pid_ch - pid_c * ngroups  # Head dimension
    
    # Calculate number of blocks in N dimension and get block IDs
    num_pid_n = tl.cdiv(K, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n

    # Calculate base pointers for the current block
    a_ptr += pid_b * stride_a_batch + pid_c * chunk_size * stride_a_seqlen + pid_h * stride_a_head
    dout_ptr += pid_b * stride_dout_batch + pid_c * stride_dout_chunk + pid_h * stride_dout_head

    # Create offset arrays for the block
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_cs = tl.arange(0, BLOCK_SIZE_CS)

    # Calculate pointers for loading data
    dout_ptrs = dout_ptr + (offs_m[:, None] * stride_dout_csize_n + offs_cs[None, :] * stride_dout_csize_m)
    a_ptrs = a_ptr + (offs_cs[:, None] * stride_a_seqlen + offs_n[None, :] * stride_ak)
    
    # Handle boundary conditions
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Main computation loop
    for cs in range(0, tl.cdiv(chunk_size_limit, BLOCK_SIZE_CS)):
        # Load data with masks for boundary checking
        dout = tl.load(dout_ptrs, 
                      mask=(offs_m[:, None] < chunk_size) & (offs_cs[None, :] < chunk_size_limit - cs * BLOCK_SIZE_CS),
                      other=0.0).to(dot_dtype)
        a = tl.load(a_ptrs,
                   mask=(offs_cs[:, None] < chunk_size_limit - cs * BLOCK_SIZE_CS) & (offs_n[None, :] < K),
                   other=0.0).to(dot_dtype)
        
        # Perform matrix multiplication
        acc += tl.dot(dout, a)
        
        # Update pointers
        dout_ptrs += BLOCK_SIZE_CS * stride_dout_csize_m
        a_ptrs += BLOCK_SIZE_CS * stride_a_seqlen

    # Handle residual if present
    if HAS_RESIDUAL:
        res_ptr += pid_b * stride_res_batch + pid_c * chunk_size * stride_res_seqlen + pid_h * stride_res_head
        res_ptrs = res_ptr + (offs_m[:, None] * stride_res_seqlen + offs_n[None, :] * stride_res_k)
        res = tl.load(res_ptrs, 
                     mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < K)).to(tl.float32)
        acc += res

    # Store results
    db = acc.to(db_ptr.dtype.element_ty)
    db_ptr += pid_b * stride_db_batch + pid_c * chunk_size * stride_db_seqlen + pid_h * stride_db_head
    db_ptrs = db_ptr + (offs_m[:, None] * stride_db_seqlen + offs_n[None, :] * stride_db_k)
    tl.store(db_ptrs, db, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < K))

def _bmm_chunk_bwd(a, dout, residual=None, out=None):
    """
    Arguments:
        a: (batch, seqlen, k) or (batch, seqlen, ngroups, k)
        dout: (batch, nchunks, chunk_size, chunk_size) or (batch, nchunks, ngroups, chunk_size, chunk_size)
        residual: Optional (batch, seqlen, k) or (batch, seqlen, ngroups, k)
    Returns:
        out: (batch, seqlen, k) or (batch, seqlen, ngroups, k)
    """
    has_groups = a.dim() == 4
    if not has_groups:
        batch, seqlen, k = a.shape
    else:
        batch, seqlen, ngroups, k = a.shape
    
    nchunks, chunk_size = dout.shape[1], dout.shape[-1]
    
    # Ensure contiguous memory layout
    if a.stride(-1) != 1 and a.stride(-2) != 1:
        a = a.contiguous()
    if dout.stride(-1) != 1 and dout.stride(-2) != 1:
        dout = dout.contiguous()
    
    # Handle residual tensor
    if residual is not None:
        if residual.stride(-1) != 1 and residual.stride(1) != 1:
            residual = residual.contiguous()
    
    # Create output tensor if not provided
    if out is None:
        out = torch.empty_like(a)
    
    # Determine dot product dtype
    dot_dtype = (tl.bfloat16 if a.dtype == torch.bfloat16 or dout.dtype == torch.bfloat16
                 else tl.float16 if a.dtype == torch.float16 or dout.dtype == torch.float16
                 else tl.float32)
    
    # Calculate grid dimensions
    grid = lambda META: (
        triton.cdiv(chunk_size, META['BLOCK_SIZE_M']) * triton.cdiv(k, META['BLOCK_SIZE_N']),
        batch,
        nchunks if not has_groups else nchunks * ngroups
    )
    
    # Launch kernel
    with torch.cuda.device(a.device.index):
        _bmm_chunk_bwd_kernel[grid](
            a, dout, out, residual,
            seqlen, chunk_size, k, ngroups if has_groups else 1,
            *get_strides(a, dout, out, residual, has_groups),
            dot_dtype,
            HAS_RESIDUAL=residual is not None
        )
    
    return out
