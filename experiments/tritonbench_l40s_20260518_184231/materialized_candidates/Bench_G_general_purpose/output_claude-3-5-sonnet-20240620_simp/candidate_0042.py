import torch
import triton
import triton.language as tl

# Define the backward kernel
@triton.jit
def _bmm_chunk_bwd_kernel(
    # Pointers to matrices
    a_ptr, dout_ptr, db_ptr, da_ptr, res_ptr,
    # Matrix dimensions
    M, N, K,
    # Batch info and strides
    batch, a_batch_stride, db_batch_stride, dout_batch_stride,
    # Matrix strides
    a_stride_m, a_stride_k,
    db_stride_k, db_stride_n,
    dout_stride_m, dout_stride_n,
    # Output strides
    da_batch_stride, da_stride_m, da_stride_k,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, 
    BLOCK_SIZE_K: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = num_pid_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * num_pid_m
    group_size_m = min(num_pid_m, num_pid_m - first_pid_m)
    pid_m = (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Block pointers
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    
    # Iterate through K dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_idx = k * BLOCK_SIZE_K + offs_k
        # Load dout and b blocks
        dout_block = tl.load(dout_ptr + offs_m[:, None] * dout_stride_m + 
                            k_idx[None, :] * dout_stride_n,
                            mask=offs_m[:, None] < M and k_idx[None, :] < K)
        b_block = tl.load(db_ptr + k_idx[:, None] * db_stride_k + 
                         offs_n[None, :] * db_stride_n,
                         mask=k_idx[:, None] < K and offs_n[None, :] < N)
        
        # Compute matrix multiplication
        acc += tl.dot(dout_block, b_block)

    # Add residual if needed
    if HAS_RESIDUAL:
        res_block = tl.load(res_ptr + offs_m[:, None] * da_stride_m + 
                           offs_k[None, :] * da_stride_k,
                           mask=offs_m[:, None] < M and offs_k[None, :] < K)
        acc += res_block

    # Store result
    mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
    tl.store(da_ptr + offs_m[:, None] * da_stride_m + 
             offs_k[None, :] * da_stride_k, acc, mask=mask)

# Python wrapper function
def _bmm_chunk_bwd(a, dout, db, res=None):
    batch, M, K = a.shape
    _, _, N = db.shape
    
    # Compute strides
    a_batch_stride = a.stride(0)
    db_batch_stride = db.stride(0)
    dout_batch_stride = dout.stride(0)
    
    # Initialize output gradient
    da = torch.empty_like(a)
    da_batch_stride = da.stride(0)
    
    # Configure block sizes
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 16
    
    # Compute grid size
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N) * batch,)
    
    # Launch kernel
    _bmm_chunk_bwd_kernel[grid](
        a.data_ptr(), dout.data_ptr(), db.data_ptr(), da.data_ptr(),
        res.data_ptr() if res is not None else None,
        M, N, K,
        batch, a_batch_stride, db_batch_stride, dout_batch_stride,
        a.stride(1), a.stride(2),
        db.stride(1), db.stride(2),
        dout.stride(1), dout.stride(2),
        da_batch_stride, da.stride(1), da.stride(2),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        HAS_RESIDUAL=res is not None,
    )
    
    return da
