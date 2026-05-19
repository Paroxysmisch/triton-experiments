import triton
import triton.language as tl
import torch

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_CS': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_CS': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_CS': 32}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'CS', 'BATCH'],
)
@triton.jit
def _bmm_chunk_bwd_kernel(
    # Pointers to matrices
    a_ptr, dout_ptr, db_ptr, res_ptr,
    # Matrix dimensions
    M, N, CS, BATCH,
    # Strides for accessing memory
    stride_a_batch, stride_a_cs_m,
    stride_dout_batch, stride_dout_csize_m, stride_dout_n,
    stride_db_batch, stride_db_cs_n,
    stride_res_batch, stride_res_cs_n,
    # Block sizes for the kernel
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, 
    BLOCK_SIZE_CS: tl.constexpr,
    # Whether to add residual
    HAS_RESIDUAL: tl.constexpr,
):
    """
    Computes backward pass for chunked batched matrix multiplication
    db = a @ dout (with optional residual)
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = num_pid_m * num_pid_n
    group_id = pid // num_pid_in_group
    pid_in_group = pid % num_pid_in_group
    pid_m = pid_in_group // num_pid_n
    pid_n = pid_in_group % num_pid_n

    # Block pointers
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_CS)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Iterate through chunks
    for k in range(0, CS, BLOCK_SIZE_CS):
        k_remaining = tl.minimum(BLOCK_SIZE_CS, CS - k)
        
        # Load chunks from A and dout
        a = tl.load(a_ptr + group_id * stride_a_batch + 
                   (k + offs_k[:k_remaining])[:, None] * stride_a_cs_m +
                   offs_am[None, :], mask=(offs_am[None, :] < M))
        
        dout = tl.load(dout_ptr + group_id * stride_dout_batch +
                      offs_am[:, None] * stride_dout_csize_m +
                      offs_bn[None, :] * stride_dout_n,
                      mask=(offs_am[:, None] < M) & (offs_bn[None, :] < N))
        
        # Compute partial dot product
        acc += tl.dot(a, dout)
    
    # Add residual if needed
    if HAS_RESIDUAL:
        res = tl.load(res_ptr + group_id * stride_res_batch +
                     offs_am[:, None] * stride_res_cs_n +
                     offs_bn[None, :],
                     mask=(offs_am[:, None] < M) & (offs_bn[None, :] < N))
        acc += res
    
    # Store result
    tl.store(db_ptr + group_id * stride_db_batch +
             offs_am[:, None] * stride_db_cs_n +
             offs_bn[None, :],
             acc, mask=(offs_am[:, None] < M) & (offs_bn[None, :] < N))

def _bmm_chunk_bwd(a, dout, res=None):
    """
    Wrapper function for the backward pass of chunked batched matrix multiplication
    Args:
        a: Input tensor of shape (batch, chunk_size, M)
        dout: Gradient tensor of shape (batch, M, N)
        res: Optional residual tensor of shape (batch, chunk_size, N)
    Returns:
        db: Gradient tensor of shape (batch, chunk_size, N)
    """
    # Get tensor dimensions
    batch, chunk_size, M = a.shape
    _, _, N = dout.shape
    
    # Ensure contiguous tensors
    a = a.contiguous()
    dout = dout.contiguous()
    
    # Initialize output tensor
    db = torch.empty((batch, chunk_size, N), 
                    device=a.device, 
                    dtype=a.dtype)
    
    # Compute grid size
    grid = (triton.cdiv(M, 128) * triton.cdiv(N, 256) * batch,)
    
    # Launch kernel
    _bmm_chunk_bwd_kernel[grid](
        a_ptr=a, 
        dout_ptr=dout,
        db_ptr=db,
        res_ptr=res if res is not None else db,  # Use db as dummy if no residual
        M=M, N=N, CS=chunk_size, BATCH=batch,
        stride_a_batch=a.stride(0),
        stride_a_cs_m=a.stride(1),
        stride_dout_batch=dout.stride(0),
        stride_dout_csize_m=dout.stride(1),
        stride_dout_n=dout.stride(2),
        stride_db_batch=db.stride(0),
        stride_db_cs_n=db.stride(1),
        stride_res_batch=res.stride(0) if res is not None else 0,
        stride_res_cs_n=res.stride(1) if res is not None else 0,
        HAS_RESIDUAL=res is not None,
    )
    
    return db
