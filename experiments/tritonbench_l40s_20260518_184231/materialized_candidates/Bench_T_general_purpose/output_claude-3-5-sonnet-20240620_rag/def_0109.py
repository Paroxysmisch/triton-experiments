import torch
import triton
import triton.language as tl

@triton.jit
def pseudoinverse_svd_kernel(
    U_ptr, S_ptr, Vt_ptr, A_inv_ptr,
    m, n, k, rcond,
    U_stride_0, U_stride_1, U_stride_2,
    S_stride_0, S_stride_1,
    Vt_stride_0, Vt_stride_1, Vt_stride_2,
    A_inv_stride_0, A_inv_stride_1, A_inv_stride_2,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    
    # Load S (singular values)
    s_ptr = S_ptr + pid * S_stride_0
    s = tl.load(s_ptr + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < k, other=0.0)
    
    # Compute threshold
    s_max = tl.max(s)
    threshold = rcond * s_max
    
    # Compute S_inv
    s_inv = tl.where(s > threshold, 1.0 / s, 0.0)
    
    # Compute A_inv = V * S_inv * U^H
    for i in range(0, n, BLOCK_SIZE):
        for j in range(0, m, BLOCK_SIZE):
            acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
            
            for l in range(0, k, BLOCK_SIZE):
                v_ptr = Vt_ptr + pid * Vt_stride_0 + i * Vt_stride_1 + l * Vt_stride_2
                u_ptr = U_ptr + pid * U_stride_0 + j * U_stride_1 + l * U_stride_2
                
                v = tl.load(v_ptr + tl.arange(0, BLOCK_SIZE)[:, None] * Vt_stride_2 + tl.arange(0, BLOCK_SIZE)[None, :],
                            mask=(tl.arange(0, BLOCK_SIZE)[:, None] < n - i) & (tl.arange(0, BLOCK_SIZE)[None, :] < k - l))
                u = tl.load(u_ptr + tl.arange(0, BLOCK_SIZE)[:, None] * U_stride_2 + tl.arange(0, BLOCK_SIZE)[None, :],
                            mask=(tl.arange(0, BLOCK_SIZE)[:, None] < m - j) & (tl.arange(0, BLOCK_SIZE)[None, :] < k - l))
                
                s_inv_block = s_inv[l:l+BLOCK_SIZE]
                
                acc += tl.dot(v * s_inv_block[None, :], tl.trans(u))
            
            a_inv_ptr = A_inv_ptr + pid * A_inv_stride_0 + i * A_inv_stride_1 + j * A_inv_stride_2
            tl.store(a_inv_ptr + tl.arange(0, BLOCK_SIZE)[:, None] * A_inv_stride_1 + tl.arange(0, BLOCK_SIZE)[None, :],
                     acc, mask=(tl.arange(0, BLOCK_SIZE)[:, None] < n - i) & (tl.arange(0, BLOCK_SIZE)[None, :] < m - j))

def pseudoinverse_svd(A, *, full_matrices=True, rcond=1e-15, out=None) -> torch.Tensor:
    assert A.dim() >= 2, "Input tensor must have at least 2 dimensions"
    
    # Compute SVD
    U, S, Vt = torch.linalg.svd(A, full_matrices=full_matrices)
    
    # Determine shapes
    *batch_dims, m, n = A.shape
    k = S.shape[-1]
    
    # Allocate output tensor if not provided
    if out is None:
        out = torch.empty((*batch_dims, n, m), dtype=A.dtype, device=A.device)
    
    # Determine grid and block sizes
    BLOCK_SIZE = triton.next_power_of_2(max(m, n, k))
    grid = (prod(batch_dims),) if batch_dims else (1,)
    
    # Launch kernel
    pseudoinverse_svd_kernel[grid](
        U, S, Vt, out,
        m, n, k, rcond,
        U.stride(0) if U.dim() > 2 else 0, U.stride(-2), U.stride(-1),
        S.stride(0) if S.dim() > 1 else 0, S.stride(-1),
        Vt.stride(0) if Vt.dim() > 2 else 0, Vt.stride(-2), Vt.stride(-1),
        out.stride(0) if out.dim() > 2 else 0, out.stride(-2), out.stride(-1),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4
    )
    
    return out

def prod(iterable):
    result = 1
    for x in iterable:
        result *= x
    return result
