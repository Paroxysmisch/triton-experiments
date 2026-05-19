import torch
import triton
import triton.language as tl

@triton.jit
def pseudoinverse_kernel(
    out_ptr, u_ptr, s_ptr, v_ptr, s_inv_ptr, m, n, k, rcond, full_matrices: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    batch_size = tl.num_programs(0)
    
    for b in range(batch_size):
        # Load U, S, and V for the current batch
        u_batch_ptr = u_ptr + b * m * k
        v_batch_ptr = v_ptr + b * n * k
        s_batch_ptr = s_ptr + b * k
        s_inv_batch_ptr = s_inv_ptr + b * k
        
        # Load U and V
        u = tl.load(u_batch_ptr, mask=tl.arange(0, m * k) < m * k, other=0.0)
        v = tl.load(v_batch_ptr, mask=tl.arange(0, n * k) < n * k, other=0.0)
        
        # Load S and compute S_inv
        s = tl.load(s_batch_ptr, mask=tl.arange(0, k) < k, other=0.0)
        s_max = tl.max(s, axis=0)
        s_inv = tl.where(s > rcond * s_max, 1.0 / s, 0.0)
        tl.store(s_inv_batch_ptr, s_inv, mask=tl.arange(0, k) < k)
        
        # Compute U * S_inv
        u_s_inv = tl.zeros((m, k), dtype=tl.float32)
        for i in range(m):
            for j in range(k):
                u_s_inv[i, j] = u[i * k + j] * s_inv[j]
        
        # Compute V^H * (U * S_inv)
        out_batch_ptr = out_ptr + b * m * n
        out = tl.zeros((m, n), dtype=tl.float32)
        for i in range(m):
            for j in range(n):
                for l in range(k):
                    out[i, j] += u_s_inv[i, l] * v[l * n + j]
        
        # Store the result
        tl.store(out_batch_ptr, out, mask=tl.arange(0, m * n) < m * n)

def pseudoinverse_svd(A, *, full_matrices=True, rcond=1e-15, out=None) -> torch.Tensor:
    # Check input tensor shape
    *batch_dims, m, n = A.shape
    batch_size = 1
    for dim in batch_dims:
        batch_size *= dim
    
    # Perform SVD
    U, S, Vh = torch.linalg.svd(A, full_matrices=full_matrices)
    
    # Allocate output tensor if not provided
    if out is None:
        out = torch.empty_like(A)
    
    # Compute the block size
    BLOCK_SIZE = triton.next_power_of_2(max(m, n))
    
    # Compute the number of warps
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
    
    # Enqueue the kernel
    pseudoinverse_kernel[(batch_size,)](
        out, U, S, Vh, out, m, n, min(m, n), rcond, full_matrices, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps
    )
    
    return out
