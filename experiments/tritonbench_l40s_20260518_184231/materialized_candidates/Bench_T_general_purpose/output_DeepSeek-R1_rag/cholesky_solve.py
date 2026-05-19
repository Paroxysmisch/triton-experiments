import triton
import triton.language as tl
import torch

@triton.jit
def forward_substitution_lower_kernel(
    B_ptr, L_ptr, Y_ptr,
    n, k,
    stride_batch_B, stride_n_B, stride_k_B,
    stride_batch_L, stride_n_L, stride_k_L,
    stride_batch_Y, stride_n_Y, stride_k_Y,
    BLOCK_SIZE: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_col = tl.program_id(1)
    
    # Create masks for valid rows/columns
    row_range = tl.arange(0, BLOCK_SIZE)
    batch_mask = pid_batch < stride_batch_B
    col_mask = pid_col < k
    
    # Offsets for current batch and column
    B_offset = pid_batch * stride_batch_B + pid_col * stride_k_B
    L_offset = pid_batch * stride_batch_L
    Y_offset = pid_batch * stride_batch_Y + pid_col * stride_k_Y
    
    # Iterate over each row in sequence
    for i in range(n):
        # Load B[i, current column]
        b_i = tl.load(B_ptr + B_offset + i * stride_n_B, mask=(i < n) & batch_mask & col_mask, other=0.0)
        
        # Compute sum(L[i, :i] * Y[:i, current column])
        sum_val = 0.0
        for j in range(i):
            l_ij = tl.load(L_ptr + L_offset + i * stride_n_L + j * stride_k_L, mask=(j < i) & (i < n) & (j < n) & batch_mask, other=0.0)
            y_j = tl.load(Y_ptr + Y_offset + j * stride_n_Y, mask=(j < i) & (j < n) & batch_mask & col_mask, other=0.0)
            sum_val += l_ij * y_j
        
        # Load L[i, i]
        l_ii = tl.load(L_ptr + L_offset + i * stride_n_L + i * stride_k_L, mask=(i < n) & batch_mask, other=1.0)
        
        # Compute Y[i] = (b_i - sum_val) / L[i, i]
        y_i = (b_i - sum_val) / l_ii
        
        # Store Y[i]
        tl.store(Y_ptr + Y_offset + i * stride_n_Y, y_i, mask=(i < n) & batch_mask & col_mask)

@triton.jit
def backward_substitution_upper_kernel(
    Y_ptr, L_ptr, X_ptr,
    n, k,
    stride_batch_Y, stride_n_Y, stride_k_Y,
    stride_batch_L, stride_n_L, stride_k_L,
    stride_batch_X, stride_n_X, stride_k_X,
    CONJ: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_col = tl.program_id(1)
    
    # Create masks for valid rows/columns
    row_range = tl.arange(0, BLOCK_SIZE)
    batch_mask = pid_batch < stride_batch_Y
    col_mask = pid_col < k
    
    # Offsets for current batch and column
    Y_offset = pid_batch * stride_batch_Y + pid_col * stride_k_Y
    L_offset = pid_batch * stride_batch_L
    X_offset = pid_batch * stride_batch_X + pid_col * stride_k_X
    
    # Iterate over rows in reverse order
    for i in range(n-1, -1, -1):
        # Load Y[i, current column]
        y_i = tl.load(Y_ptr + Y_offset + i * stride_n_Y, mask=(i < n) & batch_mask & col_mask, other=0.0)
        
        # Compute sum(L^H[i, i+1:] * X[i+1:, current column])
        sum_val = 0.0
        for j in range(i+1, n):
            l_ji = tl.load(L_ptr + L_offset + j * stride_n_L + i * stride_k_L, mask=(j < n) & (i < n) & batch_mask, other=0.0)
            if CONJ:
                l_ji = tl.math.conj(l_ji)
            x_j = tl.load(X_ptr + X_offset + j * stride_n_X, mask=(j < n) & batch_mask & col_mask, other=0.0)
            sum_val += l_ji * x_j
        
        # Load L[i, i]
        l_ii = tl.load(L_ptr + L_offset + i * stride_n_L + i * stride_k_L, mask=(i < n) & batch_mask, other=1.0)
        if CONJ:
            l_ii = tl.math.conj(l_ii)
        
        # Compute X[i] = (y_i - sum_val) / L[i, i]
        x_i = (y_i - sum_val) / l_ii
        
        # Store X[i]
        tl.store(X_ptr + X_offset + i * stride_n_X, x_i, mask=(i < n) & batch_mask & col_mask)

def cholesky_solve(B, L, upper=False, *, out=None):
    assert B.dtype == L.dtype, "B and L must have the same dtype"
    assert B.shape[:-2] == L.shape[:-2], "Batch dimensions of B and L must match"
    assert L.shape[-1] == L.shape[-2], "L must be square"
    assert B.shape[-2] == L.shape[-1], "Incompatible matrix sizes"
    
    B = B.contiguous()
    L = L.contiguous()
    if out is None:
        out = torch.empty_like(B)
    else:
        out = out.contiguous()
    
    n = L.size(-1)
    k = B.size(-1)
    batch_dims = B.shape[:-2]
    num_batches = int(torch.prod(torch.tensor(batch_dims))) if batch_dims else 1
    
    # Reshape batches into a single dimension
    B_flat = B.view(num_batches, n, k)
    L_flat = L.view(num_batches, n, n)
    out_flat = out.view(num_batches, n, k)
    
    # Intermediate tensor for the first solve
    Y = torch.empty_like(B_flat)
    
    # Determine block sizes
    BLOCK_SIZE = 16
    
    # First triangular solve
    if upper:
        # Solve U^H Y = B (backward substitution with conjugate)
        grid = (num_batches, k)
        backward_substitution_upper_kernel[grid](
            B_flat, L_flat, Y,
            n, k,
            B_flat.stride(0), B_flat.stride(1), B_flat.stride(2),
            L_flat.stride(0), L_flat.stride(1), L_flat.stride(2),
            Y.stride(0), Y.stride(1), Y.stride(2),
            CONJ=True,
            BLOCK_SIZE=BLOCK_SIZE
        )
        # Second solve U X = Y (forward substitution)
        grid = (num_batches, k)
        forward_substitution_lower_kernel[grid](
            Y, L_flat, out_flat,
            n, k,
            Y.stride(0), Y.stride(1), Y.stride(2),
            L_flat.stride(0), L_flat.stride(1), L_flat.stride(2),
            out_flat.stride(0), out_flat.stride(1), out_flat.stride(2),
            BLOCK_SIZE=BLOCK_SIZE
        )
    else:
        # Solve L Y = B (forward substitution)
        grid = (num_batches, k)
        forward_substitution_lower_kernel[grid](
            B_flat, L_flat, Y,
            n, k,
            B_flat.stride(0), B_flat.stride(1), B_flat.stride(2),
            L_flat.stride(0), L_flat.stride(1), L_flat.stride(2),
            Y.stride(0), Y.stride(1), Y.stride(2),
            BLOCK_SIZE=BLOCK_SIZE
        )
        # Second solve L^H X = Y (backward substitution with conjugate)
        grid = (num_batches, k)
        backward_substitution_upper_kernel[grid](
            Y, L_flat, out_flat,
            n, k,
            Y.stride(0), Y.stride(1), Y.stride(2),
            L_flat.stride(0), L_flat.stride(1), L_flat.stride(2),
            out_flat.stride(0), out_flat.stride(1), out_flat.stride(2),
            CONJ=True,
            BLOCK_SIZE=BLOCK_SIZE
        )
    
    return out.view(*batch_dims, n, k)
