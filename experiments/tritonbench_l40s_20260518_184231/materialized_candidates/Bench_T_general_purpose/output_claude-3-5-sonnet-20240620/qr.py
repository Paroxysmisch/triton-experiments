import triton
import triton.language as tl
import torch

@triton.jit
def qr_kernel(
    # Pointers to matrices
    a_ptr, q_ptr, r_ptr,
    # Matrix dimensions
    M, N, 
    # Strides for accessing matrices
    stride_am, stride_an,
    stride_qm, stride_qn,
    stride_rm, stride_rn,
    # Other parameters
    BLOCK_SIZE: tl.constexpr,
):
    """QR decomposition kernel using Householder reflections"""
    pid = tl.program_id(0)
    
    # Load the matrix block
    offs_am = pid * stride_am + tl.arange(0, BLOCK_SIZE)
    offs_an = tl.arange(0, BLOCK_SIZE)
    
    # Initialize Q as identity and R as copy of A
    a = tl.load(a_ptr + offs_am[:, None] * stride_am + offs_an[None, :] * stride_an)
    r = a.copy()
    q = tl.zeros([BLOCK_SIZE, BLOCK_SIZE], dtype=tl.float32)
    q += tl.where(tl.arange(0, BLOCK_SIZE)[:, None] == tl.arange(0, BLOCK_SIZE)[None, :], 1.0, 0.0)
    
    # Main QR loop using Householder reflections
    for k in range(min(M, N)):
        # Get column k
        x = r[k:, k]
        
        # Compute Householder vector
        alpha = tl.sqrt(tl.sum(x * x))
        alpha = tl.where(x[0] > 0, -alpha, alpha)
        r[k, k] = alpha
        
        if k < M - 1:
            # Compute v = x + alpha*e1
            v = x.copy()
            v[0] += alpha
            
            # Normalize v
            beta = tl.sqrt(tl.sum(v * v))
            v = tl.where(beta > 0, v / beta, v)
            
            # Update R: R = R - 2v(v^T R)
            for j in range(k + 1, N):
                dot = tl.sum(v * r[k:, j])
                r[k:, j] -= 2.0 * dot * v
            
            # Update Q: Q = Q - 2v(v^T Q)
            for j in range(M):
                dot = tl.sum(v * q[k:, j])
                q[k:, j] -= 2.0 * dot * v

    # Store results
    tl.store(q_ptr + offs_am[:, None] * stride_qm + offs_an[None, :] * stride_qn, q)
    tl.store(r_ptr + offs_am[:, None] * stride_rm + offs_an[None, :] * stride_rn, r)

def qr(A, mode='reduced', *, out=None):
    """
    Computes QR decomposition of a matrix or batch of matrices.
    
    Args:
        A (Tensor): Input tensor of shape (*, m, n) where * represents batch dimensions
        mode (str, optional): One of 'reduced', 'complete', 'r'. Default: 'reduced'
        out (tuple, optional): Output tuple of (Q, R) tensors. Default: None
    
    Returns:
        tuple: (Q, R) where:
            - Q is orthogonal/unitary matrix
            - R is upper triangular matrix
    """
    assert mode in ['reduced', 'complete', 'r'], f"Invalid mode: {mode}"
    
    # Get matrix dimensions
    *batch_dims, m, n = A.shape
    k = min(m, n)
    
    # Handle different modes
    if mode == 'reduced':
        q_shape = (*batch_dims, m, k)
        r_shape = (*batch_dims, k, n)
    elif mode == 'complete':
        q_shape = (*batch_dims, m, m)
        r_shape = (*batch_dims, m, n)
    else:  # mode == 'r'
        q_shape = (*batch_dims, 0, 0)  # Empty tensor for Q
        r_shape = (*batch_dims, k, n)
    
    # Initialize output tensors
    if out is None:
        Q = torch.empty(q_shape, dtype=A.dtype, device=A.device)
        R = torch.empty(r_shape, dtype=A.dtype, device=A.device)
    else:
        Q, R = out
        assert Q.shape == q_shape and R.shape == r_shape
    
    # Configure kernel parameters
    BLOCK_SIZE = 32
    grid = (triton.cdiv(m, BLOCK_SIZE),)
    
    # Launch kernel
    qr_kernel[grid](
        A, Q, R,
        m, n,
        A.stride(-2), A.stride(-1),
        Q.stride(-2), Q.stride(-1),
        R.stride(-2), R.stride(-1),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return Q, R
