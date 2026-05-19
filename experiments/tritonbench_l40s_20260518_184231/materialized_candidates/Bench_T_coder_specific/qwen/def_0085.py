triton
import triton
import triton.language as tl

@triton.jit
def matrix_power_eig_kernel(
    A_ptr,
    V_ptr,
    Lambda_k_ptr,
    inv_V_ptr,
    A_out_ptr,
    n: tl.constexpr,
    batch_size: tl.constexpr,
    k: tl.float32,
):
    pid = tl.program_id(axis=0)
    row = pid % n
    col = pid // n
    
    # Load elements from A, V, and inv_V
    A_elem = tl.load(A_ptr + pid * n)
    V_elem = tl.load(V_ptr + pid * n)
    inv_V_elem = tl.load(inv_V_ptr + pid * n)
    
    # Initialize sum for A^k
    A_k_elem = 0.0
    
    # Compute A^k using the formula A^k = V diag(Λ^k) V^-1
    for i in range(n):
        lambda_k_i = Lambda_k_ptr[i] ** k
        A_k_elem += lambda_k_i * V_elem * inv_V_elem
    
    # Store the result in the output tensor
    tl.store(A_out_ptr + pid * n, A_k_elem)

# Wrapper function
def matrix_power_eig(A, k, *, out=None):
    n = A.shape[-1]
    batch_size = A.shape[:-2].numel()
    
    # Allocate memory for intermediate results
    V = torch.zeros_like(A)
    Lambda_k = torch.zeros((n,), dtype=A.dtype)
    inv_V = torch.zeros_like(A)
    
    # Compute eigenvalues and eigenvectors
    eigs = torch.linalg.eigh(A)
    Lambda_k[:] = eigs[0] ** k
    V[:] = eigs[1]
    inv_V[:] = torch.linalg.inv(eigs[1])
    
    # Allocate memory for output tensor
    if out is None:
        out = torch.zeros_like(A)
    
    # Launch Triton kernel
    grid = (batch_size * n,)
    block = (n,)
    matrix_power_eig_kernel[grid, block](A.data_ptr(), V.data_ptr(), Lambda_k.data_ptr(), inv_V.data_ptr(), out.data_ptr(), n, batch_size, k)
    
    return out
