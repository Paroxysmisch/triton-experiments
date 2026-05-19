import triton
import triton.language as tl
import triton.cudalib as cu

@triton.jit
def svd_kernel(A_ptr, U_ptr, S_ptr, Vh_ptr, m, n, batch_size, full_matrices, driver, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    if pid < batch_size:
        # Compute the SVD using cuSOLVER
        A = tl.load(A_ptr + pid * m * n, mask=tl.arange(0, m * n) < m * n, other=0.0)
        A = A.to(tl.float32)  # Ensure the data type is compatible with cuSOLVER

        # Allocate memory for U, S, and Vh
        U = tl.zeros((m, m), dtype=tl.float32) if full_matrices else tl.zeros((m, min(m, n)), dtype=tl.float32)
        S = tl.zeros((min(m, n),), dtype=tl.float32)
        Vh = tl.zeros((n, n), dtype=tl.float32) if full_matrices else tl.zeros((min(m, n), n), dtype=tl.float32)

        # Call cuSOLVER SVD
        cu.svd(A, U, S, Vh, driver=driver)

        # Store the results
        tl.store(U_ptr + pid * m * m, U, mask=tl.arange(0, m * m) < m * m)
        tl.store(S_ptr + pid * min(m, n), S, mask=tl.arange(0, min(m, n)) < min(m, n))
        tl.store(Vh_ptr + pid * n * n, Vh, mask=tl.arange(0, n * n) < n * n)

import torch
import triton
import triton.runtime.driver as driver

def linalg_svd(A, full_matrices=True, *, driver=None, out=None):
    # Validate input
    if not isinstance(A, torch.Tensor):
        raise ValueError("Input A must be a torch.Tensor")
    if A.dtype not in [torch.float32, torch.float64, torch.complex64, torch.complex128]:
        raise ValueError("Input A must be of type float, double, cfloat, or cdouble")
    
    # Determine batch size and matrix dimensions
    batch_size = A.shape[:-2] if A.dim() > 2 else (1,)
    m, n = A.shape[-2:]
    
    # Allocate output tensors
    if out is None:
        U_shape = (*batch_size, m, m) if full_matrices else (*batch_size, m, min(m, n))
        S_shape = (*batch_size, min(m, n))
        Vh_shape = (*batch_size, n, n) if full_matrices else (*batch_size, min(m, n), n)
        U = torch.empty(U_shape, dtype=A.dtype, device=A.device)
        S = torch.empty(S_shape, dtype=torch.float32 if A.dtype in [torch.float32, torch.complex64] else torch.float64, device=A.device)
        Vh = torch.empty(Vh_shape, dtype=A.dtype, device=A.device)
    else:
        U, S, Vh = out
        if U.shape != U_shape or S.shape != S_shape or Vh.shape != Vh_shape:
            raise ValueError("Output tensors must have the correct shape")
    
    # Determine grid configuration
    grid = (batch_size, 1, 1)
    
    # Call the Triton kernel
    svd_kernel[grid](
        A.data_ptr(), U.data_ptr(), S.data_ptr(), Vh.data_ptr(),
        m, n, batch_size, full_matrices, driver,
        BLOCK_SIZE=1024
    )
    
    return U, S, Vh
