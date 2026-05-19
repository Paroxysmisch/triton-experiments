import torch
import triton
import triton.language as tl

@triton.jit
def matrix_power_kernel(V_ptr, L_ptr, V_inv_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Program IDs
    row_id = tl.program_id(0)
    col_id = tl.program_id(1)

    # Load eigenvectors and eigenvalues
    row_offsets = tl.arange(0, BLOCK_SIZE)
    col_offsets = tl.arange(0, BLOCK_SIZE)

    # Calculate the starting position for V, L, V_inv, and out
    V_start = V_ptr + row_id * n + col_offsets
    L_start = L_ptr + col_id * n + col_offsets
    V_inv_start = V_inv_ptr + col_id * n + row_offsets
    out_start = out_ptr + row_id * n + col_offsets

    # Load the data
    V = tl.load(V_start, mask=col_offsets < n)
    L = tl.load(L_start, mask=col_offsets < n)
    V_inv = tl.load(V_inv_start, mask=row_offsets < n)

    # Compute diag(L^k)
    L_k = tl.pow(L, k)

    # Compute V * diag(L^k) * V_inv
    VL = V * L_k
    out = tl.dot(VL, V_inv)

    # Store the result
    tl.store(out_start, out, mask=col_offsets < n)

def matrix_power_eig(A, k, *, out=None):
    # Get the shape of the input tensor
    *batch_dims, n, _ = A.shape

    # Compute eigenvalues and eigenvectors
    eigvals, eigvecs = torch.linalg.eig(A)

    # Compute the inverse of the eigenvectors
    eigvecs_inv = torch.linalg.inv(eigvecs)

    # Allocate output if not provided
    if out is None:
        out = torch.empty_like(A, dtype=torch.cdouble)

    # Flatten the batch dimensions for processing
    A_flat = A.reshape(-1, n, n)
    eigvals_flat = eigvals.reshape(-1, n)
    eigvecs_flat = eigvecs.reshape(-1, n, n)
    eigvecs_inv_flat = eigvecs_inv.reshape(-1, n, n)
    out_flat = out.reshape(-1, n, n)

    # Launch the kernel for each batch
    for i in range(A_flat.shape[0]):
        matrix_power_kernel[(n, n)](
            eigvecs_flat[i],
            eigvals_flat[i],
            eigvecs_inv_flat[i],
            out_flat[i],
            n,
            BLOCK_SIZE=n
        )

    # Reshape the output back to the original batch dimensions
    return out.reshape(*batch_dims, n, n)

# Example usage
A = torch.randn(2, 3, 3, dtype=torch.cdouble)
k = 2.5
result = matrix_power_eig(A, k)
print(result)
