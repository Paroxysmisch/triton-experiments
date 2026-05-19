import triton
import triton.language as tl

@triton.jit
def pseudoinverse_svd_kernel(A, *, full_matrices=True, rcond=1e-15, out=None):
    # Decompose the matrix A into its singular value components
    U, S, V = tl.svd(A, full_matrices=full_matrices)

    # Compute the pseudoinverse
    pseudoinverse = tl.matmul(V.transpose(), tl.rdiv(1, S), U.transpose())

    # Set the singular values below a certain threshold to zero to avoid numerical instability
    pseudoinverse = tl.where(S > rcond * tl.max(S), pseudoinverse, 0)

    # Return the pseudoinverse
    return pseudoinverse

def pseudoinverse_svd(A, *, full_matrices=True, rcond=1e-15, out=None):
    # Call the Triton kernel
    return pseudoinverse_svd_kernel(A, full_matrices=full_matrices, rcond=rcond, out=out)
