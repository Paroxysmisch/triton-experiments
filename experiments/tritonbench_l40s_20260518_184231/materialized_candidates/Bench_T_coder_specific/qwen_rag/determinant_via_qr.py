import triton
import triton.language as tl

# Helper function to perform a Householder reflection
@triton.jit
def householder_reflection(a_ptr, a_row_stride, a_col_stride, n_cols, k, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start_ptr = a_ptr + row_idx * a_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets

    # Load the vector a
    a_k = tl.load(input_ptrs + k * a_col_stride, mask=col_offsets < n_cols, other=0.0)
    a_norm_sq = tl.sum(a_k * a_k)

    # Compute the reflection vector v
    v_k = a_k + tl.where(k == 0, tl.sqrt(a_norm_sq), 0.0)
    v_norm = tl.norm(v_k, ord=2)
    v_k /= v_norm

    # Apply the reflection to the matrix
    for j in range(k, n_cols):
        v_j = tl.load(input_ptrs + j * a_col_stride, mask=col_offsets < n_cols, other=0.0)
        alpha = 2.0 * tl.dot(v_k, v_j) / v_norm
        for i in range(k, n_cols):
            v_i = tl.load(input_ptrs + i * a_col_stride, mask=col_offsets < n_cols, other=0.0)
            tl.atomic_add(input_ptrs + i * a_col_stride + j * a_col_stride, v_i * alpha)

# Helper function to compute the determinant from the upper triangular matrix R
@triton.jit
def determinant_from_R(r_ptr, r_row_stride, r_col_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    det_Q = 1.0
    det_R = 1.0
    row_idx = tl.program_id(0)
    row_start_ptr = r_ptr + row_idx * r_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets

    for k in range(n_cols):
        r_kk = tl.load(input_ptrs + k * r_col_stride, mask=col_offsets < n_cols, other=0.0)
        det_R *= r_kk

    return det_Q * det_R

@triton.jit
def determinant_via_qr_kernel(out_ptr, a_ptr, a_row_stride, a_col_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start_ptr = a_ptr + row_idx * a_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets

    # Perform QR decomposition using Householder reflections
    for k in range(n_cols):
        householder_reflection(a_ptr, a_row_stride, a_col_stride, n_cols, k, BLOCK_SIZE=BLOCK_SIZE)

    # Extract the upper triangular matrix R
    r_ptr = a_ptr

    # Compute the determinant from the upper triangular matrix R
    det = determinant_from_R(r_ptr, a_row_stride, a_col_stride, n_cols, BLOCK_SIZE=BLOCK_SIZE)

    # Store the determinant in the output buffer
    tl.store(out_ptr, det)

@triton.jit
def determinant_via_qr(A, *, mode='reduced', out=None) -> tl.Tensor:
    n_cols = A.shape[-1]
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    assert A.shape[-1] == A.shape[-2], "Matrix must be square"

    if out is None:
        out = tl.zeros((A.shape[0], 1), dtype=A.dtype)

    determinant_via_qr_kernel[(A.shape[0], )](out, A, A.stride(0), A.stride(1), n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return out

# Example usage
if __name__ == "__main__":
    import torch
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    det = determinant_via_qr(A)
    print("Determinant:", det)
