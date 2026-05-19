import torch
import triton
import triton.language as tl

@triton.jit
def scale_columns_kernel(
    V_ptr, lambda_ptr, output_ptr,
    n, batch_size,
    stride_v_batch, stride_v_row, stride_v_col,
    stride_lambda_batch, stride_lambda_col,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total_operations = batch_size * n * n
    if pid >= total_operations:
        return
    batch_idx = pid // (n * n)
    pid_in_batch = pid % (n * n)
    row_idx = pid_in_batch // n
    col_idx = pid_in_batch % n

    v_offset = batch_idx * stride_v_batch + row_idx * stride_v_row + col_idx * stride_v_col
    lambda_offset = batch_idx * stride_lambda_batch + col_idx * stride_lambda_col

    v = tl.load(V_ptr + v_offset)
    l_val = tl.load(lambda_ptr + lambda_offset)
    scaled = v * l_val
    tl.store(output_ptr + v_offset, scaled)

def scale_columns_triton(V, lambda_k):
    assert V.dim() >= 2 and lambda_k.dim() >= 1
    assert V.shape[-1] == V.shape[-2], "V must be square matrices"
    n = V.shape[-1]
    original_shape = V.shape
    V_flat = V.contiguous().view(-1, n, n)
    lambda_flat = lambda_k.contiguous().view(-1, n)
    batch_size_flat = V_flat.shape[0]

    output_flat = torch.empty_like(V_flat)
    stride_v_batch = V_flat.stride(0)
    stride_v_row = V_flat.stride(1)
    stride_v_col = V_flat.stride(2)
    stride_lambda_batch = lambda_flat.stride(0)
    stride_lambda_col = lambda_flat.stride(1)

    total_operations = batch_size_flat * n * n
    grid = lambda meta: (total_operations,)
    scale_columns_kernel[grid](
        V_flat, lambda_flat, output_flat,
        n, batch_size_flat,
        stride_v_batch, stride_v_row, stride_v_col,
        stride_lambda_batch, stride_lambda_col,
        BLOCK_SIZE=128,
    )
    return output_flat.view(original_shape)

def matrix_power_eig(A, k, *, out=None) -> torch.Tensor:
    assert A.shape[-1] == A.shape[-2], "A must be a square matrix."
    eigenvalues, eigenvectors = torch.linalg.eig(A)
    lambda_k = torch.pow(eigenvalues, k)
    V_scaled = scale_columns_triton(eigenvectors, lambda_k)
    V_inv = torch.linalg.inv(eigenvectors)
    result = torch.matmul(V_scaled, V_inv)
    if out is not None:
        dtype = A.dtype
        if result.is_complex() and not torch.is_complex(out):
            raise RuntimeError("output tensor must be complex dtype if result is complex.")
        out.copy_(result)
        return out
    return result
