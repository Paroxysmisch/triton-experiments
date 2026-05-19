import triton
import triton.language as tl
import torch

@triton.jit
def fused_mv_logsoftmax_dropout_kernel(
    input_ptr, vec_ptr, output_ptr,
    n, m, p, training, stride_input_n, stride_input_m,
    BLOCK_SIZE: tl.constexpr
):
    # Define block indices
    row_idx = tl.program_id(0)
    
    # Compute the starting position for this block
    input_offset = row_idx * stride_input_n
    vec_offset = 0

    # Load the input matrix row and vector
    row = tl.load(input_ptr + input_offset + tl.arange(0, BLOCK_SIZE) % m, mask=tl.arange(0, BLOCK_SIZE) < m)
    vec = tl.load(vec_ptr + vec_offset + tl.arange(0, BLOCK_SIZE) % m, mask=tl.arange(0, BLOCK_SIZE) < m)

    # Matrix-vector multiplication
    z = tl.dot(row, vec)

    # Compute max for numerical stability in log-softmax
    z_max = tl.max(z, axis=0)

    # Subtract max and exponentiate
    z_exp = tl.exp(z - z_max)

    # Compute sum of exponentials
    sum_exp = tl.sum(z_exp, axis=0)

    # Compute log-softmax
    s = z - z_max - tl.log(sum_exp)

    # Apply dropout
    if training:
        keep_prob = 1 - p
        dropout_mask = tl.rand([BLOCK_SIZE], seed=row_idx) < keep_prob
        s = tl.where(dropout_mask, s / keep_prob, 0.0)

    # Store the result
    tl.store(output_ptr + row_idx, s)


def fused_mv_logsoftmax_dropout(input, vec, p=0.5, training=True, inplace=False, dim=0, *, out=None):
    assert input.dim() == 2, "Input must be a 2D matrix"
    assert vec.dim() == 1, "Vec must be a 1D vector"
    assert input.size(1) == vec.size(0), "Incompatible dimensions for matrix-vector multiplication"
    assert dim in (0, -1), "Log-softmax dimension must be 0 or -1"

    n, m = input.shape
    input_ptr = input.data_ptr()
    vec_ptr = vec.data_ptr()

    # Allocate output tensor
    if out is None:
        out = torch.empty((n,), device=input.device, dtype=input.dtype)

    output_ptr = out.data_ptr()

    # Launch the Triton kernel
    grid = (n,)
    fused_mv_logsoftmax_dropout_kernel[grid](
        input_ptr, vec_ptr, output_ptr,
        n, m, p, training, input.stride(0), input.stride(1),
        BLOCK_SIZE=1024  # Assuming block size, adjust as needed
    )

    return out
