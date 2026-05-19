import triton
import triton.language as tl
import torch

@triton.jit
def bmm_dropout_gelu_kernel(
    X_ptr, Y_ptr, O_ptr,
    B, N, M, P,
    drop_p, seed,
    stride_xn, stride_xm,
    stride_ym, stride_yp,
    stride_on, stride_op,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_P: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    batch_id = pid // (N * P)
    row_id = (pid // P) % N
    col_id = pid % P

    # Offsets
    x_offset = batch_id * stride_xn + row_id * stride_xm
    y_offset = batch_id * stride_ym + col_id * stride_yp
    o_offset = batch_id * stride_on + row_id * stride_op + col_id

    # Load inputs
    x = tl.load(X_ptr + x_offset)
    y = tl.load(Y_ptr + y_offset)

    # Perform the batch matrix multiplication
    z = tl.dot(x, y)

    # Apply dropout if training
    if drop_p > 0:
        random = tl.rand(seed, pid)
        z = tl.where(random < drop_p, 0, z / (1 - drop_p))

    # Apply GELU
    if approximate == 'tanh':
        z = 0.5 * z * (1 + tl.tanh(0.7978845608 * (z + 0.044715 * z * z * z)))
    else:
        z = 0.5 * z * (1 + tl.erf(z / tl.sqrt(2)))

    # Store the result
    tl.store(O_ptr + o_offset, z)

def fused_bmm_dropout_gelu(input1, input2, p=0.5, training=True, inplace=False, approximate='none', *, out=None):
    assert input1.shape[0] == input2.shape[0]  # Batch size must match
    assert input1.shape[2] == input2.shape[1]  # M dimension must match for matrix multiplication

    B, N, M = input1.shape
    _, _, P = input2.shape

    if out is None:
        out = torch.empty((B, N, P), device=input1.device, dtype=input1.dtype)

    # Launch Triton kernel
    grid = (B * N * P,)
    bmm_dropout_gelu_kernel[grid](
        input1, input2, out,
        B, N, M, P,
        p if training else 0, torch.randint(0, 2**31 - 1, (1,)).item(),
        input1.stride(1), input1.stride(2),
        input2.stride(1), input2.stride(2),
        out.stride(1), out.stride(2),
        BLOCK_SIZE_M=32, BLOCK_SIZE_N=32, BLOCK_SIZE_P=32
    )

    return out
