import triton
import triton.language as tl
import torch

@triton.jit
def fused_bmm_dropout_gelu_kernel(
    X_ptr, Y_ptr, Z_ptr, mask_ptr, B, N, M, P, p, stride_xn, stride_xm, stride_ym, stride_yp, stride_zn, stride_zp, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    bid = pid // (N * P)
    row = (pid // P) % N
    col = pid % P

    # Offsets for the batch
    X_offset = bid * stride_xn + row * stride_xm
    Y_offset = bid * stride_ym + col
    Z_offset = bid * stride_zn + row * stride_zp

    # Accumulate the result of batch matrix multiplication
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for k in range(0, M, BLOCK_SIZE):
        X_block = tl.load(X_ptr + X_offset + k)
        Y_block = tl.load(Y_ptr + Y_offset + k * stride_yp)
        acc += X_block * Y_block

    # Store the result of BMM
    Z = acc.to(tl.float32)

    # Apply dropout if training
    if p > 0:
        mask = tl.load(mask_ptr + Z_offset)
        Z = tl.where(mask, Z / (1 - p), 0)

    # Apply GELU
    Z = 0.5 * Z * (1.0 + tl.erf(Z / tl.sqrt(2.0)))

    # Store the result
    tl.store(Z_ptr + Z_offset, Z)


def fused_bmm_dropout_gelu(input1, input2, p=0.5, training=True, inplace=False, approximate='none', *, out=None):
    B, N, M = input1.shape
    _, _, P = input2.shape

    # Check input dimensions
    assert M == input2.shape[1], "Incompatible dimensions for batch matrix multiplication"

    # Prepare output tensor
    if out is None:
        out = torch.empty((B, N, P), device=input1.device, dtype=input1.dtype)

    # Generate random mask for dropout
    if training and p > 0:
        mask = torch.rand((B, N, P), device=input1.device) > p
    else:
        mask = torch.ones((B, N, P), device=input1.device, dtype=torch.bool)

    # Launch Triton kernel
    BLOCK_SIZE = 128  # Example block size, tune for your hardware
    grid = (B * N * P + BLOCK_SIZE - 1) // BLOCK_SIZE
    fused_bmm_dropout_gelu_kernel[grid](
        input1, input2, out, mask, B, N, M, P, p,
        input1.stride(0), input1.stride(1), input2.stride(0), input2.stride(1),
        out.stride(0), out.stride(1), BLOCK_SIZE
    )

    return out
