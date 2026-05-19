import triton
import triton.language as tl

@triton.jit
def fused_bmm_rmsnorm_gelu_dropout_sub_kernel(
    X_ptr, Y_ptr, O_ptr, out_ptr,
    B, N, M, P,
    norm_shape, dropout_p, training, approximate, eps,
    stride_xb, stride_xn, stride_xm,
    stride_yb, stride_ym, stride_yp,
    stride_ob, stride_on, stride_op,
    stride_outb, stride_outn, stride_outp,
    rng_seed, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    # Compute the indices
    b_idx = pid // (N * P)
    n_idx = (pid // P) % N
    p_idx = pid % P

    # Load input1 and input2
    X_offset = b_idx * stride_xb + n_idx * stride_xn
    Y_offset = b_idx * stride_yb + p_idx * stride_yp
    X = tl.load(X_ptr + X_offset, mask=(n_idx < N) & (p_idx < P))
    Y = tl.load(Y_ptr + Y_offset, mask=(n_idx < N) & (p_idx < P))

    # Batch matrix multiplication
    Z = tl.dot(X, Y)

    # RMS Normalization
    mean_square = tl.sum(Z * Z, axis=-1) / norm_shape
    Z_norm = Z / tl.sqrt(mean_square + eps)

    # GELU activation
    if approximate == 'tanh':
        cdf = 0.5 * (1.0 + tl.tanh((tl.sqrt(2.0 / tl.pi) * (Z_norm + 0.044715 * Z_norm * Z_norm * Z_norm))))
        G = Z_norm * cdf
    else:
        G = 0.5 * Z_norm * (1.0 + tl.erf(Z_norm / tl.sqrt(2.0)))

    # Dropout
    if training:
        rng = tl.rand(rng_seed, pid)
        mask = rng > dropout_p
        D = G * mask / (1.0 - dropout_p)
    else:
        D = G

    # Load other tensor
    O_offset = b_idx * stride_ob + n_idx * stride_on + p_idx * stride_op
    O = tl.load(O_ptr + O_offset, mask=(n_idx < N) & (p_idx < P))

    # Subtract other
    Y = D - O

    # Store the result
    out_offset = b_idx * stride_outb + n_idx * stride_outn + p_idx * stride_outp
    tl.store(out_ptr + out_offset, Y, mask=(n_idx < N) & (p_idx < P))

import torch
import triton
import triton.language as tl

def fused_bmm_rmsnorm_gelu_dropout_sub(input1, input2, other, normalized_shape, dropout_p=0.5, training=True, approximate='none', eps=1e-5, *, out=None):
    # Ensure inputs are compatible
    B, N, M = input1.shape
    _, _, P = input2.shape
    assert input1.shape[-1] == input2.shape[-2], "Incompatible shapes for batch matrix multiplication"
    assert normalized_shape == P, "normalized_shape must match the last dimension of the output"

    # Prepare output tensor
    if out is None:
        out = torch.empty((B, N, P), device=input1.device, dtype=input1.dtype)

    # Launch Triton kernel
    BLOCK_SIZE = 128  # Define a suitable block size
    grid = (B * N * P + BLOCK_SIZE - 1) // BLOCK_SIZE

    fused_bmm_rmsnorm_gelu_dropout_sub_kernel[grid](
        input1, input2, other, out,
        B, N, M, P,
        normalized_shape, dropout_p, training, approximate, eps,
        input1.stride(0), input1.stride(1), input1.stride(2),
        input2.stride(0), input2.stride(1), input2.stride(2),
        other.stride(0), other.stride(1), other.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        torch.randint(0, 2**32, (1,), device=input1.device).item(),
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out
