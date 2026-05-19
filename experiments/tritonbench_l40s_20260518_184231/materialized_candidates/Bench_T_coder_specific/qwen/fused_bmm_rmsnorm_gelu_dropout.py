import torch
import triton
import triton.language as tl

def fused_bmm_rmsnorm_gelu_dropout(
    input1,
    input2,
    normalized_shape,
    dropout_p=0.1,
    eps=1e-5,
    training=True,
    approximate='none',
    out=None,
):
    batch_size, n, m = input1.shape
    _, m, p = input2.shape
    normalized_shape = [p]

    # Initialize output tensor
    if out is None:
        out = torch.empty((batch_size, n, p), device=input1.device, dtype=input1.dtype)

    # Initialize gamma for RMS normalization
    gamma = torch.ones(normalized_shape, device=input1.device, dtype=input1.dtype)

    # Launch Triton kernel
    fused_bmm_rmsnorm_gelu_dropout(
        input1,
        input2,
        out,
        gamma,
        batch_size,
        n,
        m,
        p,
        eps,
        dropout_p,
        training,
        len(normalized_shape) > 0,
    )

    return out
