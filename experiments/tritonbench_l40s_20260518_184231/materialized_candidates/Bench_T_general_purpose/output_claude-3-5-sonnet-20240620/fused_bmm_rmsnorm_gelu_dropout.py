import triton
import triton.language as tl

@triton.jit
def fused_bmm_rmsnorm_gelu_dropout_kernel(input1, input2, output, normalized_shape, dropout_mask, dropout_p, eps, B, N, M, P):
    # Batch matrix multiplication
    Z1 = tl.zeros((B, N, P), dtype=tl.float32)
    for b in range(B):
        for n in range(N):
            for p in range(P):
                Z1[b, n, p] = tl.dot(input1[b, n, :], input2[b, :, p])

    # RMS Normalization
    mean_sq = tl.mean(Z1**2, axis=-1, keepdims=True)
    Z2 = Z1 / tl.sqrt(mean_sq + eps)

    # GELU Activation
    Z3 = 0.5 * Z2 * (1 + tl.tanh(tl.sqrt(2 / tl.pi) * (Z2 + 0.044715 * Z2**3)))

    # Dropout
    if tl.load(dropout_mask) < dropout_p:
        Z3 = 0  # Zero out the value based on dropout probability

    # Write output
    output[:] = Z3

import torch
import triton

def fused_bmm_rmsnorm_gelu_dropout(input1: torch.Tensor, input2: torch.Tensor, normalized_shape: torch.Size, 
                                    dropout_p: float = 0.1, eps: float = 1e-5, training: bool = True, 
                                    approximate: str = 'none', out: torch.Tensor = None) -> torch.Tensor:
    B, N, M = input1.shape
    _, M2, P = input2.shape

    if M != M2:
        raise ValueError("The shapes of input1 and input2 must be compatible for batch matrix multiplication.")

    # Prepare output tensor
    if out is None:
        out = torch.empty((B, N, P), dtype=input1.dtype, device=input1.device)

    # Prepare dropout mask
    dropout_mask = torch.empty((B, N, P), dtype=torch.float32, device=input1.device)
    if training:
        dropout_mask.uniform_()
        dropout_mask = (dropout_mask > dropout_p).float()  # Create a mask based on dropout probability

    # Launch Triton kernel
    grid = (B, N, P)
    fused_bmm_rmsnorm_gelu_dropout_kernel[grid](input1, input2, out, normalized_shape, dropout_mask, dropout_p, eps, B, N, M, P)

    return out
