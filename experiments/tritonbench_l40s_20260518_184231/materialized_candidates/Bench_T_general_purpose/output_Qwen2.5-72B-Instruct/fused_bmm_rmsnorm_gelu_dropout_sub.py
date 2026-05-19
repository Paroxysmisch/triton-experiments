import triton
import triton.language as tl

@triton.jit
def fused_bmm_rmsnorm_gelu_dropout_sub_kernel(
    X_ptr, Y_ptr, O_ptr, Z_ptr, B, N, M, P, dropout_p, training, approximate, eps, out_ptr,
    X_batch_stride, X_n_stride, X_m_stride,
    Y_batch_stride, Y_m_stride, Y_p_stride,
    O_batch_stride, O_n_stride, O_p_stride,
    Z_batch_stride, Z_n_stride, Z_p_stride,
    OUT_BATCH_STRIDE, OUT_N_STRIDE, OUT_P_STRIDE,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_P: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_n = tl.cdiv(P, BLOCK_SIZE_P)
    num_pid_k = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_in_group = num_pid_m * num_pid_n
    group_id = pid // num_pid_in_group
    b = group_id
    pid_mn = pid % num_pid_in_group
    pid_m = pid_mn // num_pid_n
    pid_n = pid_mn % num_pid_n
    offs_m = pid_m * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_n = pid_n * BLOCK_SIZE_P + tl.arange(0, BLOCK_SIZE_P)
    offs_k = tl.arange(0, BLOCK_SIZE_M)
    X = tl.load(X_ptr + b * X_batch_stride + offs_m[:, None] * X_n_stride + offs_k[None, :] * X_m_stride, mask=offs_m[:, None] < N and offs_k[None, :] < M, other=0.0)
    Y = tl.load(Y_ptr + b * Y_batch_stride + offs_k[:, None] * Y_m_stride + offs_n[None, :] * Y_p_stride, mask=offs_k[:, None] < M and offs_n[None, :] < P, other=0.0)
    Z = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_P), dtype=tl.float32)
    for k in range(0, M, BLOCK_SIZE_M):
        Z += tl.dot(X, Y)
        X = tl.load(X_ptr + b * X_batch_stride + offs_m[:, None] * X_n_stride + (k + offs_k)[None, :] * X_m_stride, mask=offs_m[:, None] < N and (k + offs_k)[None, :] < M, other=0.0)
        Y = tl.load(Y_ptr + b * Y_batch_stride + (k + offs_k)[:, None] * Y_m_stride + offs_n[None, :] * Y_p_stride, mask=(k + offs_k)[:, None] < M and offs_n[None, :] < P, other=0.0)
    
    # RMS normalization
    mean = tl.sum(Z * Z, axis=1) / P
    Z = Z / tl.sqrt(mean[:, None] + eps)
    
    # GELU activation
    if approximate == 'none':
        Z = 0.5 * Z * (1 + tl.erf(Z / 1.4142135623730951))
    elif approximate == 'tanh':
        Z = 0.5 * Z * (1 + tl.tanh(0.7978845608028654 * (Z + 0.044715 * Z * Z * Z)))
    
    # Dropout
    if training:
        mask = tl.rand(seed=pid) < dropout_p
        Z = tl.where(mask, 0.0, Z / (1 - dropout_p))
    
    # Subtraction
    O = tl.load(O_ptr + b * O_batch_stride + offs_n[None, :] * O_p_stride, mask=offs_n[None, :] < P, other=0.0)
    Z = Z - O
    
    # Store the result
    tl.store(Z_ptr + b * Z_batch_stride + offs_m[:, None] * Z_n_stride + offs_n[None, :] * Z_p_stride, Z, mask=offs_m[:, None] < N and offs_n[None, :] < P)

import torch
import triton
import triton.language as tl

def fused_bmm_rmsnorm_gelu_dropout_sub(input1, input2, other, normalized_shape, dropout_p=0.5, training=True, approximate='none', eps=1e-5, out=None):
    B, N, M = input1.shape
    _, _, P = input2.shape
    if out is None:
        out = torch.empty((B, N, P), dtype=input1.dtype, device=input1.device)
    
    # Define the grid and block sizes
    BLOCK_SIZE_B = 1
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_P = 16
    grid = (B * (N // BLOCK_SIZE_N) * (P // BLOCK_SIZE_P),)
    
    # Launch the kernel
    fused_bmm_rmsnorm_gelu_dropout_sub_kernel[grid](
        input1, input2, other, out, B, N, M, P, dropout_p, training, approximate, eps, out,
        input1.stride(0), input1.stride(1), input1.stride(2),
        input2.stride(0), input2.stride(1), input2.stride(2),
        other.stride(0), other.stride(1), other.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_SIZE_B, BLOCK_SIZE_N, BLOCK_SIZE_M, BLOCK_SIZE_P
    )
    
    return out

import torch

# Example tensors
B, N, M, P = 2, 32, 64, 128
input1 = torch.randn(B, N, M, device='cuda')
input2 = torch.randn(B, M, P, device='cuda')
other = torch.randn(B, 1, P, device='cuda')  # Broadcastable to (B, N, P)
normalized_shape = P

# Reference implementation using PyTorch
def reference_fused_bmm_rmsnorm_gelu_dropout_sub(input1, input2, other, normalized_shape, dropout_p=0.5, training=True, approximate='none', eps=1e-5):
    Z = torch.bmm(input1, input2)
    Z_norm = Z / torch.sqrt(torch.mean(Z**2, dim=-1, keepdim=True) + eps)
    if approximate == 'none':
        G = 0.5 * Z_norm * (1 + torch.erf(Z_norm / 1.4142135623730951))
    elif approximate == 'tanh':
        G = 0.5 * Z_norm * (1 + torch.tanh(0.7978845608028654 * (Z_norm + 0.044715 * Z_norm**3)))
    if training:
        D = torch.nn.functional.dropout(G, p=dropout_p, training=training)
    else:
        D = G
    Y = D - other
    return Y

# Test the Triton implementation
triton_output = fused_bmm_rmsnorm_gelu_dropout_sub(input1, input2, other, normalized_shape, dropout_p=0.5, training=True, approximate='none', eps=1e-5)
pytorch_output = reference_fused_bmm_rmsnorm_gelu_dropout_sub(input1, input2, other, normalized_shape, dropout_p=0.5, training=True, approximate='none', eps=1e-5)

# Check if the outputs are close
print(torch.allclose(triton_output, pytorch_output, atol=1e-4))
