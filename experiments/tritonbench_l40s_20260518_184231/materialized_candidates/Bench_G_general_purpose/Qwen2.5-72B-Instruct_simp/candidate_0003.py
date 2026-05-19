import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Q, K, V, Out,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_km,
    stride_vb, stride_vh, stride_vm,
    stride_ob, stride_oh, stride_om,
    n_head, n_ctx, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    # Get the current block index
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(n_ctx, BLOCK_M)
    num_pid_n = tl.cdiv(n_ctx, BLOCK_N)
    num_pid_in_batch = num_pid_m * num_pid_n
    batch_id = pid // num_pid_in_batch
    pid %= num_pid_in_batch
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # Compute the offsets for the current block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    # Compute the base offsets for the current head and batch
    q_ptrs = Q + (batch_id * stride_qb + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qh)
    k_ptrs = K + (batch_id * stride_kb + offs_n[:, None] * stride_km + offs_d[None, :] * stride_kh)
    v_ptrs = V + (batch_id * stride_vb + offs_n[:, None] * stride_vm + offs_d[None, :] * stride_vh)
    o_ptrs = Out + (batch_id * stride_ob + offs_m[:, None] * stride_om + offs_d[None, :] * stride_oh)

    # Initialize the output and the attention scores
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    # Compute the dot product of Q and K
    q = tl.load(q_ptrs)
    k = tl.load(k_ptrs)
    qk = tl.dot(q, k, trans_b=True) * scale

    # Apply the softmax function to the attention scores
    qk = tl.softmax(qk, axis=1)

    # Compute the weighted sum of V
    v = tl.load(v_ptrs)
    acc += tl.dot(qk, v)

    # Store the result to the output tensor
    tl.store(o_ptrs, acc)

import triton
import triton.language as tl
import torch

def context_attention_fwd(Q, K, V, Out, n_head, n_ctx, scale):
    # Get the dimensions of the input tensors
    batch_size, _, _ = Q.shape

    # Define the block and grid sizes
    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_D = 64
    grid = (triton.cdiv(n_ctx, BLOCK_M) * triton.cdiv(n_ctx, BLOCK_N) * batch_size,)

    # Define the strides for the input and output tensors
    stride_qb = Q.stride(0)
    stride_qh = Q.stride(1)
    stride_qm = Q.stride(2)

    stride_kb = K.stride(0)
    stride_kh = K.stride(1)
    stride_km = K.stride(2)

    stride_vb = V.stride(0)
    stride_vh = V.stride(1)
    stride_vm = V.stride(2)

    stride_ob = Out.stride(0)
    stride_oh = Out.stride(1)
    stride_om = Out.stride(2)

    # Launch the Triton kernel
    _fwd_kernel[grid](
        Q, K, V, Out,
        stride_qb, stride_qh, stride_qm,
        stride_kb, stride_kh, stride_km,
        stride_vb, stride_vh, stride_vm,
        stride_ob, stride_oh, stride_om,
        n_head, n_ctx, scale,
        BLOCK_M, BLOCK_N, BLOCK_D
    )

# Example usage
batch_size = 2
n_head = 4
n_ctx = 128
d_model = 64
scale = 1.0 / (d_model ** 0.5)

Q = torch.randn(batch_size, n_head, n_ctx, d_model, device='cuda')
K = torch.randn(batch_size, n_head, n_ctx, d_model, device='cuda')
V = torch.randn(batch_size, n_head, n_ctx, d_model, device='cuda')
Out = torch.empty_like(Q)

context_attention_fwd(Q, K, V, Out, n_head, n_ctx, scale)

print(Out)
