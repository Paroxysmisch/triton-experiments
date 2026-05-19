import triton
import triton.language as tl

@triton.jit
def ff_llama(
    x_ptr, w1_ptr, w3_ptr, rms_w_ptr, output_ptr,
    M, N, K,
    stride_xm, stride_xn,
    stride_w1k, stride_w1n,
    stride_w3k, stride_w3n,
    stride_rms_n,
    stride_outm, stride_outn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = num_pid_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * num_pid_m
    group_size = num_pid_in_group
    pid_m = first_pid_m + (pid % num_pid_m)
    pid_n = (pid % group_size) // num_pid_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xn)
    w1_ptrs = w1_ptr + (offs_k[:, None] * stride_w1k + offs_n[None, :] * stride_w1n)
    w3_ptrs = w3_ptr + (offs_k[:, None] * stride_w3k + offs_n[None, :] * stride_w3n)

    accumulator1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    accumulator3 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        x = tl.load(x_ptrs)
        w1 = tl.load(w1_ptrs)
        w3 = tl.load(w3_ptrs)

        accumulator1 += tl.dot(x, w1)
        accumulator3 += tl.dot(x, w3)

        x_ptrs += BLOCK_SIZE_K * stride_xn
        w1_ptrs += BLOCK_SIZE_K * stride_w1k
        w3_ptrs += BLOCK_SIZE_K * stride_w3k

    # Apply SILU activation to accumulator1
    accumulator1 = tl.sigmoid(accumulator1) * accumulator1

    # Normalize using L2 norm
    rms_w = tl.load(rms_w_ptr + offs_n)
    l2_norm = tl.sqrt(tl.sum(accumulator1 * accumulator1, axis=1) + 1e-6)
    l2_norm = l2_norm[:, None]
    accumulator1 /= l2_norm

    # Scale by RMS weight
    accumulator1 *= rms_w

    # Combine results
    output = accumulator1 * accumulator3

    # Store the result
    out_ptrs = output_ptr + (offs_m[:, None] * stride_outm + offs_n[None, :] * stride_outn)
    tl.store(out_ptrs, output)

import torch

def kernel_ff(x, w1, w3, rms_w):
    assert x.dtype in [torch.float16, torch.int8]
    assert w1.dtype in [torch.float16, torch.int8]
    assert w3.dtype in [torch.float16, torch.int8]
    assert rms_w.dtype == torch.float16

    M, K = x.shape
    N = w1.shape[1]

    output = torch.empty((M, N), dtype=torch.float16, device=x.device)

    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 32

    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)

    ff_llama[grid](
        x, w1, w3, rms_w, output,
        M, N, K,
        x.stride(0), x.stride(1),
        w1.stride(0), w1.stride(1),
        w3.stride(0), w3.stride(1),
        rms_w.stride(0),
        output.stride(0), output.stride(1),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )

    return output
