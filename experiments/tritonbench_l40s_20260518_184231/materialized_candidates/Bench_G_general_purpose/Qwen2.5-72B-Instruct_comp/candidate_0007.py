import triton
import triton.language as tl

# Constants
BLOCK_SIZE_M = 16
BLOCK_SIZE_N = 16
BLOCK_SIZE_K = 16
GROUP_SIZE_M = 8
EPS = 1e-5

@triton.jit
def ff_llama(
    x_ptr, w1_ptr, w3_ptr, rms_w_ptr, output_ptr,
    M, N, K, stride_xm, stride_xk, stride_w1k, stride_w1n, stride_w3k, stride_w3n, stride_rmsw,
    use_fp8: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Offsets for the input and output matrices
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    w1_ptrs = w1_ptr + (offs_k[:, None] * stride_w1k + offs_n[None, :] * stride_w1n)
    w3_ptrs = w3_ptr + (offs_k[:, None] * stride_w3k + offs_n[None, :] * stride_w3n)
    rms_w_ptrs = rms_w_ptr + offs_m

    # Load the weights and input
    x = tl.load(x_ptrs)
    w1 = tl.load(w1_ptrs)
    w3 = tl.load(w3_ptrs)
    rms_w = tl.load(rms_w_ptrs)

    # Compute the two matrix multiplications
    acc1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        x = tl.load(x_ptrs)
        w1 = tl.load(w1_ptrs)
        w3 = tl.load(w3_ptrs)
        acc1 += tl.dot(x, w1)
        acc2 += tl.dot(x, w3)
        x_ptrs += BLOCK_SIZE_K * stride_xk
        w1_ptrs += BLOCK_SIZE_K * stride_w1k
        w3_ptrs += BLOCK_SIZE_K * stride_w3k

    # Apply RMS scaling
    l2_norm = tl.sqrt(tl.sum(acc1 * acc1, axis=1) / N + EPS)
    acc1 = acc1 / l2_norm[:, None]

    # Apply scaled sigmoid activation
    acc1 = acc1 * tl.sigmoid(acc1)
    output = acc1 * acc2

    # Store the output
    output_ptrs = output_ptr + (offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xk)
    tl.store(output_ptrs, output)

import torch

def kernel_ff(x, w1, w3, rms_w, output):
    # Assert correct types and shapes
    assert x.dtype == torch.float32
    assert w1.dtype == torch.float32
    assert w3.dtype == torch.float32
    assert rms_w.dtype == torch.float32
    assert output.dtype == torch.float32
    assert x.shape[1] == w1.shape[0] == w3.shape[0]
    assert w1.shape[1] == w3.shape[1] == output.shape[1]
    assert x.shape[0] == output.shape[0]

    # Transpose weight matrices
    w1 = w1.t().contiguous()
    w3 = w3.t().contiguous()

    # Set up grid dimensions
    M, K = x.shape
    N = w1.shape[0]
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )

    # Invoke the Triton kernel
    ff_llama[grid](
        x, w1, w3, rms_w, output,
        M, N, K,
        x.stride(0), x.stride(1),
        w1.stride(0), w1.stride(1),
        w3.stride(0), w3.stride(1),
        rms_w.stride(0),
        use_fp8=False
    )

# Example usage
M, K, N = 1024, 1024, 1024
x = torch.randn((M, K), device='cuda', dtype=torch.float32)
w1 = torch.randn((K, N), device='cuda', dtype=torch.float32)
w3 = torch.randn((K, N), device='cuda', dtype=torch.float32)
rms_w = torch.randn((M,), device='cuda', dtype=torch.float32)
output = torch.empty((M, N), device='cuda', dtype=torch.float32)

kernel_ff(x, w1, w3, rms_w, output)
