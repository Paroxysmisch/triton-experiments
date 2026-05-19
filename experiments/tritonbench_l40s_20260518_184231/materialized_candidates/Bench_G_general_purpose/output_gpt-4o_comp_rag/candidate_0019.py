import triton
import triton.language as tl
import torch

@triton.jit
def ff_llama(x_ptr, w1_ptr, w3_ptr, rms_w_ptr, output_ptr, M, N, K, EPS, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K):
    pid = tl.program_id(axis=0)
    block_m = pid // (N // BLOCK_SIZE_N)
    block_n = pid % (N // BLOCK_SIZE_N)

    # Offsets
    offs_m = block_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = block_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Pointers to the current block
    x_ptrs = x_ptr + offs_m[:, None] * K + offs_k[None, :]
    w1_ptrs = w1_ptr + offs_k[:, None] * N + offs_n[None, :]
    w3_ptrs = w3_ptr + offs_k[:, None] * N + offs_n[None, :]
    rms_w_ptrs = rms_w_ptr + offs_k[:, None] * N + offs_n[None, :]

    # Load data
    x = tl.load(x_ptrs, mask=offs_m[:, None] < M, other=0.0)
    w1 = tl.load(w1_ptrs, mask=offs_n[None, :] < N, other=0.0)
    w3 = tl.load(w3_ptrs, mask=offs_n[None, :] < N, other=0.0)
    rms_w = tl.load(rms_w_ptrs, mask=offs_n[None, :] < N, other=0.0)

    # Compute accumulators
    acc1 = tl.dot(x, w1)
    acc2 = tl.dot(x, w3)

    # Apply RMS normalization
    rms = tl.sqrt(tl.sum(acc1 * acc1, axis=1) / K + EPS)
    acc1_norm = acc1 / rms[:, None]

    # Element-wise operations
    silu = acc1_norm * tl.sigmoid(acc1_norm)
    result = silu * acc2

    # Store the result
    output_ptrs = output_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(output_ptrs, result, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

### Wrapper Function

Next, we will define the `kernel_ff` wrapper function. This function will:
1. Reshape and prepare the inputs.
2. Set up the grid dimensions for parallel execution.
3. Invoke the Triton kernel.


# Input and weights
x = torch.randn(512, 256, device='cuda', dtype=torch.float32)
w1 = torch.randn(256, 512, device='cuda', dtype=torch.float32)
w3 = torch.randn(256, 512, device='cuda', dtype=torch.float32)
rms_w = torch.randn(256, 512, device='cuda', dtype=torch.float32)

# Call the kernel wrapper
output = kernel_ff(x, w1, w3, rms_w)
print(output)
