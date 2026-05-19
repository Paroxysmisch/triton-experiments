import triton
import triton.language as tl

@triton.jit
def rbe_triton(
    x_ptr,  # Pointer to the input tensor
    out_ptr,  # Pointer to the output tensor
    freq_ptr,  # Pointer to the precomputed frequency values
    batch,  # Batch size
    M,  # Dimension M
    K,  # Dimension K
    BLOCK_SIZE_M: tl.constexpr,  # Block size for M dimension
    BLOCK_SIZE_K: tl.constexpr,  # Block size for K dimension
):
    # Get the program ID
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    # Compute the offsets for the block
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_batch = tl.arange(0, batch)

    # Create a mask to avoid out-of-bounds memory access
    mask = (offs_m < M) & (offs_k < K)

    # Load the input data
    x = tl.load(x_ptr + offs_batch[:, None, None] * M * K + offs_m[None, :, None] * K + offs_k[None, None, :], mask=mask, other=0.0)

    # Separate real and imaginary parts
    real = tl.where((offs_m[None, :, None] % 2 == 0), x, 0.0)
    imag = tl.where((offs_m[None, :, None] % 2 != 0), x, 0.0)

    # Load the precomputed frequency values
    freq = tl.load(freq_ptr + offs_k[None, None, :], mask=mask, other=0.0)

    # Compute the position-dependent transformation
    out_real = real * tl.cos(freq) - imag * tl.sin(freq)
    out_imag = real * tl.sin(freq) + imag * tl.cos(freq)

    # Store the output data
    tl.store(out_ptr + offs_batch[:, None, None] * M * K + offs_m[None, :, None] * K + offs_k[None, None, :], out_real + out_imag, mask=mask)

import triton
import triton.language as tl
import torch

def get_freq_multi_tokens(K, theta=10000.0):
    k = torch.arange(0, K, dtype=torch.float32)
    freq = 1.0 / (theta ** (k / K))
    return freq

def rbe_triton_wrapper(x, freq):
    batch, M, K = x.shape
    BLOCK_SIZE_M = 2
    BLOCK_SIZE_K = 1024

    # Ensure the input tensor is on the GPU
    x = x.cuda()
    freq = freq.cuda()

    # Allocate the output tensor
    out = torch.empty_like(x).cuda()

    # Define the grid and block dimensions
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(K, BLOCK_SIZE_K), batch)

    # Launch the kernel
    rbe_triton[grid](
        x, out, freq,
        batch, M, K,
        BLOCK_SIZE_M, BLOCK_SIZE_K
    )

    return out

# Example usage
if __name__ == "__main__":
    batch = 2
    M = 10
    K = 2048
    x = torch.randn(batch, M, K, dtype=torch.float32)
    freq = get_freq_multi_tokens(K)

    out = rbe_triton_wrapper(x, freq)
    print(out)
