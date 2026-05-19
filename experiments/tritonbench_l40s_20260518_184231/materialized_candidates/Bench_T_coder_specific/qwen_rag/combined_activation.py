import torch
import triton
import triton.language as tl

# Define constants
BLOCK_SIZE = 64

@triton.jit
def sigmoid_kernel(X, Y, N):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + offsets)
    sigm = 1 / (1 + tl.exp(-x))
    tl.store(Y + offsets, sigm)

@triton.jit
def matmul_kernel(X, W, Y, N, K, M):
    pid = tl.program_id(axis=0)
    block_x = pid % M
    block_y = pid // M
    m = block_y * BLOCK_SIZE
    n = block_x * BLOCK_SIZE
    partial = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for k in range(K):
        a = tl.load(X + m * K + k)
        b = tl.load(W + k * M + n)
        partial += a * b
    tl.store(Y + m * M + n, partial)

@triton.jit
def add_bias_kernel(X, B, Y, N, M):
    pid = tl.program_id(axis=0)
    block_x = pid % M
    block_y = pid // M
    m = block_y * BLOCK_SIZE
    n = block_x * BLOCK_SIZE
    x = tl.load(X + m * M + n)
    b = tl.load(B + n)
    y = x + b
    tl.store(Y + m * M + n, y)

@triton.jit
def tanh_kernel(X, Y, N):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + offsets)
    tanh_val = (tl.exp(x) - tl.exp(-x)) / (tl.exp(x) + tl.exp(-x))
    tl.store(Y + offsets, tanh_val)

@triton.jit
def elementwise_mul_kernel(X, Y, Z, N):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + offsets)
    y = tl.load(Y + offsets)
    z = x * y
    tl.store(Z + offsets, z)

def combined_activation(input, weight1, weight2, bias, out=None):
    if out is None:
        out = torch.empty_like(input)

    batch_size, _, din = input.shape
    dout, dout_din = weight2.shape
    din_out = weight1.shape[-1]

    # Step 1: Compute sigmoid(X @ weight1)
    intermed = torch.empty(batch_size, dout, din_out, device=input.device, dtype=torch.float32)
    matmul_kernel[input.numel() // (batch_size * dout)](input, weight1, intermed, batch_size, dout, din_out)
    sigmoid_kernel[intermed.numel() // BLOCK_SIZE](intermed, intermed, din_out)

    # Step 2: Element-wise multiply with weight2
    elementwise_mul_kernel[intermed.numel() // BLOCK_SIZE](intermed, weight2, out, dout, din_out)

    # Step 3: Add bias
    add_bias_kernel[out.numel() // BLOCK_SIZE](out, bias, out, dout, din_out)

    return out

# Example usage
if __name__ == "__main__":
    batch_size = 2
    N = 5
    D_in = 3
    D_out = 4
    input = torch.randn(batch_size, N, D_in, device='cuda')
    weight1 = torch.randn(D_in, D_out, device='cuda')
    weight2 = torch.randn(D_out, device='cuda')
    bias = torch.randn(D_out, device='cuda')

    output = combined_activation(input, weight1, weight2, bias)
    print(output)
