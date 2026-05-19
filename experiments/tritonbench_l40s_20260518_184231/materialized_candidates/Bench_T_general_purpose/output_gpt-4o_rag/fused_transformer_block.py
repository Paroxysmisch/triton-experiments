import torch
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    pid = tl.program_id(0)
    # Compute the row and column of the block
    row = pid // (N // BLOCK_SIZE_N)
    col = pid % (N // BLOCK_SIZE_N)

    # Create pointers for the block
    a_block_ptr = a_ptr + row * BLOCK_SIZE_M * K
    b_block_ptr = b_ptr + col * BLOCK_SIZE_N

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        # Load blocks of A and B
        a_block = tl.load(a_block_ptr + k, mask=True)
        b_block = tl.load(b_block_ptr + k * N, mask=True)

        # Compute matrix multiplication for the block
        acc += tl.dot(a_block, b_block)

    # Store the result
    c_block_ptr = c_ptr + row * BLOCK_SIZE_M * N + col * BLOCK_SIZE_N
    tl.store(c_block_ptr, acc)

def fused_transformer_block(input, weight1, weight2, residual, dropout_p=0.1, eps=1e-5, *, out=None):
    # Matrix multiplication: Z1 = X W1
    X = input
    W1 = weight1
    Z1 = torch.empty((*X.shape[:-1], W1.shape[-1]), device=X.device, dtype=X.dtype)
    M, N, K = X.shape[-2], W1.shape[-1], X.shape[-1]
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 32, 32, 32  # Example block sizes

    grid = (M * N // (BLOCK_SIZE_M * BLOCK_SIZE_N),)
    matmul_kernel[grid](
        X, W1, Z1,
        M, N, K,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K
    )

    # Softmax: Z2 = softmax(Z1)
    Z2 = F.softmax(Z1, dim=-1)

    # Dropout: Z3 = dropout(Z2, p)
    Z3 = F.dropout(Z2, p=dropout_p, training=True)

    # Matrix multiplication: Z4 = Z3 W2
    W2 = weight2
    Z4 = torch.empty((*Z3.shape[:-1], W2.shape[-1]), device=Z3.device, dtype=Z3.dtype)
    M, N, K = Z3.shape[-2], W2.shape[-1], Z3.shape[-1]

    grid = (M * N // (BLOCK_SIZE_M * BLOCK_SIZE_N),)
    matmul_kernel[grid](
        Z3, W2, Z4,
        M, N, K,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K
    )

    # Addition with residual: Z4 = Z4 + R
    Z4 += residual

    # Layer Normalization: Y = LayerNorm(Z4, eps)
    Y = F.layer_norm(Z4, Z4.shape[-1:], eps=eps)

    # Output
    if out is not None:
        out.copy_(Y)
        return out
    return Y
