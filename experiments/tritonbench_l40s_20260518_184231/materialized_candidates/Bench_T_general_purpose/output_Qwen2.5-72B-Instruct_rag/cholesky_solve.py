import triton
import triton.language as tl
import torch

# Triton kernel to solve the system of linear equations using Cholesky decomposition
@triton.jit
def cholesky_solve_kernel(
    B,  # Pointer to the right-hand side tensor
    L,  # Pointer to the Cholesky decomposition tensor
    X,  # Pointer to the output tensor
    batch,  # Number of batches
    n,  # Size of the matrix (n x n)
    k,  # Number of right-hand side vectors (n x k)
    upper,  # Flag indicating if L is upper triangular
    BLOCK_SIZE: tl.constexpr,  # Block size for processing
):
    pid = tl.program_id(0)
    bid = tl.program_id(1)

    if bid >= batch:
        return

    B += bid * n * k
    L += bid * n * n
    X += bid * n * k

    for i in range(0, n, BLOCK_SIZE):
        for j in range(0, k, BLOCK_SIZE):
            # Load B block
            b_block = tl.load(B + i * k + j, mask=(i + tl.arange(0, BLOCK_SIZE)[:, None] < n) & (j + tl.arange(0, BLOCK_SIZE)[None, :] < k), other=0.0)

            for l in range(0, n, BLOCK_SIZE):
                # Load L block
                l_block = tl.load(L + i * n + l, mask=(i + tl.arange(0, BLOCK_SIZE)[:, None] < n) & (l + tl.arange(0, BLOCK_SIZE)[None, :] < n), other=0.0)

                # Compute the solution block
                if upper:
                    x_block = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
                    for m in range(l, -1, -BLOCK_SIZE):
                        l_block_m = tl.load(L + m * n + l, mask=(m + tl.arange(0, BLOCK_SIZE)[:, None] < n) & (l + tl.arange(0, BLOCK_SIZE)[None, :] < n), other=0.0)
                        x_block -= tl.dot(l_block_m, x_block, allow_tf32=False)
                    x_block /= l_block
                else:
                    x_block = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
                    for m in range(0, l + BLOCK_SIZE, BLOCK_SIZE):
                        l_block_m = tl.load(L + i * n + m, mask=(i + tl.arange(0, BLOCK_SIZE)[:, None] < n) & (m + tl.arange(0, BLOCK_SIZE)[None, :] < n), other=0.0)
                        x_block -= tl.dot(l_block_m, x_block, allow_tf32=False)
                    x_block /= l_block

                # Store the solution block
                tl.store(X + i * k + j, x_block, mask=(i + tl.arange(0, BLOCK_SIZE)[:, None] < n) & (j + tl.arange(0, BLOCK_SIZE)[None, :] < k))

# Wrapper function to call the Triton kernel
def cholesky_solve(B, L, upper=False, *, out=None):
    B = B.contiguous()
    L = L.contiguous()
    if out is None:
        out = torch.empty_like(B)
    else:
        out = out.contiguous()

    batch = B.shape[:-2] if B.dim() > 2 else 1
    n = B.shape[-2]
    k = B.shape[-1]

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]), triton.cdiv(batch, meta["BLOCK_SIZE"]))
    cholesky_solve_kernel[grid](B, L, out, batch, n, k, upper, BLOCK_SIZE=32)

    return out
