|import|import triton
import triton.language as tl
import torch

# Triton kernel to compute the upper triangular part of a 2D matrix
@triton.jit
def triu_kernel(
    X,  # Pointer to the input matrix
    Y,  # Pointer to the output matrix
    M,  # Number of rows in the input matrix
    N,  # Number of columns in the input matrix
    diagonal,  # Diagonal offset
    M_BLOCK_SIZE: tl.constexpr,  # Block size for rows
    N_BLOCK_SIZE: tl.constexpr,  # Block size for columns
):
    pid = tl.program_id(0)
    row = pid * M_BLOCK_SIZE + tl.arange(0, M_BLOCK_SIZE)[:, None]
    m_mask = row < M
    X += row * N
    Y += row * N

    for n_offset in range(0, N, N_BLOCK_SIZE):
        cols = n_offset + tl.arange(0, N_BLOCK_SIZE)[None, :]
        n_mask = cols < N
        mask = m_mask and n_mask

        x = tl.load(X + cols, mask, other=0.0)
        y = tl.where(row + diagonal <= cols, x, 0.0)
        tl.store(Y + cols, y, mask=mask)

# Triton kernel to compute the upper triangular part for batched matrices
@triton.jit
def triu_batch_kernel(
    X,  # Pointer to the input matrix batch
    Y,  # Pointer to the output matrix batch
    batch,  # Number of matrices in the batch
    MN,  # Number of elements in each matrix (flattened)
    N,  # Number of columns in each matrix
    diagonal,  # Diagonal offset
    BATCH_BLOCK_SIZE: tl.constexpr,  # Block size for batch processing
    MN_BLOCK_SIZE: tl.constexpr,  # Block size for elements within a matrix
):
    batch_id = tl.program_id(0)
    mn_id = tl.program_id(1)
    row = batch_id * BATCH_BLOCK_SIZE + tl.arange(0, BATCH_BLOCK_SIZE)[:, None]
    batch_mask = row < batch
    X += row * MN
    Y += row * MN

    cols = mn_id * MN_BLOCK_SIZE + tl.arange(0, MN_BLOCK_SIZE)[None, :]
    mn_mask = cols < MN
    mask = batch_mask and mn_mask
    x = tl.load(X + cols, mask, other=0.0)
    m = cols // N
    n = cols % N
    y = tl.where(m + diagonal <= n, x, 0.0)
    tl.store(Y + cols, y, mask=mask)

# Function to compute the upper triangular part of a matrix or batch of matrices
def triu(A, diagonal=0):
    A = A.contiguous()
    out = torch.empty_like(A)
    assert len(A.shape) > 1, "Input tensor must have at least 2 dimensions"
    M, N = A.shape[-2:]
    with torch.cuda.device(A.device):
        if len(A.shape) == 2:
            grid = lambda meta: (triton.cdiv(M, meta["M_BLOCK_SIZE"]),)
            triu_kernel[grid](A, out, M, N, diagonal, M_BLOCK_SIZE=32, N_BLOCK_SIZE=8)
        else:
            batch = int(torch.numel(A) / M / N)
            B = A.view(batch, -1)
            grid = lambda meta: (
                triton.cdiv(batch, meta["BATCH_BLOCK_SIZE"]),
                triton.cdiv(M * N, meta["MN_BLOCK_SIZE"]),
            )
            triu_batch_kernel[grid](B, out, batch, M * N, N, diagonal, BATCH_BLOCK_SIZE=32, MN_BLOCK_SIZE=8)
            out = out.view(A.shape)
    return out

# Triton kernel to solve a system of linear equations given its Cholesky decomposition
@triton.jit
def cholesky_solve_kernel(
    B,  # Pointer to the right-hand side matrix
    L,  # Pointer to the lower triangular Cholesky decomposition matrix
    n,  # Dimension of the matrix
    BLOCK_SIZE: tl.constexpr,  # Block size for processing rows
):
    pid = tl.program_id(0)
    row = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[:, None]
    mask = row < n
    B += row
    L += row * n

    b = tl.load(B, mask, other=0.0)
    x = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    for j in range(n):
        l = tl.load(L + j, mask=mask, other=0.0)
        x += l[None, :] * b[None, j]

    tl.store(B, x.to(B.dtype.element_ty), mask=mask)

# Triton kernel to solve a batch of systems of linear equations given their Cholesky decompositions
@triton.jit
def cholesky_solve_batch_kernel(
    B,  # Pointer to the right-hand side matrix batch
    L,  # Pointer to the lower triangular Cholesky decomposition matrix batch
    batch,  # Number of matrices in the batch
    n,  # Dimension of each matrix
    BATCH_BLOCK_SIZE: tl.constexpr,  # Block size for batch processing
):
    batch_id = tl.program_id(0)
    row = batch_id * BATCH_BLOCK_SIZE + tl.arange(0, BATCH_BLOCK_SIZE)[:, None]
    batch_mask = row < batch
    B += row * n
    L += row * n

    b = tl.load(B, batch_mask, other=0.0)
    x = tl.zeros((BATCH_BLOCK_SIZE, n), dtype=tl.float32)

    for j in range(n):
        l = tl.load(L + j, batch_mask, other=0.0)
        x += l[None, :] * b[:, None]

    tl.store(B, x.to(B.dtype.element_ty), mask=batch_mask)

# Function to solve a system of linear equations given its Cholesky decomposition
def cholesky_solve(B, L, upper=False, *, out=None):
    if not isinstance(B, torch.Tensor):
        raise TypeError("Argument B must be a torch.Tensor")
    if not isinstance(L, torch.Tensor):
        raise TypeError("Argument L must be a torch.Tensor")
    if B.is_complex() and L.is_complex():
        dtype = torch.complex64
    elif B.is_floating_point() and L.is_floating_point():
        dtype = torch.float32
    else:
        raise ValueError("Both arguments must be complex or floating point")

    if B.ndim < 2:
        raise ValueError("Argument B must be at least 2-dimensional")
    if L.ndim < 2:
        raise ValueError("Argument L must be at least 2-dimensional")
    if B.shape[-2] != L.shape[-1]:
        raise ValueError("Incompatible dimensions")
    if out is not None and B.ndim != out.ndim:
        raise ValueError("Incompatible dimensions")

    if out is None:
        out = torch.empty_like(B, dtype=dtype)
    else:
        out = out.to(d
