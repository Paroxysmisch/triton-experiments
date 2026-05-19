import triton
import triton.language as tl

@triton.jit
def svd_kernel(A_ptr, U_ptr, S_ptr, Vh_ptr, m, n, k, batch_size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    if pid >= batch_size:
        return

    # Pointers for the current batch
    A_batch_ptr = A_ptr + pid * m * n
    U_batch_ptr = U_ptr + pid * m * k
    S_batch_ptr = S_ptr + pid * k
    Vh_batch_ptr = Vh_ptr + pid * k * n

    # Load the matrix A
    A = tl.load(A_batch_ptr, eviction_policy="evict_last")

    # Perform SVD
    U, S, Vh = tl.linalg.svd(A, full_matrices=False)

    # Truncate to rank-k
    U_k = U[:, :k]
    S_k = S[:k]
    Vh_k = Vh[:k, :]

    # Store the results
    tl.store(U_batch_ptr, U_k)
    tl.store(S_batch_ptr, S_k)
    tl.store(Vh_batch_ptr, Vh_k)

import torch

def low_rank_svd_approximation(A, k, *, full_matrices=True, out=None) -> torch.Tensor:
    # Validate inputs
    if not isinstance(A, torch.Tensor):
        raise TypeError("A must be a torch.Tensor")
    if not (1 <= k <= min(A.shape[-2], A.shape[-1])):
        raise ValueError("k must satisfy 1 <= k <= min(m, n)")
    
    # Determine the output shape
    batch_shape = A.shape[:-2]
    m, n = A.shape[-2], A.shape[-1]
    U_shape = (*batch_shape, m, k)
    S_shape = (*batch_shape, k)
    Vh_shape = (*batch_shape, k, n)
    
    # Allocate output tensors if not provided
    if out is None:
        U = torch.empty(U_shape, dtype=A.dtype, device=A.device)
        S = torch.empty(S_shape, dtype=A.dtype, device=A.device)
        Vh = torch.empty(Vh_shape, dtype=A.dtype, device=A.device)
    else:
        U, S, Vh = out
        if U.shape != U_shape or S.shape != S_shape or Vh.shape != Vh_shape:
            raise ValueError("Output tensor shapes do not match expected shapes")
    
    # Determine the number of batches
    batch_size = 1 if len(batch_shape) == 0 else torch.prod(torch.tensor(batch_shape)).item()
    
    # Launch the Triton kernel
    grid = (batch_size,)
    svd_kernel[grid](A, U, S, Vh, m, n, k, batch_size, BLOCK_SIZE=1024)
    
    # Construct the rank-k approximation
    U_k = U
    S_k = torch.diag_embed(S)
    Vh_k = Vh
    A_k = U_k @ S_k @ Vh_k
    
    return A_k
