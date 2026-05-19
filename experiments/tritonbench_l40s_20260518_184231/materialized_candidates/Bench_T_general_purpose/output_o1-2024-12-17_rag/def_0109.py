import torch
import triton
import triton.language as tl

@triton.jit
def _threshold_invert_kernel(
    sigma_out_ptr, sigma_in_ptr,
    largest_sigma, rcond,
    n_elems,
    BLOCK_SIZE: tl.constexpr
):
    """
    Triton kernel to invert singular values (sigma), applying threshold:
    sigma_i^+ = 1 / sigma_i if sigma_i > rcond * largest_sigma else 0
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elems

    # Load input singular values
    val = tl.load(sigma_in_ptr + offsets, mask=mask)
    threshold_val = rcond * largest_sigma

    # Compute the masked inversion
    inv_val = tl.where(val > threshold_val, 1.0 / val, 0.0)
    tl.store(sigma_out_ptr + offsets, inv_val, mask=mask)

def pseudoinverse_svd(A, *, full_matrices=True, rcond=1e-15, out=None):
    """
    Computes the Moore-Penrose pseudoinverse of a matrix using SVD.

    Args:
        A (Tensor): Input tensor of shape (*, m, n), where * is zero or more batch dimensions.
    Keyword args:
        full_matrices (bool, optional): If True (default), compute the full SVD; else reduced SVD.
        rcond (float, optional): Relative condition number threshold. Singular values smaller
                                 than rcond * largest_singular_value are set to zero. Default: 1e-15.
        out (Tensor, optional): Output tensor. Ignored if None. Default: None.

    Returns:
        Tensor: The pseudoinverse of A.
    Math:
        A+ = V^H * Sigma+ * U^H
        Sigma+_i = 1 / sigma_i   if sigma_i > rcond * sigma_max
                    0           otherwise
    """
    # Perform SVD using PyTorch (supports batches and complex dtypes)
    U, S, Vh = torch.linalg.svd(A, full_matrices=full_matrices)

    # Flatten S for block-wise Triton processing
    # S is shape (*, min(m, n)) after torch.linalg.svd
    orig_shape = S.shape
    S_flat = S.reshape(-1)
    n_elems = S_flat.numel()

    # Find the largest singular value in the entire batch
    largest_sigma = float(S_flat.max())

    # Prepare output buffer for inverted singular values
    S_inv_flat = torch.empty_like(S_flat)

    # Determine block size for kernel
    BLOCK_SIZE = 1024
    grid = lambda meta: ( (n_elems + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'], )

    # Enqueue the Triton kernel for threshold-based inversion
    _threshold_invert_kernel[grid](
        S_inv_flat,           # sigma_out_ptr
        S_flat,               # sigma_in_ptr
        largest_sigma,        # largest_sigma
        rcond,                # rcond
        n_elems,              # n_elems
        BLOCK_SIZE=BLOCK_SIZE
    )

    # Reshape inverted singular values back
    S_inv = S_inv_flat.reshape(orig_shape)

    # Reconstruct the pseudoinverse: V^H * Sigma^+ * U^H
    # Sigma^+ must match final shape for multiplication
    # Expand S_inv into a diagonal matrix and multiply
    # Torch can handle batch multiplications
    m, n = A.shape[-2], A.shape[-1]
    # For convenient matmul shapes, use diag_embed
    S_inv_mat = torch.diag_embed(S_inv)

    # If not full_matrices, U, Vh shapes are (.., m, k) and (.., k, n) for k = min(m, n)
    # If full_matrices, shapes are (.., m, m) and (.., n, n), with S of shape (.., min(m,n))
    # In either case, we can multiply directly:
    if m >= n:
        # A is tall or square, S_inv_mat is (.., n, n), so result = Vh^H * S_inv_mat * U^H
        # => dims: (.., n, n) x (.., n, m)
        # but Vh is (.., n, n) if full_matrices else (.., n, min(m,n))
        # V^H is just Vh.conj().transpose(-2, -1)
        VH = Vh.conj().transpose(-2, -1)
        # result shape: (*, n, m)
        X = torch.matmul(VH, S_inv_mat)
        X = torch.matmul(X, U.conj().transpose(-2, -1))
    else:
        # A is wide, S_inv_mat is (.., m, m), so result = Vh^H * S_inv_mat * U^H
        # but effectively the same approach
        VH = Vh.conj().transpose(-2, -1)
        X = torch.matmul(VH, S_inv_mat)
        X = torch.matmul(X, U.conj().transpose(-2, -1))

    # If out is provided, copy results there
    if out is not None and isinstance(out, torch.Tensor):
        out[...] = X
        return out
    return X
