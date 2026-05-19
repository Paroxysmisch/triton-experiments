import torch
import triton
import triton.language as tl
from triton.runtime.driver import register_driver


# The cuSOLVER gesvdj driver is currently only available in CUDA 11.4+.
@register_driver("gesvdj")
def gesvdj_driver(svd_kwargs):
    return "gesvdj", "gesvdj_batched"


# The cuSOLVER gesvda driver is currently only available in CUDA 11.4+.
@register_driver("gesvda")
def gesvda_driver(svd_kwargs):
    return "gesvda", "gesvda_batched"


@triton.jit
def svd_fwd_kernel(A, U, S, V, m, n, full_matrices, min_mn, reduction_factor, **meta):
    # This kernel computes the singular value decomposition of a matrix A.
    # The result is given as the product U \diag(S) V^{H}.
    # If A is a batch of matrices, then the output has the same batch dimensions.
    # The matrices U and V are not unique, nor are they continuous with respect to A.
    # Gradients computed using U or Vh will only be finite when A does not have repeated singular values.
    # Differences with numpy.linalg.svd: Unlike numpy.linalg.svd, this function always returns a tuple of three tensors and it doesn't support compute_uv argument.
    # Please use torch.linalg.svdvals, which computes only the singular values, instead of compute_uv=False.
    # When full_matrices=True, the gradients with respect to U[..., :, min(m, n):] and Vh[..., min(m, n):, :] will be ignored, as those vectors can be arbitrary bases of the corresponding subspaces.
    # A = U \operatorname{diag}(S) V^{\text{H}} \mathrlap{\qquad U \in \mathbb{K}^{m \times m}, S \in \mathbb{R}^k, V \in \mathbb{K}^{n \times n}}

    # CUDA 11.4+ only: use the gesvdj driver for SVD, which is faster and more accurate.
    # If the driver is not available, fall back to the gesvd or gesvda driver.
    if meta["driver"] in ("gesvdj", "gesvdj_batched"):
        return gesvdj_fwd_kernel(A, U, S, V, m, n, full_matrices, min_mn, reduction_factor, **meta)
    elif meta["driver"] in ("gesvda", "gesvda_batched"):
        return gesvda_fwd_kernel(A, U, S, V, m, n, full_matrices, min_mn, reduction_factor, **meta)
    else:
        return gesvd_fwd_kernel(A, U, S, V, m, n, full_matrices, min_mn, reduction_factor, **meta)


@triton.jit
def gesvdj_fwd_kernel(A, U, S, V, m, n, full_matrices, min_mn, reduction_factor, **meta):
    # This kernel uses the cuSOLVER gesvdj driver to compute the singular value decomposition of a matrix A.
    # The result is given as the product U \diag(S) V^{H}.
    # If A is a batch of matrices, then the output has the same batch dimensions.
    # The matrices U and V are not unique, nor are they continuous with respect to A.
    # Gradients computed using U or Vh will only be finite when A does not have repeated singular values.
    # Differences with numpy.linalg.svd: Unlike numpy.linalg.svd, this function always returns a tuple of three tensors and it doesn't support compute_uv argument.
    # Please use torch.linalg.svdvals, which computes only the singular values, instead of compute_uv=False.
    # When full_matrices=True, the gradients with respect to U[..., :, min(m, n):] and Vh[..., min(m, n):, :] will be ignored, as those vectors can be arbitrary bases of the corresponding subspaces.
    # A = U \operatorname{diag}(S) V^{\text{H}} \mathrlap{\qquad U \in \mathbb{K}^{m \times m}, S \in \mathbb{R}^k, V \in \mathbb{K}^{n \times n}}

    # The gesvdj driver computes the SVD with a reduced-SVD algorithm when m or n is less than or equal to 16.
    # To allow for larger m and n, we can use the gesvdj_batched driver, which computes the SVD in a number of smaller batches.
    # The reduction_factor argument specifies the size of each batch.
    batch_size = triton.cdiv(min_mn, reduction_factor)
    batch_idx = tl.program_id(0)
    A += batch_idx * m * n
    U += batch_idx * m * m
    V += batch_idx * n * n
    if full_matrices:
        gesvdj_fwd_kernel_impl(A, U, S, V, m, n, batch_size, **meta)
    else:
        gesvdj_fwd_kernel_impl_r(A, U, S, V, m, n, batch_size, **meta)


@triton.jit
def gesvdj_fwd_kernel_impl(A, U, S, V, m, n, batch_size, **meta):
    # This kernel computes the singular value decomposition of a batch of matrices A.
    # The result is given as the product U \diag(S) V^{H}.
    # If A is a batch of matrices, then the output has the same batch dimensions.
    # The matrices U and V are not unique, nor are they continuous with respect to A.
    # Gradients computed using U or Vh will only be finite when A does not have repeated singular values.
    # Differences with numpy.linalg.svd: Unlike numpy.linalg.svd, this function always returns a tuple of three tensors and it doesn't support compute_uv argument.
    # Please use torch.linalg.svdvals, which computes only the singular values, instead of compute_uv=False.
    # When full_matrices=True, the gradients with respect to U[..., :, min(m, n):] and Vh[..., min(m, n):, :] will be ignored, as those vectors can be arbitrary bases of the corresponding subspaces.
    # A = U \operatorname{diag}(S) V^{\text{H}} \mathrlap{\qquad U \in \mathbb{K}^{m \times m}, S \in \mathbb{R}^k, V \in \mathbb{K}^{n \times n}}

    # Compute the SVD of each matrix in the batch.
    for i in range(batch_size):
        gesvdj_fwd_kernel_impl_one(A, U, S, V, m, n, i, **meta)
        A += m * n
        U += m * m
        V += n * n


@triton.jit
def gesvdj_fwd_kernel_impl_r(A, U, S, V, m, n, batch_size, **meta):
    # This kernel computes the singular value decomposition of a batch of matrices A.
    # The result is given as the product U \diag(S) V^{H}.
    # If A is a batch of matrices, then the output has the same batch dimensions.
    # The matrices U and V are not unique, nor are they continuous with respect to A.
    # Gradients computed using U or Vh will only be finite when A does not have repeated singular values.
    # Differences with numpy.linalg.svd: Unlike numpy.linalg.svd, this function always returns a tuple of three tensors and it doesn't support compute_uv argument.
    # Please use torch.linalg.svdvals, which computes only the singular values, instead of compute_uv=False.
    # When full_matrices=True
