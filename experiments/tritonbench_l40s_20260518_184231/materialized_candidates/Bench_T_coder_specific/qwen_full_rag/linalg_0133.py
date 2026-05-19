import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.autograd.function import Function
from torch.library import custom_bwd, custom_fwd
from .utils import calculate_settings

class _LdlFactor(Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, A: Tensor, hermitian: bool = False, out: tuple = None):
        dtype = A.dtype
        assert dtype in (torch.float32, torch.float64, torch.complex64, torch.complex128), "Unsupported dtype."

        is_complex = dtype in (torch.complex64, torch.complex128)
        M = A.shape[-1]
        N = A.shape[-2]

        batch_shape = A.shape[:-2]
        if len(batch_shape) > 0:
            A = A.reshape(-1, N, M)
        
        if A.stride(-1) != 1 and A.stride(-2) != 1:
            A = A.contiguous()

        k = M

        if is_complex and not hermitian:
            # For non-Hermitian complex matrices, we know that U is just the transpose of L
            # So we can save some work by only computing the lower triangular part
            k = M // 2

        # We want to allocate as little memory as possible. For this reason, we will use strided
        # storage for the output buffers. This means that the output tensors will not be contiguous.
        # As long as we expose a well-defined API, this should not be a problem for users.
        if out is None:
            # Allocate output tensors. Stride is set to 1 for all but the last dimension to enable
            # broadcasting along the batch dimensions.
            LD = torch.empty((N, k), dtype=dtype, device=A.device, stride=(A.stride(-2), A.stride(-1)))
            pivots = torch.empty((M,), dtype=torch.int32, device=A.device, stride=(A.stride(-1),))
        else:
            # User-provided output buffers
            LD, pivots = out
        
        # The algorithm requires the input tensor to be in column-major order, so we will store the
        # strides accordingly. This way, we can simply ignore the strides when indexing the tensors.
        # In the end, we will convert the result back to the original input layout.
        original_layout = A.layout
        A = A.clone().contiguous()

        grid, _, _ = calculate_settings(M, N)

        _ldl_factor_kernel[grid](
            A,
            LD,
            pivots,
            M,
            N,
            k,
            A.stride(0),
            A.stride(1),
            LD.stride(0),
            LD.stride(1),
            pivots.stride(0),
            is_complex,
            hermitian,
            1,
            1,
        )

        # Convert the result back to the original layout
        if original_layout == torch.strided:
            LD = LD.contiguous()
            pivots = pivots.contiguous()

        if len(batch_shape) > 0:
            m = M if not is_complex or hermitian else M // 2
            ld_strides = (LD.stride(0), LD.stride(1)) if LD.is_contiguous() else (LD.stride(1), LD.stride(0))
            LD = LD.view(batch_shape + (m, N)).clone().reshape(batch_shape + (N, k))
            pivots = pivots.view(batch_shape + (m,))

        ctx.save_for_backward(LD, A)
        ctx.hermitian = hermitian
        ctx.is_complex = is_complex

        return LD, pivots
    
    @staticmethod
    @custom_bwd
    def backward(ctx, dLD: Tensor, dpivots: Tensor):
        raise RuntimeError("The LDL factorization is a factorization and thus differentiable by construction. "
                           "Therefore, there is no need to manually define a backward pass.")

def ldl_factor(A: Tensor, *, hermitian: bool = False, out: tuple = None) -> (Tensor, Tensor):
    r"""
    Computes a compact representation of the LDL factorization of a Hermitian or symmetric (possibly indefinite) matrix.

    .. note:: 
        The implementation assumes that the input matrix is either Hermitian (if complex) or symmetric (if real).
        There is no runtime check for symmetry/hersmitian-ness, and passing a non-symmetric/hermitian matrix may lead
        to incorrect results.

    Args:
        A (Tensor): tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions consisting of symmetric or Hermitian matrices.
    
    Keyword Args: 
        hermitian (bool, optional): whether to consider the input to be Hermitian or symmetric. For real-valued matrices, this switch has no effect. Default: ``False``.
        out (tuple, optional): tuple of two tensors to write the output to. Ignored if ``None``. Default: ``None``.
    
    Returns:
        A namedtuple `(LD, pivots)` containing the factorized matrix and the pivot indices. The `LD` tensor contains both the lower triangular matrix ``L`` and the diagonal matrix ``D`` in a compact form, following the format specified by LAPACK's :obj:`sytrf` routine. If input tensors are broadcasted to shape :obj:`(*, K, N)`, the output tensors will have the shape :obj:`(*, N, K)`.

    Example:
    
        >>> A = torch.tensor([[4.0, 3.0], [3.0, 4.0]], dtype=torch.float32, device='cuda')
        >>> A
        tensor([[4.0000, 3.0000],
                [3.0000, 4.0000]], dtype=torch.float32, device='cuda')
        >>> LD, pivots = linalg.ldl_factor(A)
        >>> LD
        tensor([[[ 2.0000,  0.7500],
                 [ 0.0000,  1.3333]],
        
                [[ 2.0000, -0.7500],
                 [ 0.0000,  1.3333]]], dtype=torch.float32, device='cuda')
        >>> pivots
        tensor([[1, 2]], dtype=torch.int32, device='cuda')
    """
    return _LdlFactor.apply(A, hermitian, out)

@triton.jit
def _ldl_factor_kernel(
    A,
    LD,
    pivots,
    M,
    N,
    k,
    stride_am,
    stride_ak,
    stride_ld,
    stride_lk,
    stride_p,
    is_complex: tl.constexpr,
    hermitian: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    # This program processes BLOCK_SIZE_M rows and BLOCK_SIZE_N columns.
    # It will produce a compact upper triangular matrix of size (BLOCK_SIZE_M, BLOCK_SIZE_N)
    # which contains the U part of the factorization.
    # For example, if M=3, N=4 and the program is invoked with pid=0, it will process
    # the submatrix consisting of the first 3 rows and the first 4 columns.
    offset_m = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_n = tl.arange(0, BLOCK_SIZE_N)
    rn = tl.max_contiguous(tl.multiple_of(offset_n % N, BLOCK_SIZE_N), BLOCK_SIZE_N)
    A = A + (offset_m[:, None] * stride_am + offset_n[None, :] * stride_ak)
    LD = LD + (offset_m[:, None] * stride_ld + offset_n[None, :] * stride_lk)
    eps = 1e-9 if A.dtype in [tl.float32, tl.bfloat16] else 1e-15
    o = tl.zeros((BLOCK_SIZE_M, ), dtype=tl.float32) + eps

    _do = tl.load(pivots + offset_m, mask=offset_m < M, other=0)
    p = _do.to(tl.int32)
    # We now construct the U matrix in compact form. It will look something like this (assuming M=3, N=4):
    # [ u11 u12 u13 ]
    # [   u22 u23 ]
    # [     u33 ]
    # The elements of this matrix are produced column-wise, hence the order in which we load elements from A.
    # We start by loading the first column, which is [4, 3]. Note that we skip the diagonal element 4 when
    # constructing U because we store it in a separate variable. The rest of the column goes into U as is.
    # Next, we load the second column, which is [3, 4]. Again, we skip the diagonal element 3, which becomes
    # the next element of U. The element 4 is already in the correct position in U, so we do nothing with it.
    # Finally, we load the third column, which is [4]. This is the last element of U.
    for j in range(rn[-1] + BLOCK_SIZE_N, N, BLOCK_SIZE_N):
        # Loop over remaining
