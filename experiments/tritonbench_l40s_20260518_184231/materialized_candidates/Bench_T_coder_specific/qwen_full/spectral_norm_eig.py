import torch
import triton
import triton.language as tl
from flag_gems.utils.shape_utils import volume

def get_triton_function(dtype):
    # Kernel for float32
    if dtype == torch.float32:
        @triton.jit
        def triton_function(A, out):
            # Compute the spectral norm using eigenvalues
            lambda_max = tl.max(tl.abs(eig(A)))
            tl.store(out, lambda_max)
        return triton_function
    # Kernel for float64
    elif dtype == torch.float64:
        @triton.jit
        def triton_function(A, out):
            # Compute the spectral norm using eigenvalues
            lambda_max = tl.max(tl.abs(eig(A)))
            tl.store(out, lambda_max)
        return triton_function
    # Kernel for complex64
    elif dtype == torch.complex64:
        @triton.jit
        def triton_function(A, out):
            # Compute the spectral norm using eigenvalues
            lambda_max = tl.max(tl.abs(eig(A)))
            tl.store(out, lambda_max)
        return triton_function
    # Kernel for complex128
    elif dtype == torch.complex128:
        @triton.jit
        def triton_function(A, out):
            # Compute the spectral norm using eigenvalues
            lambda_max = tl.max(tl.abs(eig(A)))
            tl.store(out, lambda_max)
        return triton_function
    else:
        assert False

@torch.inference_mode()
def spectral_norm_eig(A, *, out=None):
    # Function to compute the spectral norm of a matrix using its eigenvalues
    layout = A.layout
    device = A.device
    dtype = A.dtype
    if dtype in (torch.float32, torch.float64, torch.complex64, torch.complex128):
        if out is None:
            out = torch.empty([], dtype=dtype, device=device)
        else:
            assert out.shape == () and out.dtype == dtype and out.device == device
        triton_function = get_triton_function(dtype)
        n = A.shape[-1]
        # Reshape A for Triton kernel
        A = A.reshape([-1, n, n])
        batch_shape = A.shape[:-2]
        A = A.reshape([volume(batch_shape), n, n])
        M = A.shape[0]
        grid = (M, )
        triton_function[grid](A, out)
        out = out.reshape(batch_shape)
        return out
    else:
        assert dtype in (torch.float32, torch.float64, torch.complex64, torch.complex128)
