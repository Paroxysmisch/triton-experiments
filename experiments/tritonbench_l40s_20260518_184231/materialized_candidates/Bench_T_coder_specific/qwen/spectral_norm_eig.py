import torch
import triton
import triton.language as tl

@torch.jit.script
def spectral_norm_eig(A, out=None):
    if A.dtype not in [torch.float32, torch.double, torch.cfloat, torch.cdouble]:
        raise ValueError("Unsupported dtype. Supported dtypes are float32, double, cfloat, and cdouble.")
    
    if len(A.shape) < 2 or A.shape[-2] != A.shape[-1]:
        raise ValueError("Input must be a square matrix.")
    
    N = A.shape[-1]
    num_batches = A.numel() // (N * N)
    
    if out is None:
        out = torch.empty(num_batches, device=A.device, dtype=torch.float32)
    
    grid = lambda meta: (num_batches,)
    spectral_norm_kernel[grid](A.data_ptr(), out.data_ptr(), N, A.stride(-2), out.stride(-1), num_batches)
    
    return out
