import torch
import triton
import triton.language as tl

def svd(A, full_matrices=True, *, driver=None, out=None):
    # Check if CUDA is available and set the driver appropriately
    if A.is_cuda:
        if driver not in [None, 'gesvd', 'gesvdj', 'gesvda']:
            raise ValueError(f"Invalid driver '{driver}'. Supported options are: None, 'gesvd', 'gesvdj', 'gesvda'")
    else:
        if driver is not None:
            raise RuntimeError("The 'driver' argument is only supported on CUDA inputs")
    
    # Delegate computation to PyTorch's SVD with driver option handling
    U, S, Vh = torch.linalg.svd(A, full_matrices=full_matrices, driver=driver)
    
    # Handle output tuple if provided
    if out is not None:
        if len(out) != 3:
            raise ValueError("The 'out' tuple must have three elements")
        out[0].resize_(U.shape).copy_(U)
        out[1].resize_(S.shape).copy_(S)
        out[2].resize_(Vh.shape).copy_(Vh)
        return tuple(out)
    return (U, S, Vh)

# Register the function into Triton's namespace
triton.linalg.svd = svd
