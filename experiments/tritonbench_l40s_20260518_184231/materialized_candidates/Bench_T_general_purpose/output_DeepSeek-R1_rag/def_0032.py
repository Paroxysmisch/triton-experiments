import torch

def linalg_eig(A, *, out=None):
    # Synchronize CUDA device if necessary
    if A.is_cuda:
        torch.cuda.current_stream().synchronize()
    
    # Compute eigenvalue decomposition using PyTorch's implementation
    eigenvalues, eigenvectors = torch.linalg.eig(A)
    
    # Handle output tuple
    if out is not None:
        if not isinstance(out, tuple) or len(out) != 2:
            raise TypeError("argument 'out' must be a tuple of two tensors")
        out[0].resize_(eigenvalues.shape).copy_(eigenvalues)
        out[1].resize_(eigenvectors.shape).copy_(eigenvectors)
        return (out[0], out[1])
    
    return (eigenvalues, eigenvectors)

# Update PyTorch's __dict__ for namespace addition
torch.linalg.eig = linalg_eig
