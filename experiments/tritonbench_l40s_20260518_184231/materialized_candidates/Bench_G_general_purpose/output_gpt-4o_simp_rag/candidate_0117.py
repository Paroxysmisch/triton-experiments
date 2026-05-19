import triton
import triton.language as tl
import torch

@triton.jit
def logsumexp_fwd_kernel(X, Z, scale, D, B, **meta):
    pid = tl.program_id(0)
    # Create a range for the last dimension
    rx = tl.arange(0, B)
    # Calculate offset for the block
    offset = pid * B + rx

    # Load the data for the current block
    x = tl.load(X + offset, mask=offset < D, other=-float('inf'))
    
    # Optionally scale the input
    if scale is not None:
        x *= scale

    # Compute the maximum value in the block
    c = tl.max(x, axis=0)

    # Compute the log-sum-exp
    out = tl.log(tl.sum(tl.exp(x - c), axis=0)) + c

    # Store the result
    tl.store(Z + pid, out)

def logsumexp_fwd(x, scale=None, dtype=None):
    # Get the input shape
    shape = x.shape
    D = shape[-1]
    
    # Determine the block size B
    B = 128  # Example block size, can be tuned based on the GPU and input size
    B = min(B, D)
    
    # Create an empty output tensor
    z = torch.empty(shape[:-1], dtype=x.dtype if dtype is None else dtype, device=x.device)
    
    # Launch the kernel
    grid = (triton.cdiv(D, B),)
    logsumexp_fwd_kernel[grid](x, z, scale, D, B)
    
    # Reshape the result back to the original shape minus the last dimension
    return z

# Example usage
x = torch.randn(32, 128, device='cuda')  # Example input tensor
result = logsumexp_fwd(x)
print(result)
