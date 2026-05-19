import torch
import triton
import triton.language as tl

if triton.__version__ >= "2.1.0":

    @triton.jit
    def std(
        input, dim=None, *, correction=1, keepdim=False, out=None
    ):
        # Convert to float32
        x = input.to(tl.float32)
        
        # Calculate mean
        c1 = tl.reduce(x, dim, tl.sum) / x.numel()
        
        # Calculate variance
        x_diff_c1 = tl.where(x != c1, x - c1, 0.)
        c2 = tl.reduce(x_diff_c1 * x_diff_c1, dim, tl.sum) / x.numel()
        
        # Apply Bessel's correction
        if correction != 0:
            c2 *= x.numel() / (x.numel() - correction)
        
        # Return standard deviation
        return tl.sqrt(c2)

else:

    @triton.jit
    def std(
        input, dim=None, *, correction=1, keepdim=False, out=None
    ):
        # Convert to float32
        x = input.to(tl.float32)
        
        # Flatten input if dim is None
        if dim is None:
            dim = range(x.ndim)
            x = x.reshape(-1)
        
        # Calculate mean
        x_mean = tl.reduce(x, dim, tl.mean)
        
        # Subtract mean and take absolute value
        x_abs = tl.abs(x - x_mean)
        
        # Calculate mean of absolute values
        x_mean_abs = tl.reduce(x_abs, dim, tl.mean)
        
        # Adjust for Bessel's correction if necessary
        if correction != 0:
            x_mean_abs *= correction / (correction - 1)
        
        # Return standard deviation
        return x_mean_abs
