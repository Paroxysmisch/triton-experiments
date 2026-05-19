import triton
import triton.language as tl

@triton.jit
def batch_norm_sigmoid(input, running_mean, running_var, weight, bias, training, momentum, eps):
    N, C, L = input.shape
    if training:
        # Calculate mean and variance
        mean = tl.zeros((N, C), dtype=input.dtype)
        var = tl.zeros((N, C), dtype=input.dtype)
        for i in range(L):
            mean += input[:, :, i]
            var += input[:, :, i] ** 2
        mean /= L
        var /= L
        
        # Update running statistics
        running_mean *= (1 - momentum)
        running_var *= (1 - momentum)
        running_mean += momentum * mean
        running_var += momentum * var
        
        # Normalize input
        inv_std = tl.rsqrt(var + eps)
        normalized_input = (input - mean[:, :, None]) * inv_std[:, :, None]
        
        # Apply weight and bias
        if weight is not None and bias is not None:
            normalized_input = normalized_input * weight[:, None, :] + bias[:, None, :]
        
        # Apply sigmoid activation
        out = 1 / (1 + tl.exp(-normalized_input))
    
    else:
        # Use running statistics
        inv_std = tl.rsqrt(running_var + eps)
        normalized_input = (input - running_mean[:, :, None]) * inv_std[:, :, None]
        
        # Apply weight and bias
        if weight is not None and bias is not None:
            normalized_input = normalized_input * weight[:, None, :] + bias[:, None, :]
        
        # Apply sigmoid activation
        out = 1 / (1 + tl.exp(-normalized_input))

    return out
