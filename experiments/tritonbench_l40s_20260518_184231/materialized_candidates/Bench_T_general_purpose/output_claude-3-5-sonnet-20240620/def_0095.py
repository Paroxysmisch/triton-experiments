@triton.jit
def batch_norm_kernel(
    input_ptr, running_mean_ptr, running_var_ptr, weight_ptr, bias_ptr,
    output_ptr, N, C, H, W, momentum, eps, training):
    
    # Calculate the index for each element
    n = triton.program_id(0)
    c = triton.program_id(1)
    
    # Ensure we are within bounds
    if n >= N or c >= C:
        return
    
    # Load input, running mean, and running variance
    input_val = input_ptr[n, c, :, :]
    running_mean = running_mean_ptr[c]
    running_var = running_var_ptr[c]
    
    # Compute the mean and variance if training
    if training:
        batch_mean = triton.mean(input_val)
        batch_var = triton.var(input_val)
        
        # Update running mean and variance
        running_mean = momentum * running_mean + (1 - momentum) * batch_mean
        running_var = momentum * running_var + (1 - momentum) * batch_var
    else:
        batch_mean = running_mean
        batch_var = running_var
    
    # Normalize the input
    normalized = (input_val - batch_mean) / triton.sqrt(batch_var + eps)
    
    # Apply scale and shift if weight and bias are provided
    if weight_ptr is not None:
        normalized *= weight_ptr[c]
    if bias_ptr is not None:
        normalized += bias_ptr[c]
    
    # Store the output
    output_ptr[n, c, :, :] = normalized

def batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-05) -> Tensor:
    N, C, H, W = input.shape
    output = torch.empty_like(input)

    # Launch the Triton kernel
    batch_norm_kernel[(N, C)](
        input_ptr=input,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        N=N,
        C=C,
        H=H,
        W=W,
        momentum=momentum,
        eps=eps,
        training=training
    )
    
    return output
