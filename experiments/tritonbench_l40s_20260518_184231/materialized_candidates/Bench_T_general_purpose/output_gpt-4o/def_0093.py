import torch
import triton
import triton.language as tl

@triton.jit
def log_softmax_kernel(X, Y, dim_size, stride, BLOCK_SIZE: tl.constexpr):
    # Compute the linear index within the block
    idx = tl.arange(0, BLOCK_SIZE)
    
    # Load input elements
    x = tl.load(X + idx * stride, mask=idx < dim_size, other=0.0)
    
    # Apply the natural logarithm
    log_x = tl.log(x)
    
    # Compute max log value for numerical stability
    max_log_x = tl.max(log_x, axis=0)
    
    # Compute exponentials
    exp_log_x = tl.exp(log_x - max_log_x)
    
    # Compute the sum of exponentials
    sum_exp_log_x = tl.sum(exp_log_x, axis=0)
    
    # Compute softmax values
    softmax_log_x = exp_log_x / sum_exp_log_x
    
    # Store results
    tl.store(Y + idx * stride, softmax_log_x, mask=idx < dim_size)

def softmax_log(input, dim=-1, dtype=None):
    # Ensure the input tensor is on the correct device
    input = input.to(device='cuda')
    
    # Optionally cast the input tensor to the specified dtype
    if dtype is not None:
        input = input.to(dtype=dtype)
    
    # Prepare the output tensor
    output = torch.empty_like(input)
    
    # Determine the dimension size and stride
    dim_size = input.size(dim)
    stride = input.stride(dim)
    
    # Launch the Triton kernel
    grid = (triton.cdiv(dim_size, 1024),)
    log_softmax_kernel[grid](input, output, dim_size, stride, BLOCK_SIZE=1024)
    
    return output

# Example usage
if __name__ == "__main__":
    # Define input tensor
    input_tensor = torch.rand(3, 4, device='cuda') * 10
    
    # Apply softmax_log along the last dimension
    result = softmax_log(input_tensor, dim=1)
    print(result)

    # Apply softmax_log along a different dimension
    result = softmax_log(input_tensor, dim=0)
    print(result)
