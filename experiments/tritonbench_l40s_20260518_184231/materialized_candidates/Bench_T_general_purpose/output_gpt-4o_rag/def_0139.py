import triton
import triton.language as tl
import torch

@triton.jit
def std_kernel(input_ptr, output_ptr, stride, N, correction, BLOCK_SIZE: tl.constexpr):
    # Define block indices
    pid = tl.program_id(0)
    
    # Create pointers to input and output
    input_offset = pid * stride
    input_block = tl.load(input_ptr + input_offset, mask=pid < N, other=0.0)
    
    # Compute mean
    mean = tl.sum(input_block, axis=0) / N
    
    # Compute variance
    diff = input_block - mean
    variance = tl.sum(diff * diff, axis=0) / tl.max(0, N - correction)
    
    # Compute standard deviation
    std_dev = tl.sqrt(variance)
    
    # Store the result
    tl.store(output_ptr + pid, std_dev)

def std(input, dim=None, *, correction=1, keepdim=False, out=None):
    # Convert input to a contiguous tensor if not already
    input = input.contiguous()
    
    # Determine the size of the reduction dimension
    if dim is None:
        dim = tuple(range(input.ndim))
    elif isinstance(dim, int):
        dim = (dim,)
    
    # Calculate the size of the reduction dimension
    N = 1
    for d in dim:
        N *= input.size(d)
    
    # Calculate stride for the input tensor
    stride = input.stride(dim[-1])
    
    # Prepare output tensor
    if out is None:
        output_shape = list(input.shape)
        for d in dim:
            output_shape[d] = 1 if keepdim else 0
        output_shape = [s for s in output_shape if s != 0]
        out = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    
    # Launch Triton kernel
    grid = (input.numel() // N,)
    std_kernel[grid](input, out, stride, N, correction, BLOCK_SIZE=128)
    
    return out

# Example usage
input_tensor = torch.randn(256, 256, device='cuda')
result = std(input_tensor, dim=(0, 1), correction=1, keepdim=True)
print(result)
