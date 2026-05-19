# triton_kernels.py

import triton
import triton.language as tl

@triton.jit
def fused_tile_exp_kernel(input_ptr, output_ptr, dims, n_elements):
    # Calculate the index for the input tensor
    idx = tl.program_id(0)
    
    # Compute the number of dimensions
    num_dims = len(dims)
    
    # Calculate the corresponding input index
    input_index = idx % n_elements
    
    # Tiling operation
    tiled_index = input_index
    for dim in range(num_dims):
        tiled_index //= dims[dim]
    
    # Load the input tensor
    x = tl.load(input_ptr + tiled_index)
    
    # Apply the exponential function
    result = tl.exp(x)
    
    # Store the result in the output tensor
    tl.store(output_ptr + idx, result)

def fused_tile_exp(input, dims, *, out=None):
    # Ensure input is a tensor
    if not isinstance(input, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")
    
    # Prepare the dimensions for tiling
    input_shape = input.shape
    num_dims = len(input_shape)
    dims = (1,) * (num_dims - len(dims)) + dims  # Prepend ones if necessary
    
    # Calculate the output shape
    output_shape = tuple(input_shape[i] * dims[i] for i in range(num_dims))
    
    # Create output tensor if not provided
    if out is None:
        out = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    
    # Launch the Triton kernel
    n_elements = out.numel()
    grid = (n_elements,)
    fused_tile_exp_kernel[grid](input, out, dims, n_elements)
    
    return out
