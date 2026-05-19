import torch
import triton
import triton.language as tl

# Define the Triton kernel
softmax_kernel = triton.compile(softmax_kernel, 
                               signature=("float32[:]", "float32[:]", "int32", "int32", "int32", "int32", "int32"), 
                               constants={"BLOCK_SIZE": 128})

def softmax(input, dim, dtype=None):
    # Determine the number of elements and slices
    n_elements = input.size(dim)
    n_slices = input.numel() // n_elements
    
    # Get strides
    stride_slice = input.stride(dim)
    stride_elem = input.stride(-1)
    
    # Create output tensor
    if dtype is None:
        dtype = input.dtype
    output = torch.empty_like(input, dtype=dtype)
    
    # Cast input to float32 if needed
    if input.dtype != torch.float32:
        input = input.to(torch.float32)
    
    # Call the Triton kernel
    softmax_kernel[grid=(n_slices,), block=(128,)](
        input.data_ptr(), 
        output.data_ptr(), 
        n_elements, 
        n_slices, 
        stride_slice, 
        stride_elem, 
        max_shift=0
    )
    
    return output
