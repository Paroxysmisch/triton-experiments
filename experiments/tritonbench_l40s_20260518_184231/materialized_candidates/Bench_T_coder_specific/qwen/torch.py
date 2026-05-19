import triton
import numpy as np

def torch_permute_copy(input_tensor, dims):
    # Get the shape and strides of the input tensor
    input_shape = input_tensor.shape
    input_strides = input_tensor.strides
    
    # Calculate the shape and strides of the output tensor
    output_shape = tuple(input_shape[dims[i]] for i in range(len(dims)))
    output_strides = tuple(input_strides[dims[i]] for i in range(len(dims)))
    
    # Allocate memory for the output tensor
    output_tensor = np.empty(output_shape, dtype=input_tensor.dtype)
    
    # Prepare the arguments for the Triton kernel
    n_elements = np.prod(output_shape)
    block_size = 32
    
    # Launch the Triton kernel
    grid_size = (n_elements // block_size,) if n_elements > 0 else (1,)
    permute_kernel[(grid_size)](input_tensor.ctypes.data, 
                                output_tensor.ctypes.data, 
                                n_elements, 
                                input_strides, 
                                output_strides, 
                                len(dims))
    
    return output_tensor

# Example usage:
# input_tensor = np.array([[1, 2], [3, 4]], dtype=np.float32)
# dims = (1, 0)
# result = torch_permute_copy(input_tensor, dims)
# print(result)  # Output: [[1., 3.], [2., 4.]]
