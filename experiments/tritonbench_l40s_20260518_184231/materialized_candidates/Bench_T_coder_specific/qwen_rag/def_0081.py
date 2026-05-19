import triton
import triton.language as tl
from typing import Union, Tuple

# Function to perform 2D adaptive average pooling
@triton.jit
def adaptive_avg_pool2d_kernel(X_ptr, X_shape, X_strides, Y_ptr, Y_shape, Y_strides, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_blocks = tl.cdiv(Y_shape[0] * Y_shape[1], BLOCK_SIZE)
    
    # Calculate the coordinates of the current block in the output tensor
    y = pid // Y_shape[1]
    x = pid % Y_shape[1]
    
    # Calculate the coordinates of the corresponding region in the input tensor
    input_y_start = y * Y_strides[0]
    input_x_start = x * Y_strides[1]
    
    # Initialize sum and count for averaging
    sum_val = tl.zeros([], dtype=tl.float32)
    count = tl.zeros([], dtype=tl.int32)
    
    # Iterate over the region in the input tensor
    for i in range(BLOCK_SIZE):
        for j in range(BLOCK_SIZE):
            input_y = input_y_start + i
            input_x = input_x_start + j
            
            # Check if the coordinates are within bounds
            if input_y < X_shape[0] and input_x < X_shape[1]:
                sum_val += X_ptr[input_y * X_strides[0] + input_x * X_strides[1]]
                count += 1
    
    # Average the values and store them in the output tensor
    avg_val = sum_val / count
    Y_ptr[y * Y_strides[0] + x * Y_strides[1]] = avg_val

# Function to apply the sigmoid activation function
@triton.jit
def sigmoid_kernel(X_ptr, X_shape, X_strides, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_elements = X_shape[0] * X_shape[1]
    
    # Calculate the coordinates of the current element in the input tensor
    idx = pid
    
    # Apply the sigmoid function
    x = X_ptr[idx * X_strides[0]]
    sigm = 1.0 / (1.0 + tl.exp(-x))
    X_ptr[idx * X_strides[0]] = sigm

# Wrapper function for sigmoid adaptive average pooling
def sigmoid_adaptive_avg_pool2d(input: tl.Tensor, output_size: Union[int, Tuple[int, int]]) -> tl.Tensor:
    # Determine the shape of the input and output tensors
    input_shape = input.shape
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    output_shape = (input_shape[0] // output_size[0], input_shape[1] // output_size[1])
    
    # Create output tensor
    output = tl.zeros(output_shape, dtype=input.dtype)
    
    # Define block size for parallelism
    BLOCK_SIZE = 16
    
    # Launch adaptive_avg_pool2d kernel
    grid = tl.grid((output_shape[0] * output_shape[1]))
    adaptive_avg_pool2d_kernel[grid](input.data, input.shape, input.stride, output.data, output_shape, output.stride, BLOCK_SIZE)
    
    # Launch sigmoid kernel
    grid = tl.grid((output_shape[0] * output_shape[1]))
    sigmoid_kernel[grid](output.data, output_shape, output.stride, BLOCK_SIZE)
    
    return output

# Example usage
if __name__ == "__main__":
    import torch
    input_tensor = torch.randn(16, 16, device="cuda")
    output_tensor = sigmoid_adaptive_avg_pool2d(input_tensor, (4, 4))
    print(output_tensor)
