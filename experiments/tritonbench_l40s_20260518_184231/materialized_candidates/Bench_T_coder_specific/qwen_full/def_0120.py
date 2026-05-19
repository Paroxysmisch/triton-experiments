import torch
import triton
import triton.language as tl

@triton.jit
def matrix_vector_dot_kernel(A, x, y, alpha, beta):
    # Convert inputs to float32
    x_fp32 = x.to(tl.float32)
    y_fp32 = y.to(tl.float32)
    
    # Compute matrix-vector product and update y
    y_fp32 = alpha * tl.dot(A, x_fp32) + beta * y_fp32
    
    # Compute dot product of updated y and x
    result = tl.dot(y_fp32, x_fp32)
    
    return result

def matrix_vector_dot_triton(A: torch.Tensor, x: torch.Tensor, y: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    # Ensure inputs are on CUDA device
    assert A.is_cuda and x.is_cuda and y.is_cuda
    # Call the Triton kernel
    return matrix_vector_dot_kernel(A, x, y, alpha, beta)
