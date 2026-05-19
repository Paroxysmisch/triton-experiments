import torch
import triton
import triton.language as tl

@triton.jit
def _kernel_matrix_multiply_symmetric(A, B, C, alpha, beta):
    # Convert inputs to float32
    A = A.to(tl.float32)
    B = B.to(tl.float32)
    C = C.to(tl.float32)

    # Matrix multiplication and scaling
    C = alpha * tl.dot(A, B) + beta * C

    # In-place matrix multiplication and scaling
    C = alpha * tl.dot(C, C, out_dtype=tl.float32).to(C.dtype) + beta * C

    return C

def matrix_multiply_symmetric(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    # Ensure A, B, C are 2D tensors
    if A.dim() != 2 or B.dim() != 2 or C.dim() != 2:
        raise ValueError("All tensors must be 2D")
    
    # Call the Triton kernel
    return _kernel_matrix_multiply_symmetric(A, B, C, alpha, beta)
