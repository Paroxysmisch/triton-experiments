import torch
import triton
import triton.language as tl

def least_squares_qr(A, b, *, mode='reduced', out=None):
    # Perform QR decomposition using PyTorch
    Q, R = torch.linalg.qr(A, mode=mode)
    
    # Compute Q^H b
    Q_H_b = torch.matmul(Q.transpose(-2, -1).conj(), b)
    
    # Solve R x = Q^H b for x using back substitution
    x = torch.linalg.solve_triangular(R, Q_H_b, upper=True)
    
    # If out is provided, copy result to out
    if out is not None:
        out.copy_(x)
        return out
    
    return x

# Example usage
A = torch.randn(5, 3)
b = torch.randn(5)
x = least_squares_qr(A, b)
print(x)
