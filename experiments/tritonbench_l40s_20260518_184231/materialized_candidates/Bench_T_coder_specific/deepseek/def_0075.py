import triton
import numpy as np

def fused_cholesky_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    # Perform Cholesky decomposition on A
    L = np.linalg.cholesky(A)
    
    # Solve for x in Ax = b
    x = np.linalg.solve(L, b)
    
    return x
