import torch

def least_squares_qr(A, b, *, mode='reduced', out=None) -> torch.Tensor:
    # Perform QR decomposition on A
    Q, R = torch.linalg.qr(A, mode=mode)
    
    # Compute Q^H b (Q conjugate transpose multiplied by b)
    Q_H = Q.conj().mT  # Matrix transpose and conjugate
    Q_H_b = torch.matmul(Q_H, b)
    
    # Handle different modes to extract appropriate R and Q_H_b
    n = A.shape[-1]
    if mode == 'complete':
        # Slice R to take the first n rows and all columns
        R_eff = R[..., :n, :]
        # Slice Q_H_b to take the first n elements along the respective dimension
        Q_H_b_eff = Q_H_b[..., :n, :]
    else:
        R_eff = R
        Q_H_b_eff = Q_H_b
    
    # Solve the upper triangular system R_eff x = Q_H_b_eff
    x = torch.linalg.solve_triangular(R_eff, Q_H_b_eff, upper=True)
    
    # Handle the output tensor if provided
    if out is not None:
        if not out.is_complex() and x.is_complex():
            raise RuntimeError("Cannot cast complex x to real out tensor.")
        out.copy_(x)
        return out
    return x
