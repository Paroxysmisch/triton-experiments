def least_squares_qr(A, b, *, mode='reduced', out=None):
    # Check if A and b have the correct shapes
    assert A.shape[-2:] == b.shape[:-1], "Input shapes must match"
    
    # Compute the QR decomposition of A
    Q, R = torch.linalg.qr(A, mode=mode)
    
    # Compute the least squares solution
    x = torch.linalg.solve_triangular(R, Q.conj().transpose(-2, -1) @ b, upper=True)
    
    return x
