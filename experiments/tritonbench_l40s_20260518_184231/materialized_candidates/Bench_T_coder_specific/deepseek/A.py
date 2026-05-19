def triton_solve_linear_system(A, B, *, left=True, out=None):
    # Ensure that the input types are supported
    if A.dtype not in [torch.float32, torch.float64, torch.complex64, torch.complex128]:
        raise ValueError(f"Unsupported dtype for A: {A.dtype}")
    if B.dtype not in [torch.float32, torch.float64, torch.complex64, torch.complex128]:
        raise ValueError(f"Unsupported dtype for B: {B.dtype}")

    # Ensure that A is invertible
    if not torch.allclose(A @ torch.linalg.inv(A), B):
        raise ValueError("Matrix A is not invertible")

    # Compute the solution
    if left:
        X = torch.linalg.solve(A, B)
    else:
        X = torch.linalg.solve(B, A)

    # Return the solution
    return X
