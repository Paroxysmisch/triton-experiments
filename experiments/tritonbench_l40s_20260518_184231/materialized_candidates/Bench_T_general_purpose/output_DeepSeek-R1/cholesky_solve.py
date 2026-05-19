import torch

def cholesky_solve(B: torch.Tensor, L: torch.Tensor, upper: bool = False, *, out: torch.Tensor = None) -> torch.Tensor:
    # First triangular solve to get Y
    if upper:
        # Solve U^H Y = B, where U is upper triangular; U^H is lower triangular
        Y = torch.linalg.solve_triangular(L.mH, B, upper=False)
    else:
        # Solve L Y = B, where L is lower triangular
        Y = torch.linalg.solve_triangular(L, B, upper=False)
    
    # Second triangular solve to get X
    if upper:
        # Solve U X = Y, where U is upper triangular
        X = torch.linalg.solve_triangular(L, Y, upper=True)
    else:
        # Solve L^H X = Y, where L^H is upper triangular
        X = torch.linalg.solve_triangular(L.mH, Y, upper=True)
    
    # Handle output tensor if provided
    if out is not None:
        out.copy_(X)
        return out
    return X
