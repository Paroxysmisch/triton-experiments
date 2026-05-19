import torch
import triton
import triton.language as tl

def solve(A: torch.Tensor, B: torch.Tensor, *, left: bool = True, out: torch.Tensor = None) -> torch.Tensor:
    # Check input types
    if A.dtype not in [torch.float32, torch.float64, torch.complex64, torch.complex128]:
        raise ValueError("A must be of type float, double, cfloat, or cdouble")
    if B.dtype not in [torch.float32, torch.float64, torch.complex64, torch.complex128]:
        raise ValueError("B must be of type float, double, cfloat, or cdouble")
    if A.dtype != B.dtype:
        raise ValueError("A and B must have the same dtype")

    # Check input shapes
    if A.dim() < 2 or B.dim() < 2:
        raise ValueError("A and B must be at least 2D tensors")
    if A.shape[-2] != A.shape[-1]:
        raise ValueError("A must be a square matrix")
    if left:
        if A.shape[-1] != B.shape[-2]:
            raise ValueError("The last dimension of A must match the second-to-last dimension of B when left=True")
    else:
        if A.shape[-2] != B.shape[-1]:
            raise ValueError("The second-to-last dimension of A must match the last dimension of B when left=False")

    # Determine batch size
    batch_size = 1
    if A.dim() > 2:
        batch_size = A.shape[:-2].numel()
    if B.dim() > 2:
        if batch_size == 1:
            batch_size = B.shape[:-2].numel()
        elif B.shape[:-2] != A.shape[:-2]:
            raise ValueError("A and B must have the same batch dimensions")

    # Prepare output tensor
    if out is None:
        out_shape = list(A.shape[:-2]) + [B.shape[-2] if left else B.shape[-1], B.shape[-1] if left else B.shape[-2]]
        out = torch.empty(out_shape, dtype=A.dtype, device=A.device)
    else:
        if out.shape != (list(A.shape[:-2]) + [B.shape[-2] if left else B.shape[-1], B.shape[-1] if left else B.shape[-2]]):
            raise ValueError("out must have the correct shape")

    # Strides for A, B, and out
    A_strides = A.stride()
    B_strides = B.stride()
    X_strides = out.stride()

    # Launch kernel
    grid = (batch_size, 1, 1)
    if A.dtype in [torch.float32, torch.complex64]:
        solve_kernel[grid](A, B, out, A_strides, B_strides, X_strides, batch_size, A.shape[-1], B.shape[-1] if left else B.shape[-2])
    elif A.dtype in [torch.float64, torch.complex128]:
        solve_kernel[grid](A, B, out, A_strides, B_strides, X_strides, batch_size, A.shape[-1], B.shape[-1] if left else B.shape[-2])

    # Synchronize device
    if A.device.type == 'cuda':
        torch.cuda.synchronize()

    return out
