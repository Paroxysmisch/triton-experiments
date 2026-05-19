import torch
import triton

def cholesky_solve(B, L, upper=False, out=None):
    """
    Solves the system of linear equations AX = B using the Cholesky decomposition.
    
    Args:
        B (torch.Tensor): Right-hand side tensor of shape (*, n, k).
        L (torch.Tensor): Tensor of shape (*, n, n) containing the lower or upper triangular Cholesky decompositions.
        upper (bool, optional): Flag indicating whether L is upper triangular. Default: False.
        out (torch.Tensor, optional): Output tensor. Ignored if None. Default: None.
        
    Returns:
        torch.Tensor: Solution tensor of shape (*, n, k).
    """
    if out is None:
        out = torch.empty_like(B)
    
    # Get the shapes and strides
    B_shape = B.shape
    B_stride_n = B.stride(-2)
    B_stride_k = B.stride(-1)
    L_shape = L.shape
    L_stride_n = L.stride(-2)
    L_stride_ld = L.stride(-1)
    
    # Launch the Triton kernel
    grid = (triton.cdiv(B_shape[0], 32), triton.cdiv(B_shape[-2], 32))
    block = (32,)
    triton.launch_kernel(
        cholesky_solve_kernel,
        grid=grid,
        block=block,
        args=[
            B.data_ptr(), B_shape, B_stride_n, B_stride_k,
            L.data_ptr(), L_shape, L_stride_n, L_stride_ld,
            out.data_ptr(), out.shape, out.stride(-2), out.stride(-1),
            upper, B.dtype
        ],
        stream=torch.cuda.current_stream().cuda_stream
    )
    
    return out
