import torch
import triton
import triton.language as tl

@triton.jit
def cholesky_kernel(
    A_ptr,  # Pointer to the input matrix data
    L_ptr,  # Pointer to the output matrix data
    n: int,  # Size of the matrix (n x n)
    stride_Ab: int, stride_Ah: int, stride_Aw: int,  # Strides for the input tensor
    stride_Lb: int, stride_Lh: int, stride_Lw: int,  # Strides for the output tensor
    BLOCK_SIZE: tl.constexpr,  # Block size for kernel execution
    is_complex: tl.constexpr  # Whether the input is complex
):
    pid = tl.program_id(0)
    # Iterate over each matrix in the batch
    for batch in range(pid, pid + 1):  # Simplified for example; adjust grid appropriately
        for i in range(n):
            for j in range(i + 1):
                a_idx = batch * stride_Ab + i * stride_Ah + j * stride_Aw
                l_idx = batch * stride_Lb + i * stride_Lh + j * stride_Lw
                if j == i:
                    sum_val = 0.0
                    for k in range(j):
                        l_ik = tl.load(L_ptr + batch * stride_Lb + i * stride_Lh + k * stride_Lw)
                        l_jk = tl.load(L_ptr + batch * stride_Lb + j * stride_Lh + k * stride_Lw)
                        if is_complex:
                            sum_val += l_ik * tl.conj(l_jk)
                        else:
                            sum_val += l_ik * l_jk
                    a_jj = tl.load(A_ptr + a_idx)
                    l_jj = tl.sqrt(a_jj - sum_val)
                    tl.store(L_ptr + l_idx, l_jj)
                else:
                    sum_val = 0.0
                    for k in range(j):
                        l_ik = tl.load(L_ptr + batch * stride_Lb + i * stride_Lh + k * stride_Lw)
                        l_jk = tl.load(L_ptr + batch * stride_Lb + j * stride_Lh + k * stride_Lw)
                        if is_complex:
                            sum_val += l_ik * tl.conj(l_jk)
                        else:
                            sum_val += l_ik * l_jk
                    a_ij = tl.load(A_ptr + a_idx)
                    l_jj = tl.load(L_ptr + batch * stride_Lb + j * stride_Lb + j * stride_Lw)
                    l_ij = (a_ij - sum_val) / l_jj
                    tl.store(L_ptr + l_idx, l_ij)

def linalg_cholesky(A: torch.Tensor, *, upper: bool = False, out: torch.Tensor = None) -> torch.Tensor:
    # Check input dimensions
    if A.dim() < 2 or A.shape[-1] != A.shape[-2]:
        raise RuntimeError("A must be a batch of square matrices with shape (*, n, n)")
    
    # Check dtype
    if A.dtype not in [torch.float32, torch.float64, torch.complex64, torch.complex128]:
        raise RuntimeError(f"Unsupported dtype {A.dtype}")
    
    # Allocate output tensor
    if out is None:
        out = torch.empty_like(A)
    else:
        if out.shape != A.shape:
            raise RuntimeError("Output tensor must have the same shape as A")
        if out.dtype != A.dtype:
            raise RuntimeError("Output tensor must have the same dtype as A")
    
    # Ensure contiguous memory layout
    if not A.is_contiguous():
        A = A.contiguous()
    if not out.is_contiguous():
        out = out.contiguous()
    
    # Flatten batch dimensions
    batch_dims = A.shape[:-2]
    n = A.size(-1)
    A_flat = A.view(-1, n, n)
    out_flat = out.view(-1, n, n)
    
    # Determine grid size based on batch size
    grid = (A_flat.size(0),)
    
    # Check if complex
    is_complex = A.is_complex()
    
    # Launch Triton kernel
    cholesky_kernel[grid](
        A_flat, out_flat, n,
        A_flat.stride(0), A_flat.stride(1), A_flat.stride(2),
        out_flat.stride(0), out_flat.stride(1), out_flat.stride(2),
        BLOCK_SIZE=16, is_complex=is_complex
    )
    
    # If upper is True, transpose and conjugate
    if upper:
        out_flat = out_flat.transpose(-2, -1)
        if is_complex:
            out_flat = out_flat.conj()
    
    # Reshape to original dimensions
    out = out_flat.view(batch_dims + (n, n))
    
    # Synchronize if on CUDA
    if A.is_cuda:
        torch.cuda.synchronize()
    
    return out
