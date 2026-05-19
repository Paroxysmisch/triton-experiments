import torch
import triton
import triton.language as tl

@triton.jit
def _swiglu_forward_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N,
    # The stride variables represent how much to increase the ptr by when moving by 1
    stride_am, stride_an,
    stride_bm, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(axis=0)
    
    # Row index
    row_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Create mask to handle edge cases
    row_mask = row_idx < M
    
    # Compute SWiGLU for each element in the row
    for n in range(0, N, BLOCK_SIZE):
        col_idx = n + tl.arange(0, BLOCK_SIZE)
        col_mask = col_idx < N
        mask = row_mask[:, None] & col_mask[None, :]
        
        # Load inputs
        a = tl.load(a_ptr + row_idx[:, None] * stride_am + col_idx[None, :] * stride_an, mask=mask)
        b = tl.load(b_ptr + row_idx[:, None] * stride_bm + col_idx[None, :] * stride_bn, mask=mask)
        
        # Compute SiLU(a) * b
        silu_a = a * tl.sigmoid(a)
        c = silu_a * b
        
        # Store output
        tl.store(c_ptr + row_idx[:, None] * stride_cm + col_idx[None, :] * stride_cn, c, mask=mask)

@triton.jit
def _swiglu_backward_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, dc_ptr, da_ptr, db_ptr,
    # Matrix dimensions
    M, N,
    # The stride variables
    stride_am, stride_an,
    stride_bm, stride_bn,
    stride_dcm, stride_dcn,
    stride_dam, stride_dan,
    stride_dbm, stride_dbn,
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    row_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row_mask = row_idx < M

    for n in range(0, N, BLOCK_SIZE):
        col_idx = n + tl.arange(0, BLOCK_SIZE)
        col_mask = col_idx < N
        mask = row_mask[:, None] & col_mask[None, :]
        
        # Load inputs
        a = tl.load(a_ptr + row_idx[:, None] * stride_am + col_idx[None, :] * stride_an, mask=mask)
        b = tl.load(b_ptr + row_idx[:, None] * stride_bm + col_idx[None, :] * stride_bn, mask=mask)
        dc = tl.load(dc_ptr + row_idx[:, None] * stride_dcm + col_idx[None, :] * stride_dcn, mask=mask)
        
        # Compute gradients
        sig_a = tl.sigmoid(a)
        dsig_a = sig_a * (1 - sig_a)
        da = dc * b * (sig_a + a * dsig_a)
        db = dc * (a * sig_a)
        
        # Store gradients
        tl.store(da_ptr + row_idx[:, None] * stride_dam + col_idx[None, :] * stride_dan, da, mask=mask)
        tl.store(db_ptr + row_idx[:, None] * stride_dbm + col_idx[None, :] * stride_dbn, db, mask=mask)

def swiglu_forward(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape == b.shape, "Input tensors must have the same shape"
    assert a.is_cuda and b.is_cuda, "Input tensors must be on GPU"
    
    M, N = a.shape
    c = torch.empty_like(a)
    
    # Launch kernel with appropriate grid size
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE']),)
    
    _swiglu_forward_kernel[grid](
        a_ptr=a, b_ptr=b, c_ptr=c,
        M=M, N=N,
        stride_am=a.stride(0), stride_an=a.stride(1),
        stride_bm=b.stride(0), stride_bn=b.stride(1),
        stride_cm=c.stride(0), stride_cn=c.stride(1),
        BLOCK_SIZE=128,
        num_warps=4
    )
    
    return c

def swiglu_backward(a: torch.Tensor, b: torch.Tensor, dc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert a.shape == b.shape == dc.shape, "All tensors must have the same shape"
    assert all(t.is_cuda for t in [a, b, dc]), "All tensors must be on GPU"
    
    M, N = a.shape
    da = torch.empty_like(a)
    db = torch.empty_like(b)
    
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE']),)
    
    _swiglu_backward_kernel[grid](
        a_ptr=a, b_ptr=b, dc_ptr=dc,
        da_ptr=da, db_ptr=db,
        M=M, N=N,
        stride_am=a.stride(0), stride_an=a.stride(1),
        stride_bm=b.stride(0), stride_bn=b.stride(1),
        stride_dcm=dc.stride(0), stride_dcn=dc.stride(1),
        stride_dam=da.stride(0), stride_dan=da.stride(1),
        stride_dbm=db.stride(0), stride_dbn=db.stride(1),
        BLOCK_SIZE=128,
        num_warps=4
    )
    
    return da, db
