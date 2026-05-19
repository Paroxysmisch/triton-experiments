import torch
import triton
import triton.language as tl

def logsumexp(input, dim, keepdim=False, *, out=None) -> torch.Tensor:
    assert isinstance(input, torch.Tensor)
    assert dim >= 0 and dim < len(input.shape)
    
    N = input.size(dim)
    stride_n = input.stride(dim)
    M = input.numel() // N
    
    if out is None:
        out = torch.empty_like(input, device=input.device, dtype=input.dtype)
    
    grid = (
        triton.cdiv(N, triton.next_power_of_2(64)),
        triton.cdiv(M, triton.next_power_of_2(64)),
    )
    block = (64, 64, 1)
    
    logsumexp_kernel[
        grid, block
    ](
        input.contiguous().data_ptr(),
        out.data_ptr(),
        N,
        stride_n,
        M,
        input.stride(0),
        BLOCK_SIZE_N=64,
        BLOCK_SIZE_M=64,
    )
    
    if not keepdim:
        out = out.squeeze(dim)
    
    return out
