import torch
import triton
import triton.language as tl

@triton.jit
def truncate_kernel(x_ptr: tl.tensor, y_ptr: tl.tensor,
                   n_elements: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * tl.cdiv(n_elements, tl.num_warps())
    block_end = min(block_start + tl.cdiv(n_elements, tl.num_warps()), n_elements)
    for i in range(block_start, block_end):
        x_val = tl.load(x_ptr + i)
        # Truncate integer values
        if tl.is_floating_point(x_val.dtype):
            y_val = tl.floor(x_val)
        else:
            y_val = x_val
        tl.store(y_ptr + i, y_val)

def trunc(input: torch.Tensor, *, out=None) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(input)
    
    n_elements = input.numel()
    truncate_kernel[(n_elements // tl.block_dim(0) + 1)](
        x_ptr=input, y_ptr=out,
        n_elements=n_elements,
    )
    
    return out
