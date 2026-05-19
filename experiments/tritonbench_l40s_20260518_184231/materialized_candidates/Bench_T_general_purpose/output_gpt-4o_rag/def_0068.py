import torch
import triton
import triton.language as tl

@triton.jit
def add_mean_kernel(
    input_ptr, other_ptr, output_ptr,
    alpha, N, M, HAS_DIM: tl.constexpr, REDUCE_DIM: tl.constexpr
):
    pid = tl.program_id(0)
    
    if HAS_DIM:
        offsets = pid * M + tl.arange(0, M)
        mask = offsets < N * M
        input_vals = tl.load(input_ptr + offsets, mask=mask)
        other_vals = tl.load(other_ptr + offsets, mask=mask)
        
        # Add scaled other to input
        result = input_vals + alpha * other_vals
        
        # Compute mean along the specified dimension
        mean_val = tl.sum(result, axis=0) / M
        tl.store(output_ptr + pid, mean_val)
    else:
        # Flatten case
        offsets = tl.arange(0, N * M)
        input_vals = tl.load(input_ptr + offsets)
        other_vals = tl.load(other_ptr + offsets)
        
        # Add scaled other to input
        result = input_vals + alpha * other_vals
        
        # Compute mean over all elements
        mean_val = tl.sum(result) / (N * M)
        tl.store(output_ptr, mean_val)

def add_mean(input, other, dim=None, alpha=1, keepdim=False, dtype=None, out=None):
    if dtype is not None:
        input = input.to(dtype)
        other = other.to(dtype)
    
    if isinstance(other, (int, float, complex)):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    
    input_broadcasted, other_broadcasted = torch.broadcast_tensors(input, other)
    
    if dim is None:
        N, M = input_broadcasted.numel(), 1
        HAS_DIM = False
        REDUCE_DIM = 0
    else:
        if isinstance(dim, int):
            dim = (dim,)
        input_broadcasted = input_broadcasted.permute(*dim, *range(input_broadcasted.dim()))
        N, M = input_broadcasted.shape[0], input_broadcasted.shape[1]
        HAS_DIM = True
        REDUCE_DIM = 1

    output_shape = input_broadcasted.shape if keepdim else tuple(d for i, d in enumerate(input_broadcasted.shape) if i not in dim)
    output = input.new_empty(output_shape) if out is None else out

    add_mean_kernel[(N,)](
        input_ptr=input_broadcasted,
        other_ptr=other_broadcasted,
        output_ptr=output,
        alpha=alpha,
        N=N,
        M=M,
        HAS_DIM=HAS_DIM,
        REDUCE_DIM=REDUCE_DIM
    )
    
    return output
