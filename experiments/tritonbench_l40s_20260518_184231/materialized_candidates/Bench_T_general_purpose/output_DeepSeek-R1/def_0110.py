import torch
import triton
import triton.language as tl

@triton.jit
def _exp_mean_kernel_1d(
    input_ptr, output_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    exp_x = tl.exp(x)
    partial_sum = tl.sum(exp_x, axis=0)
    tl.atomic_add(output_ptr, partial_sum)

@triton.jit
def _exp_mean_kernel_2d(
    input_ptr, output_ptr,
    input_row_stride, output_row_stride,
    reduce_size,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    input_row_start = input_ptr + row_idx * input_row_stride
    output_row = output_ptr + row_idx * output_row_stride

    sum_exp = 0.0
    for offset in range(0, reduce_size, BLOCK_SIZE):
        col_offsets = offset + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < reduce_size
        x = tl.load(input_row_start + col_offsets, mask=mask, other=0.0)
        exp_x = tl.exp(x)
        sum_exp += tl.sum(exp_x, axis=0)
    mean = sum_exp / reduce_size
    tl.store(output_row, mean)

def exp_mean(input, dim=None, keepdim=False, dtype=None, out=None) -> torch.Tensor:
    if input.numel() == 0:
        raise RuntimeError("input tensor cannot be empty")
    
    input = input.contiguous()
    device = input.device
    input_dtype = input.dtype
    
    if dim is None:
        n_elements = input.numel()
        input_flat = input.view(-1)
        output_shape = [1] * input.dim() if keepdim else []
        output = torch.zeros((1,), dtype=input_dtype, device=device)
        if n_elements == 0:
            return output.reshape(output_shape)
        
        BLOCK_SIZE = 1024
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        _exp_mean_kernel_1d[grid](input_flat, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        output = (output / n_elements).to(dtype=dtype if dtype is not None else input_dtype)
        output = output.reshape(output_shape)
    else:
        reduce_dims = dim if isinstance(dim, (list, tuple)) else [dim]
        reduce_dims = sorted([d if d >= 0 else d + input.dim() for d in reduce_dims])
        reduce_dims = list(set(reduce_dims))
        
        for d in reduce_dims:
            if d < 0 or d >= input.dim():
                raise ValueError(f"Dimension {d} is out of range for input of dimension {input.dim()}")
        
        non_reduce_dims = [d for d in range(input.dim()) if d not in reduce_dims]
        non_reduced_size = 1
        for d in non_reduce_dims:
            non_reduced_size *= input.size(d)
        reduced_size = 1
        for d in reduce_dims:
            reduced_size *= input.size(d)
        
        input_flat = input.reshape(non_reduced_size, reduced_size).contiguous()
        output = torch.empty(non_reduced_size, 1, dtype=input_dtype, device=device)
        
        BLOCK_SIZE = 1024
        grid = (non_reduced_size,)
        _exp_mean_kernel_2d[grid](
            input_flat, output,
            input_flat.stride(0), output.stride(0),
            reduced_size,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        output_shape = list(input.shape)
        for d in reduce_dims:
            output_shape[d] = 1
        output = output.reshape(output_shape)
        if not keepdim:
            output = output.squeeze(dim)
    
    if dtype is not None:
        output = output.to(dtype=dtype)
    
    if out is not None:
        if not out.is_contiguous():
            raise RuntimeError("out tensor must be contiguous")
        out.copy_(output)
        return out
    else:
        return output
