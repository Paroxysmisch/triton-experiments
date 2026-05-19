import torch
import triton
import triton.language as tl

@triton.jit
def mean_kernel(
    input_ptr,
    output_ptr,
    reduce_size,
    num_elements,
    input_row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    input_row_ptr = input_ptr + row_idx * input_row_stride
    accumulator = tl.zeros((1,), tl.float32)  # Assumes input is cast to float32
    for offset in range(0, reduce_size, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < reduce_size
        elements = tl.load(input_row_ptr + cols, mask=mask, other=0.0)
        accumulator += tl.sum(elements, axis=0)
    mean_val = accumulator / num_elements
    tl.store(output_ptr + row_idx, mean_val)

def add_mean(input, other, dim=None, alpha=1, keepdim=False, dtype=None, out=None):
    # Cast input to dtype if specified
    if dtype is not None:
        input = input.to(dtype)
    # Scale other by alpha and add to input, broadcasting as needed
    sum_tensor = torch.add(input, other, alpha=alpha)
    # Handle reduction dimensions and num_elements calculation
    if dim is not None:
        reduce_dims = dim if isinstance(dim, tuple) else (dim,)
        for d in reduce_dims:
            if d < 0 or d >= sum_tensor.dim():
                raise ValueError(f"Dimension {d} out of range for tensor with {sum_tensor.dim()} dimensions")
        num_elements = 1
        for d in reduce_dims:
            num_elements *= sum_tensor.size(d)
        non_reduced_dims = [d for d in range(sum_tensor.dim()) if d not in reduce_dims]
        sum_permuted = sum_tensor.permute(*non_reduced_dims, *reduce_dims)
        A = sum_permuted.size(0) if len(non_reduced_dims) > 0 else 1
        B = num_elements
        sum_2d = sum_permuted.contiguous().view(A, B)
    else:
        num_elements = sum_tensor.numel()
        sum_2d = sum_tensor.view(1, -1)
        A, B = sum_2d.shape
    # Allocate output tensor
    output = torch.empty(A, dtype=sum_2d.dtype, device=sum_tensor.device)
    # Configure and launch Triton kernel
    BLOCK_SIZE = min(triton.next_power_of_2(B), 1024)
    grid = (A,)
    mean_kernel[grid](
        sum_2d,
        output,
        B,
        num_elements,
        sum_2d.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    # Reshape output to match keepdim and original dimensions
    if dim is not None:
        out_shape = list(sum_tensor.shape)
        for d in reduce_dims:
            out_shape[d] = 1 if keepdim else None
        out_shape = [s if s is not None else 1 for s in out_shape]
        output = output.reshape([size if idx not in reduce_dims or keepdim else 1 for idx, size in enumerate(sum_tensor.shape)])
        if not keepdim:
            output = output.squeeze(dim)
    else:
        if keepdim:
            output = output.view(*([1] * sum_tensor.dim()))
        else:
            output = output.squeeze()
    # Handle output tensor
    if out is not None:
        out.copy_(output)
        return out
    return output
