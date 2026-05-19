import torch
import triton
import triton.language as tl

@triton.jit
def add_mean_fwd_kernel(
    x,
    z,
    D: tl.constexpr,
    B: tl.constexpr,
    ND: tl.constexpr,
):
    i_n = tl.program_id(0)
    i_d = tl.program_id(1)

    o_d = i_d * B + tl.arange(0, B)
    mask = o_d < D

    x_ptr = x + i_n * D
    x_vals = tl.load(x_ptr + o_d, mask=mask, other=0.0)

    sum_x = tl.sum(x_vals, axis=0)
    tl.store(z + (i_n * ND + i_d), sum_x)


def add_mean(input, other, dim=None, alpha=1, keepdim=False, dtype=None, out=None) -> torch.Tensor:
    # Cast input to dtype if provided
    if dtype is not None:
        input = input.to(dtype)
    # Convert other to a tensor if it's a number and cast to input's dtype
    other_tensor = other if isinstance(other, torch.Tensor) else torch.tensor(other, device=input.device)
    other_tensor = other_tensor.to(input.dtype)
    # Scale other by alpha
    other_tensor = other_tensor * alpha
    # Broadcast input and other to a common shape
    try:
        input, other_tensor = torch.broadcast_tensors(input, other_tensor)
    except RuntimeError as e:
        raise RuntimeError(f"input and other could not be broadcast together. input shape: {input.shape}, other shape: {other_tensor.shape}") from e
    sum_tensor = input + other_tensor

    # Determine reduction dimensions
    if dim is None:
        dim = tuple(range(sum_tensor.dim()))
    elif isinstance(dim, int):
        dim = (dim,)
    else:
        dim = tuple(sorted(dim))
    # Check for empty reduction
    num_elements = 1
    for d in dim:
        num_elements *= sum_tensor.size(d)
    if num_elements == 0:
        raise RuntimeError("cannot reduce over zero elements")
    # Short-circuit if no reduction needed
    if num_elements == 1:
        result = sum_tensor
        if keepdim:
            new_shape = [size if i not in dim else 1 for i, size in enumerate(sum_tensor.shape)]
            result = result.reshape(new_shape)
        else:
            result = result.squeeze(dim)
        if out is not None:
            out.copy_(result)
        return result

    # Reshape sum_tensor to (N, D) where N is product of non-reduction dims, D product of reduction dims
    non_reduce_dims = [d for d in range(sum_tensor.dim()) if d not in dim]
    permute_dims = non_reduce_dims + list(dim)
    sum_permuted = sum_tensor.permute(permute_dims)
    N = torch.Size(sum_permuted.shape[:len(non_reduce_dims)]).numel()
    D = torch.Size(sum_permuted.shape[len(non_reduce_dims):]).numel()
    sum_reshaped = sum_permuted.reshape(N, D)

    # Configure kernel parameters
    B_size = min(triton.next_power_of_2(D), 1024)  # Adjust block size as needed
    ND_blocks = triton.cdiv(D, B_size)

    # Allocate partial sums tensor
    partial_sums = torch.empty((N, ND_blocks), dtype=sum_reshaped.dtype, device=sum_reshaped.device)

    # Launch kernel
    add_mean_fwd_kernel[(N, ND_blocks)](
        sum_reshaped,
        partial_sums,
        D=D,
        B=B_size,
        ND=ND_blocks,
    )

    # Sum the partial sums along the last dimension to get total sum
    total_sums = partial_sums.sum(dim=-1)

    # Compute mean
    means = total_sums / num_elements

    # Reshape to output shape
    output_shape = list(sum_tensor.shape)
    for d in sorted(dim, reverse=True):
        if keepdim:
            output_shape[d] = 1
        else:
            del output_shape[d]
    means = means.reshape(output_shape)

    # Handle output tensor
    if out is not None:
        out.copy_(means)
    return means
