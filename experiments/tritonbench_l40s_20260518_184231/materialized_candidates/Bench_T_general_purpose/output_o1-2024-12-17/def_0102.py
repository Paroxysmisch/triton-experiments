import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_mul_kernel(
    input_ptr, other_ptr, output_ptr,
    stride_input, stride_other, stride_output,
    dim_size, other_is_scalar,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * dim_size + tl.arange(0, BLOCK_SIZE)
    mask = offsets < ((pid + 1) * dim_size)
    # Load input values
    x = tl.where(
        mask,
        tl.load(input_ptr + offsets * stride_input),
        float("-inf")
    )
    # Compute max for numerical stability
    x_max = tl.max(x, axis=0)
    x = x - x_max
    x_exp = tl.exp(x)
    # Compute sum of exponentials
    s = tl.sum(x_exp, axis=0)
    softmax_val = x_exp / s

    # Multiply by "other"
    if other_is_scalar:
        scalar_val = tl.load(other_ptr)  # single value
        mul_val = softmax_val * scalar_val
    else:
        other_x = tl.where(
            mask,
            tl.load(other_ptr + offsets * stride_other),
            0.0
        )
        mul_val = softmax_val * other_x

    # Store result
    tl.store(output_ptr + offsets * stride_output, mul_val, mask=mask)


def softmax_mul(input, other, dim, dtype=None, out=None):
    """
    Applies the softmax function to the input tensor along the specified
    dimension, and then multiplies the softmaxed values by 'other'.

    Args:
        input (torch.Tensor): The input tensor to apply softmax on.
        other (torch.Tensor or Number): The tensor or number to multiply
                                        with the softmaxed values.
        dim (int): The dimension along which softmax will be computed.
        dtype (torch.dtype, optional): Desired dtype of returned tensor.
                                       If specified, the input tensor is
                                       cast to dtype before the operation.
                                       Default is None.
        out (torch.Tensor, optional): The output tensor.

    Returns:
        torch.Tensor: Result of softmax(input) multiplied by other.
    """
    if dtype is not None:
        input = input.to(dtype)

    # If out is None, create a new tensor for output
    if out is None:
        out = torch.empty_like(input, dtype=input.dtype, device=input.device)

    # Ensure 'dim' is in range
    dim = dim % input.ndim

    # Flatten the dimension to be computed
    # We'll launch a block for each "slice" along 'dim'.
    dim_size = input.shape[dim]
    slices = input.numel() // dim_size

    # Strides for the dimension of interest
    stride_input = input.stride(dim)
    stride_output = out.stride(dim)

    # Handle 'other', checking if it's a scalar or tensor
    other_is_scalar = not isinstance(other, torch.Tensor)
    if other_is_scalar:
        # Convert scalar to a tensor on the same device for Triton
        other_val = torch.tensor(float(other), dtype=input.dtype, device=input.device)
        other_ptr = other_val.data_ptr()
        stride_other = 0
    else:
        # Ensure shapes are compatible
        if other.numel() != input.numel():
            raise ValueError("Shapes of 'other' and 'input' must match if 'other' is a tensor.")
        other = other.to(input.device, dtype=input.dtype)
        stride_other = other.stride(dim)
        other_ptr = other.data_ptr()

    grid = (slices,)
    # Launch kernel
    _softmax_mul_kernel[grid](
        input.data_ptr(),
        other_ptr,
        out.data_ptr(),
        stride_input,
        stride_other,
        stride_output,
        dim_size,
        other_is_scalar,
        BLOCK_SIZE=dim_size  # each block handles the entire dim for that slice
    )
    return out
