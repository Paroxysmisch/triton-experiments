import torch
import triton
import triton.language as tl

@triton.jit
def trunc_kernel(
    input_ptr: tl.tensor, output_ptr: tl.tensor,
    M: tl.constexpr, N: tl.constexpr,
    input_stride_x, input_stride_y,
):
    """Kernel to truncate the values in the input tensor and store them in the output tensor.
    
    For integer inputs, return the input tensor unchanged, following the array-api convention.

    Args:
        input_ptr: Pointer to the input tensor.
        output_ptr: Pointer to the output tensor.
        M: Number of rows in the input tensor.
        N: Number of columns in the input tensor.
        input_stride_x: Stride of the input tensor along the row dimension.
        input_stride_y: Stride of the input tensor along the column dimension.
    """
    program_id = tl.program_id(axis=0)
    
    # Loading the input tensor in blocks
    input_block_ptr = tl.make_block_ptr(
        base=input_ptr,
        shape=(M, N),
        strides=(input_stride_x, input_stride_y),
        offsets=(program_id, 0),
        block_shape=(1, N),
        order=(1, 0),
    )
    
    # Output tensor pointer (1D result)
    output_block_ptr = tl.make_block_ptr(
        base=output_ptr,
        shape=(M, ),
        strides=(1, ),
        offsets=(program_id, ),
        block_shape=(1, ),
        order=(0, ),
    )
    
    # Load the input values
    input_block = tl.load(input_block_ptr)
    
    # Apply truncation (only for non-integer inputs)
    output_block = tl.trunc(input_block)
    
    # Store the results in the output tensor
    tl.store(output_block_ptr, output_block)


def trunc(input: torch.Tensor, *, out=None) -> torch.Tensor:
    """Returns a new tensor with the truncated integer values of the elements of the input tensor.
    If the input tensor contains integer values, it returns a copy of the input tensor.
    
    Args:
        input: Tensor containing the input values.
        out (optional): Output tensor to store results. If not provided, a new tensor is returned.

    Returns:
        A tensor containing the truncated values.
    """
    if input.is_integer():
        # For integer inputs, return a copy of the input tensor
        return input.clone()

    M, N = input.shape
    device = input.device
    dtype = input.dtype

    # Prepare the output tensor (if not provided)
    if out is None:
        out = torch.empty_like(input, device=device, dtype=dtype)

    # Launch the Triton kernel
    trunc_kernel[(M, )](
        input_ptr=input, output_ptr=out,
        M=M, N=N,
        input_stride_x=input.stride(0), input_stride_y=input.stride(1),
    )

    return out
