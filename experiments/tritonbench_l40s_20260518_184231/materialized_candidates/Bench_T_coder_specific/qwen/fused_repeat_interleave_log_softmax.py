import torch
import triton
import triton.language as tl

def fused_repeat_interleave_log_softmax(input, repeats, dim=None, *, output_size=None, dtype=None, out=None):
    # Flatten the input if dim is None
    if dim is None:
        input = input.view(-1)
        repeats = repeats.view(-1)
        dim = 0

    # Get the shape of the input tensor
    input_shape = list(input.shape)
    repeats_shape = list(repeats.shape)

    # Calculate the total number of elements in the output tensor
    if output_size is None:
        output_size = input.numel() * repeats.max().item()
    else:
        assert output_size >= input.numel() * repeats.max().item(), "Output size must be at least the product of input size and max repeat count"

    # Allocate memory for the output tensor
    if out is None:
        out = torch.empty(output_size, dtype=dtype or input.dtype, device=input.device)

    # Prepare the Triton kernel arguments
    x_ptr = input.contiguous().data_ptr()
    repeats_ptr = repeats.contiguous().data_ptr()
    y_ptr = out.data_ptr()
    repeats_count = repeats.size(0)

    # Launch the Triton kernel
    fused_repeat_interleave_log_softmax_kernel[triton.cdiv(out.numel(), BLOCK_SIZE), BLOCK_SIZE](
        x_ptr, repeats_ptr, y_ptr, output_size, repeats_count
    )

    return out
