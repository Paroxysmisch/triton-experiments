import triton
import triton.language as tl

def sqrt(input, *, out=None):
    # Get the device type
    device_type = input.device.type

    # Create a new tensor for the output if none is provided
    if out is None:
        out = tl.zeros_like(input)

    # Launch the Triton kernel
    grid_size = (tl.cdiv(input.shape[0], 256),)
    block_size = (256,)
    sqrt_kernel[input.shape[0]](input.data_ptr(), out.data_ptr(), input.shape[0])

    return out
