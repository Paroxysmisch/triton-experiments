import triton
import triton.language as tl

# Define the Triton kernel
@triton.jit
def broadcast_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    x_shape,
    y_shape,
    out_shape,
    stride_x,
    stride_y,
    stride_out,
    n_elements,
    block_size=256
):
    pid = tl.program_id(axis=0)
    coords = pid * block_size + tl.arange(0, block_size)

    # Compute the indices for x and y
    x_idx = coords // stride_x
    y_idx = coords // stride_y

    # Compute the linear index for the output
    out_idx = coords // stride_out

    # Broadcast x and y to match the output shape
    x_val = tl.load(x_ptr + x_idx, mask=(coords < n_elements))
    y_val = tl.load(y_ptr + y_idx, mask=(coords < n_elements))

    # Store the result in the output buffer
    tl.store(out_ptr + out_idx, x_val + y_val, mask=(coords < n_elements))

# Define the wrapper function
def broadcast_tensors(*tensors):
    import torch
    from typing import List

    # Get the shapes of the input tensors
    shapes = [tensor.shape for tensor in tensors]
    dtype = tensors[0].dtype

    # Determine the output shape through broadcasting
    out_shape = torch.broadcast_shapes(*shapes)

    # Calculate strides for each tensor
    strides = [tuple(t.stride()) for t in tensors]

    # Create the output tensor
    out_tensor = torch.empty(out_shape, dtype=dtype, device=tensors[0].device)

    # Prepare the grid and block sizes for Triton
    num_blocks = (out_tensor.numel() + 255) // 256
    block_size = 256

    # Launch the Triton kernel
    broadcast_kernel[grid=num_blocks, block=block_size](
        tensors[0].data_ptr(),
        tensors[1].data_ptr(),
        out_tensor.data_ptr(),
        tensors[0].shape,
        tensors[1].shape,
        out_shape,
        tensors[0].stride()[0],
        tensors[1].stride()[0],
        out_tensor.stride()[0],
        out_tensor.numel()
    )

    return [out_tensor] + list(tensors[2:])  # Return the broadcasted tensor and the original tensors

# Example usage
x = torch.tensor([[0], [1]], dtype=torch.float32)
y = torch.tensor([0, 1, 2], dtype=torch.float32)
a, b = broadcast_tensors(x, y)
print(a.size())  # Should print torch.Size([2, 3])
print(a)         # Should print tensor([[0., 1., 2.],
                  #                   [0., 1., 2.]])
