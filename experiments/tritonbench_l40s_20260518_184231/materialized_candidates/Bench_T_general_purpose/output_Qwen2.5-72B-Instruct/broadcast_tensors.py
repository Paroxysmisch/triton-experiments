import triton
import triton.language as tl

@triton.jit
def broadcast_kernel(
    output_ptr,  # Pointer to the output tensor
    input_ptrs,  # Pointers to the input tensors
    output_shape,  # Shape of the output tensor
    input_shapes,  # Shapes of the input tensors
    input_strides,  # Strides of the input tensors
    output_strides,  # Strides of the output tensor
    num_inputs,  # Number of input tensors
    num_elements,  # Total number of elements in the output tensor
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Iterate over the elements in the block
    for i in range(BLOCK_SIZE):
        if block_start + i < num_elements:
            output_index = offsets[i]
            output_coords = tl.zeros((len(output_shape),), dtype=tl.int32)
            input_coords = [tl.zeros((len(input_shapes[0]),), dtype=tl.int32) for _ in range(num_inputs)]

            # Convert flat index to coordinates
            for dim in range(len(output_shape) - 1, -1, -1):
                output_coords[dim] = output_index % output_shape[dim]
                output_index //= output_shape[dim]

            # Compute the coordinates for each input tensor
            for input_idx in range(num_inputs):
                for dim in range(len(input_shapes[input_idx])):
                    if input_shapes[input_idx][dim] == 1:
                        input_coords[input_idx][dim] = 0
                    else:
                        input_coords[input_idx][dim] = output_coords[dim]

            # Compute the linear index for each input tensor
            input_indices = [0] * num_inputs
            for input_idx in range(num_inputs):
                input_indices[input_idx] = sum(input_coords[input_idx][dim] * input_strides[input_idx][dim] for dim in range(len(input_shapes[input_idx])))

            # Load the values from the input tensors
            input_values = [tl.load(input_ptrs[input_idx] + input_indices[input_idx]) for input_idx in range(num_inputs)]

            # Write the value to the output tensor
            output_value = input_values[0]  # Assuming element-wise operation is just copying the first input
            tl.store(output_ptr + offsets[i], output_value)

import torch
import triton
import triton.language as tl

def broadcast_tensors(*tensors):
    # Check if all tensors are of the same type
    if not all(isinstance(t, torch.Tensor) for t in tensors):
        raise ValueError("All inputs must be tensors.")

    # Determine the output shape using broadcasting rules
    output_shape = []
    for dim in range(max(t.dim() for t in tensors)):
        max_size = 1
        for t in tensors:
            if dim < t.dim():
                max_size = max(max_size, t.size(-1 - dim))
        output_shape.insert(0, max_size)

    # Prepare the input and output tensors
    input_ptrs = [t.data_ptr() for t in tensors]
    input_shapes = [t.shape for t in tensors]
    input_strides = [t.stride() for t in tensors]
    output_tensor = torch.empty(output_shape, dtype=tensors[0].dtype, device=tensors[0].device)
    output_ptr = output_tensor.data_ptr()
    output_strides = output_tensor.stride()

    # Flatten the shapes and strides for the kernel
    flat_input_shapes = [item for shape in input_shapes for item in shape]
    flat_input_strides = [item for stride in input_strides for item in stride]
    flat_output_strides = [item for item in output_strides]

    # Launch the Triton kernel
    num_elements = output_tensor.numel()
    num_inputs = len(tensors)
    grid = (num_elements // 1024 + 1,)
    block = (1024,)
    broadcast_kernel[grid, block](
        output_ptr,
        input_ptrs,
        flat_input_shapes,
        flat_output_strides,
        num_inputs,
        num_elements,
        BLOCK_SIZE=1024
    )

    # Return the broadcasted tensors
    return [output_tensor] * len(tensors)

# Example usage
x = torch.arange(3).view(1, 3)
y = torch.arange(2).view(2, 1)
a, b = broadcast_tensors(x, y)
print(a.size())  # Should be torch.Size([2, 3])
print(a)  # Should be tensor([[0, 1, 2], [0, 1, 2]])

x = torch.arange(3).view(1, 3)
y = torch.arange(2).view(2, 1)
a, b = broadcast_tensors(x, y)
print(a.size())  # Should be torch.Size([2, 3])
print(a)  # Should be tensor([[0, 1, 2], [0, 1, 2]])
