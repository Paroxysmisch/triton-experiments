import torch
import triton
import triton.language as tl

@triton.jit
def mean_kernel(output_ptr, input_ptr, input_shape, input_strides, output_strides, reduce_dims, num_reduce_dims: tl.constexpr,
                BLOCK_SIZE: tl.constexpr, num_stages: tl.constexpr):
    # Calculate the total number of elements in the input tensor
    total_elements = 1
    for dim in range(len(input_shape)):
        total_elements *= input_shape[dim]

    # Calculate the starting index for the program
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    start_idx = pid * BLOCK_SIZE
    end_idx = min(start_idx + BLOCK_SIZE, total_elements)

    # Initialize the sum and count for the mean calculation
    sum_val = tl.zeros((1,), dtype=tl.float32)
    count = 0

    # Iterate over the elements in the block
    for idx in range(start_idx, end_idx):
        # Calculate the multi-dimensional index from the linear index
        multi_idx = [0] * len(input_shape)
        remaining = idx
        for dim in range(len(input_shape) - 1, -1, -1):
            multi_idx[dim] = remaining % input_shape[dim]
            remaining //= input_shape[dim]

        # Check if the current index is in the reduction dimensions
        in_reduce_dim = False
        for reduce_dim in range(num_reduce_dims):
            if multi_idx[reduce_dims[reduce_dim]] == 0:
                in_reduce_dim = True
                break

        if in_reduce_dim:
            continue

        # Load the element from the input tensor
        input_offset = 0
        for dim in range(len(input_shape)):
            input_offset += multi_idx[dim] * input_strides[dim]
        element = tl.load(input_ptr + input_offset)

        # Accumulate the sum and count
        sum_val += element
        count += 1

    # Calculate the mean
    mean_val = sum_val / count

    # Calculate the output index
    output_idx = 0
    for dim in range(len(input_shape)):
        if dim not in reduce_dims:
            output_idx += multi_idx[dim] * output_strides[dim]

    # Store the result in the output tensor
    tl.store(output_ptr + output_idx, mean_val)

def mean(input, dim, keepdim=False, dtype=None, out=None):
    if dtype is not None:
        input = input.to(dtype)

    input_shape = input.shape
    input_strides = input.stride()
    input_numel = input.numel()

    if isinstance(dim, int):
        dim = [dim]
    reduce_dims = dim
    num_reduce_dims = len(reduce_dims)

    # Calculate the output shape
    output_shape = list(input_shape)
    for d in reduce_dims:
        if not keepdim:
            output_shape[d] = 1
    output_shape = [s for s in output_shape if s != 1]

    # Allocate the output tensor
    if out is None:
        out = torch.empty(output_shape, device=input.device, dtype=input.dtype)
    else:
        assert out.shape == output_shape, "Output tensor shape mismatch"
        assert out.device == input.device, "Output tensor device mismatch"
        assert out.dtype == input.dtype, "Output tensor dtype mismatch"

    # Calculate the output strides
    output_strides = [0] * len(output_shape)
    stride = 1
    for i in range(len(output_shape) - 1, -1, -1):
        output_strides[i] = stride
        stride *= output_shape[i]

    # Determine the block size
    BLOCK_SIZE = 1024

    # Number of software pipelining stages
    num_stages = 4

    # Launch the kernel
    grid = (input_numel // BLOCK_SIZE + (input_numel % BLOCK_SIZE > 0),)
    mean_kernel[grid](
        out.data_ptr(),
        input.data_ptr(),
        input_shape,
        input_strides,
        output_strides,
        reduce_dims,
        num_reduce_dims,
        BLOCK_SIZE,
        num_stages
    )

    return out

# Example usage
input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device='cuda')
output_tensor = mean(input_tensor, dim=1, keepdim=True)
print(output_tensor)
