import torch
import triton
import triton.language as tl

device = 'cuda:0'

@triton.jit
def permute_kernel(
    input_ptr,
    output_ptr,
    input_shape,
    output_shape,
    input_strides,
    output_strides,
    num_elements: tl.constexpr,
    num_dims: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_size = tl.num_programs(axis=0)
    start = pid * num_elements // block_size
    end = (pid + 1) * num_elements // block_size

    for idx in range(start, end):
        input_idx = [0] * num_dims
        output_idx = [0] * num_dims
        temp_idx = idx

        # Calculate the output index
        for i in range(num_dims - 1, -1, -1):
            output_idx[i] = temp_idx // output_strides[i]
            temp_idx = temp_idx % output_strides[i]

        # Calculate the input index
        for i in range(num_dims):
            input_idx[i] = output_idx[i]

        # Compute the linear index for input and output
        input_linear_idx = 0
        output_linear_idx = 0
        for i in range(num_dims):
            input_linear_idx += input_idx[i] * input_strides[i]
            output_linear_idx += output_idx[i] * output_strides[i]

        # Load and store the value
        value = tl.load(input_ptr + input_linear_idx)
        tl.store(output_ptr + output_linear_idx, value)

def torch_permute_copy(input: torch.Tensor, dims: tuple) -> torch.Tensor:
    # Validate the input and dims
    assert input.is_cuda, 'Input tensor must be on GPU'
    assert len(dims) == input.dim(), 'The number of dimensions in dims must match the input tensor'
    assert sorted(dims) == list(range(input.dim())), 'dims must be a permutation of input tensor dimensions'

    # Prepare the output tensor
    output_shape = [input.shape[d] for d in dims]
    output = torch.empty(size=output_shape, dtype=input.dtype, device=device)

    # Calculate strides for input and output
    input_strides = [input.stride(i) for i in range(input.dim())]
    output_strides = [output.stride(i) for i in range(output.dim())]

    # Number of elements
    num_elements = input.numel()

    # Launch the Triton kernel
    grid = (triton.cdiv(num_elements, 1024),)
    permute_kernel[grid](
        input_ptr=input, output_ptr=output,
        input_shape=input.shape, output_shape=output.shape,
        input_strides=input_strides, output_strides=output_strides,
        num_elements=num_elements, num_dims=input.dim()
    )

    return output
