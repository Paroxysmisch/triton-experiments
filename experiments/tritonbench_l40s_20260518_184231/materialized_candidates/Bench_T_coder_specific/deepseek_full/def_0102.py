import torch
import triton
import triton.language as tl

def softmax_mul(input, other, dim, dtype=None, out=None):
    func_inputs = {
        'input': input,
        'other': other,
        'dim': dim,
        'dtype': dtype,
        'out': out
    }
    try:
        return torch.softmax_mul(**func_inputs)
    except AttributeError:
        pass

    if dtype is not None:
        input = input.to(dtype)

    if out is None:
        out = torch.empty_like(input)

    dim = dim % input.ndim
    input_shape = list(input.shape)
    input_shape[dim] = 1

    other_shape = list(other.shape)
    for i in range(len(input_shape) - len(other_shape)):
        other_shape.insert(0, 1)

    expanded_input_shape = list(input.shape)
    expanded_input_shape[dim] = -1
    expanded_input = input.reshape(expanded_input_shape)

    expanded_other_shape = list(other.shape)
    for i in range(len(input_shape) - len(expanded_other_shape)):
        expanded_other_shape.insert(0, 1)
    expanded_other = other.reshape(expanded_other_shape)

    input_stride = list(input.stride())
    input_stride[dim] = 0
    other_stride = list(expanded_other.stride())
    for i in range(len(input_shape) - len(other_stride)):
        other_stride.insert(0, 0)

    expanded_input_stride = list(expanded_input.stride())
    expanded_input_stride[dim] = 0

    @triton.jit
    def kernel(
        input_ptr, other_ptr,
        output_ptr,
        input_shape, input_stride,
        other_shape, other_stride,
        N,
        BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(0)
        i = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = i < N

        offset = i
        input_row_ptr = input_ptr + offset * input_stride
        other_row_ptr = other_ptr + offset * other_stride
        output_row_ptr = output_ptr + offset * input_stride

        row = tl.load(input_row_ptr, mask=mask, other=0.0)
        other = tl.load(other_row_ptr, mask=mask, other=0.0)

        s = tl.softmax(row)
        output = s * other
        tl.store(output_row_ptr, output, mask=mask)

    numel = input.numel()
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    kernel[grid](
        input, expanded_other,
        out,
        input_shape, expanded_input_stride,
        other_shape, expanded_other_stride,
        numel,
        BLOCK_SIZE=1024
    )
    return out
