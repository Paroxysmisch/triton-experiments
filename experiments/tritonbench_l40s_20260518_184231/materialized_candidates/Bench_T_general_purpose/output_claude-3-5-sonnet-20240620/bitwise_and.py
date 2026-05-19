import triton
import triton.language as tl

@triton.jit
def bitwise_and_kernel(input_ptr, other_ptr, output_ptr, n_elements):
    # Compute the bitwise AND for integral types or logical AND for boolean types
    pid = tl.program_id(0)
    start = pid * tl.num_warps() * tl.warp_size
    end = min(start + tl.num_warps() * tl.warp_size, n_elements)

    for i in range(start, end):
        input_val = tl.load(input_ptr + i)
        other_val = tl.load(other_ptr + i)
        result = input_val & other_val  # Bitwise AND operation
        tl.store(output_ptr + i, result)

def bitwise_and(input: tl.Tensor, other: tl.Tensor, *, out: tl.Tensor = None) -> tl.Tensor:
    # Ensure input tensors are of integral or boolean types
    if not (input.dtype in [tl.int32, tl.int64, tl.bool] and other.dtype in [tl.int32, tl.int64, tl.bool]):
        raise ValueError("Input tensors must be of integral or Boolean types.")

    # Create output tensor if not provided
    if out is None:
        out = tl.empty_like(input)

    # Launch the kernel
    n_elements = input.numel()
    grid = (triton.cdiv(n_elements, triton.warp_size),)
    bitwise_and_kernel[grid](input.data_ptr(), other.data_ptr(), out.data_ptr(), n_elements)

    return out
