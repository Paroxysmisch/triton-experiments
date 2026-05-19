import triton
import triton.language as tl

# Define the Triton kernel
@triton.jit
def pow_func_scalar_tensor_kernel_rank_1(
    input_ptr,
    output_ptr,
    exponent,
    n_elements: tl.constexpr,
):
    # Get the thread's index
    idx = tl.program_id(axis=0)
    
    # Check if the thread index is within the bounds of the input array
    if idx < n_elements:
        # Compute the power of the element and store it in the output array
        tl.store(output_ptr + idx, tl.pow(tl.load(input_ptr + idx), exponent))

# Define the Triton wrapper
def pow_func_scalar_tensor_wrapper_rank_1(input, output, exponent):
    # Determine the grid size
    n_elements = output.numel()
    grid = lambda meta: triton.jit.tagged.Grid(meta, num_warps_x=16, num_warps_y=8)
    
    # Determine the number of warps
    n_warps = triton.cuda.num_warps_per_block()
    if n_warps <= 4:
        num_warps = 8
    elif n_warps <= 8:
        num_warps = 16
    else:
        num_warps = 32
    
    # Launch the Triton kernel
    pow_func_scalar_tensor_kernel_rank_1[(num_warps, )](
        input_ptr=input.data_ptr(),
        output_ptr=output.data_ptr(),
        exponent=exponent,
        n_elements=n_elements,
        num_warps=num_warps,
        grid=grid,
    )
