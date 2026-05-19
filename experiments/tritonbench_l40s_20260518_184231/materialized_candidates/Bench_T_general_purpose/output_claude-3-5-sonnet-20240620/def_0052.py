import triton
import triton.language as tl

@triton.jit
def sum_std_kernel(input_ptr, output_ptr, dim, N, correction, BLOCK_SIZE: tl.constexpr):
    # Initialize thread index
    pid = tl.program_id(0)
    # Calculate the start and end indices for this block
    start = pid * BLOCK_SIZE
    end = tl.min(start + BLOCK_SIZE, N)

    # Initialize sum and count
    sum_val = 0.0
    count = 0

    # Compute the sum
    for i in range(start, end):
        sum_val += input_ptr[i]
        count += 1

    # Store the result in output
    output_ptr[pid] = sum_val

    # Synchronize threads
    tl.barrier()

    # Calculate standard deviation if count > 1
    if count > 1:
        mean = sum_val / count
        variance = 0.0
        for i in range(start, end):
            variance += (input_ptr[i] - mean) ** 2
        variance /= (count - correction)
        std_dev = tl.sqrt(variance)
        output_ptr[pid + 1] = std_dev

def sum_std(input, dim=None, keepdim=False, dtype=None, correction=1, out=None) -> Tensor:
    # Ensure input is a tensor
    if not isinstance(input, Tensor):
        raise TypeError("Input must be a Tensor")

    # Determine the dimensions to reduce
    if dim is None:
        dim = tuple(range(input.dim()))
    elif isinstance(dim, int):
        dim = (dim,)
    
    # Prepare output tensor
    output_shape = list(input.shape)
    for d in dim:
        output_shape[d] = 1 if not keepdim else output_shape[d]
    output = torch.empty(output_shape, dtype=dtype) if out is None else out

    # Launch the Triton kernel
    N = input.numel()
    sum_std_kernel[(N + BLOCK_SIZE - 1) // BLOCK_SIZE](input.data_ptr(), output.data_ptr(), dim, N, correction)

    return output
